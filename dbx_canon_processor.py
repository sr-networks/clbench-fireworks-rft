"""dbx rollout processor: the official CLBench database_exploration task with the notepad
memory scaffold — the RFT-side twin of dbx_engine.run_episode (same episode construction,
same evidence bookkeeping, same QSTAT emission), ported onto the battle-tested spectrum
canon processor skeleton (unkillable turns, robust LLM calls, think-stripping).

  - Task content: products_v{variant}.db + questions_pool_v{variant}.json via
    dbx_engine.make_episode; row_id "dbx-v{variant}-s{seed}-{i}" selects both.
  - Memory mechanism: context WIPED between questions (payload = [system] + current
    question's turns only); the notepad (notepad_update field of act) is the sole carrier.
  - Reward inputs: one QSTAT line per question in the ANSWER's tool message —
      QSTAT pos=<k> grp=<g> correct=<0|1> nq=<n> ev=<result|notepad|prophecy|none>
    parsed by dbx_reward.compute_dbx_reward. ev is the anti-answer-baking evidence tag
    (see dbx_engine docstring).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import List

import litellm
litellm.drop_params = True

from eval_protocol.mcp.execution.policy import LiteLLMPolicy
from eval_protocol.models import EvaluationRow, Message, Status
from eval_protocol.types.types import TerminationReason
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.pytest.utils import normalize_fireworks_model_for_litellm

from dbx_engine import (make_episode, notepad_block, TOOLS, NOTEPAD_MAX,
                        _num_tokens, _val_in, _scrub)
from src.interface import Response  # type: ignore
from src.tasks.database_exploration.task import DatabaseAction  # type: ignore

N_QUESTIONS = 15
BUDGET = 8       # 6 was too tight: probe showed honest-but-naive discovery costs ~7,
                 # 10/15 questions died budget-exceeded; one-shot floor stays 0.000
MAX_INNER_CALLS = BUDGET + 5     # queries + answer + malformed-response retries
ROW_RE = re.compile(r"dbx-v(\d+)-s(\d+)")
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
print("[dbx] processor v1 (unkillable questions; notepad-only persistence)", flush=True)


class DbxCanonRolloutProcessor(RolloutProcessor):
    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        sem = config.semaphore
        cp = normalize_fireworks_model_for_litellm(config.completion_params) or {}
        for row in rows:
            row.input_metadata.completion_params = cp
        model_id = str(cp.get("model") or "")
        temperature = float(cp.get("temperature", 1.2))
        max_tokens = int(cp.get("max_tokens", 4096))

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            t0 = time.perf_counter()
            policy = LiteLLMPolicy(model_id=model_id, temperature=temperature, max_tokens=max_tokens)
            m = ROW_RE.search(row.input_metadata.row_id or "")
            variant, seed = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
            task, picked = make_episode(seed=seed, variant=variant,
                                        n_questions=N_QUESTIONS, budget=BUDGET)
            msgs: List[Message] = list(row.messages)      # [system] from the dataset
            row.messages = msgs
            notepad = ""
            seen_tokens: set = set()                      # numeric tokens observed in ANY result/feedback
            seen_text: list = []
            done = False
            q = task.build_current_query()
            for _qi in range(N_QUESTIONS):
                pos = q.metadata["question_num"]
                grp = picked[pos - 1]["group"]
                notepad_start = notepad                   # what the model actually READ
                prior_tokens = set(seen_tokens)           # provenance snapshot BEFORE this question
                prior_text = "\n".join(seen_text)
                last_obs = ""
                nq = 0
                msgs.append(Message(role="user", content=q.prompt + notepad_block(notepad)))
                q_start = len(msgs) - 1
                answered = False
                for _ in range(MAX_INNER_CALLS):
                    # context wipe: the model sees only [system] + the CURRENT question's turns
                    payload = []
                    for msg in [msgs[0]] + msgs[q_start:]:
                        d = msg.model_dump()
                        if d.get("role") == "assistant" and d.get("content"):
                            d["content"] = THINK.sub("", d["content"]).strip()
                        payload.append(d)
                    am = None
                    for attempt in range(3):
                        try:
                            resp = await policy._make_llm_call(messages=payload, tools=TOOLS)
                            choices = (resp or {}).get("choices") or []
                            if choices:
                                am = choices[0]["message"]
                                break
                        except Exception:
                            pass
                        await asyncio.sleep(20 * (attempt + 1))
                    if am is None:
                        break                             # -> force-advance below
                    try:
                        tcs = [tc if isinstance(tc, dict) else tc.model_dump() for tc in (am.get("tool_calls") or [])]
                        msgs.append(Message(role="assistant", content=am.get("content") or "", tool_calls=tcs or None))
                    except Exception:
                        tcs = []
                    if not tcs:
                        break                             # no tool call -> force-advance
                    for tc in tcs:
                        try:
                            fn = tc.get("function") or {}
                            if fn.get("name") != "act" or answered:
                                msgs.append(Message(role="tool",
                                                    content="(unknown tool)" if fn.get("name") != "act"
                                                    else "(question already answered)",
                                                    tool_call_id=tc.get("id")))
                                continue
                            try:
                                args = json.loads(fn.get("arguments") or "{}")
                            except Exception:
                                args = {}
                            if not isinstance(args, dict):
                                args = {}
                            nu = args.get("notepad_update")
                            if isinstance(nu, str) and nu.strip():
                                notepad = nu[:NOTEPAD_MAX]
                            action = "ANSWER" if args.get("action") == "ANSWER" else "QUERY"
                            content = str(args.get("content", ""))
                            sr = task.step(Response(action=DatabaseAction(action=action, content=content), metadata={}))
                            obs = sr.observation.content
                            if action == "QUERY" and not sr.observation.instance_complete:
                                nq += 1
                                last_obs = obs
                                seen_tokens |= _num_tokens(obs)
                                seen_text.append(obs)
                                msgs.append(Message(role="tool", content=_scrub(obs), tool_call_id=tc.get("id")))
                                continue
                            # instance complete: ANSWER, or budget exceeded during QUERY
                            oc = getattr(sr, "instance_outcome", None)
                            correct = bool(getattr(oc, "success", False))
                            used = task._question_history[-1]["num_queries"]
                            in_result = _val_in(content, _num_tokens(last_obs), last_obs)
                            in_np = _val_in(content, _num_tokens(notepad_start), notepad_start)
                            has_prov = in_np and _val_in(content, prior_tokens, prior_text)
                            ev = ("result" if in_result else
                                  "notepad" if has_prov else
                                  "prophecy" if in_np else "none")
                            seen_tokens |= _num_tokens(obs)   # feedback (incl. revealed GT) is observed too
                            seen_text.append(obs)
                            msgs.append(Message(
                                role="tool",
                                content=(f"QSTAT pos={pos} grp={grp} correct={int(correct)} "
                                         f"nq={used} ev={ev}\n{_scrub(obs)}"),
                                tool_call_id=tc.get("id")))
                            answered = True
                            done = bool(sr.done)
                            nxt = getattr(sr, "next_query", None)
                            if nxt is not None:
                                q = nxt
                        except Exception:
                            msgs.append(Message(role="tool", content="(turn skipped)",
                                                tool_call_id=tc.get("id") if isinstance(tc, dict) else None))
                    if answered:
                        break
                if not answered:                          # force-advance with an empty answer
                    try:
                        sr = task.step(Response(action=DatabaseAction(action="ANSWER", content=""), metadata={}))
                        used = task._question_history[-1]["num_queries"]
                        msgs.append(Message(role="user",
                                            content=(f"QSTAT pos={pos} grp={grp} correct=0 "
                                                     f"nq={used} ev=none\n(no valid act call; advanced)")))
                        done = bool(sr.done)
                        nxt = getattr(sr, "next_query", None)
                        if nxt is not None:
                            q = nxt
                    except Exception:
                        break                             # end gracefully; questions-so-far still score
                if done:
                    break

            row.execution_metadata.rollout_duration_seconds = time.perf_counter() - t0
            row.rollout_status = Status.rollout_finished(termination_reason=TerminationReason.CONTROL_PLANE_SIGNAL)
            return row

        async def _wrap(r: EvaluationRow) -> EvaluationRow:
            async with sem:
                try:
                    return await process_row(r)
                except Exception as e:
                    r.rollout_status = Status.rollout_error(str(e)[:300])
                    raise

        return [asyncio.create_task(_wrap(r)) for r in rows]
