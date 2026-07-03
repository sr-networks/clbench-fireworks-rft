"""BENCH-PURE (1:1) rollout processor: the official CLBench blind_spectrum_monitoring task, unmodified.

  - Task content: the OFFICIAL variants (canonical fixed 13-channel band, official Jinja-rendered scan
    prompts), 30 instances per rollout exactly as in the default schedule stages. One dataset row = one
    variant series; the row seed only drives the activity/noise draws (the layout is canonical by design).
  - Response format: NATIVE — the model reports center_freqs AND bandwidths (no fixed-width simplification,
    no report cap beyond a sanity limit).
  - Metric/reward: the BENCH'S OWN available-spectrum IoU (task.instance_outcome.reward), emitted per scan
    as SCAN_AVAIL for the reward fn (occupied-IoU is emitted too, as a diagnostic only).
  - Memory mechanism: the bench's own — FULL conversation history (ICL). No windowing, no echo scaffold,
    no notepad. The ONLY experimental manipulation lives in the dataset's system prompt (nudge vs neutral).
  - Context hygiene: <think>...</think> blocks are stripped from PAST assistant turns (standard Qwen3
    practice); otherwise 30 thinking turns overflow the context. The current turn still reasons freely.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List

import litellm
litellm.drop_params = True

from eval_protocol.mcp.execution.policy import LiteLLMPolicy
from eval_protocol.models import EvaluationRow, Message, Status
from eval_protocol.types.types import TerminationReason
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.pytest.utils import normalize_fireworks_model_for_litellm

from spectrum_adapter import band_seed  # per-row deterministic seed (activity/noise; layout is canonical)
from bench_eval import load_default_schedule, resolved_gt, occ_iou  # official variants + scoring helpers
from src.interface import Response  # type: ignore
from src.tasks.blind_spectrum_monitoring.task import ScanReport, Transmitter  # type: ignore

MAX_INNER_CALLS = 3
SANITY_MAX_REPORT = 40           # generous; the canonical band has 13 channels
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

# NOTEPAD mode (bench icl_notepad semantics; env var set by the test entry BEFORE import): context CLEARED
# between instances; a notepad is shown with every scan and updated via the optional notepad_update field of
# the report — the notepad is the ONLY cross-instance carrier (memory in the strict sense). Default (off) =
# ICL mode (full conversation history).
NOTEPAD_MODE = os.environ.get("SPECTRUM_CANON_NOTEPAD") == "1"
NOTEPAD_MAX = 4000
print(f"[canon] processor v5 (unkillable scans; mode={'NOTEPAD' if NOTEPAD_MODE else 'ICL'})", flush=True)

_PROPS: Dict[str, Any] = {
    "center_freqs": {"type": "array", "items": {"type": "number"}},
    "bandwidths": {"type": "array", "items": {"type": "number"}},
}
if NOTEPAD_MODE:
    _PROPS["notepad_update"] = {
        "type": "string",
        "description": ("Optional: replace your notepad with this text (it persists to the next scan; "
                        "omit to keep the current notepad unchanged)."),
    }
TOOLS: List[Dict[str, Any]] = [{"type": "function", "function": {
    "name": "submit_report",
    "description": ("Submit your occupancy report for the CURRENT scan. center_freqs: the center frequency "
                    "(MHz) of every occupied region; bandwidths: the width (MHz) of each region, same order."),
    "parameters": {"type": "object", "properties": _PROPS, "required": ["center_freqs", "bandwidths"]},
}}]

_VARIANTS = {st["variant"]: st["kwargs"] for st in load_default_schedule()}


def _variant_for(row_id: str) -> str:
    for name in _VARIANTS:
        if name in (row_id or ""):
            return name
    return "five_ch_wide"


class SpectrumCanonRolloutProcessor(RolloutProcessor):
    """Official-task rollout: user turns carry the task's own scan prompts; full history; native scoring."""

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
            variant = _variant_for(row.input_metadata.row_id or "")
            kwargs = dict(_VARIANTS[variant])
            kwargs["seed"] = band_seed(row.input_metadata.row_id) % (2 ** 31 - 1)
            from src.registry import get_task_class  # type: ignore
            task = get_task_class("blind_spectrum_monitoring")(**kwargs)
            task.build_canonical_run_state()
            gt = resolved_gt(task, float(kwargs.get("W", 15.0)), float(kwargs.get("G", 9.0)))
            query = task.build_current_query()

            msgs: List[Message] = list(row.messages)          # [system] from the dataset (the ONLY knob)
            row.messages = msgs
            num_instances = int(kwargs.get("num_instances", 30))
            notepad = ""
            done = False
            for _scan in range(num_instances):
                content = query.prompt
                if NOTEPAD_MODE:
                    content += ("\n=== YOUR NOTEPAD ===\n"
                                + (notepad if notepad else "(empty)")
                                + "\n(You cannot see earlier scans; the notepad above is the only thing that "
                                  "persists. Update it via the notepad_update field of submit_report.)\n")
                msgs.append(Message(role="user", content=content))
                scan_user_idx = len(msgs) - 1
                submitted = False
                for _ in range(MAX_INNER_CALLS):
                    if NOTEPAD_MODE:
                        # bench icl_notepad default: context CLEARED between instances — the model sees only
                        # [system] + the current instance's turns; the notepad is the sole carrier.
                        src = [msgs[0]] + msgs[scan_user_idx:]
                    else:
                        src = msgs                            # ICL mode: FULL history
                    payload = []
                    for m in src:                             # strip think from PAST assistant turns
                        d = m.model_dump()
                        if d.get("role") == "assistant" and d.get("content"):
                            d["content"] = THINK.sub("", d["content"]).strip()
                        payload.append(d)
                    # Robust call: litellm already retries 8x internally; on top of that, tolerate transient
                    # deployment unhealthiness (cold start / "no healthy upstream") with backoff, and treat a
                    # still-failed call as a skipped turn instead of killing the rollout (ejuyuo2l died from
                    # resp["choices"][0] IndexError on failed calls -> 36% row errors -> job abort).
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
                        break  # -> force-advance below with an empty report
                    # UNKILLABLE: any malformed response / weird tool call / task hiccup degrades THIS turn,
                    # never the rollout (the neutral-prompt jobs died from response-dependent exceptions that
                    # escaped narrower guards — >20% dead rows aborts the whole job).
                    try:
                        tcs = [tc if isinstance(tc, dict) else tc.model_dump() for tc in (am.get("tool_calls") or [])]
                        msgs.append(Message(role="assistant", content=am.get("content") or "", tool_calls=tcs or None))
                    except Exception:
                        tcs = []
                    if not tcs:
                        break
                    for tc in tcs:
                        try:
                            fn = tc.get("function") or {}
                            if fn.get("name") != "submit_report":
                                msgs.append(Message(role="tool", content="(unknown tool)", tool_call_id=tc.get("id")))
                                continue
                            try:
                                args = json.loads(fn.get("arguments") or "{}")
                            except Exception:
                                args = {}
                            if not isinstance(args, dict):
                                args = {}
                            if NOTEPAD_MODE:
                                nu = args.get("notepad_update")
                                if isinstance(nu, str) and nu.strip():
                                    notepad = nu[:NOTEPAD_MAX]
                            cfs, bws = [], []
                            for x in (args.get("center_freqs") or [])[:SANITY_MAX_REPORT]:
                                try:
                                    cfs.append(float(x))
                                except Exception:
                                    pass
                            for x in (args.get("bandwidths") or [])[:SANITY_MAX_REPORT]:
                                try:
                                    bws.append(float(x))
                                except Exception:
                                    pass
                            bws = bws[:len(cfs)] + [8.0] * max(0, len(cfs) - len(bws))
                            txs = [Transmitter(center_freq=c, bandwidth=b, currently_active=True, estimated_power=-30.0)
                                   for c, b in zip(cfs, bws)]
                            sr = task.step(Response(action=ScanReport(transmitters=txs), metadata={}))
                            oc = getattr(sr, "instance_outcome", None)
                            avail = float(getattr(oc, "reward", 0.0) or 0.0)   # THE bench metric
                            try:
                                occ = occ_iou(cfs, bws, gt)                    # diagnostic only
                            except Exception:
                                occ = 0.0
                            msgs.append(Message(role="tool",
                                                content=f"ok\nSCAN_AVAIL: {avail:.4f} SCAN_OCC: {occ:.4f}",
                                                tool_call_id=tc.get("id")))
                            submitted = True
                            done = bool(sr.done)
                            nq = getattr(sr, "next_query", None)
                            if nq is not None:
                                query = nq
                        except Exception:
                            msgs.append(Message(role="tool", content="(turn skipped)", tool_call_id=tc.get("id") if isinstance(tc, dict) else None))
                    if submitted:
                        break
                if not submitted:                              # force-advance with an empty report
                    try:
                        sr = task.step(Response(action=ScanReport(transmitters=[]), metadata={}))
                        done = bool(sr.done)
                        nq = getattr(sr, "next_query", None)
                        if nq is not None:
                            query = nq
                    except Exception:
                        break                                  # end the rollout gracefully; scans-so-far still score
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
