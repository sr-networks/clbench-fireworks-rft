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

TOOLS: List[Dict[str, Any]] = [{"type": "function", "function": {
    "name": "submit_report",
    "description": ("Submit your occupancy report for the CURRENT scan. center_freqs: the center frequency "
                    "(MHz) of every occupied region; bandwidths: the width (MHz) of each region, same order."),
    "parameters": {"type": "object", "properties": {
        "center_freqs": {"type": "array", "items": {"type": "number"}},
        "bandwidths": {"type": "array", "items": {"type": "number"}},
    }, "required": ["center_freqs", "bandwidths"]},
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
            done = False
            for _scan in range(num_instances):
                msgs.append(Message(role="user", content=query.prompt))
                submitted = False
                for _ in range(MAX_INNER_CALLS):
                    payload = []
                    for m in msgs:                            # FULL history; strip think from PAST turns
                        d = m.model_dump()
                        if d.get("role") == "assistant" and d.get("content"):
                            d["content"] = THINK.sub("", d["content"]).strip()
                        payload.append(d)
                    resp = await policy._make_llm_call(messages=payload, tools=TOOLS)
                    am = resp["choices"][0]["message"]
                    tcs = [tc if isinstance(tc, dict) else tc.model_dump() for tc in (am.get("tool_calls") or [])]
                    msgs.append(Message(role="assistant", content=am.get("content") or "", tool_calls=tcs or None))
                    if not tcs:
                        break
                    for tc in tcs:
                        fn = tc.get("function", {})
                        if fn.get("name") != "submit_report":
                            msgs.append(Message(role="tool", content="(unknown tool)", tool_call_id=tc.get("id")))
                            continue
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except Exception:
                            args = {}
                        cfs = [float(x) for x in (args.get("center_freqs") or [])][:SANITY_MAX_REPORT]
                        bws = [float(x) for x in (args.get("bandwidths") or [])][:SANITY_MAX_REPORT]
                        if len(bws) < len(cfs):               # tolerate ragged answers, don't invent widths
                            bws += [8.0] * (len(cfs) - len(bws))
                        txs = [Transmitter(center_freq=c, bandwidth=b, currently_active=True, estimated_power=-30.0)
                               for c, b in zip(cfs, bws)]
                        sr = task.step(Response(action=ScanReport(transmitters=txs), metadata={}))
                        oc = getattr(sr, "instance_outcome", None)
                        avail = float(getattr(oc, "reward", 0.0) or 0.0)   # THE bench metric
                        occ = occ_iou(cfs, bws, gt)                        # diagnostic only
                        msgs.append(Message(role="tool",
                                            content=f"ok\nSCAN_AVAIL: {avail:.4f} SCAN_OCC: {occ:.4f}",
                                            tool_call_id=tc.get("id")))
                        submitted = True
                        done = bool(sr.done)
                        nq = getattr(sr, "next_query", None)
                        if nq is not None:
                            query = nq
                    if submitted:
                        break
                if not submitted:                              # force-advance with an empty report
                    sr = task.step(Response(action=ScanReport(transmitters=[]), metadata={}))
                    done = bool(sr.done)
                    nq = getattr(sr, "next_query", None)
                    if nq is not None:
                        query = nq
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
