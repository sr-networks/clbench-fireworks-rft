"""BENCH-SHAPED custom RolloutProcessor for the spectrum notepad task (Option A).

Unlike MCP-Gym (which delivers env observations as TOOL results), this processor drives the rollout itself
and produces the bench's turn structure: it pushes each scan as a **user** message, lets the model reply
with **assistant** turns using notepad tools + submit_report, scores each report, and pushes the next scan.
It runs in the Fireworks cloud RFT exactly like McpGym (the cloud runs the uploaded evaluator via pytest,
calling whatever `rollout_processor` the @evaluation_test names) — NO MCP server, NO hosted endpoint.

Memory = the agent's notepad. The processor WINDOWS the model's input to [system + current scan + this
scan's own tool turns], so earlier scans are invisible and the notepad is the only cross-scan memory. Per
scan occ-IoU is emitted as `SCAN_OCC:` in the submit_report tool result for spectrum_reward.py (it sits
inside the current scan's window only, so it never leaks into a later report decision).

Reuses SpectrumEnv for the per-row deterministic band, scan generation, and occ-IoU scoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional

import litellm
litellm.drop_params = True   # Fireworks rejects unsupported params (e.g. tool_choice) with UnsupportedParamsError;
                             # drop them instead of failing every call (io2d7zlp: 5544 failed calls on tool_choice).

from eval_protocol.mcp.execution.policy import LiteLLMPolicy
from eval_protocol.models import EvaluationRow, Message, Status
from eval_protocol.types.types import TerminationReason
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.pytest.types import RolloutProcessorConfig
from eval_protocol.pytest.utils import normalize_fireworks_model_for_litellm

from spectrum_adapter import (
    SpectrumEnv, DEFAULT_TASK_NAME, DEFAULT_TASK_KWARGS, CH_BW, MAX_REPORT, NOTEPAD_MAX_CHARS,
    SCAFFOLD, SCRAMBLE, EPOCH_SALT, band_seed,
)
from src.tasks.blind_spectrum_monitoring.task import ScanReport, Transmitter  # type: ignore

NUM_SCANS = int(DEFAULT_TASK_KWARGS["num_instances"])  # 12
MAX_INNER_CALLS = 6   # model calls allowed within one scan before we force-advance (notepad_read/write/submit)

# Streamlog markers for the epoch-salt probe: one distinct SALT per epoch-process => salting is group-stable
# and epoch-varying (safe). Also list epoch-identifying env-var NAMES (names only, never values — no secrets):
# if the harness exposes the epoch number we can switch to a DETERMINISTIC per-epoch salt (paired arms).
print(f"[spectrum] SALT={EPOCH_SALT or 'off'} SCAFFOLD={SCAFFOLD} SCRAMBLE={SCRAMBLE}", flush=True)
_epochish = sorted(k for k in __import__("os").environ if any(s in k.upper() for s in ("EPOCH", "STEP", "ROUND", "ITER")))
print(f"[spectrum] epoch-ish env var NAMES: {_epochish}", flush=True)

TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "notepad_read",
        "description": "Read your private notepad — the only memory that persists across scans (empty until you write).",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "notepad_write",
        "description": "Overwrite your private notepad with the FULL new contents (replaces what was there). Use it however helps you solve the task.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    }},
    {"type": "function", "function": {
        "name": "submit_report",
        "description": "Submit your answer for the CURRENT scan: center_freqs = center frequencies (MHz) of the occupied regions (each fixed 8 MHz wide). Ends your turn for this scan.",
        "parameters": {"type": "object", "properties": {
            "center_freqs": {"type": "array", "items": {"type": "number"}}}, "required": ["center_freqs"]},
    }},
]
if SCAFFOLD:
    # Scaffold arms: the env maintains the memory (running-list echo); notepad tools would confound the
    # memory-channel attribution, so expose submit_report only (matches the original scaffold-run shape).
    TOOLS = [t for t in TOOLS if t["function"]["name"] == "submit_report"]


# Band seeding is centralized in spectrum_adapter.band_seed (per-row deterministic + optional epoch salt).
_band_seed = band_seed


def _exec_tool(name: str, args: Dict[str, Any], notepad: Dict[str, str], env: SpectrumEnv):
    """Execute a notepad/submit tool inline. Returns (tool_result_str, scored_occ_or_None)."""
    if name == "notepad_read":
        return (notepad["text"] or "(notepad is empty — you have written nothing yet)"), None
    if name == "notepad_write":
        notepad["text"] = str(args.get("text") or "")[:NOTEPAD_MAX_CHARS]
        return "ok", None
    if name == "submit_report":
        freqs = [float(c) for c in (args.get("center_freqs") or [])][:MAX_REPORT]
        txs = [Transmitter(center_freq=c, bandwidth=CH_BW, currently_active=True, estimated_power=-30.0)
               for c in freqs]
        _obs, occ, _done, _trunc, info = env.step(ScanReport(transmitters=txs))
        return f"ok\nSCAN_OCC: {occ:.4f} SCAN_AVAIL: {float(info.get('scan_avail', 0.0)):.4f}", occ
    return f"(unknown tool: {name})", None


class SpectrumTurnRolloutProcessor(RolloutProcessor):
    """Drives the bench-shaped user/assistant rollout for the spectrum notepad task."""

    def __call__(self, rows: List[EvaluationRow], config: RolloutProcessorConfig) -> List[asyncio.Task[EvaluationRow]]:
        sem = config.semaphore
        # The CLOUD injects the in-training (hot-reload) model into config.completion_params, NOT into
        # row.input_metadata — exactly as MCPGymRolloutProcessor reads it (default_mcp_gym_rollout_processor
        # lines 285-289). Resolve it HERE and copy into each row; reading row.input_metadata gave the wrong
        # model -> 404 -> retry-hang (job tjiml7ra stuck at pct=0).
        cp = normalize_fireworks_model_for_litellm(config.completion_params) or {}
        for row in rows:
            row.input_metadata.completion_params = cp
        model_id = str(cp.get("model") or "")
        temperature = float(cp.get("temperature", 1.2))
        max_tokens = int(cp.get("max_tokens", 2048))

        async def process_row(row: EvaluationRow) -> EvaluationRow:
            t0 = time.perf_counter()
            policy = LiteLLMPolicy(
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                # NOTE: no tool_choice — Fireworks doesn't support it (litellm.drop_params handles it anyway).
                # With /no_think + the tool schema, the model still calls a tool; the inner loop force-advances
                # if it ever doesn't.
            )

            env = SpectrumEnv(DEFAULT_TASK_NAME, dict(DEFAULT_TASK_KWARGS))
            obs, _ = env.reset(seed=_band_seed(row.input_metadata.row_id))
            notepad = {"text": ""}
            msgs: List[Message] = list(row.messages)               # starts with [system]
            row.messages = msgs                                     # same object -> config.logger.log(row) sees live state
            sys_msgs = [m for m in msgs if m.role == "system"]
            scan_text = obs["prompt"]

            def emit(m: Message):
                """Append a message AND incrementally log the row, so the dashboard trace shows each turn
                distinctly: system / user(scan) / assistant(tool_calls) / tool(result). Mirrors how the
                built-in AgentRolloutProcessor logs via append_message_and_log."""
                msgs.append(m)
                try:
                    config.logger.log(row)
                except Exception:
                    pass

            for _scan in range(NUM_SCANS):
                emit(Message(role="user", content=scan_text))          # push the scan as a USER turn
                scan_start = len(msgs) - 1                              # window boundary for this scan
                submitted = False
                for _inner in range(MAX_INNER_CALLS):
                    windowed = sys_msgs + msgs[scan_start:]             # NOTEPAD is the only cross-scan memory
                    payload = [m.model_dump() for m in windowed]
                    resp = await policy._make_llm_call(messages=payload, tools=TOOLS)
                    am = resp["choices"][0]["message"]
                    # Normalize tool_calls to plain dicts: litellm may return pydantic objects, which (a) fail
                    # Message validation locally and (b) serialize non-canonically for the dashboard viewer.
                    tcs = [tc if isinstance(tc, dict) else tc.model_dump() for tc in (am.get("tool_calls") or [])]
                    emit(Message(role="assistant", content=am.get("content") or "", tool_calls=tcs or None))
                    if not tcs:
                        break                                          # no tool (rare w/ required) -> end scan
                    for tc in tcs:
                        fn = tc.get("function", {})
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except Exception:
                            args = {}
                        result, occ = _exec_tool(fn.get("name", ""), args, notepad, env)
                        emit(Message(role="tool", content=result, tool_call_id=tc.get("id")))
                        if occ is not None:
                            submitted = True
                    if submitted:
                        break
                if not submitted:
                    # model never submitted this scan -> force-advance with an empty report (occ ~ 0)
                    env.step(ScanReport(transmitters=[]))
                scan_text = env.pending_scan
                if env.done or not scan_text:
                    break

            row.messages = msgs
            row.execution_metadata.rollout_duration_seconds = time.perf_counter() - t0
            # THE dashboard-trace fix: rollout_status defaults to rollout_running and the trace viewer only
            # renders finished rollouts. McpGym sets this in manager.py:147; a custom processor must do it
            # itself — without it every row stays "running" forever and the UI shows no traces.
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
