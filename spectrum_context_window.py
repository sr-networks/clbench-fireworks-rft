"""ICL-OFF (notepad-style) windowing for the spectrum memory task.

By default eval-protocol sends the FULL accumulating conversation to the model every turn, so the agent
can in-context-learn over the whole scan history. To test memory *without* that crutch — i.e. so the
model's ONLY memory of prior scans is the RUNNING LIST echoed into the current observation — we window
the model's input to [system messages] + [everything from the last SCAN_MARKER onward]. That leaves the
model seeing only the current scan (its peaks + its running list), never the earlier scans/reports.

Mirrors poker's context_window.py: monkeypatch LiteLLMPolicy._make_llm_call so only the MODEL'S input is
windowed; the full trajectory is untouched in the trace (reward.py / SCAN_OCC parsing still see all scans).
Imported for its side effect by test_spectrum_rft.py (runs in the rollout/policy process).
"""

import logging
from typing import Any, Dict, List

from spectrum_adapter import SCAN_MARKER

logger = logging.getLogger(__name__)


def window_to_current_scan(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep system messages + all messages from the last SCAN_MARKER onward (current scan only)."""
    if not messages:
        return messages
    last = None
    for i, m in enumerate(messages):
        c = m.get("content")
        if isinstance(c, str) and SCAN_MARKER in c:
            last = i
    if last is None:
        return messages  # scan 1 (no prior scans yet) — nothing to cut
    system = [m for m in messages[:last] if m.get("role") == "system"]
    return system + messages[last:]


_PATCHED = False
_CUT_LOGGED = 0  # throttle: log only the first N CUT events to the streamlog (proof it fired) without spam


def install() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from eval_protocol.mcp.execution.policy import LiteLLMPolicy
    except Exception as e:  # pragma: no cover
        logger.warning("spectrum_context_window: could not import LiteLLMPolicy (%s); ICL NOT switched off", e)
        return False

    orig = LiteLLMPolicy._make_llm_call

    async def _make_llm_call(self, messages, tools):  # type: ignore[no-untyped-def]
        global _CUT_LOGGED
        windowed = window_to_current_scan(messages)
        if len(windowed) < len(messages) and _CUT_LOGGED < 30:
            _CUT_LOGGED += 1
            # Cloud-side proof: grep the epoch streamlog for "spectrum_context_window: CUT". Each line shows
            # the model's input shrank from the full history to [system]+[current scan] — i.e. ICL is OFF.
            logger.info(
                "spectrum_context_window: CUT %d->%d msgs (model sees system + current scan only)",
                len(messages), len(windowed),
            )
        return await orig(self, windowed, tools)

    LiteLLMPolicy._make_llm_call = _make_llm_call
    _PATCHED = True
    logger.info("spectrum_context_window: installed current-scan-only windowing (ICL OFF; memory = running list)")
    return True


install()
