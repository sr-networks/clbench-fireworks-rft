"""Enforce notepad-only cross-hand memory by windowing the policy's LLM context.

eval-protocol accumulates the full conversation history and sends ALL of it to the model every
turn, so by default the agent in hand i sees the complete trace of hands j<i. That makes the
notepad redundant and turns the experiment into "in-context learning over the full transcript"
rather than a test of learning *with the notepad as memory*.

This module monkeypatches the policy's LLM call so that, at hand boundaries, the model only sees:

    [system messages]  +  [everything from the LAST "FEEDBACK FROM PREVIOUS INSTANCE" boundary onward]

This reproduces the canonical icl_notepad behaviour `clear_context_between_instances=True`: the
conversation is cleared at each new hand (instance), and the new-hand message — assembled by the gym
(poker_adapter.py) in the canonical order FEEDBACK -> notepad -> query — carries the persistent
notepad, so the agent's ONLY memory of prior hands is what it wrote there. The marker string is the
exact one the reference agent emits (systems/icl_notepad/system.py), so windowing on it is faithful,
not a custom sentinel. The FULL history is left untouched in the trajectory (so reward.py still sees
every hand's outcome); only the model's *input* is windowed.

Imported for its side effect by test_poker_rft.py (which runs in the rollout/policy process).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Exact canonical marker injected at each new instance (hand) — see icl_notepad respond().
BOUNDARY = "FEEDBACK FROM PREVIOUS INSTANCE"


def window_to_current_hand(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep system messages + all messages from the last hand boundary onward."""
    if not messages:
        return messages
    last_boundary = None
    for i, m in enumerate(messages):
        c = m.get("content")
        if isinstance(c, str) and BOUNDARY in c:
            last_boundary = i
    if last_boundary is None:
        return messages  # still in hand 1 (the cold/no-memory hand) — nothing to cut
    system = [m for m in messages[:last_boundary] if m.get("role") == "system"]
    tail = messages[last_boundary:]
    out = system + tail
    global _CUTS
    if _CUTS < 5:
        _CUTS += 1
        logger.info("context_window: CUT prior-hand trace %d -> %d messages (notepad-only)", len(messages), len(out))
    return out


_CUTS = 0


_PATCHED = False


def install() -> bool:
    """Monkeypatch LiteLLMPolicy so its LLM input is windowed to the current hand. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from eval_protocol.mcp.execution.policy import LiteLLMPolicy
    except Exception as e:  # pragma: no cover
        logger.warning("context_window: could not import LiteLLMPolicy (%s); memory NOT isolated", e)
        return False

    orig = LiteLLMPolicy._make_llm_call

    async def _make_llm_call(self, messages, tools):  # type: ignore[no-untyped-def]
        return await orig(self, window_to_current_hand(messages), tools)

    LiteLLMPolicy._make_llm_call = _make_llm_call
    _PATCHED = True
    logger.info("context_window: installed notepad-only context windowing on LiteLLMPolicy")
    return True


install()
