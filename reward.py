"""Memory-gain reward for eval-protocol.

Objective: train the model to LEARN WITH MEMORY, not to bake poker skill into the weights.

Each rollout is `num_instances` consecutive poker hands vs the same fixed opponent, with an
accumulating context + notepad. We score the rollout by the **memory gain**:

    gain = mean(chip reward over the LATER hands) - mean(chip reward over the EARLIER hands)

Rationale (why this can't be satisfied by learning poker into the weights):
- If the policy simply gets better at poker, EVERY hand (early and late) improves by the same
  amount, so the gain is unchanged -> no reward.
- The gain only rises if the policy gets better at *using information that accumulates across
  hands* (e.g. noticing the opponent is a calling station and exploiting it in later hands).
- The FIRST hand of every rollout has no prior memory, so it is the de-facto "without memory"
  baseline; later hands are "with memory". gain == (with memory) - (without memory).

A small illegal-action penalty keeps the model playing legal poker (a competence prerequisite,
not strategy). Per-hand chips are clipped to bound poker variance.
"""

from __future__ import annotations

import re
from typing import Dict, List

from eval_protocol.models import EvaluationRow, EvaluateResult, MetricResult

HAND_CLIP = 6.0             # clip each hand's chip outcome (in big blinds) to [-HAND_CLIP, HAND_CLIP]
GAIN_CLIP = 8.0             # clip the final gain
MEMORY_GAIN_WEIGHT = 1.0
# The reward must be ~= the memory gain so GRPO optimizes WITH-memory minus WITHOUT-memory directly.
# An earlier run with ILLEGAL_PENALTY=-0.25 let the illegal term (~12 illegal/rollout x -0.25 = -3.0)
# DOMINATE the score, so training chased legality (and lost) instead of raising memory_gain (which went
# flat 0.69->0.52). With a tiny illegal nudge and a modest incomplete nudge, the score tracks the gain.
ILLEGAL_PENALTY = -0.02     # gentle: discourage illegal spam without dominating the memory signal
INCOMPLETE_PENALTY = -0.5   # mild: encourage completing >=2 hands (prerequisite to measuring a gain)

# Authoritative per-hand profit vector (big blinds) the env emits at episode end. This is the ground
# truth from the task's hand_history and includes hands that auto-resolve between the model's actions
# (which transcript scraping drops asymmetrically by opponent). Preferred source; regex is fallback.
_PROFITS = re.compile(r"HAND_PROFITS_BB:\s*([-\d.,\s]+)")
_NET_CHIPS = re.compile(r"Net chip change this hand:\s*([+-]?\d+)\s*chips")
_BIG_BLIND = 10.0


def _clip(x: float) -> float:
    return max(-HAND_CLIP, min(HAND_CLIP, x))


def _per_hand_chips(row: EvaluationRow) -> List[float]:
    """Ordered per-hand outcome (big blinds). Prefer the authoritative HAND_PROFITS_BB line emitted at
    episode end; fall back to scraping each hand's 'Net chip change' line if it is absent."""
    # 1) authoritative vector (last occurrence wins, in case of any duplication)
    for m in reversed(row.messages):
        content = getattr(m, "content", "") or ""
        mt = _PROFITS.search(content)
        if mt:
            vals = [v for v in mt.group(1).replace("\n", " ").split(",") if v.strip()]
            try:
                return [_clip(float(v)) for v in vals]
            except ValueError:
                break
    # 2) fallback: scrape every per-hand 'Net chip change' line (chips -> big blinds)
    chips: List[float] = []
    for m in row.messages:
        content = getattr(m, "content", "") or ""
        for mt in _NET_CHIPS.finditer(content):
            chips.append(_clip(float(mt.group(1)) / _BIG_BLIND))
    return chips


def _count_illegal(row: EvaluationRow) -> int:
    return sum(
        1 for m in row.messages
        if getattr(m, "role", None) == "tool" and "Invalid poker action" in (getattr(m, "content", "") or "")
    )


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _metrics(gain: float, illegal: int, n: int, early: float, late: float) -> Dict[str, MetricResult]:
    """Track the components separately so the PROOF of memory training isolates the memory gain from
    the illegal-move confound in the raw Score (Score = memory_gain + illegal_penalty)."""
    return {
        # THE proof signal: with-memory (late hands) minus without-memory (early hands).
        "memory_gain": MetricResult(score=gain, is_score_valid=True, reason=f"late-early={gain:+.3f}"),
        "illegal_actions": MetricResult(score=float(illegal), is_score_valid=True, reason=f"{illegal} illegal"),
        "hands_completed": MetricResult(score=float(n), is_score_valid=True, reason=f"{n} hands scored"),
        "early_mean": MetricResult(score=early, is_score_valid=True, reason=f"{early:+.3f} BB/hand"),
        "late_mean": MetricResult(score=late, is_score_valid=True, reason=f"{late:+.3f} BB/hand"),
    }


def compute_poker_reward(row: EvaluationRow) -> EvaluateResult:
    chips = _per_hand_chips(row)
    illegal = _count_illegal(row)
    n = len(chips)

    if n < 2:
        score = INCOMPLETE_PENALTY + ILLEGAL_PENALTY * illegal
        return EvaluateResult(
            score=score, is_score_valid=True,
            reason=f"hands_completed={n} (<2, no gain measurable); illegal={illegal}; score={score:.2f}",
            metrics=_metrics(0.0, illegal, n, 0.0, 0.0),
        )

    half = n // 2
    early = _mean(chips[:half])      # "without memory" portion (incl. hand 1)
    late = _mean(chips[half:])       # "with memory" portion
    gain = max(-GAIN_CLIP, min(GAIN_CLIP, late - early))
    score = MEMORY_GAIN_WEIGHT * gain + ILLEGAL_PENALTY * illegal
    return EvaluateResult(
        score=score, is_score_valid=True,
        reason=(
            f"hands={n}; hand1={chips[0]:+.1f}; early_mean={early:+.2f}; late_mean={late:+.2f}; "
            f"memory_gain={gain:+.2f}; illegal={illegal}; score={score:.2f}"
        ),
        metrics=_metrics(gain, illegal, n, early, late),
    )
