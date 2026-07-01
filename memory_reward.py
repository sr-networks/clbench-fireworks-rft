"""Reward for the learn-and-recall task = recall accuracy (the memory gain).

Per-round rewards come from the control plane (round 0 = LEARN, reward 0; rounds 1..N-1 = RECALL,
reward 1 if the model recalled the fact, else 0). The "without memory" baseline is structurally 0
(you cannot recall what you can't see and didn't write down), so:

    score = mean(recall-round rewards)   (== with-memory − without-memory gain)

A model that learns to write the facts to its notepad in round 0 and read them back later scores
~1; a model that ignores the notepad scores ~0. The facts are random per rollout, so this cannot be
solved by memorising into the weights — only by learning the write-then-recall skill.
"""

from __future__ import annotations

from typing import List

from eval_protocol.models import EvaluationRow, EvaluateResult


def _round_rewards(row: EvaluationRow) -> List[float]:
    out: List[float] = []
    for m in row.messages:
        cps = getattr(m, "control_plane_step", None)
        if isinstance(cps, dict) and getattr(m, "role", None) == "tool":
            out.append(float(cps.get("reward", 0.0) or 0.0))
    return out


def compute_memory_reward(row: EvaluationRow) -> EvaluateResult:
    rounds = _round_rewards(row)
    recall = rounds[1:] if len(rounds) > 1 else []
    score = sum(recall) / len(recall) if recall else 0.0
    return EvaluateResult(
        score=score, is_score_valid=True,
        reason=(
            f"rounds={len(rounds)}; recall_rounds={len(recall)}; "
            f"recall_accuracy(with-memory gain)={score:.3f}"
        ),
        metrics={},
    )
