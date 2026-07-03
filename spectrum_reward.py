"""Memory-gain reward for the blind-spectrum task.

Objective: train the model to LEARN WITH MEMORY — to recover the persistent transmitter set by
ACCUMULATING evidence across scans. The reward is the per-scan IoU of the OCCUPIED spectrum
(`SCAN_OCC`, emitted by the env against the hidden ground-truth layout); `memory_gain` (late-minus-early
occupied-IoU) is the PROOF of memory and is MEASURED, not rewarded (rewarding the gain invites
sandbagging — tanking early scans to inflate the delta).

Why OCCUPIED-IoU, not the task-native AVAILABLE-IoU (tracked as `mean_avail`): the available spectrum is
dominated by the large truly-empty region, so a memoryless agent that reports only the ~n_active current
peaks already scores ~0.47 available-IoU and memory adds only ~0.05 — too weak to train, and reporting
recalled transmitters can even hurt it. Occupied-IoU instead scores |reported_occ ∩ true_occ| / |union|:
a memoryless agent covers only n_active/total of the occupied band (~3/12 ≈ 0.25) while a full-memory
agent reaches ~1.0 — a 4x signal that REQUIRES recalling dormant transmitters. Carpeting the band is
capped (~0.5, union blows up), so accurate accumulation is the only way up. And because early scans have
seen few transmitters (low ceiling) while late scans have seen most, maximizing occupied-IoU raises LATE
scans more than EARLY — the late-minus-early gain emerges from the task structure itself.

Diagnosis that motivated this (run yysvs5nh, base qwen3-1p7b): when the model reports a transmitter it is
NEVER hallucinated (0%) — always a real current (93.9%) or recalled prior-seen (6.1%) peak. So the memory
behaviour already exists in the policy with zero precision downside; it just wasn't being PAID for under
available-IoU. Occupied-IoU pays for it directly.
"""

from __future__ import annotations

import re
from typing import Dict, List

from eval_protocol.models import EvaluationRow, EvaluateResult, MetricResult

INCOMPLETE_PENALTY = -0.2      # only if ZERO scans scored (a first-turn format failure)
OCC_SCALE = 3.0                # scale mean occupied-IoU into a wider reward range for GRPO advantages

_OCC = re.compile(r"SCAN_OCC:\s*([0-9.]+)")
_AVAIL = re.compile(r"SCAN_AVAIL:\s*([0-9.]+)")


def _collect(row: EvaluationRow, pat: re.Pattern) -> List[float]:
    """All per-scan scores the env emitted across the conversation (one per completed scan). Per-step
    emission makes this robust to rollouts that die mid-episode on a bare-JSON tool lapse."""
    out: List[float] = []
    for m in row.messages:
        content = getattr(m, "content", "") or ""
        for v in pat.findall(content):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _metrics(gain: float, n: int, early: float, late: float, mean_occ: float, mean_avail: float) -> Dict[str, MetricResult]:
    return {
        "memory_gain": MetricResult(score=gain, is_score_valid=True, reason=f"late-early occ={gain:+.3f}"),
        "scans_completed": MetricResult(score=float(n), is_score_valid=True, reason=f"{n} scans"),
        "early_mean": MetricResult(score=early, is_score_valid=True, reason=f"{early:.3f} occ-IoU"),
        "late_mean": MetricResult(score=late, is_score_valid=True, reason=f"{late:.3f} occ-IoU"),
        "mean_occ": MetricResult(score=mean_occ, is_score_valid=True, reason=f"{mean_occ:.3f} occ-IoU"),
        "mean_avail": MetricResult(score=mean_avail, is_score_valid=True, reason=f"{mean_avail:.3f} avail-IoU"),
    }


def compute_spectrum_avail_reward(row: EvaluationRow) -> EvaluateResult:
    """BENCH-PURE reward: the task's NATIVE available-spectrum IoU (SCAN_AVAIL) — exactly the CLBench metric.
    occupied-IoU is tracked as a diagnostic only. Gains (late-early) measured on both, never rewarded."""
    avail = _collect(row, _AVAIL)
    occ = _collect(row, _OCC)
    n = len(avail)
    if n < 1:
        return EvaluateResult(
            score=INCOMPLETE_PENALTY, is_score_valid=True,
            reason="scans_completed=0 (first-turn format failure)",
            metrics=_metrics(0.0, 0, 0.0, 0.0, 0.0, 0.0),
        )
    mean_avail = _mean(avail)
    mean_occ = _mean(occ)
    half = max(1, n // 2)
    a_early, a_late = _mean(avail[:half]), _mean(avail[half:])
    o_early, o_late = (_mean(occ[:half]), _mean(occ[half:])) if occ else (0.0, 0.0)
    score = OCC_SCALE * mean_avail          # scaling is GRPO-internal; the METRIC is mean_avail itself
    m = _metrics(o_late - o_early, n, a_early, a_late, mean_occ, mean_avail)
    m["avail_gain"] = MetricResult(score=a_late - a_early, is_score_valid=True,
                                   reason=f"late-early avail={a_late - a_early:+.3f}")
    return EvaluateResult(
        score=score, is_score_valid=True,
        reason=(f"scans={n}; mean_avail={mean_avail:.3f}; early_avail={a_early:.3f}; late_avail={a_late:.3f}; "
                f"avail_gain={a_late - a_early:+.3f}; mean_occ={mean_occ:.3f}; occ_gain={o_late - o_early:+.3f}; "
                f"score={score:.3f}"),
        metrics=m,
    )


def compute_spectrum_reward(row: EvaluationRow) -> EvaluateResult:
    occ = _collect(row, _OCC)
    avail = _collect(row, _AVAIL)
    n = len(occ)
    if n < 1:
        return EvaluateResult(
            score=INCOMPLETE_PENALTY, is_score_valid=True,
            reason="scans_completed=0 (first-turn format failure)",
            metrics=_metrics(0.0, 0, 0.0, 0.0, 0.0, 0.0),
        )
    mean_occ = _mean(occ)
    mean_avail = _mean(avail)
    half = max(1, n // 2)
    early = _mean(occ[:half])            # less memory accumulated (information-limited ceiling)
    late = _mean(occ[half:])             # more memory accumulated
    gain = late - early                  # PROOF metric only (NOT in the reward -> no sandbagging)
    score = OCC_SCALE * mean_occ         # REWARD = per-scan occupied-IoU (requires recalling dormant tx)
    return EvaluateResult(
        score=score, is_score_valid=True,
        reason=(
            f"scans={n}; mean_occ={mean_occ:.3f}; early_occ={early:.3f}; late_occ={late:.3f}; "
            f"memory_gain={gain:+.3f}; mean_avail={mean_avail:.3f}; score={score:.3f}"
        ),
        metrics=_metrics(gain, n, early, late, mean_occ, mean_avail),
    )
