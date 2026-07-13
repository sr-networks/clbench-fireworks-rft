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
ANCHOR_N = 2                   # scans used as the memoryless anchor in the abs-contrast reward (scans 1-2)

_OCC = re.compile(r"SCAN_OCC:\s*([0-9.]+)")
_AVAIL = re.compile(r"SCAN_AVAIL:\s*([0-9.]+)")
_REC = re.compile(r"SCAN_REC:\s*([0-9.]+)")
_DORM = re.compile(r"SCAN_DORM:\s*([0-9.]+)")
_ACQ = re.compile(r"SCAN_ACQ:\s*([0-9.]+)")
_RAREA = re.compile(r"SCAN_RAREA:\s*([0-9.]+)")
_WMAX = re.compile(r"SCAN_WMAX:\s*([0-9.]+)")

# Dormant-reward guards. ANCHOR_MARGIN: the anchor hinge fires only past a0+margin — honest anchor_dev sits
# at ~+0.03 (measured, jobs b5h41zlu/n4g52553), so the margin keeps the hinge at exact zero for every honest
# rollout (no dilution noise) while grid-baking (anchor occ ~0.4-1.0) is taxed hard. CARPET_*: the area-ratio
# hinge; honest reports have FP~0 (measured: 0% hallucinated freqs) so rarea <= recall <= ~1.0 + width slop —
# threshold 1.15 with a steep weight makes any blanket strictly dominated.
ANCHOR_MARGIN = 0.10
CARPET_THRESH = 1.15
CARPET_W = 4.0
# Width guard: max reported region width / max GT channel width. Honest reports never exceed the widest
# channel (census: 0% hallucinated regions across all arms), while a blanket needs a band-scale entry
# (168MHz -> ratio ~10) — the tell that catches late-start carpets which dodge the anchor hinge and slip
# under the area threshold (scripted latecarpet audit: 2.06 without this guard, deeply negative with it).
WMAX_THRESH = 1.5
WMAX_W = 1.0
# Completion guard (dormant-completion arm). The lr-2e-4 run (pyl9pkw8) found an UNGUARDED exploit: by
# inflating its own thinking trace it hit the 8192-token turn cap mid-report and DIED around scan 24,
# dropping the hardest late scans from its own average -> truncated episodes OUTSCORED complete ones
# (mean_dorm 0.63 truncated vs 0.58 full at ep4). Nothing in the base reward requires finishing, so GRPO
# selected for dying. pen_complete taxes the scan shortfall below NUM_SCANS: exact-zero inside a grace
# window (honest episodes score 28-30, measured on pscoc9fp: 272/288 at 30, 13 at 29, 2 at 28), rising
# linearly for genuine truncation. Kills the exploit, leaves honest play untouched — same discipline as the
# other three guards. Red-teamed in sim_dormant_complete.py (quit20/quit24/gridquit must net below accumulate).
NUM_SCANS = 30
COMPLETE_GRACE = 2
COMPLETE_W = 1.5
# Acquisition-credit weight (dormacq arm). SCAN_ACQ = dormant coverage restricted to the YOUNGEST recallable
# cohort (channels first sighted exactly one scan ago, invisible now) — a direct per-event test of "did you
# merge the previous scan's new detections into the notepad". mean_dorm already pays holding memory; the
# census shows acquisition (merge-OK ~43%) is the bottleneck, so it gets an explicit bonus. 0.5 keeps
# mean_dorm the dominant term (acq is a subset of dorm, so honest play earns both; a policy cannot trade
# retention away for acquisition without paying more in dorm-FN than it gains here).
ACQ_W = 0.5


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


def compute_spectrum_recall_reward(row: EvaluationRow) -> EvaluateResult:
    """RECALL-WEIGHTED reward: grades the per-scan Tversky overlap (SCAN_REC) instead of symmetric occ-IoU.
    Tversky discounts false positives (alpha<1) and keeps a full penalty for missing a true transmitter
    (beta=1), so the gradient PAYS for keeping the persistent set instead of punishing it — the direct test
    of the 'reward erodes accumulation' diagnosis. mean_occ (symmetric) and mean_avail (bench-native) are
    STILL measured as untouched diagnostics so this arm is comparable apples-to-apples with the occ arms."""
    rec = _collect(row, _REC)
    occ = _collect(row, _OCC)
    avail = _collect(row, _AVAIL)
    n = len(rec)
    if n < 1:
        return EvaluateResult(
            score=INCOMPLETE_PENALTY, is_score_valid=True,
            reason="scans_completed=0 (first-turn format failure)",
            metrics=_metrics(0.0, 0, 0.0, 0.0, 0.0, 0.0),
        )
    mean_rec = _mean(rec)
    mean_occ = _mean(occ)
    mean_avail = _mean(avail)
    half = max(1, n // 2)
    r_early, r_late = _mean(rec[:half]), _mean(rec[half:])
    o_early, o_late = (_mean(occ[:half]), _mean(occ[half:])) if occ else (0.0, 0.0)
    score = OCC_SCALE * mean_rec         # REWARD = per-scan recall-weighted overlap (pays for accumulation)
    m = _metrics(o_late - o_early, n, o_early, o_late, mean_occ, mean_avail)
    m["mean_rec"] = MetricResult(score=mean_rec, is_score_valid=True, reason=f"{mean_rec:.3f} recall-Tversky")
    m["recall_gain"] = MetricResult(score=r_late - r_early, is_score_valid=True,
                                    reason=f"late-early rec={r_late - r_early:+.3f}")
    return EvaluateResult(
        score=score, is_score_valid=True,
        reason=(f"scans={n}; mean_rec={mean_rec:.3f}; early_rec={r_early:.3f}; late_rec={r_late:.3f}; "
                f"recall_gain={r_late - r_early:+.3f}; mean_occ={mean_occ:.3f}; occ_gain={o_late - o_early:+.3f}; "
                f"mean_avail={mean_avail:.3f}; score={score:.3f}"),
        metrics=m,
    )


# Memoryless early-half floor per variant, from measure_canon_a0.py (scripted current-scan reporter on the
# canon 24 rows — a TASK property the policy cannot move; recompute if the task/schedule ever changes).
A0 = {"five_ch_wide": 0.2082, "five_plus_four_mixed": 0.2141, "full_grid_active": 0.2368}


def compute_spectrum_abs_contrast_reward(row: EvaluationRow) -> EvaluateResult:
    """CONTRAST reward, two-sided (user's formula): score = late - |anchor - a0|, where `anchor` is the mean
    occ over the first ANCHOR_N scans (scans 1-2) rather than the whole first half. WHY the short anchor:
    occ starts at a0 on scan 1 (truly memoryless) but SATURATES by scan ~10, so the first HALF (scans 1-15)
    is mostly post-saturation memory-rich territory (mean ~0.40 >> a0 ~0.22). Pinning that whole window to a0
    would tax the exact early memory use we want and select for hide-until-scan-16 policies. Scans 1-2 are
    still essentially memoryless (only ~1 prior scan of accumulation), so they capture task PROFICIENCY with
    minimal memory contamination: anchor ABOVE a0 means the model got better at the BASE task (penalized back
    down), anchor BELOW a0 means sandbagging (penalized up). The optimum is anchor==a0, so the only way up is
    to raise LATE above a fixed proficiency anchor -> isolates memory-specific gain. Scans 3-15 are neither in
    the anchor nor in late, so they carry NO gradient (no suppression pressure) and stay memory-rich. Two
    scans (vs scan 1 alone) halve the per-rollout anchor variance for only ~+0.03 bias. Trajectory-level."""
    occ = _collect(row, _OCC)
    avail = _collect(row, _AVAIL)
    n = len(occ)
    if n < 1:
        return EvaluateResult(
            score=INCOMPLETE_PENALTY, is_score_valid=True,
            reason="scans_completed=0 (first-turn format failure)",
            metrics=_metrics(0.0, 0, 0.0, 0.0, 0.0, 0.0),
        )
    rid = str(getattr(row.input_metadata, "row_id", None) or "")
    a0 = next((v for k, v in A0.items() if k in rid), 0.21)
    mean_occ = _mean(occ)
    mean_avail = _mean(avail)
    half = max(1, n // 2)
    early = _mean(occ[:half])                 # full first half — PROOF/diagnostic only (memory_gain)
    late = _mean(occ[half:])
    anchor = _mean(occ[:ANCHOR_N])            # memoryless proficiency anchor (scans 1-2) — drives the penalty
    gain = late - early
    contrast = late - abs(anchor - a0)
    score = OCC_SCALE * contrast
    m = _metrics(gain, n, early, late, mean_occ, mean_avail)
    m["anchor_mean"] = MetricResult(score=anchor, is_score_valid=True,
                                    reason=f"{anchor:.3f} occ over scans 1-{ANCHOR_N}")
    m["anchor_dev"] = MetricResult(score=anchor - a0, is_score_valid=True,
                                   reason=f"anchor-a0={anchor - a0:+.3f} (a0={a0:.3f})")
    m["early_dev"] = MetricResult(score=early - a0, is_score_valid=True,
                                  reason=f"early-a0={early - a0:+.3f} (a0={a0:.3f})")
    return EvaluateResult(
        score=score, is_score_valid=True,
        reason=(
            f"scans={n}; late={late:.3f}; anchor={anchor:.3f}; a0={a0:.3f}; early={early:.3f}; "
            f"contrast={contrast:.3f}; memory_gain={gain:+.3f}; score={score:.3f}"
        ),
        metrics=m,
    )


def compute_spectrum_dormant_reward(row: EvaluationRow) -> EvaluateResult:
    """DORMANT-COVERAGE reward: score = OCC_SCALE * (mean_dorm - anchor_hinge - carpet_hinge).

    mean_dorm is the per-scan Tversky coverage of the RECALLABLE channels (seen in an earlier scan AND not
    visible now) — the one quantity a memoryless policy cannot earn, so task proficiency carries ZERO
    gradient by construction (the failure of the hinge-contrast arm, where ~70% of the late gain was
    proficiency, and of the abs-contrast arm, where neutralizing proficiency left nothing trainable).
    Dense (~29 per-scan values/rollout, the density that trained in the proc arm), and it pays the missing
    behavior directly: a channel merged into the notepad at scan k earns credit at EVERY later scan it is
    dormant, a dropped one costs FN at every later scan — compound interest for merge->persist->read-back,
    inverting the copy-forward attractor's payoff. Within a GRPO group all 12 candidates share the row's
    band/schedule, so the recallable sets are identical and the group advantage is pure policy signal.

    Guards (both exact-zero for honest policies -> no dilution): anchor hinge max(0, anchor-a0-margin)
    taxes grid-baking (reporting the memorized 13-slot grid lifts no-notepad occ — the thing the goal
    forbids); carpet hinge CARPET_W*max(0, mean_rarea-thresh) taxes blanket reports (also FP-taxed inside
    dorm itself via the allowed-set: paint on never-seen channels counts as FP). GOAL METRICS: late_mean
    (occ) must rise by halves with anchor_dev flat — both logged here unchanged from the contrast arms."""
    occ = _collect(row, _OCC)
    avail = _collect(row, _AVAIL)
    dorm = _collect(row, _DORM)
    rarea = _collect(row, _RAREA)
    n = len(occ)
    if n < 1:
        return EvaluateResult(
            score=INCOMPLETE_PENALTY, is_score_valid=True,
            reason="scans_completed=0 (first-turn format failure)",
            metrics=_metrics(0.0, 0, 0.0, 0.0, 0.0, 0.0),
        )
    rid = str(getattr(row.input_metadata, "row_id", None) or "")
    a0 = next((v for k, v in A0.items() if k in rid), 0.21)
    mean_occ = _mean(occ)
    mean_avail = _mean(avail)
    half = max(1, n // 2)
    early = _mean(occ[:half])
    late = _mean(occ[half:])
    anchor = _mean(occ[:ANCHOR_N])
    gain = late - early
    mean_dorm = _mean(dorm)                    # 0.0 if no dorm scans completed (died on scan 1) — no reward
    dh = max(1, len(dorm) // 2)
    d_early, d_late = _mean(dorm[:dh]), _mean(dorm[dh:])
    mean_rarea = _mean(rarea)
    wmax = _collect(row, _WMAX)
    mean_wmax = _mean(wmax)
    pen_anchor = max(0.0, anchor - a0 - ANCHOR_MARGIN)
    pen_carpet = CARPET_W * max(0.0, mean_rarea - CARPET_THRESH)
    pen_wmax = WMAX_W * max(0.0, mean_wmax - WMAX_THRESH)
    score = OCC_SCALE * (mean_dorm - pen_anchor - pen_carpet - pen_wmax)
    m = _metrics(gain, n, early, late, mean_occ, mean_avail)
    m["mean_dorm"] = MetricResult(score=mean_dorm, is_score_valid=True,
                                  reason=f"{mean_dorm:.3f} dormant-Tversky over {len(dorm)} scans")
    m["dorm_gain"] = MetricResult(score=d_late - d_early, is_score_valid=True,
                                  reason=f"late-early dorm={d_late - d_early:+.3f}")
    m["rarea_mean"] = MetricResult(score=mean_rarea, is_score_valid=True,
                                   reason=f"report/GT area={mean_rarea:.3f}")
    m["anchor_mean"] = MetricResult(score=anchor, is_score_valid=True,
                                    reason=f"{anchor:.3f} occ over scans 1-{ANCHOR_N}")
    m["anchor_dev"] = MetricResult(score=anchor - a0, is_score_valid=True,
                                   reason=f"anchor-a0={anchor - a0:+.3f} (a0={a0:.3f})")
    m["early_dev"] = MetricResult(score=early - a0, is_score_valid=True,
                                  reason=f"early-a0={early - a0:+.3f} (a0={a0:.3f})")
    m["pen_anchor"] = MetricResult(score=pen_anchor, is_score_valid=True, reason=f"anchor hinge {pen_anchor:.3f}")
    m["pen_carpet"] = MetricResult(score=pen_carpet, is_score_valid=True, reason=f"carpet hinge {pen_carpet:.3f}")
    m["pen_wmax"] = MetricResult(score=pen_wmax, is_score_valid=True,
                                 reason=f"width hinge {pen_wmax:.3f} (mean_wmax={mean_wmax:.2f})")
    return EvaluateResult(
        score=score, is_score_valid=True,
        reason=(
            f"scans={n}; mean_dorm={mean_dorm:.3f} ({len(dorm)} dorm-scans); rarea={mean_rarea:.3f}; "
            f"wmax={mean_wmax:.2f}; anchor={anchor:.3f}; a0={a0:.3f}; pen_anchor={pen_anchor:.3f}; "
            f"pen_carpet={pen_carpet:.3f}; pen_wmax={pen_wmax:.3f}; late={late:.3f}; "
            f"memory_gain={gain:+.3f}; score={score:.3f}"
        ),
        metrics=m,
    )


def compute_spectrum_dormant_completion_reward(row: EvaluationRow) -> EvaluateResult:
    """DORMANT + COMPLETION-GUARD reward (dormcomplete arm):
        score = OCC_SCALE * (mean_dorm - anchor_hinge - carpet_hinge - width_hinge - completion_hinge).

    Identical to compute_spectrum_dormant_reward (same mean_dorm currency, same three exact-zero guards)
    plus ONE subtractive term, pen_complete = COMPLETE_W * max(0, NUM_SCANS - n - COMPLETE_GRACE)/NUM_SCANS,
    that taxes episodes ending early. Motivation (measured): the lr-2e-4 run pyl9pkw8 learned to truncate
    its OWN episode — a bloated thinking trace overruns the 8192-token turn cap mid-report, the processor
    breaks the rollout (spectrum_canon_processor.py:269), and the hardest late scans never get scored, so the
    surviving per-scan average LOOKS higher (mean_dorm 0.63 truncated vs 0.58 full at ep4). The base reward
    averages only over scans actually played, so GRPO was rewarding DYING. This hinge is exact-zero inside a
    grace window (honest episodes score 28-30 — pscoc9fp ep0: 272/288 at 30, 13 at 29, 2 at 28, so
    NUM_SCANS-GRACE=28 is free) and rises linearly for genuine truncation, making finishing dominant while
    leaving honest play byte-for-byte identical to the dormant arm. Directly comparable to that arm."""
    occ = _collect(row, _OCC)
    avail = _collect(row, _AVAIL)
    dorm = _collect(row, _DORM)
    rarea = _collect(row, _RAREA)
    n = len(occ)
    if n < 1:
        return EvaluateResult(
            score=INCOMPLETE_PENALTY, is_score_valid=True,
            reason="scans_completed=0 (first-turn format failure)",
            metrics=_metrics(0.0, 0, 0.0, 0.0, 0.0, 0.0),
        )
    rid = str(getattr(row.input_metadata, "row_id", None) or "")
    a0 = next((v for k, v in A0.items() if k in rid), 0.21)
    mean_occ = _mean(occ)
    mean_avail = _mean(avail)
    half = max(1, n // 2)
    early = _mean(occ[:half])
    late = _mean(occ[half:])
    anchor = _mean(occ[:ANCHOR_N])
    gain = late - early
    mean_dorm = _mean(dorm)                    # 0.0 if no dorm scans completed (died on scan 1) — no reward
    dh = max(1, len(dorm) // 2)
    d_early, d_late = _mean(dorm[:dh]), _mean(dorm[dh:])
    mean_rarea = _mean(rarea)
    wmax = _collect(row, _WMAX)
    mean_wmax = _mean(wmax)
    pen_anchor = max(0.0, anchor - a0 - ANCHOR_MARGIN)
    pen_carpet = CARPET_W * max(0.0, mean_rarea - CARPET_THRESH)
    pen_wmax = WMAX_W * max(0.0, mean_wmax - WMAX_THRESH)
    pen_complete = COMPLETE_W * max(0, NUM_SCANS - n - COMPLETE_GRACE) / NUM_SCANS
    score = OCC_SCALE * (mean_dorm - pen_anchor - pen_carpet - pen_wmax - pen_complete)
    m = _metrics(gain, n, early, late, mean_occ, mean_avail)
    m["mean_dorm"] = MetricResult(score=mean_dorm, is_score_valid=True,
                                  reason=f"{mean_dorm:.3f} dormant-Tversky over {len(dorm)} scans")
    m["dorm_gain"] = MetricResult(score=d_late - d_early, is_score_valid=True,
                                  reason=f"late-early dorm={d_late - d_early:+.3f}")
    m["rarea_mean"] = MetricResult(score=mean_rarea, is_score_valid=True,
                                   reason=f"report/GT area={mean_rarea:.3f}")
    m["anchor_mean"] = MetricResult(score=anchor, is_score_valid=True,
                                    reason=f"{anchor:.3f} occ over scans 1-{ANCHOR_N}")
    m["anchor_dev"] = MetricResult(score=anchor - a0, is_score_valid=True,
                                   reason=f"anchor-a0={anchor - a0:+.3f} (a0={a0:.3f})")
    m["early_dev"] = MetricResult(score=early - a0, is_score_valid=True,
                                  reason=f"early-a0={early - a0:+.3f} (a0={a0:.3f})")
    m["pen_anchor"] = MetricResult(score=pen_anchor, is_score_valid=True, reason=f"anchor hinge {pen_anchor:.3f}")
    m["pen_carpet"] = MetricResult(score=pen_carpet, is_score_valid=True, reason=f"carpet hinge {pen_carpet:.3f}")
    m["pen_wmax"] = MetricResult(score=pen_wmax, is_score_valid=True,
                                 reason=f"width hinge {pen_wmax:.3f} (mean_wmax={mean_wmax:.2f})")
    m["pen_complete"] = MetricResult(score=pen_complete, is_score_valid=True,
                                     reason=f"completion hinge {pen_complete:.3f} (n={n}/{NUM_SCANS})")
    return EvaluateResult(
        score=score, is_score_valid=True,
        reason=(
            f"scans={n}; mean_dorm={mean_dorm:.3f} ({len(dorm)} dorm-scans); rarea={mean_rarea:.3f}; "
            f"wmax={mean_wmax:.2f}; anchor={anchor:.3f}; a0={a0:.3f}; pen_anchor={pen_anchor:.3f}; "
            f"pen_carpet={pen_carpet:.3f}; pen_wmax={pen_wmax:.3f}; pen_complete={pen_complete:.3f}; "
            f"late={late:.3f}; memory_gain={gain:+.3f}; score={score:.3f}"
        ),
        metrics=m,
    )


def compute_spectrum_dormant_acq_reward(row: EvaluationRow) -> EvaluateResult:
    """DORMANT + ACQUISITION-CREDIT reward (dormacq arm):
        score = OCC_SCALE * (mean_dorm + ACQ_W * mean_acq - anchor_hinge - carpet_hinge - width_hinge).

    Identical to compute_spectrum_dormant_reward (same guards, all exact-zero for honest play) plus ONE
    additive term: mean SCAN_ACQ = dormant coverage restricted to the YOUNGEST recallable cohort (channels
    first sighted exactly one scan ago and invisible now). mean_dorm pays for HOLDING memory and only
    indirectly for acquiring it (compound interest); the pscoc9fp census shows acquisition is the stuck
    behavior (merge-OK 41.4->43.2% while retention/read-back moved), so the merge event is paid directly at
    the first scan where it becomes verifiable. SCAN_ACQ emission scans are schedule-determined (identical
    across all 12 GRPO candidates of a row); acq targets are a subset of the dorm targets and share the
    allowed-set FP accounting, so the term adds gradient toward merging without opening any surface a
    dorm-honest policy doesn't already have."""
    occ = _collect(row, _OCC)
    avail = _collect(row, _AVAIL)
    dorm = _collect(row, _DORM)
    acq = _collect(row, _ACQ)
    rarea = _collect(row, _RAREA)
    n = len(occ)
    if n < 1:
        return EvaluateResult(
            score=INCOMPLETE_PENALTY, is_score_valid=True,
            reason="scans_completed=0 (first-turn format failure)",
            metrics=_metrics(0.0, 0, 0.0, 0.0, 0.0, 0.0),
        )
    rid = str(getattr(row.input_metadata, "row_id", None) or "")
    a0 = next((v for k, v in A0.items() if k in rid), 0.21)
    mean_occ = _mean(occ)
    mean_avail = _mean(avail)
    half = max(1, n // 2)
    early = _mean(occ[:half])
    late = _mean(occ[half:])
    anchor = _mean(occ[:ANCHOR_N])
    gain = late - early
    mean_dorm = _mean(dorm)                    # 0.0 if no dorm scans completed (died on scan 1) — no reward
    mean_acq = _mean(acq)                      # 0.0 if no acq scans (schedule-determined, group-constant)
    dh = max(1, len(dorm) // 2)
    d_early, d_late = _mean(dorm[:dh]), _mean(dorm[dh:])
    mean_rarea = _mean(rarea)
    wmax = _collect(row, _WMAX)
    mean_wmax = _mean(wmax)
    pen_anchor = max(0.0, anchor - a0 - ANCHOR_MARGIN)
    pen_carpet = CARPET_W * max(0.0, mean_rarea - CARPET_THRESH)
    pen_wmax = WMAX_W * max(0.0, mean_wmax - WMAX_THRESH)
    score = OCC_SCALE * (mean_dorm + ACQ_W * mean_acq - pen_anchor - pen_carpet - pen_wmax)
    m = _metrics(gain, n, early, late, mean_occ, mean_avail)
    m["mean_dorm"] = MetricResult(score=mean_dorm, is_score_valid=True,
                                  reason=f"{mean_dorm:.3f} dormant-Tversky over {len(dorm)} scans")
    m["mean_acq"] = MetricResult(score=mean_acq, is_score_valid=True,
                                 reason=f"{mean_acq:.3f} fresh-cohort coverage over {len(acq)} acq-scans")
    m["dorm_gain"] = MetricResult(score=d_late - d_early, is_score_valid=True,
                                  reason=f"late-early dorm={d_late - d_early:+.3f}")
    m["rarea_mean"] = MetricResult(score=mean_rarea, is_score_valid=True,
                                   reason=f"report/GT area={mean_rarea:.3f}")
    m["anchor_mean"] = MetricResult(score=anchor, is_score_valid=True,
                                    reason=f"{anchor:.3f} occ over scans 1-{ANCHOR_N}")
    m["anchor_dev"] = MetricResult(score=anchor - a0, is_score_valid=True,
                                   reason=f"anchor-a0={anchor - a0:+.3f} (a0={a0:.3f})")
    m["early_dev"] = MetricResult(score=early - a0, is_score_valid=True,
                                  reason=f"early-a0={early - a0:+.3f} (a0={a0:.3f})")
    m["pen_anchor"] = MetricResult(score=pen_anchor, is_score_valid=True, reason=f"anchor hinge {pen_anchor:.3f}")
    m["pen_carpet"] = MetricResult(score=pen_carpet, is_score_valid=True, reason=f"carpet hinge {pen_carpet:.3f}")
    m["pen_wmax"] = MetricResult(score=pen_wmax, is_score_valid=True,
                                 reason=f"width hinge {pen_wmax:.3f} (mean_wmax={mean_wmax:.2f})")
    return EvaluateResult(
        score=score, is_score_valid=True,
        reason=(
            f"scans={n}; mean_dorm={mean_dorm:.3f} ({len(dorm)} dorm-scans); mean_acq={mean_acq:.3f} "
            f"({len(acq)} acq-scans); rarea={mean_rarea:.3f}; wmax={mean_wmax:.2f}; anchor={anchor:.3f}; "
            f"a0={a0:.3f}; pen_anchor={pen_anchor:.3f}; pen_carpet={pen_carpet:.3f}; "
            f"pen_wmax={pen_wmax:.3f}; late={late:.3f}; memory_gain={gain:+.3f}; score={score:.3f}"
        ),
        metrics=m,
    )


def compute_spectrum_contrast_reward(row: EvaluationRow) -> EvaluateResult:
    """CONTRAST reward, one-sided hinge: score = late - max(0, a0 - early). late-half occ is paid directly;
    early scans are penalized only for dipping BELOW the memoryless floor a0 (deliberate sandbagging), never
    for legitimate early memory use (a working notepad puts early ~0.40 vs a0 ~0.22 — the abs variant would
    tax exactly the behavior we want and select for hide-memory-until-scan-16 policies). Within a GRPO group
    the -a0 constant cancels, so for non-sandbagging candidates this is effectively pure late-half occ.
    Trajectory-level: ONE scalar per 30-scan rollout."""
    occ = _collect(row, _OCC)
    avail = _collect(row, _AVAIL)
    n = len(occ)
    if n < 1:
        return EvaluateResult(
            score=INCOMPLETE_PENALTY, is_score_valid=True,
            reason="scans_completed=0 (first-turn format failure)",
            metrics=_metrics(0.0, 0, 0.0, 0.0, 0.0, 0.0),
        )
    rid = str(getattr(row.input_metadata, "row_id", None) or "")
    a0 = next((v for k, v in A0.items() if k in rid), 0.21)
    mean_occ = _mean(occ)
    mean_avail = _mean(avail)
    half = max(1, n // 2)
    early = _mean(occ[:half])
    late = _mean(occ[half:])
    gain = late - early
    contrast = late - max(0.0, a0 - early)
    score = OCC_SCALE * contrast
    m = _metrics(gain, n, early, late, mean_occ, mean_avail)
    m["early_dev"] = MetricResult(score=early - a0, is_score_valid=True,
                                  reason=f"early-a0={early - a0:+.3f} (a0={a0:.3f})")
    return EvaluateResult(
        score=score, is_score_valid=True,
        reason=(
            f"scans={n}; late={late:.3f}; early={early:.3f}; a0={a0:.3f}; "
            f"contrast={contrast:.3f}; memory_gain={gain:+.3f}; score={score:.3f}"
        ),
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
