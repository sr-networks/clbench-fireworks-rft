"""dbx reward — RL score for the database_exploration memory arm (draft v1).

Currency: informed query-efficiency at positions >= 2. Reported STAT stays the strict
one-shot rate (correct AND nq<=1 AND evidence-backed) whose memoryless floor is 0.000;
the TRAINING reward uses a graded efficiency ladder so an untrained 1.7B is not stuck
at all-zero candidates (the n4g52553 no-gradient lesson):

    w(nq): 0 -> 1.0, 1 -> 1.0, 2 -> 0.6, 3 -> 0.3, >=4 -> 0
    eff_k = correct_k * w(nq_k) * ev_ok_k
    ev_ok_k = (ev in {result, notepad}) if nq_k <= 1 else 1     # evidence gate
    mean_eff = mean over positions 2..n

Defenses (each maps to a red-teamed exploit, see dbx_redteam.py):
  - evidence gate ......... answers not traceable to episode observations earn nothing
                            even if correct (blocks weight-baked answers + decoy queries)
  - pen_bake .............. correct nq<=1 answers with ev in {prophecy, none} are also
                            PENALIZED, so mixing baking into honest play is negative-sum
  - pen_anchor ............ position-1 efficiency above the memoryless floor a0 + margin
                            is penalized (schema facts cannot honestly exist at pos 1;
                            defense-in-depth behind the DB variants)
  - pen_complete .......... skipping questions (grace 1) is penalized, mirroring the
                            spectrum completion guard
  - empty episode ......... 0 questions answered -> flat -0.2 (format-failure floor)

    Score = 3 * (mean_eff - pen_anchor - pen_complete - pen_bake)

A0 below is the SCRIPTED memoryless anchor estimate; replace with the measured ep0
anchor from validation run 1 (same protocol as the spectrum a0 constants).
"""

import re

W = {0: 1.0, 1: 1.0, 2: 0.6, 3: 0.3}
A0 = 0.002         # MEASURED: w7guqqae ep0 anchor mean 0.0016 (191/192 rollouts at 0);
                   # hinge now bites at anchor > 0.152 (was 0.30 with the 0.15 placeholder)
A_MARGIN = 0.15
PEN_BAKE_WT = 2.0
PEN_COMPLETE_WT = 1.5
EMPTY_SCORE = -0.2


def _eff(s) -> float:
    if not s["correct"]:
        return 0.0
    w = W.get(s["nq"], 0.0)
    if s["nq"] <= 1 and s["ev"] not in ("result", "notepad"):
        return 0.0
    return w


def dbx_score(stats, n_questions: int = 15, a0: float = A0) -> dict:
    """stats: list of dicts with pos, correct, nq, ev (parsed QSTAT lines)."""
    if not stats:
        return dict(score=EMPTY_SCORE, mean_eff=0.0, anchor=0.0, one_shot=0.0,
                    pen_anchor=0.0, pen_complete=PEN_COMPLETE_WT, pen_bake=0.0)
    effs = {s["pos"]: _eff(s) for s in stats}
    late = [v for k, v in effs.items() if k >= 2]
    mean_eff = sum(late) / len(late) if late else 0.0
    anchor = effs.get(1, 0.0)
    pen_anchor = max(0.0, anchor - a0 - A_MARGIN)
    n = len(effs)
    pen_complete = PEN_COMPLETE_WT * max(0.0, (n_questions - n - 1) / n_questions)
    n_bake = sum(1 for s in stats
                 if s["correct"] and s["nq"] <= 1 and s["ev"] in ("prophecy", "none"))
    pen_bake = PEN_BAKE_WT * n_bake / n_questions
    one_shot = (sum(1 for s in stats if s["pos"] >= 2 and s["correct"] and s["nq"] <= 1
                    and s["ev"] in ("result", "notepad"))
                / max(1, sum(1 for s in stats if s["pos"] >= 2)))
    return dict(score=3.0 * (mean_eff - pen_anchor - pen_complete - pen_bake),
                mean_eff=mean_eff, anchor=anchor, one_shot=one_shot,
                pen_anchor=pen_anchor, pen_complete=pen_complete, pen_bake=pen_bake)


# ---------------- RFT-side entry (parses QSTAT lines out of the rollout) ----------------

QSTAT_RE = re.compile(r"QSTAT pos=(\d+) grp=(\w+) correct=([01]) nq=(\d+) ev=(\w+)")


def compute_dbx_reward(row):
    """EvaluateResult from a DbxCanonRolloutProcessor rollout (QSTAT lines in tool msgs)."""
    from eval_protocol.models import EvaluateResult, MetricResult
    stats = []
    # QSTAT lines live in tool messages (answered questions) and user messages
    # (force-advanced questions). Model-controllable text is scrubbed of the QSTAT
    # token at embedding time (dbx_engine._scrub), so these are processor-authored.
    for m in row.messages:
        if getattr(m, "role", None) not in ("tool", "user"):
            continue
        for pos, grp, correct, nq, ev in QSTAT_RE.findall(m.content or ""):
            stats.append(dict(pos=int(pos), grp=grp, correct=correct == "1",
                              nq=int(nq), ev=ev))
    r = dbx_score(stats)
    acc = (sum(1 for s in stats if s["correct"]) / len(stats)) if stats else 0.0
    mean_nq = (sum(s["nq"] for s in stats) / len(stats)) if stats else 0.0
    metrics = {
        "one_shot": MetricResult(score=r["one_shot"], is_score_valid=True,
                                 reason=f"informed one-shot rate pos>=2 = {r['one_shot']:.3f}"),
        "mean_eff": MetricResult(score=r["mean_eff"], is_score_valid=True,
                                 reason=f"graded efficiency pos>=2 = {r['mean_eff']:.3f}"),
        "acc": MetricResult(score=acc, is_score_valid=True, reason=f"accuracy {acc:.3f}"),
        "mean_nq": MetricResult(score=mean_nq, is_score_valid=True,
                                reason=f"mean queries/question {mean_nq:.2f}"),
        "anchor": MetricResult(score=r["anchor"], is_score_valid=True,
                               reason=f"pos-1 eff {r['anchor']:.3f}"),
        "q_answered": MetricResult(score=float(len(stats)), is_score_valid=True,
                                   reason=f"{len(stats)} questions answered"),
        "pen_anchor": MetricResult(score=r["pen_anchor"], is_score_valid=True, reason="anchor hinge"),
        "pen_complete": MetricResult(score=r["pen_complete"], is_score_valid=True, reason="completion hinge"),
        "pen_bake": MetricResult(score=r["pen_bake"], is_score_valid=True,
                                 reason="unevidenced nq<=1 correct answers"),
    }
    return EvaluateResult(
        score=r["score"],
        reason=(f"dbx score {r['score']:+.3f} (eff {r['mean_eff']:.3f}, one_shot {r['one_shot']:.3f}, "
                f"acc {acc:.3f}, pens a/c/b {r['pen_anchor']:.2f}/{r['pen_complete']:.2f}/{r['pen_bake']:.2f})"),
        metrics=metrics,
    )
