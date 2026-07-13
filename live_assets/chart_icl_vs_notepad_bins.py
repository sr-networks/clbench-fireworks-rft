"""Grouped bar chart: occ-IoU by scan-position bin (1-5, ..., 26-30), three bars per bin:
  no-mem  = scripted PERFECT memoryless agent (reports exactly the currently visible channels, perfect
            widths) run through the real task engine on the same 24 bands -> an UPPER BOUND for any
            agent without memory (same reference + caveat as chart_saturation).
  ICL     = untrained base model, full history in prompt, no notepad (g7dncu2c ep0, 288 rollouts).
  notepad = the 4 dormc16k replicate runs (s8e07n53, yp8deoer, kym4znjc, wqjyy66p — 16384 per-turn cap),
            POOLED: green bar = ep4 (trained, 4x192=768 rollouts), whisker = min-max of the 4 per-run
            means (the repeatability spread), black tick = same 4 runs at ep0 (untrained, pooled).
Per user request the notepad bars use ONLY dormc runs; the 16k cohort now replaces the 8k one.
Caveat printed + shown on page: at ep4 completion is 100/97/60/90 percent (only kym4znjc truncates
heavily), so late bins contain only scans actually played — a much smaller survivorship effect than
the 8k cohort's, but still disclosed."""
import json
import re
import sys
from glob import glob
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = "/Users/sten/Documents/Coding/fireworks/clbench-fireworks-rft"
SCRATCH = ("/private/tmp/claude-501/-Users-sten-Documents-Coding-fireworks/"
           "bc95af7b-ea9a-424a-82d1-2001ebd855ce/scratchpad")
sys.path.insert(0, HERE)

from spectrum_adapter import band_seed
from bench_eval import load_default_schedule, resolved_gt, occ_iou, PEAK
from src.registry import get_task_class  # type: ignore
from src.interface import Response  # type: ignore
from src.tasks.blind_spectrum_monitoring.task import ScanReport, Transmitter  # type: ignore

OCC = re.compile(r"SCAN_OCC:\s*([0-9.]+)")
BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
LABELS = ["scans 1–5", "6–10", "11–15", "16–20", "21–25", "26–30"]
DORMC = ["s8e07n53", "yp8deoer", "kym4znjc", "wqjyy66p"]
_VARIANTS = {st["variant"]: st["kwargs"] for st in load_default_schedule()}


def bin_stats(per_scan_lists):
    """Pool occ values by scan position; return (per-bin mean, per-bin n)."""
    pooled = [[] for _ in range(30)]
    for series in per_scan_lists:
        for i, v in enumerate(series[:30]):
            pooled[i].append(v)
    means = [mean([v for i in range(lo, hi) for v in pooled[i]]) for lo, hi in BINS]
    ns = [sum(len(pooled[i]) for i in range(lo, hi)) for lo, hi in BINS]
    return means, ns


def rollout_series(tag):
    f = glob(f"{SCRATCH}/{tag}/dataset/*/eval_results_dataset.jsonl")[0]
    out = []
    for line in open(f):
        row = json.loads(line)
        txt = "\n".join((m.get("content") or "") for m in row.get("messages", []) if isinstance(m, dict))
        out.append([float(x) for x in OCC.findall(txt)])
    return out


def memoryless_series():
    """Scripted no-memory agent on the 24 canonical rows: per scan, report exactly the visible channels."""
    rows = [json.loads(l) for l in open(f"{HERE}/canon_np_proc24.jsonl")]
    row_ids = [r.get("input_metadata", {}).get("row_id") or r.get("id") for r in rows]
    out = []
    for rid in row_ids:
        variant = next((v for v in _VARIANTS if v in rid), "five_ch_wide")
        kwargs = dict(_VARIANTS[variant])
        kwargs["seed"] = band_seed(rid) % (2 ** 31 - 1)
        task = get_task_class("blind_spectrum_monitoring")(**kwargs)
        task.build_canonical_run_state()
        gt = resolved_gt(task, float(kwargs.get("W", 15.0)), float(kwargs.get("G", 9.0)))
        query = task.build_current_query()
        occs = []
        for _ in range(int(kwargs.get("num_instances", 30))):
            vis = set()
            for f_ in PEAK.findall(query.prompt):
                fv = float(f_)
                for gi, ch in enumerate(gt):
                    if abs(fv - ch["center_freq"]) <= ch["bandwidth"] / 2:
                        vis.add(gi)
            cfs = [gt[i]["center_freq"] for i in sorted(vis)]
            bws = [gt[i]["bandwidth"] for i in sorted(vis)]
            occs.append(occ_iou(cfs, bws, gt))
            sr = task.step(Response(action=ScanReport(transmitters=[
                Transmitter(center_freq=c, bandwidth=b, currently_active=True, estimated_power=-30.0)
                for c, b in zip(cfs, bws)]), metadata={}))
            if sr.done:
                break
            nq = getattr(sr, "next_query", None)
            if nq is not None:
                query = nq
        out.append(occs)
    return out


nomem, _ = bin_stats(memoryless_series())
icl, _ = bin_stats(rollout_series("icl_g7dncu2c_ep0"))

per_run4 = {j: rollout_series(f"dormc_{j}_ep4") for j in DORMC}
per_run0 = {j: rollout_series(f"dormc_{j}_ep0") for j in DORMC}
np4, n4 = bin_stats([s for j in DORMC for s in per_run4[j]])
np0, n0 = bin_stats([s for j in DORMC for s in per_run0[j]])
run_means4 = {j: bin_stats(per_run4[j])[0] for j in DORMC}
lo4 = [min(run_means4[j][b] for j in DORMC) for b in range(len(BINS))]
hi4 = [max(run_means4[j][b] for j in DORMC) for b in range(len(BINS))]

print("bin        no-mem    ICL   np-ep0  np-ep4  [run min–max]      n_ep4")
for i, lb in enumerate(LABELS):
    print(f"{lb:10} {nomem[i]:6.3f} {icl[i]:6.3f} {np0[i]:7.3f} {np4[i]:7.3f}  "
          f"[{lo4[i]:.3f}–{hi4[i]:.3f}]  {n4[i]:6d}")

# fine-grained slices for the HTML table (scan1, scan2, scans3-5, scans16-30)
def fine(per_scan_lists):
    pooled = [[] for _ in range(30)]
    for s in per_scan_lists:
        for i, v in enumerate(s[:30]):
            pooled[i].append(v)
    sl = lambda lo, hi: mean([v for i in range(lo, hi) for v in pooled[i]])
    return sl(0, 1), sl(1, 2), sl(2, 5), sl(15, 30)

f0 = fine([s for j in DORMC for s in per_run0[j]])
f4 = fine([s for j in DORMC for s in per_run4[j]])
fi = fine(rollout_series("icl_g7dncu2c_ep0"))
print("\nfine slices    scan1   scan2   3–5     16–30")
print(f"np-ep0 pooled {f0[0]:7.3f} {f0[1]:7.3f} {f0[2]:7.3f} {f0[3]:7.3f}")
print(f"np-ep4 pooled {f4[0]:7.3f} {f4[1]:7.3f} {f4[2]:7.3f} {f4[3]:7.3f}")
print(f"ICL           {fi[0]:7.3f} {fi[1]:7.3f} {fi[2]:7.3f} {fi[3]:7.3f}")

x = np.arange(len(LABELS))
w = 0.26
fig, ax = plt.subplots(figsize=(11.5, 6.0))
ax.bar(x - w, nomem, w, color="0.62",
       label="no-mem — PERFECT scripted agent, reports only what it currently sees (upper bound, no memory)")
ax.bar(x, icl, w, color="tab:purple",
       label="ICL — untrained base, full history in prompt, no notepad (g7dncu2c ep0)")
ax.bar(x + w, np4, w, color="tab:green",
       yerr=[np.subtract(np4, lo4), np.subtract(hi4, np4)], capsize=4,
       error_kw=dict(lw=1.3, ecolor="#14521c"),
       label="notepad — RL-trained: 4 replicate dormc16k runs POOLED, ep4 "
             "(s8e07n53·yp8deoer·kym4znjc·wqjyy66p, 16k per-turn cap); whisker = min–max of the 4 runs")
ax.plot(x + w, np0, marker="_", markersize=18, markeredgewidth=2.2, ls="none", color="black",
        label="black tick = the same 4 runs UNTRAINED (ep0, pooled)")
for i in range(len(LABELS)):
    ax.annotate(f"{np4[i]:.3f}", xy=(x[i] + w, hi4[i]), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8.5, color="darkgreen", fontweight="bold")
    ax.annotate(f"{icl[i]:.3f}", xy=(x[i], icl[i]), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8.5, color="tab:purple")

ax.annotate("ICL peaks at scans 6–10, then collapses as the history\ngrows — by 26–30 it is barely above the no-memory bound",
            xy=(4.9, icl[5] + 0.012), xytext=(2.9, 0.395), fontsize=9.5, color="tab:purple",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="tab:purple", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_xlabel("scan position within the 30-scan episode")
ax.set_ylabel("occ-IoU (higher = report closer to true transmitter set)")
ax.set_title("Memory dose-response by scan position: no-memory bound vs free in-context history vs "
             "trained external notepad\n(notepad: the 4 dormc16k replicates pooled, 768 rollouts/epoch; "
             "ICL: 288 rollouts; no-mem: deterministic engine replay, same 24 bands)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115), fontsize=8.6, framealpha=0.95)
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, 0.78)
fig.set_size_inches(11.5, 7.3)
fig.tight_layout()
fig.savefig(f"{HERE}/live_assets/chart_icl_vs_notepad_bins.png", dpi=150)
print("\nwritten chart_icl_vs_notepad_bins.png")
