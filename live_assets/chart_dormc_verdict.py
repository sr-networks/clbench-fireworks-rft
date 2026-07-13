"""dormc verdict chart: the completion-guard replication (4 identical copies, lr 2e-4, 5 ep, 8 cand).
Two panels tell the whole story:
  A) complete-episode mean_dorm ep0->ep4 rose in ALL 4 runs (the repeatable memory-learning).
  B) scans_completed ep0->ep4 held ~29 only for xoi922eh; the other 3 truncated (the caveat: richer
     reports overrun the 8192 per-turn cap). Bigger dorm gain <-> heavier truncation.
Data from autopsy_dormc.py (complete-episode slices) + output_metrics curves (scans_completed ep0,ep4)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/Users/sten/Documents/Coding/fireworks/clbench-fireworks-rft"

# ordered by memory gain (also the truncation-severity order)
runs   = ["xoi922eh", "qv2mk5k0", "t37x35oj", "aic2o1up"]
dorm0  = [0.456, 0.444, 0.443, 0.448]   # complete-episode mean_dorm, ep0
dorm4  = [0.493, 0.566, 0.662, 0.715]   # complete-episode mean_dorm, ep4
scan0  = [29.911, 29.896, 29.979, 29.958]
scan4  = [29.099, 24.776, 24.557, 20.573]

x = np.arange(len(runs)); w = 0.38
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.6))

# ---- Panel A: complete-episode memory gain ----
axA.bar(x - w/2, dorm0, w, color="0.7", label="ep0 (untrained base)")
axA.bar(x + w/2, dorm4, w, color="tab:green", label="ep4 (RL-trained)")
for i in range(len(runs)):
    axA.annotate(f"+{dorm4[i]-dorm0[i]:.3f}", xy=(x[i]+w/2, dorm4[i]),
                 xytext=(0, 4), textcoords="offset points", ha="center",
                 fontsize=9, color="darkgreen", fontweight="bold")
axA.set_xticks(x); axA.set_xticklabels(runs, fontsize=9)
axA.set_ylabel("complete-episode mean_dorm (recallable-channel coverage)")
axA.set_title("A. Memory rose on FINISHED episodes in ALL 4 copies\n"
              "(scan-1 flat in every run -> clean notepad use, not weights)")
axA.set_ylim(0, 0.80); axA.legend(loc="upper left", fontsize=9, framealpha=0.95)
axA.grid(axis="y", alpha=0.25)

# ---- Panel B: scans_completed (the truncation caveat) ----
axB.axhline(28, ls="--", color="tab:red", lw=1.2, alpha=0.8)
axB.text(3.45, 28.2, "grace floor (28)\nguard = 0 above", color="tab:red",
         fontsize=8.5, va="bottom", ha="right")
axB.bar(x - w/2, scan0, w, color="0.7", label="ep0")
axB.bar(x + w/2, scan4, w, color="tab:orange", label="ep4")
for i in range(len(runs)):
    axB.annotate(f"{scan4[i]:.0f}", xy=(x[i]+w/2, scan4[i]),
                 xytext=(0, 4), textcoords="offset points", ha="center",
                 fontsize=9, color="#b45309", fontweight="bold")
axB.set_xticks(x); axB.set_xticklabels(runs, fontsize=9)
axB.set_ylabel("scans_completed (of 30)")
axB.set_title("B. ...but only xoi922eh stayed complete\n"
              "(the other 3 balloon reports -> overrun the 8192 per-turn cap)")
axB.set_ylim(0, 32); axB.legend(loc="lower left", fontsize=9, framealpha=0.95)
axB.grid(axis="y", alpha=0.25)

fig.suptitle("dormc verdict — completion guard: memory-learning is REPEATABLE (4/4), "
             "but 3/4 still truncate (token cap, not cheating)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{REPO}/live_assets/chart_dormc_verdict.png", dpi=150)
print("written chart_dormc_verdict.png")
