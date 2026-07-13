"""dormc16k verdict chart: same 4-replicate design as dormc but with the per-turn output cap
raised 8192 -> 16384 (the single lever). Two panels:
  A) complete-episode mean_dorm ep0->ep4 rose in ALL 4 runs again (scan-1 flat everywhere ->
     clean notepad use, not weight baking).
  B) scans_completed ep4: the 16k cap fixed truncation for 3 of 4 runs (>=29 scans); only
     kym4znjc still drops (27.0, 60% complete) — and it is again a big-gain run, so the
     gain<->truncation coupling is weakened but not gone.
Data from autopsy_16k.py (complete-episode slices) + Output Stats epoch curves."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/Users/sten/Documents/Coding/fireworks/clbench-fireworks-rft"

# ordered by memory gain
runs   = ["s8e07n53", "yp8deoer", "kym4znjc", "wqjyy66p"]
dorm0  = [0.447, 0.444, 0.443, 0.445]   # complete-episode mean_dorm, ep0
dorm4  = [0.488, 0.545, 0.624, 0.626]   # complete-episode mean_dorm, ep4
scan0  = [29.953, 29.969, 29.969, 29.969]
scan4  = [29.953, 29.589, 27.010, 29.182]

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
              "(scan-1 shift ≤ +0.001 in every run -> notepad, not weights)")
axA.set_ylim(0, 0.80); axA.legend(loc="upper left", fontsize=9, framealpha=0.95)
axA.grid(axis="y", alpha=0.25)

# ---- Panel B: scans_completed (truncation mostly fixed) ----
axB.axhline(28, ls="--", color="tab:red", lw=1.2, alpha=0.8)
axB.text(-0.55, 26.6, "grace floor (28)\nguard = 0 above", color="tab:red",
         fontsize=8.5, va="top", ha="left")
axB.bar(x - w/2, scan0, w, color="0.7", label="ep0")
axB.bar(x + w/2, scan4, w, color="tab:orange", label="ep4")
for i in range(len(runs)):
    axB.annotate(f"{scan4[i]:.1f}", xy=(x[i]+w/2, scan4[i]),
                 xytext=(0, 4), textcoords="offset points", ha="center",
                 fontsize=9, color="#b45309", fontweight="bold")
axB.set_xticks(x); axB.set_xticklabels(runs, fontsize=9)
axB.set_ylabel("scans_completed (of 30)")
axB.set_title("B. 16k cap fixed truncation for 3 of 4 runs\n"
              "(only kym4znjc still drops scans — again a big-gain run)")
axB.set_ylim(0, 32); axB.legend(loc="lower left", fontsize=9, framealpha=0.95)
axB.grid(axis="y", alpha=0.25)

fig.suptitle("dormc16k verdict — one lever (cap 8192 -> 16384): memory repeats 4/4,\n"
             "truncation largely cured (3/4 finish ≥ 29 scans)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(f"{REPO}/live_assets/chart_dormc16k_verdict.png", dpi=150)
print("written chart_dormc16k_verdict.png")
