"""Replication + star-sweep chart: pscoc9fp vs its exact replication (rep2) + the three one-knob arms.
The headline: identical config, wildly different trajectory -> the pscoc9fp hold was seed luck."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/sten/Documents/Coding/fireworks/clbench-fireworks-rft"

pscoc9fp = [0.4610, 0.4154, 0.6274, 0.5475, 0.5547, 0.5172, 0.5542, 0.5362]
rep2     = [0.4512, 0.4214, 0.4336, 0.4549, 0.4004, 0.3919, 0.3734, 0.3454]
r4       = [0.4593, 0.4170, 0.4064, 0.4363, 0.4520, 0.4830, 0.4262, 0.4948]
r32      = [0.4407, 0.4723, 0.5035, 0.4945, 0.5001, 0.4483]          # ep7/8, partial
lr5e5    = [0.4676, 0.4222, 0.4322, 0.4034, 0.3961, 0.3969, 0.4416, 0.4726]

fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.axhspan(0.48, 0.56, color="tab:green", alpha=0.08, lw=0)
ax.text(0.08, 0.552, "the basin (late ≈ 0.48–0.56)", color="darkgreen", fontsize=9, va="top")

ax.plot(range(8), pscoc9fp, "-o", color="tab:green", lw=2.2,
        label="pscoc9fp — original (rank 8, lr 1e-4): held 6 epochs")
ax.plot(range(8), rep2, "--s", color="black", lw=2.2,
        label="o4g4u90z — EXACT replication: declined, never entered")
ax.plot(range(8), r4, "-^", color="tab:blue", lw=1.6,
        label="zk6w6fjn — rank 4: ends on its max (0.495)")
ax.plot(range(len(r32)), r32, "-D", color="tab:orange", lw=1.6,
        label="mx9x3c7t — rank 32 (ep 7/8): 3 epochs ≈0.50, then dip")
ax.plot(range(8), lr5e5, "-v", color="tab:purple", lw=1.6,
        label="cgplwg2t — lr 5e-5: U-shape, ends at its own ep0")

ax.annotate("same config as the green run,\ndifferent seed → opposite outcome",
            xy=(7, 0.3454), xytext=(4.35, 0.318), fontsize=9, color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=1))

ax.set_xlabel("epoch")
ax.set_ylabel("late_mean (scans 16–30 occupied-spectrum IoU), training-eval level")
ax.set_title("Replication verdict: pscoc9fp's held plateau did NOT reproduce —\n"
             "run-to-run (seed) variance dominates every knob we turned")
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
ax.grid(alpha=0.25)
ax.set_ylim(0.30, 0.66)
fig.tight_layout()
fig.savefig(f"{REPO}/live_assets/chart_replication_sweep.png", dpi=150)
print("written")
