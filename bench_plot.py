#!/usr/bin/env python3
"""CLBench-style figure from bench_eval.py output: x = instance 1..90 (stage lines at 30/60),
two panels — bench-native available-IoU and our occupied-IoU. Curves = mean across runs
(rolling window for readability), one per (system, tag).

    python bench_plot.py bench_*.jsonl -o bench_headtohead.png
"""

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {  # (label, color, linestyle, width)
    ("nomem", "base"): ("no memory (base)", "#8a8a8a", ":", 2.0),
    ("icl", "base"): ("ICL — full history (base)", "#1f6fb2", "-", 2.0),
    ("echo", "base"): ("running-list memory (base)", "#7fbf7f", "--", 2.0),
    ("echo", "trained"): ("running-list memory (MEMORY-TRAINED)", "#1a7a2e", "-", 2.8),
}


def rolling(xs, w=7):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w // 2)
        seg = xs[lo:i + w // 2 + 1]
        out.append(sum(seg) / len(seg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", default="bench_headtohead.png")
    ap.add_argument("--window", type=int, default=7)
    args = ap.parse_args()

    rows = []
    for f in args.files:
        rows += [json.loads(l) for l in open(f) if l.strip()]
    series = defaultdict(lambda: defaultdict(list))   # (system,tag) -> instance -> [values per run]
    for r in rows:
        series[(r["system"], r.get("tag", ""))][r["instance"]].append(r)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for metric, ax, title in [("avail", axes[0], "CLBench-native metric: available-spectrum IoU"),
                              ("occ", axes[1], "memory-sensitive metric: occupied-spectrum IoU")]:
        for key, inst in sorted(series.items()):
            if key not in STYLE:
                continue
            label, color, ls, lw = STYLE[key]
            xs = sorted(inst)
            mean = [sum(r[metric] for r in inst[i]) / len(inst[i]) for i in xs]
            ax.plot([x + 1 for x in xs], rolling(mean, args.window), ls, color=color, lw=lw, label=label)
            ax.plot([x + 1 for x in xs], mean, ls, color=color, lw=0.7, alpha=0.25)
        for b in (30, 60):
            ax.axvline(b + 0.5, color="#bbb", lw=1, ls="--")
        ax.set_ylabel(("available-IoU" if metric == "avail" else "occupied-IoU"))
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[1].set_xlabel("instance (official CLBench default schedule: Wideband ->|<- Mixed ->|<- Full grid)")
    for ax in axes:
        ax.text(15, ax.get_ylim()[1]*0.97, "stage 1", ha="center", va="top", fontsize=9, color="#888")
        ax.text(45, ax.get_ylim()[1]*0.97, "stage 2", ha="center", va="top", fontsize=9, color="#888")
        ax.text(75, ax.get_ylim()[1]*0.97, "stage 3", ha="center", va="top", fontsize=9, color="#888")
    fig.suptitle("Qwen3-1.7B on the official CLBench blind-spectrum schedule (90 instances, 3 runs avg)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    # compact per-(system,tag) summary
    print(f"{'system':<28}{'mean avail':>11}{'mean occ':>10}{'occ s1':>8}{'occ s2':>8}{'occ s3':>8}")
    for key, inst in sorted(series.items()):
        label = STYLE.get(key, (str(key),))[0]
        allr = [r for i in inst for r in inst[i]]
        st = lambda s: (lambda v: sum(v)/len(v) if v else 0)([r["occ"] for r in allr if r["stage"] == s])
        print(f"{label:<28}{sum(r['avail'] for r in allr)/len(allr):>11.3f}{sum(r['occ'] for r in allr)/len(allr):>10.3f}"
              f"{st(0):>8.3f}{st(1):>8.3f}{st(2):>8.3f}")


if __name__ == "__main__":
    main()
