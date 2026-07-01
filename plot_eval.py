"""Plot CLBench-style stateful-vs-stateless per-hand reward curves for base vs finetuned.

    python plot_eval.py base.json finetuned.json --out poker_curve.png
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rolling(x, w=8):
    x = np.asarray(x, float)
    if len(x) < 1:
        return x
    return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="poker_curve.png")
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"base": "#9c8b86", "finetuned": "#9c3b2e"}
    for f in a.files:
        d = json.load(open(f))
        lab = d.get("label", f)
        c = colors.get(lab, None)
        for mode, style in (("stateful", "-"), ("stateless", "--")):
            y = d.get(mode, [])
            if not y:
                continue
            ax.plot(range(1, len(y) + 1), rolling(y), style, color=c, alpha=0.9,
                    label=f"{lab} · {mode} (rolling8 mean={np.mean(y):+.2f})")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("Hand index"); ax.set_ylabel("Chip reward (rolling mean)")
    ax.set_title("Poker continual learning: stateful (memory) vs stateless — base vs fine-tuned")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
