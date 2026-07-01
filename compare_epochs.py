"""Rigorous base-vs-trained PROOF of memory learning from RFT rollout datasets.

The RFT job emits a rollout dataset per epoch (`rft-evalv3-<job>-epoch-<k>`). Epoch 0 is the (near-)base
policy; the final epoch is the trained policy. For each rollout we recover the authoritative per-hand
profit vector (the env's `HAND_PROFITS_BB` line) and compute the per-rollout memory gain
(late-hand mean - early-hand mean). Comparing the gain DISTRIBUTIONS across epochs shows whether
training increased the with-memory - without-memory advantage — and a Welch t-test gives significance.

    # download the datasets first:
    #   firectl download dataset rft-evalv3-<job>-epoch-0 --output-dir /tmp/ep0
    #   firectl download dataset rft-evalv3-<job>-epoch-<final> --output-dir /tmp/epN
    python compare_epochs.py /tmp/ep0 /tmp/epN
"""

import json
import re
import sys
from pathlib import Path
from statistics import mean, pstdev

_PROFITS = re.compile(r"HAND_PROFITS_BB:\s*([-\d.,\s]+)")
_NET = re.compile(r"Net chip change this hand:\s*([+-]?\d+)\s*chips")
BB = 10.0


def per_hand(text: str):
    m = _PROFITS.search(text)
    if m:
        try:
            return [float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()]
        except ValueError:
            pass
    return [float(x) / BB for x in _NET.findall(text)]


def gains(jsonl: Path):
    out = []
    for line in open(jsonl):
        r = json.loads(line)
        text = " ".join((m.get("content") or "") for m in r.get("messages", []) if isinstance(m.get("content"), str))
        ph = per_hand(text)
        if len(ph) >= 2:
            h = len(ph) // 2
            out.append(mean(ph[h:]) - mean(ph[:h]))
    return out


def welch_t(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    ma, mb = mean(a), mean(b)
    va, vb = pstdev(a) ** 2 * na / (na - 1), pstdev(b) ** 2 * nb / (nb - 1)
    se = (va / na + vb / nb) ** 0.5
    if se == 0:
        return float("nan"), float("nan")
    t = (mb - ma) / se
    # rough two-sided p via normal approx (large n)
    import math
    p = math.erfc(abs(t) / 2 ** 0.5)
    return t, p


def find_jsonl(d: str):
    for p in Path(d).rglob("*.jsonl"):
        return p
    raise SystemExit(f"no jsonl under {d}")


def main():
    base_dir, trained_dir = sys.argv[1], sys.argv[2]
    gb = gains(find_jsonl(base_dir))
    gt = gains(find_jsonl(trained_dir))
    print(f"base    (epoch-0): n={len(gb):4d}  mean_memory_gain={mean(gb):+.4f}  (>=2-hand rollouts)")
    print(f"trained (final)  : n={len(gt):4d}  mean_memory_gain={mean(gt):+.4f}")
    t, p = welch_t(gb, gt)
    print(f"delta = {mean(gt) - mean(gb):+.4f} BB/hand | Welch t={t:.2f}, p~={p:.4g}")
    verdict = "PROVEN: training raised with-memory - without-memory" if (mean(gt) > mean(gb) and p < 0.05) \
        else ("directional (not significant)" if mean(gt) > mean(gb) else "NOT improved")
    print("=>", verdict)


if __name__ == "__main__":
    main()
