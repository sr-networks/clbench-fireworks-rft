"""CONTROL for the blind-spectrum memory task: confirm the late-minus-early IoU reward is a clean
memory signal. Drives the REAL adapter (random band per rollout) with two reference policies:

  - MEMORYLESS: report only the CURRENT scan's detected peaks (no cross-scan accumulation).
  - MEMORY:     accumulate peaks across ALL scans so far, report every region seen >=2 times.

A clean memory task requires MEMORYLESS late-early delta ~ 0 and MEMORY delta clearly > 0. Measured
(50 rollouts, 16 scans): memoryless ~ +0.00, memory ~ +0.07. The memory delta is the headroom RFT can
climb; the memoryless ~0 proves the metric is not a structural artifact.

    python control_spectrum.py
"""
from __future__ import annotations

import re
from statistics import mean, pstdev

from spectrum_adapter import SpectrumAdapter
from src.tasks.blind_spectrum_monitoring.task import ScanReport, Transmitter  # type: ignore

_PEAK = re.compile(r"freq:\s*([\d.]+)\s*MHz\s*\|\s*power:\s*(-?[\d.]+)\s*dBm\s*\|\s*width:\s*([\d.]+)")


def _peaks(prompt: str):
    return [(float(m.group(1)), float(m.group(3))) for m in _PEAK.finditer(prompt)]


BW = 8.0          # fixed reported bandwidth (matches true CH_BW) — removes a wasted DoF + bandwidth-carpet
MAX_REPORT = 16   # cap on regions per report — blocks blanketing the whole band (true count 11-14)


def _mk(centers):
    return ScanReport(transmitters=[
        Transmitter(center_freq=float(c), bandwidth=BW, currently_active=True, estimated_power=-30.0)
        for c in list(centers)[:MAX_REPORT]
    ])


def _rollout(adapter: SpectrumAdapter, policy: str):
    env = adapter.create_environment()
    obs, _ = env.reset()
    prompt = obs["prompt"]
    seen: dict = {}
    ious = []
    for _ in range(40):
        ps = _peaks(prompt)
        for f, w in ps:
            seen.setdefault(round(f / 6), []).append((f, w))
        if policy == "memoryless":
            centers = [f for f, w in ps]
        elif policy == "carpet":                 # blanket the band with MAX_REPORT evenly-spaced tiles
            centers = [4 + i * (176 / MAX_REPORT) for i in range(MAX_REPORT)]
        elif policy == "memory_all":             # accumulate everything seen (no false-alarm filter)
            centers = [mean(x[0] for x in v) for v in seen.values() if len(v) >= 1]
        else:                                    # memory: accumulate regions seen >=2 times
            centers = [mean(x[0] for x in v) for v in seen.values() if len(v) >= 2]
        obs, r, done, _t, _i = env.step(_mk(centers))
        prompt = obs["prompt"]
        ious.append(r)
        if done:
            break
    h = len(ious) // 2
    early = mean(ious[:h]) if h else 0.0
    late = mean(ious[h:]) if len(ious) >= 2 else mean(ious)
    return mean(ious), early, late, (late - early)


def _se(xs):
    n = len(xs)
    return pstdev(xs) * (n / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else 0.0


def main(reps: int = 50):
    ad = SpectrumAdapter()
    for policy in ("memoryless", "carpet", "memory_all", "memory"):
        means, earlies, lates, deltas = [], [], [], []
        for _ in range(reps):
            m, e, l, d = _rollout(ad, policy)
            means.append(m); earlies.append(e); lates.append(l); deltas.append(d)
        print(f"{policy:11s}: mean_occ={mean(means):.3f}  early={mean(earlies):.3f}  late={mean(lates):.3f}  "
              f"gain(late-early)={mean(deltas):+.3f} +/- {_se(deltas):.3f} (n={reps})")
    print("\nGOAL: pick a reward that ranks 'memory' #1. carpet should be a high-mean but ZERO-gain trap.")


if __name__ == "__main__":
    main()
