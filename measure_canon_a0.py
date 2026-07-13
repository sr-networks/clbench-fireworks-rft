"""Measure a0 — the MEMORYLESS occ-IoU floor of the canonical task, per variant.

a0 anchors the contrast reward `late - (early - a0)^2`: early-half performance is pinned to the score a
memoryless policy would get, so generic skill gains can't masquerade as memory and sandbagging is punished.
a0 must be a TASK property the policy cannot move, so it is computed by a SCRIPTED agent, not an LLM (an
LLM's early scans are already memory-contaminated by its own notepad from scan 2 on, and would drift with
training).

Memoryless policy convention (mirrors what a width-faithful current-scan reporter would emit): report every
detected peak in the CURRENT scan with its DETECTED width (`freq: X MHz | ... | width: Z MHz` lines).
A fixed-8MHz-width variant is computed as a sensitivity check.

Replicates the training series exactly: same rows as canon_np_nudge24.jsonl (variant + band_seed from
row_id), same 30-scan schedule, same occ_iou scorer. Deterministic — repeated runs give identical numbers.
"""

from __future__ import annotations

import json
import os
import re
import sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spectrum_adapter import band_seed
from bench_eval import load_default_schedule, resolved_gt, occ_iou
from src.registry import get_task_class  # type: ignore
from src.interface import Response  # type: ignore
from src.tasks.blind_spectrum_monitoring.task import ScanReport, Transmitter  # type: ignore

_PEAK = re.compile(r"freq:\s*([0-9.]+)\s*MHz\s*\|\s*power:[^|]*\|\s*width:\s*([0-9.]+)\s*MHz")
_VARIANTS = {st["variant"]: st["kwargs"] for st in load_default_schedule()}


def run_row(row_id: str, fixed_width: float | None = None):
    """Scripted memoryless rollout for one dataset row -> per-scan occ list."""
    variant = next((v for v in _VARIANTS if v in row_id), "five_ch_wide")
    kwargs = dict(_VARIANTS[variant])
    kwargs["seed"] = band_seed(row_id) % (2 ** 31 - 1)
    task = get_task_class("blind_spectrum_monitoring")(**kwargs)
    task.build_canonical_run_state()
    gt = resolved_gt(task, float(kwargs.get("W", 15.0)), float(kwargs.get("G", 9.0)))
    query = task.build_current_query()
    occs = []
    for _scan in range(int(kwargs.get("num_instances", 30))):
        peaks = [(float(f), float(w)) for f, w in _PEAK.findall(query.prompt)]
        cfs = [f for f, _ in peaks]
        bws = [fixed_width] * len(peaks) if fixed_width else [w for _, w in peaks]
        txs = [Transmitter(center_freq=c, bandwidth=b, currently_active=True, estimated_power=-30.0)
               for c, b in zip(cfs, bws)]
        sr = task.step(Response(action=ScanReport(transmitters=txs), metadata={}))
        occs.append(occ_iou(cfs, bws, gt))
        if sr.done:
            break
        nq = getattr(sr, "next_query", None)
        if nq is not None:
            query = nq
    return variant, occs


def main() -> None:
    rows = [json.loads(l) for l in open(os.path.join(HERE, "canon_np_nudge24.jsonl"))]
    row_ids = [r.get("input_metadata", {}).get("row_id") or r.get("id") for r in rows]
    for label, fw in [("detected widths (primary)", None), ("fixed 8 MHz (sensitivity)", 8.0)]:
        agg: dict[str, list[list[float]]] = {}
        for rid in row_ids:
            variant, occs = run_row(rid, fixed_width=fw)
            agg.setdefault(variant, []).append(occs)
        print(f"\n=== memoryless floor, {label} ===")
        a0 = {}
        for v, series in sorted(agg.items()):
            early = mean(mean(s[: len(s) // 2]) for s in series)
            late = mean(mean(s[len(s) // 2:]) for s in series)
            overall = mean(mean(s) for s in series)
            a0[v] = round(early, 4)
            print(f"  {v:24} early={early:.4f}  late={late:.4f}  overall={overall:.4f}  "
                  f"(n={len(series)} rows x {len(series[0])} scans)")
        print(f"  A0 = {a0}")


if __name__ == "__main__":
    main()
