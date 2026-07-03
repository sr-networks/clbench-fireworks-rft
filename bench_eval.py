#!/usr/bin/env python3
"""CLBench-style head-to-head eval on the OFFICIAL blind-spectrum default schedule (3 stages x 30 instances,
variants five_ch_wide -> five_plus_four_mixed -> full_grid_active, seeds 42/43/44, 168 MHz band, 13 channels),
comparing memory systems on the SAME model family:

    nomem      : context = [system, current scan] only (no memory at all)
    icl        : bench-style full-history ICL with FIFO truncation (token-budget approximated by chars)
    echo       : our running-list scaffold (windowed + previous-report echo)  <- matches RL training format

Run each system against a base and/or memory-trained model endpoint (Fireworks deployment), score every
instance with BOTH metrics (bench-native available-IoU from the task itself + our occupied-IoU vs ground
truth), and write one JSONL line per instance for plotting (bench_plot.py).

    python bench_eval.py --mock accumulate --systems echo --runs 1          # offline pipeline check
    python bench_eval.py --model "<model-id>[#deployment]" --systems nomem,icl,echo --runs 3 --out results.jsonl

The action format is identical across systems (one submit_report tool), so curves differ only in memory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# reuse the vendored-template redirect from the adapter, then the bench task machinery
import spectrum_adapter  # noqa: F401  (installs the Jinja template redirect)
from src.registry import get_task_class  # type: ignore
from src.interface import Response  # type: ignore
from src.tasks.blind_spectrum_monitoring.task import ScanReport, Transmitter  # type: ignore

# Schedule/variant JSONs: the pip-installed src package ships WITHOUT these data files (same issue as the
# Jinja templates), and the evaluator upload excludes gitignored vendored copies — so resolve in order:
# local vendored dir -> site-packages -> fetch from the bench's public GitHub into the vendored dir.
_VENDORED = os.path.join(HERE, "canon_fixtures")
_BENCH_RAW = "https://raw.githubusercontent.com/sr-networks/continual-learning-bench/main/src/tasks/blind_spectrum_monitoring"
_FIXTURES = ["schedules/default.json"] + [f"variants/{v}.json" for v in
             ("five_ch_wide", "five_plus_four_mixed", "full_grid_active")]

def _fixtures_dir() -> str:
    if os.path.isfile(os.path.join(_VENDORED, "schedules", "default.json")):
        return _VENDORED
    sp = os.path.join(os.path.dirname(sys.modules["src.registry"].__file__), "tasks", "blind_spectrum_monitoring")
    if os.path.isfile(os.path.join(sp, "schedules", "default.json")):
        return sp
    for rel in _FIXTURES:  # cloud: fetch once from the bench's public repo
        dst = os.path.join(_VENDORED, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        urllib.request.urlretrieve(f"{_BENCH_RAW}/{rel}", dst)
    return _VENDORED

SRC_TASK = _fixtures_dir()
BAND = 168.0
ICL_CHAR_BUDGET = 60_000          # ~ qwen3-1.7b context after headroom; FIFO-truncate beyond this (bench-style)
MAX_REPORT = 24                   # bench band has 13 channels; give some slack, still no carpeting

SYSTEM_NOMEM = (
    "You are a spectrum-monitoring analyst. Each message shows ONE scan of a radio band: noisy detected peaks "
    "(frequency, power, width). Report the center frequencies (MHz) of the transmitter regions that occupy the "
    "band via the submit_report tool. You are scored on how well your report matches the band's true occupied "
    "spectrum. Think BRIEFLY, then act."
)
SYSTEM_ICL = (
    "You are a spectrum-monitoring analyst watching ONE fixed radio-frequency band over a SERIES of scans. "
    "Each scan lists noisy detected peaks (frequency, power, width). Some transmitters are dormant in any given "
    "scan but still occupy the band. Each scan, report the center frequencies (MHz) of ALL transmitter regions "
    "that occupy the band via the submit_report tool — use the full conversation history to recall transmitters "
    "seen in earlier scans. You are scored against the band's true occupied spectrum. Think BRIEFLY, then act."
)
SYSTEM_ECHO = (
    "You are a spectrum-monitoring analyst watching ONE fixed radio-frequency band over a SERIES of scans, "
    "shown ONE AT A TIME. Each scan lists noisy detected peaks (frequency, power, width). Each scan, report "
    "the center frequencies (MHz) of the transmitter regions that occupy the band via submit_report(center_freqs). "
    "You are scored on how well your reported regions match the band's true occupied spectrum.\n\n"
    "You cannot look back at earlier scans. Each scan includes YOUR RUNNING LIST — the report you submitted "
    "for the previous scan.\n\n"
    "Think BRIEFLY, then act — keep your reasoning to a sentence or two, do not over-explain."
)

TOOLS = [{"type": "function", "function": {
    "name": "submit_report",
    "description": "Submit your report for the CURRENT scan: center_freqs = center frequencies (MHz) of the occupied regions.",
    "parameters": {"type": "object", "properties": {
        "center_freqs": {"type": "array", "items": {"type": "number"}}}, "required": ["center_freqs"]},
}}]


# ---------------- official schedule ----------------

def load_default_schedule():
    import inspect
    cls = get_task_class("blind_spectrum_monitoring")
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    sched = json.load(open(os.path.join(SRC_TASK, "schedules", "default.json")))
    stages = []
    for st in sched["stages"]:
        var = json.load(open(os.path.join(SRC_TASK, "variants", st["variant"] + ".json")))
        # filter to ctor-accepted kwargs (some variants carry extra keys like n_instances that the
        # bench's own runner strips); the schedule's num_instances/seed override variant defaults.
        kwargs = {k: v for k, v in var["defaults"].items() if k in accepted}
        kwargs["num_instances"] = st["schedule"]["num_instances"]
        kwargs["seed"] = st["schedule"]["seed"]
        stages.append({"variant": st["variant"], "kwargs": kwargs})
    return stages


def resolved_gt(task, W: float, G: float) -> list[dict]:
    """Ground-truth channels with resolved center/bandwidth, for occupied-IoU. The task's ChannelDef objects
    resolve grid slots via center_freq(W, G) / bandwidth(W), honoring narrowband overrides."""
    return [{"center_freq": float(d.center_freq(W, G)), "bandwidth": float(d.bandwidth(W))}
            for d in task._get_all_latent_channel_defs()]


def occ_iou(report: list[float], widths: list[float], gt: list[dict]) -> float:
    res, nb = 0.5, int(BAND / 0.5) + 1
    g, r = bytearray(nb), bytearray(nb)
    def paint(arr, cf, bw):
        lo, hi = max(0.0, cf - bw / 2), min(BAND, cf + bw / 2)
        for i in range(int(lo / res), min(int(hi / res) + 1, nb)):
            arr[i] = 1
    for ch in gt:
        paint(g, ch["center_freq"], ch["bandwidth"])
    for cf, bw in zip(report, widths):
        paint(r, cf, bw)
    inter = sum(1 for a, b in zip(g, r) if a and b)
    union = sum(1 for a, b in zip(g, r) if a or b)
    return inter / union if union else 0.0


# ---------------- model calls ----------------

def call_model(model: str, messages: list[dict], temperature: float, api_key: str, max_tokens: int = 4096):
    body = json.dumps({"model": model, "messages": messages, "tools": TOOLS,
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request("https://api.fireworks.ai/inference/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())["choices"][0]["message"]
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(3 * (attempt + 1))


PEAK = re.compile(r"freq:\s*([0-9.]+)\s*MHz")
NUMS = re.compile(r"-?\d+\.?\d*")

def parse_report(msg) -> list[float]:
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        if fn.get("name") == "submit_report":
            try:
                return [float(x) for x in json.loads(fn.get("arguments") or "{}").get("center_freqs") or []][:MAX_REPORT]
            except Exception:
                pass
    content = msg.get("content") or ""          # fallback: last bracketed list in text
    m = re.findall(r"\[([0-9,.\s]+)\]", content)
    if m:
        try:
            return [float(x) for x in NUMS.findall(m[-1])][:MAX_REPORT]
        except Exception:
            pass
    return []


def mock_reply(kind: str, messages: list[dict], state: dict):
    """Offline mock policies: 'current' (memoryless) or 'accumulate' (keep everything seen in this system's view)."""
    prompt = messages[-1]["content"]
    det = [float(x) for x in PEAK.findall(prompt)]
    if kind == "accumulate":
        seen = state.setdefault("seen", [])
        for f in det:
            if all(abs(f - s) > 1.0 for s in seen):
                seen.append(f)
        rep = sorted(seen)[:MAX_REPORT]
    else:
        rep = det
    return {"content": "", "tool_calls": [{"function": {"name": "submit_report",
            "arguments": json.dumps({"center_freqs": rep})}}]}


# ---------------- the three memory systems ----------------

def run_one(system: str, model: str, run_idx: int, temperature: float, api_key: str, mock: str | None):
    stages = load_default_schedule()
    rows, inst_idx = [], 0
    hist: list[dict] = []                      # icl running history
    prev_report: list[float] = []              # echo memory
    mock_state: dict = {}                      # mock accumulate memory (per full session)
    sys_prompt = {"nomem": SYSTEM_NOMEM, "icl": SYSTEM_ICL, "echo": SYSTEM_ECHO}[system]

    for si, st in enumerate(stages):
        task = get_task_class("blind_spectrum_monitoring")(**st["kwargs"])
        task.build_canonical_run_state()
        gt = resolved_gt(task, float(st["kwargs"].get("W", 15.0)), float(st["kwargs"].get("G", 9.0)))
        query = task.build_current_query()
        done = False
        while not done:
            prompt = query.prompt
            if system == "echo" and prev_report:
                prompt += ("\n=== YOUR RUNNING LIST (your previous report) ===\n"
                           + ", ".join(f"{f:.1f}" for f in prev_report)
                           + "\nKeep ALL of these, add any NEW peaks from this scan, and submit the FULL updated list.\n")
            if system == "icl":
                hist.append({"role": "user", "content": prompt})
                msgs = [{"role": "system", "content": sys_prompt}]
                tail, total = [], 0
                for m in reversed(hist):
                    total += len(m["content"] or "")
                    if total > ICL_CHAR_BUDGET:
                        break
                    tail.append(m)
                msgs += list(reversed(tail))
            else:
                msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]

            reply = mock_reply(mock, msgs, mock_state) if mock else call_model(model, msgs, temperature, api_key)
            report = parse_report(reply)
            if system == "icl":
                hist.append({"role": "assistant", "content": f"submit_report({report})"})
            prev_report = report or prev_report

            widths = [8.0] * len(report)       # reported regions: fixed nominal width (as in training)
            txs = [Transmitter(center_freq=c, bandwidth=w, currently_active=True, estimated_power=-30.0)
                   for c, w in zip(report, widths)]
            sr = task.step(Response(action=ScanReport(transmitters=txs), metadata={}))
            oc = getattr(sr, "instance_outcome", None)
            avail = float(getattr(oc, "reward", 0.0) or 0.0) if oc is not None else 0.0
            rows.append({"system": system, "model": model, "run": run_idx, "stage": si,
                         "instance": inst_idx, "occ": occ_iou(report, widths, gt), "avail": avail,
                         "n_report": len(report)})
            inst_idx += 1
            done_stage = bool(sr.done)
            nq = getattr(sr, "next_query", None)
            if nq is not None and not done_stage:
                query = nq
            else:
                done = True
        # stage ends; memory (hist/prev_report/mock_state) persists into the next stage
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock")
    ap.add_argument("--systems", default="echo")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--mock", choices=["current", "accumulate"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    all_rows = []
    for system in args.systems.split(","):
        for run in range(args.runs):
            t0 = time.time()
            rows = run_one(system.strip(), args.model, run, args.temperature, api_key, args.mock)
            occ = sum(r["occ"] for r in rows) / len(rows)
            av = sum(r["avail"] for r in rows) / len(rows)
            print(f"[{system} run{run}] {len(rows)} instances  mean_occ={occ:.3f}  mean_avail={av:.3f}  ({time.time()-t0:.0f}s)", flush=True)
            for r in rows:
                r["tag"] = args.tag
            all_rows += rows
    if args.out:
        mode = "a" if os.path.exists(args.out) else "w"
        with open(args.out, mode) as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(all_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
