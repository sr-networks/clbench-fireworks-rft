#!/usr/bin/env python3
"""Pretty-print RFT rollout traces (the rft-evalv3-<job>-epoch-N eval_results_dataset.jsonl files) on the CLI.

    firectl download dataset rft-evalv3-dtbn6lhm-epoch-0 --output-dir ./traces
    python view_trace.py traces/dataset/*/eval_results_dataset.jsonl              # list all rollouts
    python view_trace.py <file.jsonl> -i 3                                        # show rollout #3
    python view_trace.py <file.jsonl> -r sc2_7 --full                             # by row_id, untruncated
    python view_trace.py <file.jsonl> -i 0 --no-think                             # hide <think> blocks

Roles are colorized: user=cyan (scans + RUNNING LIST), assistant=green (tool calls; thinking dimmed),
tool=yellow (ok + SCAN_OCC). The eval reason (scans/mean_occ/memory_gain/score) is shown as a header.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

C = {"reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
     "user": "\033[36m", "assistant": "\033[32m", "tool": "\033[33m", "system": "\033[35m"}


def _c(color: str, s: str, on: bool) -> str:
    return f"{C[color]}{s}{C['reset']}" if on else s


def load(path: str):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def brief(row) -> str:
    er = row.get("evaluation_result") or {}
    rid = (row.get("input_metadata") or {}).get("row_id", "?")
    return f"{rid:<12} score={er.get('score', 0):.3f}  {er.get('reason', '')}"


def show(row, *, full: bool, think: bool, color: bool) -> None:
    er = row.get("evaluation_result") or {}
    rid = (row.get("input_metadata") or {}).get("row_id", "?")
    print(_c("bold", f"rollout {rid}", color))
    print(_c("bold", f"eval: {er.get('reason', '')}", color))
    print("=" * 100)
    lim = 10_000_000 if full else 700
    for m in row.get("messages", []):
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if role == "assistant":
            if content and think:
                t = content if full else content[:400]
                print(_c("dim", f"[assistant·think] {t}", color))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                print(_c("assistant", f"[assistant] {fn.get('name')}({args if full else args[:300]})", color))
        else:
            body = content if full else content[:lim]
            print(_c(role if role in C else "dim", f"[{role}] {body}", color))
        print("-" * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("-i", "--index", type=int, help="rollout index (omit to list all)")
    ap.add_argument("-r", "--row-id", help="select by row_id (first match; candidates share row_ids)")
    ap.add_argument("--full", action="store_true", help="no truncation")
    ap.add_argument("--no-think", action="store_true", help="hide assistant thinking")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()
    rows = load(a.file)
    color = not a.no_color and sys.stdout.isatty()

    if a.index is None and not a.row_id:
        print(f"{len(rows)} rollouts in {a.file}\n")
        for i, r in enumerate(rows):
            print(f"[{i:>3}] {brief(r)}")
        print("\nview one:  python view_trace.py <file> -i N   (or -r <row_id>)")
        return
    if a.row_id:
        sel = [r for r in rows if (r.get("input_metadata") or {}).get("row_id") == a.row_id]
        if not sel:
            sys.exit(f"no rollout with row_id {a.row_id}")
        row = sel[0]
    else:
        row = rows[a.index]
    show(row, full=a.full, think=not a.no_think, color=color)


if __name__ == "__main__":
    main()
