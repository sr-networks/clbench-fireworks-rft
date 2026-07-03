#!/usr/bin/env python3
"""Dataset for the BENCH-PURE experiment. One row = one official variant series (30 instances). The system
prompt is the ONLY experimental manipulation:

  --prompt neutral : task + response format only, no mention of memory/history
  --prompt nudge   : same, PLUS the memory nudge (use the conversation history; dormant transmitters persist)

    python make_canon_dataset.py --prompt nudge   --n 48 --out canon_nudge48.jsonl
    python make_canon_dataset.py --prompt neutral --n 48 --out canon_neutral48.jsonl
"""

from __future__ import annotations

import argparse
import json

VARIANTS = ["five_ch_wide", "five_plus_four_mixed", "full_grid_active"]

COMMON = (
    "You are a spectrum-monitoring analyst working through a SERIES of scans of one radio band. Each scan "
    "lists noisy detected peaks (frequency, power, width). After each scan, submit your occupancy report "
    "with the submit_report tool: center_freqs = the center frequency (MHz) of every region you believe is "
    "occupied, and bandwidths = the width (MHz) of each region, in the same order. You are scored on how "
    "well your report matches the band's true occupancy. Think BRIEFLY, then act — keep your reasoning to "
    "a sentence or two."
)

NUDGE = (
    "\n\nIMPORTANT — use your memory: this is ONE fixed band across all scans, and transmitters that are "
    "silent in the current scan are usually still there (they transmit intermittently). Your conversation "
    "history contains every earlier scan and report — re-read it, keep track of every transmitter you have "
    "EVER seen with its observed width, and report ALL of them every scan, not just the ones detected now."
)

PROMPTS = {"neutral": COMMON, "nudge": COMMON + NUDGE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["neutral", "nudge"], required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prompt = PROMPTS[args.prompt]
    with open(args.out, "w") as f:
        for i in range(args.n):
            variant = VARIANTS[i % len(VARIANTS)]
            row = {
                "messages": [{"role": "system", "content": prompt}],
                "input_metadata": {
                    "row_id": f"canon-{variant}-{i}",
                    "dataset_info": {"user_prompt_template": "{observation}", "environment_context": {}},
                },
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.n} rows ({args.prompt}) -> {args.out}")


if __name__ == "__main__":
    main()
