#!/usr/bin/env python3
"""Generate the RFT dataset (EvaluationRow JSONL) for the blind-spectrum memory task.

Each row is just one rollout (a series of scans of one randomly-generated band). The band layout is
chosen per-rollout from fresh entropy inside the env (spectrum_adapter), NOT from the dataset — McpGym
ignores per-row environment_context — so all rows are interchangeable; we only need N of them. The
system prompt frames the memory task and the submit_report tool.

    python make_spectrum_dataset.py --n 48 --out spectrum48.jsonl
"""

from __future__ import annotations

import argparse
import json

# NOTEPAD design (agent-controlled memory). The prompt explains the task MECHANICS and the tools, reminds
# the agent that memory is useful and available, but deliberately does NOT tell it WHAT to store (no "keep
# the dormant transmitters / grow your list") — RL must discover that the notepad is the way to recall and
# cover dormant transmitters. Thinking is ON (no /no_think): the agent reasons before acting.
SYSTEM_PROMPT = (
    "You are a spectrum-monitoring analyst watching ONE fixed radio-frequency band over time. You receive a "
    "SERIES of scans of that band, ONE AT A TIME. Each scan lists noisy detected peaks (frequency, power, "
    "width). Only SOME transmitters are active in any given scan; others are temporarily dormant and may "
    "reappear in later scans. Each scan, you report — via submit_report — the center frequencies (MHz) of "
    "the transmitter regions that occupy the band (each region is a fixed 8 MHz wide, so you give "
    "frequencies only). You are scored on how well your reported regions match the band's true occupied "
    "spectrum.\n\n"
    "You work entirely through tools:\n"
    "  - notepad_read(): read your private notepad.\n"
    "  - notepad_write(text): overwrite your private notepad with the full new contents.\n"
    "  - submit_report(center_freqs): submit your answer for the current scan; it returns the NEXT scan.\n\n"
    "Important: you will NOT be able to look back at earlier scans — once you submit and move on, a scan is "
    "gone. Your notepad is the ONLY thing that persists from one scan to the next. Memory is always useful "
    "and available: keeping track of what you have learned helps you solve tasks. Decide for yourself what, "
    "if anything, is worth writing down.\n\n"
    "For each scan: think it through, read and/or update your notepad as you see fit, then call submit_report "
    "with your answer — the next scan comes back as its result. Reason step by step before you act."
)

# A worked example that DEMONSTRATES in-context accumulation (the report GROWS, keeping dormant
# transmitters from memory). Used to test whether the base model CAN in-context-learn this task when
# shown how — and, if so, to give RL a strong behavioural seed to amplify. Fake frequencies, clearly
# marked as an illustration so the model copies the BEHAVIOUR, not the numbers.
FEWSHOT = (
    "\n\nWORKED EXAMPLE (a different, illustrative band — copy the METHOD, not these numbers):\n"
    "  Scan 1 peaks near 25, 70, 140 -> report: 25, 70, 140\n"
    "  Scan 2 peaks near 25, 100 (70 and 140 are dormant this scan) -> report: 25, 70, 100, 140  "
    "(you KEEP 70 and 140 from memory and ADD 100)\n"
    "  Scan 3 peaks near 70, 165 -> report: 25, 70, 100, 140, 165  (KEEP all four, ADD 165)\n"
    "Notice the report GROWS every scan and never drops a transmitter once seen. Do exactly this for the "
    "REAL band below, using its own detected peaks."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="number of rows/rollouts")
    ap.add_argument("--out", default="spectrum48.jsonl")
    ap.add_argument("--fewshot", action="store_true", help="prepend a worked accumulation example (ICL test)")
    args = ap.parse_args()

    # /no_think: thinking OFF. Qwen3-1.7B's thinking chain otherwise exhausts the response budget before the
    # tool call (finish_reason=length -> rollouts truncate mid-series). Proven on the scaffold runs.
    prompt = SYSTEM_PROMPT + (FEWSHOT if args.fewshot else "") + " /no_think"
    with open(args.out, "w") as f:
        for i in range(args.n):
            row = {
                "messages": [{"role": "system", "content": prompt}],
                "input_metadata": {
                    "row_id": f"spectrum_{i}",
                    "dataset_info": {
                        "user_prompt_template": "{observation}",
                        "environment_context": {},  # ignored by McpGym; layout comes from env entropy
                    },
                },
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.n} rows -> {args.out}")


if __name__ == "__main__":
    main()
