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

# NOTEPAD arms (bench icl_notepad semantics): the INTERFACE is explained to BOTH arms (that a notepad exists
# and how to update it — that's mechanics, not strategy); only the nudge arm is told WHAT to use it for.
NP_COMMON = (
    "You are a spectrum-monitoring analyst working through a SERIES of scans of one radio band, shown ONE AT "
    "A TIME. Each scan lists noisy detected peaks (frequency, power, width). After each scan, submit your "
    "occupancy report with the submit_report tool: center_freqs = the center frequency (MHz) of every region "
    "you believe is occupied, and bandwidths = the width (MHz) of each region, in the same order. You are "
    "scored on how well your report matches the band's true occupancy.\n\n"
    "You CANNOT see earlier scans or your earlier reports. Each scan shows YOUR NOTEPAD — a text that "
    "persists from scan to scan. You may replace it by including the optional notepad_update field in your "
    "submit_report call; otherwise it stays unchanged. The notepad is the only thing that persists.\n\n"
    "Think BRIEFLY, then act — keep your reasoning to a sentence or two."
)
NP_NUDGE = (
    "\n\nIMPORTANT — use your notepad as memory: this is ONE fixed band, and transmitters that are silent in "
    "the current scan are usually still there (they transmit intermittently). Keep a complete list in your "
    "notepad of every transmitter you have EVER seen, with its observed width, and update it every scan. "
    "Report ALL transmitters on your notepad every scan, not just the ones detected now."
)

# PROCEDURAL notepad prompt (user-authored, 2026-07-07): same bench-native system_prompt slot, but the
# nudge is replaced by an explicit per-scan procedure (read -> merge -> report -> always save) and a fixed
# notepad format. Targets the measured failure ranking of the 1.7B (77% of merge opportunities wasted,
# read-back >80% in only ~3% of scans, half the notepad junk): the merge/read-back steps are spelled out
# and the notepad format is machine-like so copy+append is cheap.
NP_PROC = (
    "You are a spectrum-monitoring analyst working through a SERIES of scans of one fixed radio band, "
    "shown ONE AT A TIME.\n\n"
    "You cannot see earlier scans or earlier reports. The only persistent state is YOUR NOTEPAD. The "
    "notepad is replaced whenever you include notepad_update in submit_report.\n\n"
    "The transmitter set is persistent but only partly visible in each scan. A transmitter may be absent "
    "from the current detections and still be part of the true occupied band.\n\n"
    "At every scan, do exactly this:\n"
    "1. Read the existing notepad as memory.\n"
    "2. Merge the current detections into memory.\n"
    "   - Keep every old remembered transmitter.\n"
    "   - Add every newly observed transmitter.\n"
    "   - Do not replace memory with only the current scan.\n"
    "3. Submit a report based on the merged memory.\n"
    "4. Always include notepad_update.\n"
    "   - notepad_update must contain the complete merged memory after this scan.\n"
    "   - If an old item is missing from notepad_update, it is forgotten forever.\n\n"
    "Report format:\n"
    "- center_freqs: center frequency MHz of every remembered occupied region.\n"
    "- bandwidths: width MHz of each region, in the same order.\n"
    "- center_freqs and bandwidths must have the same length.\n\n"
    "Notepad format:\n"
    "MEMORY_TRANSMITTERS:\n"
    "- <freq_mhz> | <width_mhz>\n"
    "- <freq_mhz> | <width_mhz>\n\n"
    "Write only this list in the notepad. No explanations in the notepad.\n\n"
    "Think briefly, then call submit_report."
)

# Reworded neutral (same semantics, different text): the original neutral text was empirically implicated in
# the ICL-neutral job failures (probe: same evaluator + nudge dataset passes), mechanism unknown.
NEUTRAL2 = (
    "You are monitoring one radio-frequency band across a sequence of scans. Every scan shows the peaks a "
    "noisy detector found, each with a frequency, a power and a width. For each scan, call the submit_report "
    "tool with two same-length arrays: center_freqs, the center frequencies (MHz) of the regions you consider "
    "occupied, and bandwidths, their widths (MHz). Your score reflects how closely your report matches the "
    "band's true occupancy. Reason briefly — a sentence or two — before acting."
)

PROMPTS = {"neutral": COMMON, "nudge": COMMON + NUDGE, "neutral2": NEUTRAL2,
           "np-neutral": NP_COMMON, "np-nudge": NP_COMMON + NP_NUDGE, "np-proc": NP_PROC}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["neutral", "nudge", "neutral2", "np-neutral", "np-nudge", "np-proc"], required=True)
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
