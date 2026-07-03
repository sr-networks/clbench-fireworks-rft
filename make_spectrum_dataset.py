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

# EXPLICIT-ACCUMULATION design. The trace diagnosis (og3fniz6) showed the 1.7B treats each scan as INDEPENDENT
# ("no prior data -> report current peaks") and never explores accumulation, so RL had nothing to reinforce.
# So we now TELL it the key fact (dormant transmitters still occupy the band -> report ALL ever-seen) and the
# exact notepad procedure. This shifts the task from "RL discovers accumulation" (beyond the 1.7B) to "RL
# improves TOLD accumulation" (scaffold btalo63n showed the 1.7B can). Thinking stays ON (user wants the
# reasoning) but with a BRIEF-thinking nudge, since the vague-prompt thinking (og3fniz6) rambled hugely.
SYSTEM_PROMPT = (
    "You are a spectrum-monitoring analyst watching ONE fixed radio-frequency band over a SERIES of scans, "
    "shown ONE AT A TIME. Each scan lists noisy detected peaks (frequency, power, width).\n\n"
    "KEY FACT: transmitters do NOT go away. A transmitter that is silent (dormant) in one scan is STILL "
    "occupying the band — it just wasn't detected this scan, and it will often reappear later. Only some "
    "transmitters are active in any given scan; the rest are dormant but STILL PRESENT. You are scored on how "
    "well your report matches the band's TRUE occupied spectrum, which includes every dormant transmitter. So "
    "you must report EVERY transmitter you have ever seen — not just the ones detected in the current scan.\n\n"
    "You cannot look back at earlier scans; your notepad is the ONLY memory that persists. Tools:\n"
    "  - notepad_read(): read your notepad.\n"
    "  - notepad_write(text): overwrite your notepad with the full new contents.\n"
    "  - submit_report(center_freqs): submit your answer for the current scan; it returns the NEXT scan.\n\n"
    "Do exactly this every scan: (1) notepad_read to recall the transmitters seen so far; (2) add any NEW "
    "peaks from the current scan to that list; (3) notepad_write the full updated list back; (4) submit_report "
    "the FULL list — every remembered transmitter, dormant or active. Keep the notepad a simple comma-separated "
    "list of center frequencies. Think BRIEFLY, then act — keep your reasoning to a sentence or two, "
    "do not over-explain."
)

# SCAFFOLD variant (env-echoed running list; submit_report is the only tool). The memory channel is
# maintained BY THE ENV — the model must USE it (keep + extend + resubmit). This is the setting where RL
# improvement replicated (btalo63n / dmzj2mz8 / f4lgszxz), now re-run with never-repeating bands.
SCAFFOLD_PROMPT = (
    "You are a spectrum-monitoring analyst watching ONE fixed radio-frequency band over a SERIES of scans, "
    "shown ONE AT A TIME. Each scan lists noisy detected peaks (frequency, power, width).\n\n"
    "KEY FACT: transmitters do NOT go away. A transmitter that is silent (dormant) in one scan is STILL "
    "occupying the band and will often reappear later. You are scored on how well your report matches the "
    "band's TRUE occupied spectrum, which includes every dormant transmitter — so report EVERY transmitter "
    "you have ever seen, not just the ones detected in the current scan.\n\n"
    "You cannot look back at earlier scans. To help you, each scan includes YOUR RUNNING LIST — the report "
    "you submitted for the previous scan. Keep ALL of it, add any NEW peaks from the current scan, and "
    "submit the FULL updated list via submit_report(center_freqs) (each region is a fixed 8 MHz wide; it "
    "returns the next scan).\n\n"
    "Think BRIEFLY, then act — keep your reasoning to a sentence or two, do not over-explain."
)

# WEAK scaffold variant (ARM C): same scaffold mechanics but the system prompt does NOT teach accumulation —
# no KEY FACT, no "report every transmitter ever seen". The echo's own "keep + extend" line is still there
# (it was in the original scaffold runs too). This recreates the old weak-prompt regime (base ~0.2-0.3) where
# RL rises were observed — now on never-repeating bands: a rise here = RL genuinely trains memory use; flat =
# the old rises were prompt-substitutable and/or band-repetition artifacts.
SCAFFOLD_WEAK_PROMPT = (
    "You are a spectrum-monitoring analyst watching ONE fixed radio-frequency band over a SERIES of scans, "
    "shown ONE AT A TIME. Each scan lists noisy detected peaks (frequency, power, width). Each scan, report "
    "the center frequencies (MHz) of the transmitter regions that occupy the band via "
    "submit_report(center_freqs) (each region is a fixed 8 MHz wide; it returns the next scan). You are "
    "scored on how well your reported regions match the band's true occupied spectrum.\n\n"
    "You cannot look back at earlier scans. Each scan includes YOUR RUNNING LIST — the report you submitted "
    "for the previous scan.\n\n"
    "Think BRIEFLY, then act — keep your reasoning to a sentence or two, do not over-explain."
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
    ap.add_argument("--think", action="store_true",
                    help="thinking ON (omit /no_think); pair with a larger --max-output-tokens to avoid truncation")
    ap.add_argument("--scaffold", action="store_true", help="running-list scaffold prompt (submit_report only)")
    ap.add_argument("--scaffold-weak", action="store_true",
                    help="scaffold WITHOUT the explicit accumulate instruction (ARM C: old weak-prompt regime)")
    ap.add_argument("--prefix", default="spectrum",
                    help="row_id namespace: row_id=f'{prefix}_{i}' seeds the band, so a NEW prefix = bands "
                         "never seen by any previous run (all runs before 2026-07-02 shared the spectrum_* bands)")
    args = ap.parse_args()

    # /no_think: thinking OFF by default. Qwen3-1.7B's thinking chain otherwise exhausts the response budget
    # before the tool call (finish_reason=length -> rollouts truncate mid-series). --think re-enables it — give
    # it a bigger --max-output-tokens (the custom processor's windowed context leaves more room than McpGym did).
    base = SCAFFOLD_WEAK_PROMPT if args.scaffold_weak else (SCAFFOLD_PROMPT if args.scaffold else SYSTEM_PROMPT)
    prompt = base + (FEWSHOT if args.fewshot else "") + ("" if args.think else " /no_think")
    with open(args.out, "w") as f:
        for i in range(args.n):
            row = {
                "messages": [{"role": "system", "content": prompt}],
                "input_metadata": {
                    "row_id": f"{args.prefix}_{i}",
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
