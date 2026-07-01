#!/usr/bin/env python3
"""Generate the RFT dataset as EvaluationRow JSONL for the FIXED-OPPONENT clbench poker setup.

Design choice (see README "Design choice: fixed opponent vs curriculum"): each rollout plays a
SINGLE fixed named opponent for the whole hand-series — one of the three canonical variants:
    calling_station (Tom) | fit_or_fold (Adam) | loose_aggressive (Alex)
chosen balanced across rollouts. This is the clbench single-variant mode (variant=...), which shows
the opponent's NAME on every hand (Tom/Adam/Alex) and uses the canonical defaults. We deliberately do
NOT use the 5-stage curriculum schedule here, so the late-vs-early memory reward is unconfounded by
opponent switches (the curriculum is still available via `schedule="default"`).

Each row differs by `seed`, which fully changes the dealt cards (variant mode reseeds per run), giving
genuine card diversity across rollouts. The system prompt is the canonical icl_notepad prompt verbatim
— generic, with NO hint of the opponent taxonomy, so the model must DISCOVER each opponent from play.

    python make_dataset.py --n 96 --num-instances 20 --out poker_fixed96.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

# The three canonical opponents (policy id -> opponent name shown to the agent). We pass
# opponent_policy + opponent_name directly rather than `variant`, which is byte-identical in what the
# model sees (same prompt, same cards, same reward — verified) but needs NO benchmark data files at
# runtime, so the cloud evaluator works without vendoring variants/ into the installed package.
VARIANTS = ["calling_station", "fit_or_fold", "loose_aggressive"]
NAMES = {"calling_station": "Tom", "fit_or_fold": "Adam", "loose_aggressive": "Alex"}

# Canonical icl_notepad system prompt (verbatim from the benchmark's icl_notepad agent config) + Qwen3
# `/no_think`. Both Qwen3 tool-calling regimes leak on the 1.7B; /no_think is the LESSER evil (measured):
#   - thinking ON  -> the <think> chain exhausts even max_tokens=2048 before the tool call -> rollout
#     dies at the first turn (hands_completed 0.49/192, ~1% complete).
#   - /no_think     -> ~5% of turns emit the poker_act call as BARE JSON in content (after an empty
#     <think></think>) instead of a structured tool_call, and one such turn ends the rollout -> ~8%
#     complete (4.59 hands avg) but the other 95% of turns ARE structured tool_calls that work.
# So we keep /no_think and get enough full rollouts via more candidates/epochs (training also raises
# completion as it learns to format). The brief reasoning still goes in poker_act's `thinking` field.
SYSTEM_PROMPT = (
    "You are playing heads-up poker. You have access to a notepad where you can track observations. "
    "Use the notepad to identify patterns and adapt your strategy to maximize profit. /no_think"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96, help="number of rows/rollouts")
    ap.add_argument("--num-instances", type=int, default=20, help="hands per rollout (>= ~8 for a late-early signal)")
    ap.add_argument("--variant", default="mixed", help="'mixed' = balanced over the 3 variants; or pin to one")
    ap.add_argument("--out", default="poker_fixed96.jsonl")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    args = ap.parse_args()

    if args.variant == "mixed":
        assign = [VARIANTS[i % len(VARIANTS)] for i in range(args.n)]
        random.Random(args.shuffle_seed).shuffle(assign)
    else:
        assign = [args.variant] * args.n

    with open(args.out, "w") as f:
        for i in range(args.n):
            row = {
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
                "input_metadata": {
                    "row_id": f"poker_{assign[i]}_s{1000 + i}",
                    "dataset_info": {
                        "user_prompt_template": "{observation}",
                        "environment_context": {
                            "opponent_policy": assign[i],
                            "opponent_name": NAMES[assign[i]],
                            "num_instances": args.num_instances,
                            "seed": 1000 + i,
                        },
                    },
                },
            }
            f.write(json.dumps(row) + "\n")
    from collections import Counter
    print(f"wrote {args.n} rows -> {args.out}  | variants: {dict(Counter(assign))} | hands/rollout: {args.num_instances}")


if __name__ == "__main__":
    main()
