#!/usr/bin/env python3
"""Dataset for the dbx (database_exploration memory) arm. One row = one episode spec:
a variant DB (schema mapping + quirk bundles, see make_dbx_variants.py) crossed with a
seed that drives the question permutation. 24 rows = 6 variants x 4 seeds, so every
GRPO batch sees all schema layouts — the layout is not learnable from the row mix.

The system prompt is the engine's own (dbx_engine.SYSTEM_PROMPT): procedural notepad
instructions, matching the np-proc style that trained in the spectrum arm.

    python make_dbx_dataset.py --n 24 --out dbx_canon24.jsonl
"""
from __future__ import annotations

import argparse
import json

from dbx_engine import SYSTEM_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.out, "w") as f:
        for i in range(args.n):
            variant, seed = i % 6, 1000 + i
            row = {
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
                "input_metadata": {
                    "row_id": f"dbx-v{variant}-s{seed}-{i}",
                    "dataset_info": {"user_prompt_template": "{observation}",
                                     "environment_context": {}},
                },
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {args.n} rows -> {args.out}")


if __name__ == "__main__":
    main()
