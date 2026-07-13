"""Build the Stage-2/3 MIXED dataset: spectrum (dormc) rows + dbx rows in one jsonl.

Each row already carries its own system prompt and a dispatch-ready row_id
("canon-<variant>-<i>" = spectrum, "dbx-v<variant>-s<seed>-<i>" = dbx), so mixing is
pure interleaving — the mix processor/reward route on the row_id prefix.

Balance note: 50/50 by ROWS (the Stage-3 pilot spec). dbx episodes are longer in tokens
(15 questions x up to 8 queries) than spectrum dormc episodes; if token-balancing is
needed later, pass --dbx-frac to change the row share (measured token ratio decides).
GRPO makes the reward scales safe to mix regardless: every candidate group is one row =
one task, so within-group advantages never compare spectrum scores with dbx scores.
"""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectrum", default="canon_np_proc24.jsonl")
    ap.add_argument("--dbx", default="dbx_canon24.jsonl")
    ap.add_argument("--dbx-frac", type=float, default=0.5,
                    help="fraction of rows that are dbx (default 0.5 = pilot spec)")
    ap.add_argument("--seed", type=int, default=0, help="interleave shuffle seed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = [json.loads(l) for l in open(HERE / args.spectrum)]
    dbx = [json.loads(l) for l in open(HERE / args.dbx)]
    if args.dbx_frac != 0.5:
        # keep ALL rows of the larger side, trim the other to hit the fraction
        if args.dbx_frac > 0.5:
            spec = spec[: round(len(dbx) * (1 - args.dbx_frac) / args.dbx_frac)]
        else:
            dbx = dbx[: round(len(spec) * args.dbx_frac / (1 - args.dbx_frac))]
    rows = spec + dbx
    for r in rows:  # dispatch key must be present and unambiguous
        rid = r["input_metadata"]["row_id"]
        assert rid.startswith(("canon-", "dbx-")), rid
    random.Random(args.seed).shuffle(rows)
    with open(HERE / args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_dbx = sum(r["input_metadata"]["row_id"].startswith("dbx-") for r in rows)
    print(f"wrote {len(rows)} rows -> {args.out} ({n_dbx} dbx, {len(rows) - n_dbx} spectrum)")


if __name__ == "__main__":
    main()
