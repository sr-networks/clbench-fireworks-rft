"""Build products_small.db — a deterministic, evaluator-shippable replica of the CLBench
database_exploration products.db (400MB -> target <10MB).

Preserves everything the memory task needs the model to DISCOVER and write to its notepad:
  - the 3-group table layout (g1 office / g2 electronics / g3 musical instruments),
    including the asymmetries: no attrs_g2, no taxn_g3, distractor fdbk_stats_g1
  - the cryptic column names (prc, vrf, ts, str_nm, rtg, hlp_ct, ...)
  - the unit quirks: g1 prc dollars REAL + ts epoch-ms + vrf 0/1; g2 prc CENTS INT +
    ts epoch-s + vrf 'true'/'false' TEXT; g3 prc dollars + ts ISO date TEXT + vrf 0/1
Sampling is deterministic (ORDER BY ref_id LIMIT N) so the DB is byte-reproducible from
the canonical products.db. Long text columns are truncated to keep prompts and size small.
"""
import sqlite3
import sys
from pathlib import Path

SRC = Path("/Users/sten/Library/Python/3.11/lib/python/site-packages/data/database_exploration/products.db")
OUT = Path(__file__).parent / "dbx_data" / "products_small.db"

ITEMS_PER_GROUP = 300
FDBK_PER_ITEM = 12          # cap reviews per item
TEXT_TRUNC = 160            # chars kept of body/desc/feat text


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    src = sqlite3.connect(str(SRC))
    dst = sqlite3.connect(str(OUT))

    # replicate CREATE TABLE statements verbatim (schema fidelity incl. types)
    for (ddl,) in src.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name"
    ):
        dst.execute(ddl)

    kept: dict[str, list] = {}
    for g in ("g1", "g2", "g3"):
        refs = [r[0] for r in src.execute(
            f"SELECT ref_id FROM items_{g} ORDER BY ref_id LIMIT {ITEMS_PER_GROUP}")]
        kept[g] = refs
        ph = ",".join("?" * len(refs))

        cols = [d[1] for d in src.execute(f"PRAGMA table_info(items_{g})")]
        for row in src.execute(f"SELECT * FROM items_{g} WHERE ref_id IN ({ph})", refs):
            row = list(row)
            for i, c in enumerate(cols):
                if c in ("desc_txt", "feat_lst") and isinstance(row[i], str):
                    row[i] = row[i][:TEXT_TRUNC]
            dst.execute(f"INSERT INTO items_{g} VALUES ({','.join('?' * len(row))})", row)

        # feedback: up to FDBK_PER_ITEM per kept item, deterministic by id
        fcols = [d[1] for d in src.execute(f"PRAGMA table_info(fdbk_{g})")]
        for ref in refs:
            for row in src.execute(
                f"SELECT * FROM fdbk_{g} WHERE ref_id = ? ORDER BY id LIMIT {FDBK_PER_ITEM}", (ref,)):
                row = list(row)
                for i, c in enumerate(fcols):
                    if c in ("ttl", "body") and isinstance(row[i], str):
                        row[i] = row[i][:TEXT_TRUNC]
                dst.execute(f"INSERT INTO fdbk_{g} VALUES ({','.join('?' * len(row))})", row)

        for aux in (f"attrs_{g}", f"taxn_{g}"):
            if src.execute("SELECT COUNT(*) FROM sqlite_master WHERE name=?", (aux,)).fetchone()[0]:
                for row in src.execute(f"SELECT * FROM {aux} WHERE ref_id IN ({ph})", refs):
                    dst.execute(f"INSERT INTO {aux} VALUES ({','.join('?' * len(row))})", row)

    ph = ",".join("?" * len(kept["g1"]))
    for row in src.execute(f"SELECT * FROM fdbk_stats_g1 WHERE ref_id IN ({ph})", kept["g1"]):
        dst.execute(f"INSERT INTO fdbk_stats_g1 VALUES ({','.join('?' * len(row))})", row)

    dst.commit()
    dst.execute("VACUUM")
    dst.close()

    chk = sqlite3.connect(str(OUT))
    print(f"{OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    for (t,) in chk.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        n = chk.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:15} {n:>6} rows")
    # quirk spot-checks
    print("g1 prc sample:", chk.execute("SELECT prc, ts, vrf FROM items_g1 JOIN fdbk_g1 USING(ref_id) LIMIT 1").fetchone())
    print("g2 prc sample:", chk.execute("SELECT prc, ts, vrf FROM items_g2 JOIN fdbk_g2 USING(ref_id) LIMIT 1").fetchone())
    print("g3 prc sample:", chk.execute("SELECT prc, ts, vrf FROM items_g3 JOIN fdbk_g3 USING(ref_id) LIMIT 1").fetchone())


if __name__ == "__main__":
    sys.exit(main())
