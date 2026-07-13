"""Build the 6 dbx VARIANT databases — the anti-gridbake defense.

Why: products_small.db is fixed, so over epochs a policy could bake into its WEIGHTS either
the category->suffix mapping ('office products are items_g1') or the quirk facts ('g2 prices
are cents'), then sandbag question 1 to dodge the anchor hinge (the latecarpet trick). With
6 variants the mapping and the quirk assignment CHANGE per dataset row, so those facts are
only knowable inside an episode — i.e. only earnable via the notepad. (Same design as the
per-row band seeds in the spectrum arm.)

Variant v (0..5) = permutation P_v of categories over suffix slots (g1,g2,g3) combined with
a DERANGED assignment of quirk bundles so no variant keeps the canonical category+quirk pair:
  bundle A: prc dollars REAL, ts epoch-ms INTEGER,  vrf 0/1 INTEGER
  bundle B: prc CENTS INTEGER, ts epoch-s INTEGER,  vrf 'true'/'false' TEXT
  bundle C: prc dollars REAL, ts ISO-date TEXT,     vrf 0/1 INTEGER
Content (items, reviews, attrs, taxonomy; the attrs/taxn/stats-table asymmetries) always
travels WITH the category. Output: dbx_data/products_v{0..5}.db + variants.json manifest.

Canonical source quirks (products_small.db): office=A, electronics=B, musical=C; office has
attrs+taxn+fdbk_stats, electronics has taxn only, musical has attrs only.
"""
import itertools
import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent / "dbx_data"
SRC = HERE / "products_small.db"

CATS = ["office", "electronics", "musical"]          # canonical content groups g1,g2,g3
SRC_SUFFIX = {"office": "g1", "electronics": "g2", "musical": "g3"}
SRC_BUNDLE = {"office": "A", "electronics": "B", "musical": "C"}
SUFFIXES = ["g1", "g2", "g3"]
BUNDLES = ["A", "B", "C"]

# 6 variants: suffix permutation x quirk assignment (chosen so no category keeps its
# canonical (suffix, bundle) pair in any variant except variant 0 = canonical itself,
# which we KEEP as one of the six — the defense is the VARIATION across rows, not
# avoiding the original).
SUFFIX_PERMS = list(itertools.permutations(SUFFIXES))       # 6 perms
BUNDLE_PERMS = [("A", "B", "C"), ("B", "C", "A"), ("C", "A", "B"),
                ("A", "C", "B"), ("B", "A", "C"), ("C", "B", "A")]


def to_epoch_s(bundle_src, v):
    if bundle_src == "A":
        return f"CAST({v} / 1000 AS INTEGER)"                 # ms -> s
    if bundle_src == "B":
        return f"CAST({v} AS INTEGER)"                        # already s
    return f"CAST(strftime('%s', {v}) AS INTEGER)"            # ISO -> s


def price_dollars(bundle_src, v):
    return f"({v} / 100.0)" if bundle_src == "B" else f"CAST({v} AS REAL)"


def vrf_int(bundle_src, v):
    if bundle_src == "B":
        return f"(CASE WHEN {v} = 'true' THEN 1 ELSE 0 END)"
    return f"CAST({v} AS INTEGER)"


def conv_price(src_b, dst_b, v):
    d = price_dollars(src_b, v)
    return f"CAST(ROUND({d} * 100) AS INTEGER)" if dst_b == "B" else f"ROUND({d}, 2)"


def conv_ts(src_b, dst_b, v):
    s = to_epoch_s(src_b, v)
    if dst_b == "A":
        return f"({s} * 1000)"
    if dst_b == "B":
        return s
    return f"date({s}, 'unixepoch')"


def conv_vrf(src_b, dst_b, v):
    i = vrf_int(src_b, v)
    return f"(CASE WHEN {i} = 1 THEN 'true' ELSE 'false' END)" if dst_b == "B" else i


PRC_TYPE = {"A": "REAL", "B": "INTEGER", "C": "REAL"}
TS_TYPE = {"A": "INTEGER", "B": "INTEGER", "C": "TEXT"}
VRF_TYPE = {"A": "INTEGER", "B": "TEXT", "C": "INTEGER"}


def build_variant(vi: int, suffix_perm, bundle_perm) -> dict:
    """category CATS[i] -> suffix suffix_perm[i], quirk bundle bundle_perm[i]."""
    out = HERE / f"products_v{vi}.db"
    if out.exists():
        out.unlink()
    src = sqlite3.connect(str(SRC))
    dst = sqlite3.connect(str(out))
    manifest = {"variant": vi, "map": {}, "bundle": {}}

    for ci, cat in enumerate(CATS):
        sg = SRC_SUFFIX[cat]          # suffix in the source db
        dg = suffix_perm[ci]          # suffix in this variant
        sb = SRC_BUNDLE[cat]          # canonical quirk bundle of this content
        db_ = bundle_perm[ci]         # quirk bundle in this variant
        manifest["map"][cat] = dg
        manifest["bundle"][cat] = db_

        # ---- items table (column sets differ per source group; keep them as content) ----
        cols = [d[1] for d in src.execute(f"PRAGMA table_info(items_{sg})")]
        sel = []
        for c in cols:
            if c == "prc":
                sel.append(conv_price(sb, db_, "prc") + " AS prc")
            else:
                sel.append(c)
        col_defs = []
        for d in src.execute(f"PRAGMA table_info(items_{sg})"):
            typ = PRC_TYPE[db_] if d[1] == "prc" else d[2]
            col_defs.append(f"{d[1]} {typ}" + (" PRIMARY KEY" if d[5] else ""))
        dst.execute(f"CREATE TABLE items_{dg} ({', '.join(col_defs)})")
        dst.executemany(
            f"INSERT INTO items_{dg} VALUES ({','.join('?' * len(cols))})",
            src.execute(f"SELECT {', '.join(sel)} FROM items_{sg}").fetchall())

        # ---- fdbk table (ts + vrf quirks live here) ----
        fcols = [d[1] for d in src.execute(f"PRAGMA table_info(fdbk_{sg})")]
        fsel = []
        for c in fcols:
            if c == "ts":
                fsel.append(conv_ts(sb, db_, "ts") + " AS ts")
            elif c == "vrf":
                fsel.append(conv_vrf(sb, db_, "vrf") + " AS vrf")
            else:
                fsel.append(c)
        fdefs = []
        for d in src.execute(f"PRAGMA table_info(fdbk_{sg})"):
            typ = {"ts": TS_TYPE[db_], "vrf": VRF_TYPE[db_]}.get(d[1], d[2])
            fdefs.append(f"{d[1]} {typ}" + (" PRIMARY KEY" if d[5] else ""))
        dst.execute(f"CREATE TABLE fdbk_{dg} ({', '.join(fdefs)})")
        dst.executemany(
            f"INSERT INTO fdbk_{dg} VALUES ({','.join('?' * len(fcols))})",
            src.execute(f"SELECT {', '.join(fsel)} FROM fdbk_{sg}").fetchall())

        # ---- aux tables travel with the CATEGORY (asymmetries preserved) ----
        for aux_kind in ("attrs", "taxn"):
            s_aux = f"{aux_kind}_{sg}"
            if src.execute("SELECT COUNT(*) FROM sqlite_master WHERE name=?", (s_aux,)).fetchone()[0]:
                ddl = src.execute("SELECT sql FROM sqlite_master WHERE name=?", (s_aux,)).fetchone()[0]
                dst.execute(ddl.replace(s_aux, f"{aux_kind}_{dg}"))
                rows = src.execute(f"SELECT * FROM {s_aux}").fetchall()
                if rows:
                    dst.executemany(
                        f"INSERT INTO {aux_kind}_{dg} VALUES ({','.join('?' * len(rows[0]))})", rows)
        if cat == "office":  # the distractor stats table follows the office content
            ddl = src.execute("SELECT sql FROM sqlite_master WHERE name='fdbk_stats_g1'").fetchone()[0]
            dst.execute(ddl.replace("fdbk_stats_g1", f"fdbk_stats_{dg}"))
            rows = src.execute("SELECT * FROM fdbk_stats_g1").fetchall()
            dst.executemany(f"INSERT INTO fdbk_stats_{dg} VALUES ({','.join('?' * len(rows[0]))})", rows)

    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    return manifest


def main():
    manifests = []
    for vi in range(6):
        m = build_variant(vi, SUFFIX_PERMS[vi], BUNDLE_PERMS[vi])
        manifests.append(m)
        print(f"v{vi}: map={m['map']} bundle={m['bundle']}")
    (HERE / "variants.json").write_text(json.dumps(manifests, indent=1))
    # sanity: canonical variant 0 must equal the source semantics
    c0 = sqlite3.connect(str(HERE / "products_v0.db"))
    cs = sqlite3.connect(str(SRC))
    for q in ["SELECT COUNT(*) FROM fdbk_g2 WHERE vrf='true'",
              "SELECT MAX(prc) FROM items_g2", "SELECT MIN(substr(ts,1,4)) FROM fdbk_g3"]:
        print("v0 == src?", q[:45], c0.execute(q).fetchone(), cs.execute(q).fetchone())


if __name__ == "__main__":
    main()
