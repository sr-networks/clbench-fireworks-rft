"""Generate per-VARIANT question pools for the database_exploration memory arm (dbx).

Design targets (see memo):
  - answerable by a 1.7B: single-group aggregates, no 3-way joins — unlike the bench's
    30 'hard' questions, these are the training curriculum; the currency is MEMORY
    (query efficiency), not SQL wizardry.
  - every question requires knowing WHICH tables belong to the category and, for many,
    a UNIT QUIRK (cents vs dollars, timestamp encoding, vrf as text vs int)
    -> unanswerable in <=1 query without prior (notepad) knowledge, comfortably
    answerable in 3-6 queries by a memoryless explorer.
  - answers must not be guessable: reject answers in {0,1,2,3} and answers equal across
    categories for the same template (else 'echo a constant' cheats the currency).

Variants (built by make_dbx_variants.py): each products_v{0..5}.db permutes BOTH the
category->table-suffix mapping AND which quirk bundle each category carries, so the SQL
for the same question text differs per variant. This is why weights cannot bake the
schema facts — only a notepad can carry them within an episode.

Output: dbx_data/questions_pool_v{0..5}.json — bench-compatible question dicts
(question_id, question, difficulty, answer, answer_type, tolerance, sql) + group
(variant table suffix) + category tags.
"""
import json
import sqlite3
from pathlib import Path

DATA = Path(__file__).parent / "dbx_data"

CAT_TEXT = {"office": "office products", "electronics": "electronics",
            "musical": "musical instruments"}
HAS_ATTRS = {"office", "musical"}     # content asymmetries travel with the category
HAS_TAXN = {"office", "electronics"}

# quirk helpers keyed by BUNDLE (see make_dbx_variants.py)
BUNDLE_PRC = {"A": "prc", "B": "prc / 100.0", "C": "prc"}                  # B stores cents
BUNDLE_YEAR = {
    "A": "CAST(strftime('%Y', ts / 1000, 'unixepoch') AS INTEGER)",       # epoch ms
    "B": "CAST(strftime('%Y', ts, 'unixepoch') AS INTEGER)",              # epoch s
    "C": "CAST(substr(ts, 1, 4) AS INTEGER)",                             # ISO date text
}
BUNDLE_VRF = {"A": "vrf = 1", "B": "vrf = 'true'", "C": "vrf = 1"}         # B is TEXT


def templates(cat, g, b):
    """(difficulty, text, sql, answer_type, tolerance) for category cat mapped to
    suffix g with quirk bundle b."""
    C, P, Y, V = CAT_TEXT[cat], BUNDLE_PRC[b], BUNDLE_YEAR[b], BUNDLE_VRF[b]
    t = [
        ("easy", f"How many reviews are recorded for {C}?",
         f"SELECT COUNT(*) FROM fdbk_{g}", "integer", 0),
        ("easy", f"How many distinct users have written reviews of {C}?",
         f"SELECT COUNT(DISTINCT uid) FROM fdbk_{g}", "integer", 0),
        ("easy", f"What is the average star rating across all reviews of {C}? Round to two decimals.",
         f"SELECT ROUND(AVG(rtg), 2) FROM fdbk_{g}", "float", 0.01),
        ("easy", f"How many reviews of {C} give a 5-star rating?",
         f"SELECT COUNT(*) FROM fdbk_{g} WHERE rtg = 5", "integer", 0),
        ("easy", f"What is the total number of helpful votes across all reviews of {C}?",
         f"SELECT SUM(hlp_ct) FROM fdbk_{g}", "integer", 0),
        ("medium", f"What is the highest listed price, in dollars, of any of the {C}?",
         f"SELECT MAX({P}) FROM items_{g}", "float", 0.01),
        ("medium", f"What is the average listed price, in dollars, of the {C}? Round to two decimals.",
         f"SELECT ROUND(AVG({P}), 2) FROM items_{g}", "float", 0.01),
        ("medium", f"How many reviews of {C} are marked as verified purchases?",
         f"SELECT COUNT(*) FROM fdbk_{g} WHERE {V}", "integer", 0),
        ("medium", f"In which year was the earliest recorded review of {C} posted?",
         f"SELECT MIN({Y}) FROM fdbk_{g}", "integer", 0),
        ("medium", f"How many {C} have a listed price above $25?",
         f"SELECT COUNT(*) FROM items_{g} WHERE {P} > 25", "integer", 0),
        ("medium", f"How many {C} have a listed price above $60?",
         f"SELECT COUNT(*) FROM items_{g} WHERE {P} > 60", "integer", 0),
        ("medium", f"How many of the {C} have an average rating of 4.5 or higher?",
         f"SELECT COUNT(*) FROM items_{g} WHERE avg_rtg >= 4.5", "integer", 0),
        ("medium", f"How many {C} are priced between $10 and $30 inclusive?",
         f"SELECT COUNT(*) FROM items_{g} WHERE {P} BETWEEN 10 AND 30", "integer", 0),
        ("medium", f"How many reviews of {C} were posted in the year 2018 or later?",
         f"SELECT COUNT(*) FROM fdbk_{g} WHERE {Y} >= 2018", "integer", 0),
        ("medium", f"How many reviews of {C} were posted in the year 2020 or later?",
         f"SELECT COUNT(*) FROM fdbk_{g} WHERE {Y} >= 2020", "integer", 0),
    ]
    if cat in HAS_ATTRS:
        t.append(("easy", f"How many product-attribute entries exist for {C}?",
                  f"SELECT COUNT(*) FROM attrs_{g}", "integer", 0))
    if cat in HAS_TAXN:
        t.append(("easy", f"How many taxonomy (category-path) entries exist for {C}?",
                  f"SELECT COUNT(*) FROM taxn_{g}", "integer", 0))
    return t


def build_pool(vi: int, manifest: dict) -> list:
    conn = sqlite3.connect(str(DATA / f"products_v{vi}.db"))
    pool, qid = [], 1
    seen_by_template: dict[str, set] = {}
    for cat in ("office", "electronics", "musical"):
        g, b = manifest["map"][cat], manifest["bundle"][cat]
        for diff, text, sql, atype, tol in templates(cat, g, b):
            val = conn.execute(sql).fetchone()[0]
            if val is None:
                continue
            if atype == "integer" and int(val) in (0, 1, 2, 3):
                continue
            tkey = text.split(" of ")[0][:40]   # template identity across categories
            seen = seen_by_template.setdefault(tkey, set())
            if val in seen:
                continue
            seen.add(val)
            pool.append({
                "question_id": qid, "question": text, "difficulty": diff,
                "answer": (int(val) if atype == "integer" else round(float(val), 2)),
                "answer_type": atype, "tolerance": tol, "sql": sql,
                "group": g, "category": cat,
            })
            qid += 1
    conn.close()
    return pool


def main():
    variants = json.loads((DATA / "variants.json").read_text())
    for m in variants:
        vi = m["variant"]
        pool = build_pool(vi, m)
        out = DATA / f"questions_pool_v{vi}.json"
        out.write_text(json.dumps(pool, indent=1))
        from collections import Counter
        by_g = dict(Counter(q["group"] for q in pool))
        print(f"v{vi}: {len(pool)} questions {by_g}")
    # cross-variant invariance check: count-type answers must MATCH across variants
    # (same content), price answers must match in dollars (cents conversion aside)
    p0 = {(q["category"], q["question"]): q["answer"]
          for q in json.loads((DATA / "questions_pool_v0.json").read_text())}
    p3 = {(q["category"], q["question"]): q["answer"]
          for q in json.loads((DATA / "questions_pool_v3.json").read_text())}
    diff = [(k, p0[k], p3[k]) for k in (p0.keys() & p3.keys())
            if abs(float(p0[k]) - float(p3[k])) > 0.05]
    print("v0 vs v3 shared-question answer mismatches (expect none):", diff[:5] or "OK")


if __name__ == "__main__":
    main()
