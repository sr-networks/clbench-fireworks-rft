# dbx (database_exploration) — run registry

Arm design, currency, and red-team: see the 2026-07-13 dbx entry in `LIVE_dormant_arm.html`
and `dbx_redteam.py`. Reward: `dbx_reward.py` (Score = 3 × (graded eff pos≥2 − pen_anchor −
pen_complete − pen_bake); evidence gate on nq≤1 credit). Scripted floors at budget 8:
oracle-notepad +2.80 / memoryless-honest +0.11 / guesser 0.00 / all bakers ≤ −5.6.

## Validation protocol (Stage 1b)

4 replicate runs, qwen3-1p7b, 5 epochs, 8 candidates, lr 1e-4, temp 1.2, max_tokens 4096,
dataset `clbench-dbx-canon24` (24 rows = 6 variants × 4 seeds), evaluator
`test-dbx-canon-test-dbx-canon`. Run-1 goes first: ep0 must (a) prove the cloud plumbing
(q_answered ≈ 15, no all-zero scores) and (b) measure the true untrained anchor a0
(A0=0.15 in dbx_reward.py is a scripted placeholder). Judge by trend, not peaks
(memory `rl-curve-analysis`).

## Runs

| run id | arm | config | change under test | outcome |
|---|---|---|---|---|
| (local smoke `spicy-state-078060`) | dbx pre-launch | gpt-oss-120b, 2 rows, local pytest | full evaluator path end-to-end | PASS — scores +0.964 / +0.129, q_answered 15/15 both, no penalty misfires |
| `w7guqqae` | dbx-r1 | 5 ep, 8 cand, lr 1e-4 | first cloud run: plumbing + true a0 from ep0 | **COMPLETED — FLAT NULL.** Score [0.006 0.012 0.004 0.009 0.007], acc [0.031 0.035 0.020 0.028 0.031], one_shot 0.000 all 5 epochs, mean_eff ~0.002. ep0 gate had passed (q_answered 15.0 ×192, 0 errors, a0=0.0016 measured → A0 set 0.002). No learning. |
| `xvmz0mxv` | dbx-r2 | 5 ep, 8 cand, lr 1e-4 | replicate (corrected A0=0.002 evaluator) | RUNNING (ep3). Confirming flat: Score [0.006 0.002 0.004], acc ~0.03, one_shot 0.000. |
| `qmi03m6j` | dbx-r3 | 5 ep, 8 cand, lr 1e-4 | replicate (corrected A0=0.002 evaluator) | RUNNING (ep3). Confirming flat: Score [0.010 0.007 0.005], one_shot 0.000. |
| `xmen1ghu` | dbx-r4 | 5 ep, 8 cand, lr 1e-4 | replicate (corrected A0=0.002 evaluator) | RUNNING (ep3). Confirming flat: Score [0.013 0.010 0.003], one_shot 0.000. |

## VERDICT (2026-07-13): dbx is a FLAT NULL on qwen3-1p7b — a reasoning ceiling, not a reward/scaffold bug

r1 complete and dead flat; r2–4 identical through ep2–3 (all four agree — the cross-replicate trend
my own rule demands). Autopsy of r1 ep4 rollouts (192) shows the null is **mechanistically clean**:
- **Scaffold used correctly:** the model issues QUERY, writes `notepad_update`, emits ANSWER in the
  prescribed loop; the procedural prompt is followed.
- **SQL mostly valid:** ~80% of queries return data (2451 ok / 637 error / 9 empty) — not a syntax wall.
- **The wall is multi-hop reasoning:** think-traces show the 1.7B cannot reliably deduce the
  category→table mapping (queries `items_g1`, sees "Home Audio & Theater", cannot decide if that is the
  musical-instruments table) and then writes the WRONG fact into the notepad. So memory cannot help —
  the "informed" in informed-one-shot requires a discovery the model can't complete even once. acc pinned
  at the ~3% base floor ⇒ the efficiency ladder has no correct answers to climb ⇒ one_shot ≡ 0.

Same capability ceiling as the spectrum notepad-only arm (see [[clbench-spectrum-memory]] null). The
reward design, evidence gate, budget-8 fix, and A0 calibration are all validated as correct — they just
have nothing to amplify on a model this small. r2–4 left running to completion for the full 4-replicate
record (near-certain confirmation; cancelling saves little and muddies the record).

**Caveat (disclosed):** r1 vs r2–4 differ in one reward constant (A0 0.15 vs 0.002 — the anchor
hinge threshold). The hinge only fires if anchor rises above it; at ep0 anchor ≈ 0.002 and
pen_anchor = 0 everywhere, so if r1's anchor stays ≤ 0.15 all epochs the difference never bites
and r1 counts as a full replicate; otherwise r1 is the pilot and r2–4 the matched trio.
`test-mix-canon` re-uploaded 2026-07-13 17:55 with the corrected A0 (Stage 3 snapshot is current).
