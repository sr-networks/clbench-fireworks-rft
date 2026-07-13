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
| `w7guqqae` | dbx-r1 | 5 ep, 8 cand, lr 1e-4 | first cloud run: plumbing + true a0 from ep0 | RUNNING. **ep0 gate PASSED**: q_answered 15.0 all 192 rollouts, 0 eval errors, all penalties 0. Measured a0 = 0.0016 (191/192 rollouts at anchor 0) → A0 placeholder 0.15 replaced with 0.002. GRPO seed thin but real: 9/24 groups have score variance (nonzero scores are exactly the eff-ladder rungs 0.064 = 1 question at W=0.3, 0.129 = 2); base acc 0.031, mean_nq 5.4 (headroom under budget 8). NOTE: r1 trains with the loose placeholder hinge (bites at 0.30 vs 0.152 corrected) — immaterial unless its anchor rises, watch per epoch. |
| `xvmz0mxv` | dbx-r2 | 5 ep, 8 cand, lr 1e-4 | replicate (corrected A0=0.002 evaluator) | RUNNING (launched 2026-07-13 17:53) |
| `qmi03m6j` | dbx-r3 | 5 ep, 8 cand, lr 1e-4 | replicate (corrected A0=0.002 evaluator) | RUNNING (launched 2026-07-13 17:53) |
| `xmen1ghu` | dbx-r4 | 5 ep, 8 cand, lr 1e-4 | replicate (corrected A0=0.002 evaluator) | RUNNING (launched 2026-07-13 17:53) |

**Caveat (disclosed):** r1 vs r2–4 differ in one reward constant (A0 0.15 vs 0.002 — the anchor
hinge threshold). The hinge only fires if anchor rises above it; at ep0 anchor ≈ 0.002 and
pen_anchor = 0 everywhere, so if r1's anchor stays ≤ 0.15 all epochs the difference never bites
and r1 counts as a full replicate; otherwise r1 is the pilot and r2–4 the matched trio.
`test-mix-canon` re-uploaded 2026-07-13 17:55 with the corrected A0 (Stage 3 snapshot is current).
