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
| `w7guqqae` | dbx-r1 | 5 ep, 8 cand, lr 1e-4 | first cloud run: plumbing + true a0 from ep0 | RUNNING (launched 2026-07-13 16:36) |
