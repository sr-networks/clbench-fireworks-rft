# Evaluating a trained model on the OFFICIAL CLBench spectrum 90-instance schedule

This is the bridge from our custom RFT task to the real benchmark. It runs a model through the bench's
`default` schedule for `blind_spectrum_monitoring` = **3 stages × 30 instances** (`five_ch_wide` →
`five_plus_four_mixed` → `full_grid_active`), with the bench's own scoring.

## Status: wired up
- The 90 instances are backed by a **held-out frozen corpus** (`mixed_grid_lifecycle.jsonl`) that is NOT
  shipped in the pip package. `eval_bench90.sh` **regenerates it deterministically** from the package's own
  `build_rollout_corpus('default')` (eval-only — never train on it). Already generated locally at
  `…/site-packages/data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl`.
- `clbench run --task blind_spectrum_monitoring --system icl_notepad --task.schedule default` is validated
  to run end-to-end; the only missing piece is a **callable model endpoint**.

## Step 1 — deploy the model (the cost gate)
`qwen3-1p7b` is NOT serverless on this account (`NOT_FOUND`), and the trained models are LoRA addons
(`HF_PEFT_ADDON`), so you must deploy:
```bash
# base deployment (scale-to-zero so it costs ~nothing when idle)
firectl create deployment accounts/fireworks/models/qwen3-1p7b \
  --accelerator-type NVIDIA_A100_80GB --min-replica-count 0 --max-replica-count 1
# load the trained LoRA addon onto it (verify exact flags with: firectl create deployed-model --help)
firectl create deployed-model accounts/<acct>/models/clbench-spectrum-icloff-30ep --deployment <dep-id>
```
The litellm model id uses the `<model>#<deployment>` form (seen in the training streamlog), e.g.:
- base:    `fireworks_ai/accounts/fireworks/models/qwen3-1p7b#accounts/<acct>/deployments/<dep>`
- trained: `fireworks_ai/accounts/<acct>/models/clbench-spectrum-icloff-30ep#accounts/<acct>/deployments/<dep>`

## Step 2 — run the eval
```bash
./eval_bench90.sh "<litellm-model-id>" icl_notepad 1   # bench-native notepad (closest to our training)
./eval_bench90.sh "<litellm-model-id>" icl 1           # full-context, for comparison
```
Run it for BOTH the base `qwen3-1p7b` and the trained model; the difference is the transfer of training to
the real benchmark. Systems also include `mem0`, `ace`, `claude`, `codex`, `human`.

## ⚠️ Two caveats before you read the numbers
1. **The official metric (available-IoU) is memory-INSENSITIVE.** This is the metric we deliberately moved
   away from: a memoryless agent already scores ~0.47 on it, because reporting only the current peaks
   already implies most of the band is "available". Our whole memory result used **occupied-IoU** instead
   (memoryless ~0.16 → 4× signal). So the official `clbench run` score may show **little/no** memory effect
   even if the model has real memory — not because memory failed, but because the metric barely rewards it.
   To see memory on these 90 scenarios, compute **occupied-IoU + late-minus-early** from the run trace
   (the per-instance reports + ground truth are in the `--output` JSON / trace). Ask me to wire up that
   post-processing once you have one successful run — I can match it to `spectrum_reward.py`'s occ-IoU.
2. **Setting mismatch (it's a transfer test).** Our model was trained with ICL-OFF windowing + a running-
   list scaffold + occ-IoU reward on RANDOM bands. The bench systems differ: `icl` = full context (no
   windowing), `icl_notepad` = the bench's own notepad (not our scaffold). So this measures **zero-shot
   transfer** to the bench's native agent, not the exact training setting. `icl_notepad` is the closest
   match. If transfer is weak, that's information about setting-specificity, not necessarily a failed model.

## TL;DR
`eval_bench90.sh` + the regenerated corpus = the official bench run, ready. Deploy the model, run base vs
trained under `icl_notepad`/`icl`. Expect the official available-IoU to under-show memory; the occ-IoU/late−
early post-processing (ask me) is what reveals it on the bench's 90 scenarios.
