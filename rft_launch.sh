#!/usr/bin/env bash
# Standard RFT launch with W&B logging. Key comes from the WANDB_API_KEY env var (in ~/.zshrc), never
# hard-coded here. Usage:
#   ./rft_launch.sh <dataset> <evaluator-id> <output-model> [epochs=8] [lr=1e-4] [candidates=12]
set -euo pipefail
source ~/.zshrc 2>/dev/null || true
ACCT=sten-ruediger-x94mx5
DS=$1; EID=$2; OUT=$3; EP=${4:-8}; LR=${5:-1e-4}; CAND=${6:-12}
: "${WANDB_API_KEY:?WANDB_API_KEY not set (add it to ~/.zshrc)}"
firectl create reinforcement-fine-tuning-job \
  --base-model accounts/fireworks/models/qwen3-1p7b \
  --dataset "$DS" \
  --evaluator "accounts/$ACCT/evaluators/$EID" \
  --output-model "$OUT" \
  --epochs "$EP" --learning-rate "$LR" --temperature 1.2 --max-output-tokens 4096 \
  --response-candidates-count "$CAND" --max-concurrent-rollouts 96 \
  --wandb --wandb-api-key "$WANDB_API_KEY" \
  --wandb-project clbench-fireworks-rft --wandb-entity sten-ruediger-capgemini \
  2>&1 | grep -ivE "DEPRECATION|updates|Current version|Latest version|upgrade|^[[:space:]]*$" \
       | grep -iE "^Name:" | sed 's#.*/##'
