#!/usr/bin/env bash
# Evaluate a (deployed) model on the OFFICIAL CLBench blind_spectrum_monitoring 90-instance schedule
# ('default' = 3 stages x 30: five_ch_wide -> five_plus_four_mixed -> full_grid_active).
#
# Usage:  ./eval_bench90.sh <litellm-model-id> [system] [runs]
#   <litellm-model-id>  a DEPLOYED Fireworks model in litellm form (qwen3-1p7b is NOT serverless here):
#       base:    fireworks_ai/accounts/fireworks/models/qwen3-1p7b#accounts/<acct>/deployments/<dep>
#       trained: fireworks_ai/accounts/<acct>/models/clbench-spectrum-icloff-30ep#accounts/<acct>/deployments/<dep>
#   [system]  icl_notepad (DEFAULT; bench-native notepad — closest to our training) | icl (full context)
#   [runs]    default 1 (the schedule's own default is 5; 1 pass = 90 model calls)
#
# Prereqs: FIREWORKS_API_KEY in env; model deployed (see EVAL_bench90.md). The held-out frozen corpus is
# regenerated deterministically below if missing (eval-only — never train on it).
source ~/.zshrc 2>/dev/null || true
export FIREWORKS_AI_API_KEY="${FIREWORKS_AI_API_KEY:-$FIREWORKS_API_KEY}"
MODEL="${1:?usage: eval_bench90.sh <litellm-model-id> [system=icl_notepad] [runs=1]}"
SYSTEM="${2:-icl_notepad}"; RUNS="${3:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/bench90_${SYSTEM}_$(date +%Y%m%d-%H%M%S).json"

python3.11 - <<'PY'
from src.tasks.blind_spectrum_monitoring.corpus import (
    build_rollout_corpus, write_scan_corpus, default_corpus_paths)
cid = "mixed_grid_lifecycle"; j, m = default_corpus_paths(cid)
if not (j.exists() and m.exists()):
    r, meta = build_rollout_corpus('default', corpus_id=cid)
    j.parent.mkdir(parents=True, exist_ok=True)
    write_scan_corpus(r, meta, jsonl_path=j, metadata_path=m)
    print("[corpus] regenerated 90-scan frozen corpus ->", j)
else:
    print("[corpus] present ->", j)
PY

echo "[eval] system=$SYSTEM runs=$RUNS model=$MODEL"
python3.11 -m src.cli run --task blind_spectrum_monitoring --system "$SYSTEM" \
  --system-params "{\"model\": \"$MODEL\"}" \
  --task-params '{"schedule": "default"}' \
  --runs "$RUNS" --no-live-dashboard --output "$OUT"
echo "[eval] official result (mean available-IoU + per-instance loss curve) -> $OUT"
echo "[note] available-IoU is memory-INSENSITIVE (memoryless ~0.47). See EVAL_bench90.md for the occ-IoU/"
echo "       late-minus-early post-processing if you want to see the MEMORY effect on these 90 scenarios."
