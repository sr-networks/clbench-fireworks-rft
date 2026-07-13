"""Mixed-task evaluator (Stage 2/3): spectrum dormc rows + dbx rows in one dataset,
dispatched by row_id prefix to each task's own processor and reward (see
mix_canon_processor.py / mix_reward.py). Reward scales don't compete: GRPO advantages
are within one row's candidate group = one task.

Dashboard note: metric names are disjoint between the tasks (mean_dorm/scans_completed…
vs one_shot/mean_eff…), so each per-epoch average aggregates only the rows of its own
task — the two curves stay separately readable in one job.
"""

import os

os.environ["SPECTRUM_CANON_NOTEPAD"] = "1"   # dormc regime; read at processor import time

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from mix_canon_processor import MixRolloutProcessor
from mix_reward import compute_mix_reward

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "accounts/fireworks/models/qwen3-1p7b"


@evaluation_test(
    input_dataset=[os.path.join(HERE, "mix48.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.2,
        "max_tokens": 16384,
    }],
    rollout_processor=MixRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_mix_canon(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_mix_reward(row)
    return row
