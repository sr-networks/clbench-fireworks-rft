"""dbx arm (database_exploration memory currency): official CLBench task, notepad-only
persistence, informed query-efficiency reward. Design + red-team: dbx_redteam.py — the
reward passed oracle>memoryless, all bakers negative, qstat-injection scrubbed.

Reward: dbx_reward.dbx_score via compute_dbx_reward (graded efficiency ladder pos>=2,
evidence gate on nq<=1 credit, anchor hinge at pos 1, completion hinge grace 1,
pen_bake on unevidenced instant answers). A0 is a scripted placeholder until run-1 ep0
measures the true 1.7B memoryless anchor (same protocol as the spectrum a0 constants).

Dataset: dbx_canon24.jsonl (6 DB variants x 4 seeds; variants permute the category->
table mapping AND the quirk bundles so schema facts are notepad-only knowledge).
"""

import os

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from dbx_canon_processor import DbxCanonRolloutProcessor
from dbx_reward import compute_dbx_reward

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "accounts/fireworks/models/qwen3-1p7b"


@evaluation_test(
    input_dataset=[os.path.join(HERE, "dbx_canon24.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.2,
        "max_tokens": 4096,
    }],
    rollout_processor=DbxCanonRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_dbx_canon(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_dbx_reward(row)
    return row
