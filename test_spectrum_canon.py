"""BENCH-PURE evaluator: the official CLBench blind_spectrum task 1:1, native available-IoU as reward,
full-history (ICL) memory. Used by BOTH training arms — the arms differ ONLY in the dataset's system prompt
(canon_nudge48 vs canon_neutral48), selected by the job's --dataset flag."""

import os

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from spectrum_canon_processor import SpectrumCanonRolloutProcessor
from spectrum_reward import compute_spectrum_avail_reward

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "accounts/fireworks/models/qwen3-8b"


@evaluation_test(
    input_dataset=[os.path.join(HERE, "canon_nudge48.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.2,
        "max_tokens": 4096,
    }],
    rollout_processor=SpectrumCanonRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_spectrum_canon(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_avail_reward(row)
    return row
