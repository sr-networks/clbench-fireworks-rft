"""BENCH-PURE NOTEPAD + DENSE-REWARD evaluator (train occ, grade avail) (memory in the strict sense): official CLBench blind_spectrum task, native
available-IoU reward, bench icl_notepad semantics — context cleared between instances, model-maintained
notepad (notepad_update field) as the only cross-instance carrier. Arms differ ONLY in the dataset's system
prompt (canon_np_nudge24 vs canon_np_neutral24), selected by the job's --dataset flag.

SPECTRUM_CANON_NOTEPAD must be set BEFORE importing the processor (read at import time)."""

import os

os.environ["SPECTRUM_CANON_NOTEPAD"] = "1"

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from spectrum_canon_processor import SpectrumCanonRolloutProcessor
from spectrum_reward import compute_spectrum_reward

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "accounts/fireworks/models/qwen3-8b"


@evaluation_test(
    input_dataset=[os.path.join(HERE, "canon_np_nudge24.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.2,
        "max_tokens": 4096,
    }],
    rollout_processor=SpectrumCanonRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_spectrum_canon_np_occ(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_reward(row)
    return row
