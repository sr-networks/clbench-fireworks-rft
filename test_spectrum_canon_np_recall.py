"""BENCH-PURE NOTEPAD + RECALL-WEIGHTED REWARD evaluator (memory in the strict sense): official CLBench
blind_spectrum task, bench icl_notepad semantics (context cleared between instances; model-maintained
notepad via notepad_update is the only cross-instance carrier). Identical to test_spectrum_canon_np_occ
EXCEPT the reward: this grades the asymmetric Tversky overlap (SCAN_REC) instead of symmetric occ-IoU, to
test whether reward pressure toward RECALL (cheap false positives, full penalty for forgetting a persistent
transmitter) rescues notepad accumulation that plain occ-IoU erodes via a trimming drift.

SPECTRUM_CANON_NOTEPAD must be set BEFORE importing the processor (read at import time). SPECTRUM_TVERSKY_ALPHA
tunes the false-positive discount (default 0.4)."""

import os

os.environ["SPECTRUM_CANON_NOTEPAD"] = "1"
os.environ.setdefault("SPECTRUM_TVERSKY_ALPHA", "0.4")

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from spectrum_canon_processor import SpectrumCanonRolloutProcessor
from spectrum_reward import compute_spectrum_recall_reward

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
def test_spectrum_canon_np_recall(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_recall_reward(row)
    return row
