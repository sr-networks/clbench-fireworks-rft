"""PROCEDURAL-PROMPT NOTEPAD arm: identical to test_spectrum_canon_np_occ8k (bench-pure notepad, dense
occ-IoU reward, 8192 tokens/call) EXCEPT the dataset carries the user-authored procedural system prompt
(canon_np_proc24: explicit read->merge->report->always-save procedure + fixed MEMORY_TRANSMITTERS notepad
format). Single experimental variable vs job ecjdkfer/evaluator occ8k: the system prompt. Same row_ids ->
same seeded bands.

Motivated by the ep0/ep4 failure-mode census (failure_stats.py): 77% of merge opportunities wasted (verbatim
notepad copies), read-back >80% in only ~3% of scans, ~half the notepad junk — the nudge prompt states the
goal but not the procedure; this arm states the procedure and pins a cheap machine-like notepad format.

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
    input_dataset=[os.path.join(HERE, "canon_np_proc24.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.2,
        "max_tokens": 8192,
    }],
    rollout_processor=SpectrumCanonRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_spectrum_canon_np_proc(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_reward(row)
    return row
