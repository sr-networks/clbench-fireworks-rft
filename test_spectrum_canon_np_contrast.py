"""CONTRAST-REWARD arm: identical to the proc-prompt arm (job i7aw83iq — bench-pure notepad, procedural
system prompt via canon_np_proc24, 8192 tokens/call) EXCEPT the reward: user-specified contrast
score = late - |early - a0|, where a0 is the scripted memoryless floor per variant (measure_canon_a0.py).
Single experimental variable vs i7aw83iq: the reward (dense per-scan mean occ-IoU -> trajectory-level
contrast). Same rows -> same seeded bands.

SPECTRUM_CANON_NOTEPAD must be set BEFORE importing the processor (read at import time)."""

import os

os.environ["SPECTRUM_CANON_NOTEPAD"] = "1"

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from spectrum_canon_processor import SpectrumCanonRolloutProcessor
from spectrum_reward import compute_spectrum_contrast_reward

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
def test_spectrum_canon_np_contrast(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_contrast_reward(row)
    return row
