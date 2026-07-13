"""ABS-CONTRAST arm (user's original formula): identical to the proc-prompt arm (i7aw83iq) and the hinge
arm (b5h41zlu) EXCEPT the reward = late - |early - a0|. Pins the early half to the memoryless floor a0 so
that generic task-proficiency gains (which lift early) are NOT credited as memory; only late-above-anchor
(the memory-specific gain) is rewarded. Runs in PARALLEL with the hinge arm for a direct comparison.
Same canon_np_proc24 dataset -> same seeded bands; single variable vs b5h41zlu = the sign of the early penalty.

SPECTRUM_CANON_NOTEPAD must be set BEFORE importing the processor (read at import time)."""

import os

os.environ["SPECTRUM_CANON_NOTEPAD"] = "1"

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from spectrum_canon_processor import SpectrumCanonRolloutProcessor
from spectrum_reward import compute_spectrum_abs_contrast_reward

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
def test_spectrum_canon_np_abscontrast(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_abs_contrast_reward(row)
    return row
