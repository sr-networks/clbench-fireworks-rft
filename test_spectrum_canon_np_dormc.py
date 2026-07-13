"""DORMANT + COMPLETION-GUARD arm (dormc): identical to the dormant arm (pscoc9fp / test_spectrum_canon_np_
dormant) EXCEPT the reward adds a fourth hinge, pen_complete = COMPLETE_W * max(0, NUM_SCANS - n - GRACE)/
NUM_SCANS, taxing episodes that end early. Motivation (measured on pyl9pkw8, the lr-2e-4 dormant run): the
policy learned to DIE early — a bloated thinking trace overruns the 8192-token turn cap mid-report, the
processor breaks the rollout, and the hardest LATE scans never get scored, so the surviving per-scan average
LOOKED higher (mean_dorm 0.63 truncated vs 0.58 complete). The base reward averages only over scans actually
played, so GRPO was rewarding DYING. This guard makes finishing dominant while staying exact-zero for honest
play (episodes score 28-30, inside the grace window). Red-teamed offline in scratchpad/sim_dormant_complete.py
(the quitter policies net below honest full-length play; accumulate base==guard). Single variable vs the
dormant arm = the completion hinge; same canon_np_proc24 dataset -> same seeded bands.

SPECTRUM_CANON_NOTEPAD must be set BEFORE importing the processor (read at import time)."""

import os

os.environ["SPECTRUM_CANON_NOTEPAD"] = "1"

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from spectrum_canon_processor import SpectrumCanonRolloutProcessor
from spectrum_reward import compute_spectrum_dormant_completion_reward

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
def test_spectrum_canon_np_dormc(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_dormant_completion_reward(row)
    return row
