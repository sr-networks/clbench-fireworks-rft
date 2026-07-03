"""ARM A — scaffold memory training on NEVER-REPEATING bands (the headline experiment).

Setting: running-list scaffold (env echoes the model's own previous report; submit_report only), ICL-off
windowing, EPOCH-SALTED bands (fresh band set every epoch => layout memorization structurally impossible;
every epoch's metrics double as a fresh-band eval of the current policy).

Claim this arm carries: RL improves USE of the provided memory channel — occ/memory_gain rising across
epochs while content never repeats. Compare against ARM B (test_spectrum_scramble: identical but the echoed
list content is destroyed): A−B isolates memory-attributable training; B bounds non-memory skill drift.

Env flags MUST be set before any spectrum import (spectrum_adapter reads them at import time).
"""

import os

os.environ["SPECTRUM_SCAFFOLD"] = "1"
os.environ["SPECTRUM_EPOCH_SALT"] = "1"

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from spectrum_turn_processor import SpectrumTurnRolloutProcessor
from spectrum_reward import compute_spectrum_reward

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "accounts/fireworks/models/qwen3-8b"


@evaluation_test(
    input_dataset=[os.path.join(HERE, "spectrum_scaf48.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.2,
        "max_tokens": 4096,          # thinking ON (brief nudge in the prompt)
    }],
    rollout_processor=SpectrumTurnRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_spectrum_scaffold(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_reward(row)
    return row
