"""ARM B — the CONTROL: identical to ARM A (test_spectrum_scaffold) except the echoed running list's
CONTENT is destroyed (random in-band frequencies, same count, same instruction, same growth). The model
gets the same structure and the same "keep + extend" pressure, but the memory channel carries no
information — so ANY training improvement in this arm is non-memory skill (formatting, transcription,
verbosity), and A−B is the memory-attributable training effect. Env flags before imports (import-time)."""

import os

os.environ["SPECTRUM_SCAFFOLD"] = "1"
os.environ["SPECTRUM_SCRAMBLE"] = "1"
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
        "max_tokens": 4096,
    }],
    rollout_processor=SpectrumTurnRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_spectrum_scramble(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_reward(row)
    return row
