"""DORMANT+ACQ arm ("dormacq"): two levers vs the dormant arm pscoc9fp, both task-agnostic by design:
  (1) reward = OCC_SCALE * (mean SCAN_DORM + ACQ_W * mean SCAN_ACQ - the same three guards). SCAN_ACQ pays
      the MERGE event directly: dormant coverage of channels first sighted exactly one scan ago — the
      behavior the pscoc9fp census shows is still the bottleneck (merge-OK 41.4->43.2%).
  (2) dataset 24 -> 48 rows (16 seeds/variant instead of 8): double the GRPO groups per epoch to attack the
      ep2 plateau (suspected group-signal exhaustion), same three canonical variants, same np-proc prompt.
Guards, hyperparams, model (qwen3-1p7b), 8192-token cap: unchanged from pscoc9fp.

SPECTRUM_CANON_NOTEPAD must be set BEFORE importing the processor (read at import time)."""

import os

os.environ["SPECTRUM_CANON_NOTEPAD"] = "1"

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from spectrum_canon_processor import SpectrumCanonRolloutProcessor
from spectrum_reward import compute_spectrum_dormant_acq_reward

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "accounts/fireworks/models/qwen3-8b"


@evaluation_test(
    input_dataset=[os.path.join(HERE, "canon_np_proc48.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.2,
        "max_tokens": 8192,
    }],
    rollout_processor=SpectrumCanonRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_spectrum_canon_np_dormacq(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_dormant_acq_reward(row)
    return row
