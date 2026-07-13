"""DORMANT-COVERAGE arm: identical to the proc/hinge/abs arms (i7aw83iq / b5h41zlu / n4g52553) EXCEPT the
reward = OCC_SCALE * (mean SCAN_DORM - anchor hinge - carpet hinge). SCAN_DORM pays ONLY for covering
recallable channels (seen in an earlier scan, invisible now) — unearnable without memory, so proficiency
carries zero gradient by construction; dense per-scan (the density that trained in the proc arm); pays
merge->persist->read-back compound interest, inverting the copy-forward attractor. Guards are exact-zero
for honest policies: anchor hinge (grid-baking; also THE goal metric — no-notepad performance must not
rise) and carpet hinge on report-area ratio (blankets; FP inside dorm also taxes paint on never-seen
channels). Same canon_np_proc24 dataset -> same seeded bands; single variable vs the other arms = reward.

SPECTRUM_CANON_NOTEPAD must be set BEFORE importing the processor (read at import time)."""

import os

os.environ["SPECTRUM_CANON_NOTEPAD"] = "1"

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from spectrum_canon_processor import SpectrumCanonRolloutProcessor
from spectrum_reward import compute_spectrum_dormant_reward

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
def test_spectrum_canon_np_dormant(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_dormant_reward(row)
    return row
