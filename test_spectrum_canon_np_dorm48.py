"""DORM48 arm: the PROVEN dormant reward (identical to pscoc9fp) on the 48-row dataset. Run 2 of the
improvement timebox, designed to disambiguate run 1 (dormacq jnlfncrh, which trained flat and DEGRADED
merge behavior): dormacq changed two things at once (acq reward term + 24->48 rows), so this run keeps the
48 rows and drops the acq term.
  - If dorm48 reproduces the pscoc9fp step (late_mean -> ~0.53): the acq term was the harm (its small-cohort,
    near-binary SCAN_ACQ signal added advantage noise that swamped the clean dorm gradient).
  - If dorm48 also trains flat: the pscoc9fp effect itself is fragile/seed-lucky and needs replication study.
The processor still emits SCAN_ACQ (inert for this reward) so acquisition can be MEASURED offline for free.

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
def test_spectrum_canon_np_dorm48(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_dormant_reward(row)
    return row
