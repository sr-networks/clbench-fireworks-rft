"""BENCH-PURE NOTEPAD + DENSE-REWARD evaluator, RAISED OUTPUT BUDGET (8192 tokens/call): identical to
test_spectrum_canon_np_occ EXCEPT max_tokens 4096 -> 8192. Tests the budget-constraint hypothesis from the
NP-OCC2/recall-arm termination analysis: recall requires verbosity (long notepad re-emission), verbose turns
hit the 4096 cap before submit_report closes, the scan goes unscored, and RL learns terse=trim=forget. If the
cap was binding, the 8192 arm should recover scan completion AND stop the notepad erosion (np@30 slid 8.2->4.7
over training at 4096); if it stays flat, the drain is reasoning verbosity, not the memory content.

Single experimental variable vs baseline job b53iisn3 (clbench-canon-np-occ2): the token cap. Same dataset
(canon_np_nudge24), same reward (compute_spectrum_reward = dense occ-IoU), same base/lr/epochs/candidates.

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
    input_dataset=[os.path.join(HERE, "canon_np_nudge24.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.2,
        "max_tokens": 8192,
    }],
    rollout_processor=SpectrumCanonRolloutProcessor(),
    mode="pointwise",
    passed_threshold=0.0,
)
def test_spectrum_canon_np_occ8k(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_reward(row)
    return row
