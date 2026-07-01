"""RE-TEST of the custom RolloutProcessor (Option A) — bench-shaped user/assistant turns — now that capacity
is back, to resolve whether the CLOUD drives a non-McpGym in-process processor (the earlier "no" coincided
with the capacity outage, so it was inconclusive). Separate evaluator (test-spectrum-turn-…) so it runs in
parallel with the McpGym notepad run without clobbering it. Thinking OFF via the /no_think dataset + 1024 tok.

KEY SIGNAL: does totalInputRequests go > 0 (custom processor runs in-cloud) or stall at 0 while capacity is
available (cloud won't drive it)?
"""

import os

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test

from spectrum_turn_processor import SpectrumTurnRolloutProcessor
from spectrum_reward import compute_spectrum_reward

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "accounts/fireworks/models/qwen3-8b"


@evaluation_test(
    input_dataset=[os.path.join(HERE, "spectrum_np48.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.2,
        "max_tokens": 1024,          # thinking OFF (dataset has /no_think)
        "tool_choice": "required",
    }],
    rollout_processor=SpectrumTurnRolloutProcessor(),   # custom user/assistant loop, no MCP server
    mode="pointwise",
    passed_threshold=0.0,
)
def test_spectrum_turn(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_reward(row)
    return row
