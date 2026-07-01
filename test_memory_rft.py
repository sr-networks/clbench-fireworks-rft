"""eval-protocol test for the learn-and-recall memory task (notepad-only memory)."""

import os

import context_window  # noqa: F401  -- side effect: notepad-only context windowing

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test, MCPGymRolloutProcessor

from memory_reward import compute_memory_reward

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "accounts/fireworks/models/qwen3-1p7b"


@evaluation_test(
    input_dataset=[os.path.join(HERE, "memory_dataset.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.0,
        "max_tokens": 512,
        "tool_choice": "required",
    }],
    rollout_processor=MCPGymRolloutProcessor(),
    server_script_path=os.path.join(HERE, "memory_server.py"),
    steps=16,                    # 8 rounds, 1 tool call each (+buffer)
    mode="pointwise",
    passed_threshold=0.0,
)
def test_memory_rft(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_memory_reward(row)
    return row
