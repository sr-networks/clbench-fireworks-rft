"""eval-protocol test for the spectrum NOTEPAD task (McpGym + agent-controlled notepad).

Consumed by BOTH local evaluation (`pytest test_spectrum_rft.py`) and RFT launch
(`eval-protocol upload --entry test_spectrum_rft.py::test_spectrum_rft`).

McpGym rollout (cloud-runnable, proven). The agent has notepad_read / notepad_write / submit_report tools
(spectrum_mcp); submit_report returns the next scan. Importing spectrum_context_window windows the model's
input to [system + current scan], so the agent's ONLY cross-scan memory is its notepad — the real fix over
the old env-echoed running-list scaffold. The reward (spectrum_reward.py) parses each scan's SCAN_OCC.
"""

import os

from eval_protocol.models import EvaluationRow
from eval_protocol.pytest import evaluation_test, MCPGymRolloutProcessor

import spectrum_context_window  # noqa: F401 — side effect: window to current scan (notepad = only memory)
from spectrum_reward import compute_spectrum_reward

HERE = os.path.dirname(os.path.abspath(__file__))

# Free-tier (<16B) base. Fireworks serves the policy internally during RFT (overridden by --base-model).
MODEL = "accounts/fireworks/models/qwen3-8b"


@evaluation_test(
    input_dataset=[os.path.join(HERE, "spectrum_np48.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.2,          # exploration so GRPO can find the notepad-memory behaviour.
        "max_tokens": 1024,          # thinking OFF (/no_think); 1024 is plenty for a single tool call.
        "tool_choice": "required",   # force a tool call each turn (notepad_read/write, submit_report)
    }],
    rollout_processor=MCPGymRolloutProcessor(),
    server_script_path=os.path.join(HERE, "spectrum_server.py"),
    # 12 scans, up to ~3 tool calls each (notepad_read, notepad_write, submit_report) + headroom.
    steps=80,
    mode="pointwise",
    passed_threshold=0.0,        # eval gate only; RFT optimizes the score directly
)
def test_spectrum_rft(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_spectrum_reward(row)
    return row
