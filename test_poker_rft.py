"""eval-protocol test wiring the Poker MCP-Gym + reward together.

This single file is what both:
  * local evaluation (`pytest test_poker_rft.py`) and
  * RFT launch (`eval-protocol create rft ... --evaluation-test ...`)
consume. The @evaluation_test decorator binds the dataset, the rollout processor (which boots
server.py as the MCP-Gym), the model, and the reward function.

The reward (see reward.py) is the port of clbench_verifiers/rubric.py.
"""

import os

import context_window  # noqa: F401  -- side effect: windows policy context to notepad-only memory

from eval_protocol.models import EvaluationRow, EvaluateResult
from eval_protocol.pytest import evaluation_test, MCPGymRolloutProcessor

from reward import compute_poker_reward

HERE = os.path.dirname(os.path.abspath(__file__))

# Free-tier (<16B) Qwen3-8B base. Fireworks serves the policy internally during RFT.
MODEL = "accounts/fireworks/models/qwen3-8b"


@evaluation_test(
    input_dataset=[os.path.join(HERE, "poker_fixed96.jsonl")],
    completion_params=[{
        "model": MODEL,
        "temperature": 1.0,      # matches prime config sampling.temperature
        "max_tokens": 1024,      # matches prime config; fine with /no_think (no long <think> to fit)
        # Force a tool call every turn. Trace analysis showed ~23% of rollouts ended early because
        # the (weak) model replied with prose and no tool call -> rollout terminated with the
        # INCOMPLETE_PENALTY, an unlearnable -3 drag unrelated to poker skill. tool_choice=required
        # makes the model always act, so the reward reflects play quality (legal moves, chip outcome).
        "tool_choice": "required",
    }],
    rollout_processor=MCPGymRolloutProcessor(),
    server_script_path=os.path.join(HERE, "server.py"),
    # Fixed-opponent rollout = num_instances hands (default 20). With the exact-canonical agent layer
    # (illegal actions are rejected and re-queried rather than auto-advanced), a weak model can spend
    # extra turns retrying illegal actions, so we give generous headroom to let all 20 hands complete
    # (more completed hands = cleaner late-vs-early memory signal). Smoke (steps=200) completed fine.
    steps=300,
    mode="pointwise",
    passed_threshold=0.0,        # eval gate only; RFT optimizes the score directly
)
def test_poker_rft(row: EvaluationRow) -> EvaluationRow:
    row.evaluation_result = compute_poker_reward(row)
    return row
