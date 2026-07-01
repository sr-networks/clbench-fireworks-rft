"""Poker MCP-Gym server — the eval-protocol equivalent of CLBenchEnv(vf.MultiTurnEnv).

Exposes a single data-plane tool, `poker_act`, whose parameters ARE the PokerAction schema
(action / thinking / amount). The MCP-Gym base class accumulates per-step reward and the
terminated flag in the control plane; the rollout processor reads those to score and to stop.
"""

from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context

from eval_protocol.mcp import McpGym

from poker_adapter import PokerAdapter


class PokerMcp(McpGym):
    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__("CLBench-ExploitablePoker", PokerAdapter(), seed, **kwargs)

    def _register_tools(self):
        @self.mcp.tool(
            name="poker_act",
            description=(
                "Take your poker action for the current decision and optionally update your notepad. "
                "Fields: thinking — your reasoning about the hand situation and why you chose this "
                "action; action — one of FOLD, CALL, CHECK, RAISE; amount — the total chips to raise "
                "to (only for RAISE; null otherwise); notepad_update — optional: update your notepad "
                "with new observations, patterns, or thoughts. If provided, it completely replaces the "
                "current notepad content; leave empty to keep it unchanged. The notepad is your only "
                "memory across hands."
            ),
        )
        def poker_act(
            action: str,
            ctx: Context,
            thinking: str = "",
            amount: int = -1,
            notepad_update: str = "",
        ) -> Dict[str, Any]:
            # NOTE: FastMCP (this version) raises when a tool param is annotated
            # Optional[...] while it scans for the Context param, so `amount` uses a
            # plain int with -1 sentinel and notepad_update uses "" for "no change".
            session_id = self._get_session_id(ctx)
            session_data = self._get_or_create_session(ctx)
            env = session_data["env"]

            # Build the clean PokerAction (the task schema has no notepad field); stash the
            # notepad update on the env so PokerEnv.step applies it as cross-hand memory.
            schema = env.response_schema
            amt = None if amount is None or amount < 0 else int(amount)
            action_obj = schema(thinking=thinking, action=action.strip().upper(), amount=amt)
            env.pending_notepad = notepad_update.strip() or None

            obs = self._execute_session_environment_step(session_id, action_obj)
            obs["action"] = action
            return obs

    def format_observation(self, obs: Dict[str, Any], env: Any) -> Dict[str, Any]:
        """Data-plane view the model sees as the tool result (no reward leakage)."""
        return {
            "observation": obs.get("prompt", ""),
            "instance_complete": obs.get("instance_complete", False),
        }
