"""MCP-Gym server for the learn-and-recall memory task. Single tool: `respond`."""

from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context

from eval_protocol.mcp import McpGym

from memory_adapter import MemoryAdapter


class MemoryMcp(McpGym):
    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__("LearnAndRecall", MemoryAdapter(), seed, **kwargs)

    def _register_tools(self):
        @self.mcp.tool(
            name="respond",
            description=(
                "Answer the current round and optionally update your notepad. "
                "answer: in the LEARN round say 'ok'; in a RECALL round give the requested fact. "
                "notepad_update: the FULL updated notepad text (overwrites). It is the ONLY thing that "
                "carries to later rounds — write down the facts you must recall."
            ),
        )
        def respond(answer: str, ctx: Context, notepad_update: str = "") -> Dict[str, Any]:
            session_id = self._get_session_id(ctx)
            session_data = self._get_or_create_session(ctx)
            env = session_data["env"]
            env.pending_notepad = notepad_update.strip() or None
            obs = self._execute_session_environment_step(session_id, answer)
            return obs

    def format_observation(self, obs: Dict[str, Any], env: Any) -> Dict[str, Any]:
        return {"observation": obs.get("prompt", ""), "instance_complete": obs.get("instance_complete", False)}
