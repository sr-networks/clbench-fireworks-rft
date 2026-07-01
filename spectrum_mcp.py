"""Blind-spectrum MCP-Gym server. Exposes one data-plane tool, `submit_report`, taking the persistent
occupied regions as two parallel float lists (center_freqs, bandwidths) — the IoU score only uses those,
so this is the simplest faithful action for a weak model. The tool builds the task's ScanReport."""

import hashlib
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context

from eval_protocol.mcp import McpGym
from eval_protocol.mcp.mcpgym import control_plane_endpoint

from spectrum_adapter import SpectrumAdapter, CH_BW, MAX_REPORT, NOTEPAD_MAX_CHARS
from src.tasks.blind_spectrum_monitoring.task import ScanReport, Transmitter  # type: ignore


def _band_seed(session_id: Optional[str]) -> Optional[int]:
    """Deterministic band seed from the PER-ROW session_id (md5 of seed+config+dataset_row_id+...). The
    cloud gives every candidate of one dataset row the SAME session_id (verified: 24 rows -> 24 sids,
    constant within a row), so seeding the band from it makes all GRPO candidates of a row face the SAME
    band+schedule (reward diffs = policy, not band luck) while different rows get different bands (so the
    model can't memorize one band -- which would itself zero out memory_gain). This is THE fix for the
    frozen-policy bug (v1-v5 used os.urandom per rollout -> every candidate a different band -> advantage
    was noise -> no gradient)."""
    if not session_id:
        return None
    return int(hashlib.md5(str(session_id).encode()).hexdigest()[:12], 16)


def _extra_of(ctx: Context) -> Dict[str, Any]:
    """The client_info._extra dict the rollout client attaches (session_id / dataset_row_id / seed /
    model_id). Used to discover what the CLOUD actually passes (vs the local eval client)."""
    try:
        ci = ctx.session.client_params.clientInfo  # type: ignore[attr-defined]
        ex = getattr(ci, "_extra", None)
        return dict(ex) if isinstance(ex, dict) else {}
    except Exception:
        return {}


class SpectrumMcp(McpGym):
    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__("CLBench-BlindSpectrumMonitoring", SpectrumAdapter(), seed, **kwargs)

    @control_plane_endpoint("/control/initial_state")
    async def get_initial_state_endpoint(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override: scan 1's env is born here. Rebuild it with a band seeded from the PER-ROW session_id
        so all candidates of a row share one band (GRPO comparability). Without this the base class env
        (created with seed=None -> our os.urandom fallback) gives each candidate a different band."""
        sid = session_data.get("session_id")
        bseed = _band_seed(sid)
        if bseed is not None:
            env, obs, info = self._new_env(seed=bseed)
            session_data["env"] = env
            session_data["obs"] = obs
        env = session_data.get("env")
        obs = session_data.get("obs")
        if env is not None and obs is not None:
            return self.format_observation(obs, env)
        return {"observation": "session_not_initialized", "session_id": sid}

    def _get_or_create_session(self, ctx: Context) -> Dict[str, Any]:
        """Override: when the env is first created via a TOOL call, seed its band from the per-row
        session_id too (defensive — covers cloud orderings where a tool runs before initial_state)."""
        session_id = self._get_session_id(ctx)
        if session_id not in self.sessions:
            env, obs, info = self._new_env(seed=_band_seed(session_id))
            with self.session_lock:
                self.sessions[session_id] = {
                    "env": env, "obs": obs, "session_data": {}, "session_id": session_id,
                }
        return self.sessions[session_id]

    def _register_tools(self):
        @self.mcp.tool(
            name="submit_report",
            description=(
                "Submit your answer for the CURRENT scan: center_freqs = the center frequencies (MHz) of "
                "the transmitter regions you believe occupy the band (each region is a fixed 8 MHz wide, so "
                "give frequencies only). It returns the NEXT scan's peaks. Earlier scans are NOT shown "
                "again — only what you saved with notepad_write persists across scans."
            ),
        )
        def submit_report(center_freqs: List[float], ctx: Context) -> Dict[str, Any]:
            session_id = self._get_session_id(ctx)
            self._get_or_create_session(ctx)
            # Fixed bandwidth (true CH_BW) and a hard cap on regions: removes a wasted DOF for a weak model
            # and makes "carpet the whole band" impossible (<=MAX_REPORT fixed-width tiles can't blanket
            # 180 MHz), so the only way to raise occupied-IoU is to place regions on REAL transmitters
            # recalled from earlier scans — i.e. via the notepad.
            freqs = [float(c) for c in (center_freqs or [])][:MAX_REPORT]
            txs = [
                Transmitter(center_freq=c, bandwidth=CH_BW, currently_active=True, estimated_power=-30.0)
                for c in freqs
            ]
            report = ScanReport(transmitters=txs)
            return self._execute_session_environment_step(session_id, report)  # -> next scan + SCAN_OCC

        @self.mcp.tool(
            name="notepad_read",
            description=(
                "Read your private notepad — the ONLY memory that persists across scans. Returns whatever "
                "you have written (empty until you write something)."
            ),
        )
        def notepad_read(ctx: Context) -> Dict[str, Any]:
            env = self._get_or_create_session(ctx)["env"]
            note = getattr(env, "notepad", "") or ""
            return {"observation": note if note else "(notepad is empty — you have written nothing yet)"}

        @self.mcp.tool(
            name="notepad_write",
            description=(
                "Overwrite your private notepad with `text` (the FULL new contents — it replaces what was "
                "there). The notepad is the only thing that carries to later scans, where earlier scans are "
                "no longer visible. Use it however helps you solve the task."
            ),
        )
        def notepad_write(text: str, ctx: Context) -> Dict[str, Any]:
            env = self._get_or_create_session(ctx)["env"]
            env.notepad = str(text or "")[:NOTEPAD_MAX_CHARS]
            return {"observation": "ok"}

    def format_observation(self, obs: Dict[str, Any], env: Any) -> Dict[str, Any]:
        # The env's obs already carries the marked, scaffold-free next scan + its SCAN_OCC line (for the
        # reward). Windowing keeps the model's view to the current scan, so the notepad is its only memory.
        return {
            "observation": obs.get("prompt", ""),
            "instance_complete": obs.get("instance_complete", False),
        }
