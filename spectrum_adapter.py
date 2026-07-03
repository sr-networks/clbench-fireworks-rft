"""EnvironmentAdapter wrapping CLBench `blind_spectrum_monitoring` for eval-protocol MCP-Gym, as a
clean MEMORY-training task (the poker port stalled because a 1.7B never learned to write useful notepad
memory; this task's memory is simpler and its reward is deterministic — see RUNS/notes).

The task: each rollout is a series of `num_instances` spectrum scans of ONE fixed but UNKNOWN band of
transmitters. Each scan shows noisy detected peaks; only a few transmitters are active per scan
(`n_active`), the rest are DORMANT (invisible this scan). The agent reports the persistent occupied
regions; it is scored by IoU of the implied AVAILABLE spectrum vs the ground-truth available set
(complement of ALL transmitters, dormant or not). So a MEMORYLESS agent that reports only the current
scan's peaks misses the dormant transmitters and scores low; an agent that ACCUMULATES peaks across
scans recovers the full band and scores higher in later scans. The reward is the late-minus-early IoU
gain (the task's own `learning_delta`), which is exactly "learning with memory": validated offline as
memoryless delta = +0.00, memory delta = +0.07 (clean, low-variance signal).

Design (mirrors the poker fixes that were hard-won):
  - Memory channel = FULL accumulating context (canonical clbench `icl` system): we do NOT window the
    conversation, so the policy sees every prior scan's peaks and its own prior reports. No notepad is
    required (that was poker's failure mode). Training improves the model's in-context aggregation.
  - Band layout VARIES per rollout, drawn from FRESH ENTROPY (McpGym passes a constant seed + ignores
    dataset config — same gotcha as poker), so the transmitter set must be DISCOVERED within the
    rollout (that discovery is the memory skill), not baked into the weights.
  - Action = a simple `submit_report(center_freqs, bandwidths)` tool (the IoU score only uses
    center_freq+bandwidth), i.e. two flat float lists — far easier for a weak model than nested objects.
"""

from __future__ import annotations

import hashlib
import os
import random
import secrets
from typing import Any, Dict, List, Optional, Tuple

from eval_protocol.mcp.adapter import EnvironmentAdapter

from src.interface import Response  # type: ignore
from src.registry import get_task_class  # type: ignore

# Redirect the task's Jinja templates to our vendored copies: the pip-installed `src` package ships
# WITHOUT tasks/blind_spectrum_monitoring/templates/ (TemplateNotFound), and the cloud evaluator installs
# the same source, so we bundle the templates here and point the task's environment at them.
import src.tasks.blind_spectrum_monitoring.task as _bsm  # type: ignore
from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TPL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectrum_templates")
_bsm._jinja_env = Environment(
    loader=FileSystemLoader(_TPL_DIR), undefined=StrictUndefined,
    keep_trailing_newline=False, trim_blocks=True, lstrip_blocks=False,
)

DEFAULT_TASK_NAME = "blind_spectrum_monitoring"
# The env uses THESE defaults (McpGym ignores per-row dataset config). num_instances scans/rollout;
# n_active transmitters visible per scan (the rest dormant -> must be remembered). Tuned offline so the
# memoryless baseline is low and the memory late-early gain is clear.
DEFAULT_TASK_KWARGS: Dict[str, Any] = {
    "num_instances": 12,
    "n_active": 3,
    "band_width": 180.0,
    "p_false_alarm": 0.2,
    "p_miss": 0.15,
    "repeat_instructions": False,
}
MIN_CH, MAX_CH = 11, 14         # transmitters per (random) band layout
CH_BW = 8.0                     # transmitter bandwidth (MHz) — also the FIXED reported width
MAX_REPORT = 16                 # cap on regions per report: blocks blanketing the band (carpet exploit)
NOTEPAD_MAX_CHARS = 2000        # cap on the agent's notepad (its persistent cross-scan memory)

# --- Experiment flags (the evaluator entry file sets these os.environ BEFORE importing this module) ---
SCAFFOLD = os.environ.get("SPECTRUM_SCAFFOLD") == "1"   # env-echoed running list (the RL-improvable memory channel)
SCRAMBLE = os.environ.get("SPECTRUM_SCRAMBLE") == "1"   # destroy the echoed list's CONTENT (memory-content control arm)

# EPOCH-SALT: fresh bands every epoch. Each RFT epoch evaluates in its own process (per-epoch streamlogs),
# so an import-time salt is CONSTANT within an epoch — all GRPO candidates of a row still share one band —
# but DIFFERENT across epochs: no band ever repeats, so weight-memorizing layouts ("content baking") is
# structurally impossible and every epoch's metrics are automatically a FRESH-BAND eval of the current
# policy. (All earlier datasets reused row_ids spectrum_{i}, i.e. the SAME 48 bands across all runs/epochs.)
# The probe job verifies the one-salt-per-epoch assumption via the [spectrum] SALT= streamlog line.
EPOCH_SALT = secrets.token_hex(4) if os.environ.get("SPECTRUM_EPOCH_SALT") == "1" else ""


def band_seed(row_id: Any) -> int:
    """Per-ROW deterministic band seed (GRPO candidates of a row share one band); EPOCH_SALT, when enabled,
    rotates the whole band set every epoch-process."""
    return int(hashlib.md5(f"{EPOCH_SALT}:{row_id}".encode()).hexdigest()[:12], 16)

# AGENT-CONTROLLED NOTEPAD design (replaces the env-echoed "running list" scaffold — which was the ENV doing
# the remembering). The agent has notepad_read / notepad_write tools (see spectrum_mcp), and ICL-OFF
# windowing hides all earlier scans, so the notepad is the agent's ONLY cross-scan memory. SCAN_MARKER is
# prepended to every scan delivery so spectrum_context_window keeps [system] + [current scan + this turn's
# notepad reads] and drops prior scans — the agent must LEARN to write/read the notepad to recall dormant
# transmitters (the system prompt does NOT tell it what to store).
ICL_OFF = True
SCAN_MARKER = "[[SCAN_BOUNDARY]]"


def _mark(prompt: str) -> str:
    """Prepend the scan-boundary marker so windowing keeps only the current scan + this turn's notepad reads."""
    if not ICL_OFF:
        return prompt
    mem = "the RUNNING LIST shown with this scan" if SCAFFOLD else "your notepad (call notepad_read)"
    return (f"{SCAN_MARKER} NEW SCAN — you cannot see earlier scans. Your only memory of them is "
            f"{mem}.\n{prompt}")


def _random_layout(ent: random.Random, band_width: float) -> List[Dict[str, Any]]:
    """A random band: MIN_CH..MAX_CH transmitters at RANDOM non-overlapping center frequencies. Positions
    must be IRREGULAR (not a grid): an earlier grid layout let the model bake "report transmitters every
    ~15 MHz" into its weights and score constant IoU WITHOUT memory (memory_gain stayed 0). With random
    positions the model cannot guess where transmitters are — it must DISCOVER each one from the scans it
    appears in (memory). Few active per scan (n_active) => most dormant each scan => memory required.
    Validated: memoryless IoU ~0.47 (flat), memory ~0.50+ (delta +0.05), even-spaced guess ~0.31."""
    n = ent.randint(MIN_CH, MAX_CH)
    centers: List[float] = []
    attempts = 0
    while len(centers) < n and attempts < 400:
        attempts += 1
        cf = round(ent.uniform(CH_BW, band_width - CH_BW), 1)
        if all(abs(cf - c) >= CH_BW + 2 for c in centers):   # non-overlapping with a small guard gap
            centers.append(cf)
    centers.sort()
    return [{"id": i, "slot": i, "center_freq": cf, "bandwidth": CH_BW} for i, cf in enumerate(centers)]


class SpectrumEnv:
    """Multi-scan blind-spectrum wrapper with full-context (icl) memory."""

    def __init__(self, task_name: str, task_kwargs: Dict[str, Any]):
        self.task_name = task_name
        self.task_kwargs = dict(task_kwargs)
        self.band_width = float(self.task_kwargs.get("band_width", 180.0))
        self.task = None
        self.current_query = None
        self.response_schema = None       # ScanReport
        self.done = False
        self.scan_ious: List[float] = []  # per-scan AVAILABLE-spectrum IoU (clbench-native, tracked)
        self.occ_ious: List[float] = []   # per-scan OCCUPIED-spectrum IoU (the memory reward signal)
        self._gt_channels: List[Dict[str, Any]] = []  # ground-truth transmitter layout (for occ-IoU)
        self._dbg_seed: Optional[int] = None          # seed the env was built with (probe)
        self.layout_n = 0
        self.notepad: str = ""                         # the agent's persistent memory (notepad_read/write)
        self.pending_scan: str = ""                    # next scan, delivered on demand via the get_scan tool
        self.last_report: List[float] = []             # center freqs of the previous report (scaffold echo)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # PER-ROW BAND for GRPO comparability: seed the band DETERMINISTICALLY from the framework-provided
        # seed (or the row seed injected by SpectrumMcp), so all candidates of the SAME dataset row share
        # ONE band and ONE activity schedule — their reward differences then reflect POLICY, not band luck.
        # (v1-v5 used os.urandom here -> every candidate got a different band -> GRPO advantage was noise
        # -> frozen policy.) Fall back to fresh entropy only if nothing was provided.
        self._dbg_seed = seed
        ent = random.Random(int(seed)) if seed is not None else random.Random(os.urandom(16))
        channels = _random_layout(ent, self.band_width)
        self.layout_n = len(channels)
        self._gt_channels = channels
        kwargs = dict(self.task_kwargs)
        kwargs.update(channels=channels, seed=(int(seed) if seed is not None else ent.randrange(1, 2 ** 31)))
        self.task = get_task_class(self.task_name)(**kwargs)
        self.task.build_canonical_run_state()
        self.current_query = self.task.build_current_query()
        self.response_schema = self.current_query.response_schema
        self.done = False
        self.scan_ious = []
        self.occ_ious = []
        self.notepad = ""
        self.pending_scan = ""
        self.last_report = []
        # Scan 1 is the initial observation (marked; no echo yet — nothing has been reported).
        return self._obs(_mark(self.current_query.prompt), instance_complete=False), {}

    def _echo(self) -> str:
        """The running-list scaffold (SCAFFOLD=1): echo the model's OWN previous report into the next scan,
        turning cross-scan accumulation into a one-step merge — the env maintains the memory channel, the
        model must USE it. SCRAMBLE=1 (control arm) replaces the content with random in-band frequencies of
        the SAME count: structure/growth/instruction identical, memory CONTENT destroyed — any training gain
        that survives scrambling is non-memory skill, any gain that disappears was real memory use."""
        if not (SCAFFOLD and self.last_report):
            return ""
        freqs = self.last_report
        if SCRAMBLE:
            rng = random.Random(os.urandom(8))
            freqs = sorted(round(rng.uniform(CH_BW, self.band_width - CH_BW), 1) for _ in freqs)
        lst = ", ".join(f"{f:.1f}" for f in freqs)
        return ("\n=== YOUR RUNNING LIST (your previous report) ===\n"
                f"{lst}\n"
                "Keep ALL of these, add any NEW peaks from this scan, and submit the FULL updated list.\n")

    def _occ_iou(self, action_obj: Any) -> float:
        """IoU of the OCCUPIED spectrum: |reported_occupied ∩ true_occupied| / |union|, over the band.

        This is the MEMORY reward (vs the task's native AVAILABLE-spectrum IoU). A memoryless agent that
        reports only the n_active current peaks covers ~n_active/total of the true occupied spectrum
        (e.g. 3/12 ≈ 0.25); an agent that recalls dormant transmitters from earlier scans covers more,
        up to 1.0. Carpeting the whole band is capped (~0.5) because the union blows up — so the only way
        past it is ACCURATE accumulation. Early scans are information-limited (few transmitters seen yet)
        so their ceiling is low; maximizing this IoU therefore raises LATE scans more than EARLY ones,
        which is exactly the late-minus-early memory gain we want to prove."""
        res = 0.5
        nb = int(self.band_width / res) + 1
        gt = bytearray(nb)
        rp = bytearray(nb)

        def paint(arr: bytearray, cf: float, bw: float) -> None:
            lo = max(0.0, cf - bw / 2.0)
            hi = min(self.band_width, cf + bw / 2.0)
            for i in range(int(lo / res), min(int(hi / res) + 1, nb)):
                arr[i] = 1

        for ch in (self._gt_channels or []):
            paint(gt, float(ch["center_freq"]), float(ch["bandwidth"]))
        for t in (getattr(action_obj, "transmitters", None) or []):
            paint(rp, float(getattr(t, "center_freq", 0.0)), float(getattr(t, "bandwidth", 0.0) or 0.0))
        inter = sum(1 for x, y in zip(gt, rp) if x and y)
        union = sum(1 for x, y in zip(gt, rp) if x or y)
        return inter / union if union else 0.0

    def step(self, action_obj: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        sr = self.task.step(Response(action=action_obj, metadata={}))
        oc = getattr(sr, "instance_outcome", None)
        avail = float(getattr(oc, "reward", 0.0) or 0.0) if oc is not None else 0.0  # native AVAILABLE-IoU
        occ = self._occ_iou(action_obj)                                              # OCCUPIED-IoU (reward)
        self.scan_ious.append(avail)
        self.occ_ious.append(occ)
        self.done = bool(sr.done)
        try:  # remember what was just reported — the scaffold echoes it into the next scan
            self.last_report = [
                float(getattr(t, "center_freq", 0.0))
                for t in (getattr(action_obj, "transmitters", None) or [])
            ][:MAX_REPORT]
        except Exception:
            self.last_report = []

        # submit_report returns the NEXT scan directly (clean + marked, NO running-list scaffold) — the
        # agent's sole cross-scan memory is its own notepad (notepad_read/notepad_write). SCAN_OCC is
        # appended for the reward parser; windowing keeps it to the current scan only, so it never leaks
        # into a later report decision.
        next_query = getattr(sr, "next_query", None)
        if next_query is not None:
            self.current_query = next_query
            self.response_schema = next_query.response_schema or self.response_schema
            nxt = next_query.prompt
        else:
            nxt = getattr(sr.observation, "content", "") or ""
        # Clean next scan (marked, no SCAN_OCC) — read by the custom SpectrumTurnRolloutProcessor via
        # env.pending_scan. McpGym's obs below appends SCAN_OCC to it (windowed to the current scan).
        # Under SCAFFOLD the running-list echo (real or scrambled) rides along inside the scan delivery.
        self.pending_scan = "" if self.done else (_mark(nxt) + self._echo())
        prompt = f"{self.pending_scan}\nSCAN_OCC: {occ:.4f} SCAN_AVAIL: {avail:.4f}"

        info = {"instance_complete": self.done, "scan_occ": occ, "scan_avail": avail, "scans_done": len(self.occ_ious)}
        return self._obs(prompt, instance_complete=self.done), occ, self.done, False, info

    @staticmethod
    def _obs(prompt: str, *, instance_complete: bool) -> Dict[str, Any]:
        return {"prompt": prompt, "instance_complete": instance_complete}


class SpectrumAdapter(EnvironmentAdapter):
    """Bridges the multi-scan blind-spectrum task into the MCP-Gym base class."""

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> SpectrumEnv:
        cfg = {**DEFAULT_TASK_KWARGS, **(config or {})}
        task_name = cfg.pop("task_name", DEFAULT_TASK_NAME)
        return SpectrumEnv(task_name=task_name, task_kwargs=cfg)

    def create_environment_with_seed(
        self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ) -> Tuple[SpectrumEnv, Dict[str, Any], Dict[str, Any]]:
        env = self.create_environment(config)
        obs, info = env.reset(seed=seed)
        return env, obs, info

    def reset_environment(self, env: SpectrumEnv, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return env.reset(seed=seed)

    def step_environment(self, env: SpectrumEnv, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        return env.step(action)

    def close_environment(self, env: SpectrumEnv) -> None:
        pass

    def get_default_config(self) -> Dict[str, Any]:
        return dict(DEFAULT_TASK_KWARGS)
