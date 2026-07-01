"""EnvironmentAdapter wrapping the CLBench `exploitable_poker` task for eval-protocol MCP-Gym, as an
EXACT reproduction of the continual-learning-bench `icl_notepad` agent on this task.

We are NOT training the model to be a good poker player (that would bake the environment into the
weights). We are training the transferable skill of *learning with memory*: across a long series of
hands the agent must identify the opponent, write what it learns to a notepad, and exploit it later —
so later hands beat earlier ones. The reward is the memory GAIN (later-hand minus earlier-hand
reward); see reward.py.

Fidelity to clbench (verified byte-identical engine; see benchmark_fixtures/ + clbench-poker-fidelity):
  - Run structure = FIXED single opponent per rollout (clbench `variant` mode): one of the three
    canonical variants — calling_station/Tom, fit_or_fold/Adam, loose_aggressive/Alex — held fixed for
    the whole hand-series, with the opponent's NAME shown every hand. This keeps the late-vs-early
    memory reward unconfounded. (The full 5-stage curriculum with opponent switches is also supported
    via `schedule="default"`; see README "Design choice: fixed opponent vs curriculum".)
  - Memory = canonical icl_notepad: the conversation is CLEARED between hands (enforced by
    context_window.py windowing the policy input), the notepad PERSISTS and is the only cross-hand
    memory; it is shown as "=== YOUR NOTEPAD ===" and the prior hand's outcome as "FEEDBACK FROM
    PREVIOUS INSTANCE", exactly as systems/icl_notepad/system.py assembles them.
  - Agent layer = canonical: generic system prompt (see make_dataset.py), opponent shown by name, no
    legal-action scaffolding. Illegal actions are rejected+re-queried exactly as canonical for the first
    ILLEGAL_RETRY_LIMIT tries; only to stop a weak model's infinite illegal-action loop (which otherwise
    kills the rollout before any memory can accumulate) do we then force a minimal legal action. See the
    ILLEGAL_RETRY_LIMIT note; this is a per-decision timeout, not a change to the game the model plays.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from eval_protocol.mcp.adapter import EnvironmentAdapter

from src.interface import Response  # type: ignore
from src.registry import get_task_class  # type: ignore

DEFAULT_TASK_NAME = "exploitable_poker"
# CRITICAL — opponent MUST vary across rollouts. eval-protocol's McpGym creates every rollout env from
# adapter.get_default_config() and IGNORES the dataset row's environment_context (mcpgym.py:436 uses
# get_default_config, not the per-row config), passing only a per-rollout SEED. So a fixed default here
# means EVERY rollout sees the same opponent — which makes the task memoryless: "beat calling_station"
# bakes into the weights and CANCELS in the late-minus-early reward (no per-rollout discovery needed),
# pinning memory_gain at ~0. (Confirmed: a whole run played calling_station/Tom in 192/192 rollouts.)
# Fix: pick the opponent from the seed in reset() (see OPPONENTS) so it varies and must be DISCOVERED
# from early hands — that discovery is exactly the memory skill we train.
OPPONENTS = [  # (policy_id, displayed_name) — the three canonical variants with OPPOSITE exploits
    ("calling_station", "Tom"),    # calls bets, never bluffs   -> value-bet, never bluff
    ("fit_or_fold", "Adam"),       # over-folds without a hand   -> bluff relentlessly
    ("loose_aggressive", "Alex"),  # floats flop, barrels turn   -> let it bluff, call down
]
# num_instances is the hands-per-rollout (the env uses THIS default, not the dataset's). >= ~12 so there
# are enough early hands to identify which of the 3 opponents it is, and enough late hands to exploit it.
DEFAULT_TASK_KWARGS: Dict[str, Any] = {
    "opponent_policy": "calling_station",  # overridden per-rollout in reset() from the seed
    "opponent_name": "Tom",
    "num_instances": 16,
    "seed": 42,
}
NOTEPAD_MAX_CHARS = 2000
# Loop-breaker for the canonical reject+retry. The canonical env rejects an illegal action and re-shows
# the SAME state (task._invalid_action_result). A strong model (gpt-5-mini, what the benchmark ran)
# self-corrects in 1 try; a weak 1.7B can sample the same illegal action over and over and burn the
# whole step budget on ONE decision, so the rollout dies at ~1 hand (measured: 624/768 canonical
# rollouts completed <2 hands, 4899 illegal actions). After this many honest canonical rejections on
# the same turn we inject a guaranteed-legal, minimal-commitment action (CHECK if free else FOLD) so the
# hand series PROGRESSES and the late-vs-early memory signal becomes measurable. This is harness
# scaffolding (a per-decision timeout) — NOT a game change: the canonical reject+retry still applies for
# the first ILLEGAL_RETRY_LIMIT tries, the forced action is still counted as illegal for the penalty,
# and because legality skill is constant across a rollout it cannot bias the early-vs-late gain.
ILLEGAL_RETRY_LIMIT = 3


def _render_notepad(notepad: str) -> str:
    return f"=== YOUR NOTEPAD (carries across hands — write what helps you win later) ===\n{notepad}\n==="


class PokerEnv:
    """Multi-hand CLBench poker wrapper with an icl_notepad memory channel."""

    def __init__(self, task_name: str, task_kwargs: Dict[str, Any]):
        self.task_name = task_name
        self.task_kwargs = dict(task_kwargs)
        # run_index is not a task __init__ arg; it permutes the hand order via task.prepare_run().
        self.run_index = int(self.task_kwargs.pop("run_index", 0))
        self.task = None
        self.current_query = None
        self.response_schema = None           # clean PokerAction (no notepad field)
        self.done = False
        self.notepad = ""
        self.pending_notepad: Optional[str] = None  # set by the tool before each step
        self.notepad_updates = 0
        self.instance_idx = 0                 # how many hands have completed so far
        self._consec_illegal = 0              # consecutive illegal tries on the current decision
        self.opponent_policy = None           # this rollout's opponent (chosen from the seed in reset)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        kwargs = dict(self.task_kwargs)
        # NB: the incoming `seed` is ignored on purpose (it is constant across cloud rollouts; see below).
        # Pick THIS rollout's opponent AND cards from FRESH ENTROPY. We measured that the cloud RFT
        # harness passes a CONSTANT seed to every rollout (all 192 rollouts in a run dealt the identical
        # first hand [J club][7 heart] = seed 42), so we cannot rely on a per-rollout seed for variation.
        # Drawing fresh entropy guarantees: (1) the opponent varies per rollout -> the task REQUIRES
        # memory (discover which of the 3 opponents you face); (2) the cards vary per rollout -> that
        # memory must be about OPPONENT TENDENCIES (random cards can't be memorized), not specific hands.
        # Trade-off: GRPO candidates of one row no longer share opponent/cards, so the advantage is
        # noisier — mitigated with more candidates/epochs; the late-minus-early reward is within-rollout
        # so it is relatively insensitive to which opponent/cards a given rollout happened to draw.
        import os as _os, random as _random
        ent = _random.Random(_os.urandom(16))
        pol, name = ent.choice(OPPONENTS)
        kwargs["opponent_policy"] = pol
        kwargs["opponent_name"] = name
        kwargs["seed"] = ent.randrange(1, 2 ** 31)   # fresh per-rollout cards
        self.opponent_policy = pol
        task_cls = get_task_class(self.task_name)
        self.task = task_cls(**kwargs)
        # Permute the hand sequence for this run (within-stage shuffle); run_index=0 = canonical order.
        if self.run_index:
            self.task.prepare_run(self.run_index)
        self.current_query = self.task.reset()
        self.response_schema = self.current_query.response_schema
        self.done = False
        self.notepad = ""
        self.pending_notepad = None
        self.notepad_updates = 0
        self.instance_idx = 0
        self._consec_illegal = 0
        return self._obs(self.current_query.prompt, instance_complete=False), {}

    def step(self, action_obj: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # 1) consume any notepad update the tool stashed for this turn
        np_update = self.pending_notepad
        self.pending_notepad = None
        if np_update:
            self.notepad = str(np_update)[:NOTEPAD_MAX_CHARS]
            self.notepad_updates += 1

        # 2) step the poker task with the clean action
        sr = self.task.step(Response(action=action_obj, metadata={}))
        obs_content = getattr(sr.observation, "content", "") or ""
        illegal = obs_content.lower().startswith("invalid")
        # CANONICAL reject+retry (task._invalid_action_result re-queries the SAME state; we do NOT
        # auto-substitute on the first tries — the model must produce a legal action itself). But to
        # prevent the weak-model death spiral (same illegal action sampled forever until the step budget
        # is gone -> rollout dies at 1 hand), after ILLEGAL_RETRY_LIMIT honest rejections on this same
        # decision we inject a guaranteed-legal minimal-commitment action so the hand series progresses.
        if illegal:
            self._consec_illegal += 1
            if self._consec_illegal >= ILLEGAL_RETRY_LIMIT:
                forced = self._forced_legal_action(sr)
                if forced is not None:
                    sr = self.task.step(Response(action=forced, metadata={}))
                    obs_content = getattr(sr.observation, "content", "") or ""
                self._consec_illegal = 0   # decision resolved (forced); start fresh next turn
        else:
            self._consec_illegal = 0

        self.done = bool(sr.done)

        reward = 0.0
        oc = getattr(sr, "instance_outcome", None)
        if oc is not None and getattr(oc, "reward", None) is not None:
            reward = float(oc.reward)

        instance_complete = bool(getattr(sr.observation, "instance_complete", False))

        hand_index = self.instance_idx          # index of the hand this step belongs to
        if instance_complete:
            self.instance_idx += 1

        next_query = getattr(sr, "next_query", None)
        if next_query is not None:
            self.current_query = next_query
            self.response_schema = next_query.response_schema or self.response_schema
            if instance_complete:
                # NEW INSTANCE (hand) boundary. Reproduce the canonical icl_notepad prompt assembly
                # (systems/icl_notepad/system.py respond()): the previous hand's outcome is injected as
                # "FEEDBACK FROM PREVIOUS INSTANCE", then the persistent notepad (only when non-empty),
                # then the new hand's query. context_window.py windows the policy input to this boundary
                # onward, so the conversation is cleared between hands and the notepad is the ONLY memory
                # carried across them — exactly the icl_notepad "clear_context_between_instances" behavior.
                parts = []
                if obs_content:
                    parts.append(f"FEEDBACK FROM PREVIOUS INSTANCE:\n{obs_content}")
                if self.notepad:
                    parts.append(f"=== YOUR NOTEPAD ===\n{self.notepad}\n===================")
                parts.append(next_query.prompt)
                prompt = "\n\n".join(parts)
            else:
                # Mid-hand turn (or a rejected illegal action): canonical injects the task observation
                # as interstitial FEEDBACK, then re-shows the (next) query for the same hand.
                prompt = f"FEEDBACK: {obs_content}\n\n{next_query.prompt}" if obs_content else next_query.prompt
        else:
            prompt = obs_content

        # AUTHORITATIVE per-hand reward vector. Against fold-happy opponents (fit_or_fold,
        # loose_aggressive) some hands auto-resolve BETWEEN the model's actions (e.g. the opponent
        # folds preflop while we are the big blind), so they never appear as a scored tool result and
        # scraping the transcript silently drops them — asymmetrically by opponent type, which would
        # bias the late-vs-early memory reward. The task's `hand_history` records EVERY hand, so at
        # episode end we emit the full per-hand profit (in big blinds) as a machine-readable line that
        # the reward function reads verbatim. Emitted only on `done`, so the model never acts on it.
        if self.done:
            hist = getattr(self.task, "hand_history", None) or []
            bb = float(getattr(self.task, "big_blind", 10) or 10)
            if hist:
                profits_bb = ",".join(f"{h.get('profit', 0) / bb:.4f}" for h in hist)
                prompt = f"{prompt}\n\nHAND_PROFITS_BB: {profits_bb}"
            # Record which opponent this rollout faced (for per-opponent analysis + verifying that the
            # seed-derived opponent actually varies across rollouts). Emitted only at done.
            if self.opponent_policy:
                prompt = f"{prompt}\nOPPONENT_POLICY: {self.opponent_policy}"

        info = {
            "instance_complete": instance_complete,
            "illegal_action": illegal,
            "hand_index": hand_index,
            # chip reward of the hand that just completed (None on mid-hand steps)
            "hand_reward": reward if instance_complete else None,
            "notepad_len": len(self.notepad),
        }
        return self._obs(prompt, instance_complete=instance_complete), reward, self.done, False, info

    def _forced_legal_action(self, sr: Any) -> Any:
        """Pick a guaranteed-legal, minimal-commitment action to break an illegal-action loop. Reads the
        canonical legal-action set from the rejected step's next_query metadata (task._legal_actions(),
        validated against the live game), preferring CHECK (free, continues) then FOLD (ends the hand,
        minimal loss) then CALL. Returns a response_schema (PokerAction) instance, or None if unknown."""
        nq = getattr(sr, "next_query", None)
        meta = getattr(nq, "metadata", None) or {}
        legal = [str(a).upper() for a in (meta.get("legal_actions") or [])]
        if not legal:
            legal = [a.upper() for a in (_extract_legal_actions(getattr(nq, "prompt", "") or "") or [])]
        schema = self.response_schema
        if schema is None:
            return None
        for act in ("CHECK", "FOLD", "CALL"):
            if act in legal:
                # model_construct bypasses field validation (the schema also requires `thinking`); the
                # task only reads .action/.amount from the response, so this is sufficient and safe.
                return schema.model_construct(action=act, amount=None)
        return None

    @staticmethod
    def _obs(prompt: str, *, instance_complete: bool) -> Dict[str, Any]:
        # EXACT CANONICAL: no extra legal-actions scaffolding. The task's hand prompt already lists the
        # available actions in natural language ("What's your action? FOLD/CALL/CHECK/RAISE X"), and an
        # illegal action is rejected and re-queried — same as the clbench reference agent sees.
        return {"prompt": prompt, "instance_complete": instance_complete}


def _extract_legal_actions(prompt_text: str):
    """Pull the per-turn legal action set from the task's "Situation:" line (ported from env.py)."""
    if not prompt_text:
        return None
    sit = None
    for line in prompt_text.splitlines():
        s = line.strip()
        if s.startswith("Situation:"):
            sit = s.lower()
            break
    if sit is None:
        return None
    if "you can check" in sit:
        legal = ["CHECK", "RAISE", "FOLD"]
    elif "you need" in sit and "to call" in sit:
        legal = ["FOLD", "CALL", "RAISE"]
    elif "all-in" in sit:
        legal = ["FOLD", "CALL"]
    else:
        return None
    seen, out = set(), []
    for a in legal:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out or None


class PokerAdapter(EnvironmentAdapter):
    """Adapter bridging the multi-hand CLBench poker task into the MCP-Gym base class."""

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> PokerEnv:
        cfg = {**DEFAULT_TASK_KWARGS, **(config or {})}
        task_name = cfg.pop("task_name", DEFAULT_TASK_NAME)
        return PokerEnv(task_name=task_name, task_kwargs=cfg)

    def create_environment_with_seed(
        self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ) -> Tuple[PokerEnv, Dict[str, Any], Dict[str, Any]]:
        env = self.create_environment(config)
        obs, info = env.reset(seed=seed)
        return env, obs, info

    def reset_environment(self, env: PokerEnv, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return env.reset(seed=seed)

    def step_environment(self, env: PokerEnv, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        return env.step(action)

    def close_environment(self, env: PokerEnv) -> None:
        pass

    def get_default_config(self) -> Dict[str, Any]:
        return dict(DEFAULT_TASK_KWARGS)
