"""A clean, deterministic learn-and-recall memory task for the MCP-Gym.

Purpose: demonstrate *learning with memory* without the noise of poker. Each rollout is N rounds:

  round 0 (LEARN):  the environment reveals a RANDOM secret ("the passphrase is <word>; the lucky
                    number is <n>"). The model should write it to its notepad. Reward 0 (nothing to
                    recall yet) — this is the "without memory" baseline round.
  rounds 1..N-1 (RECALL): the environment asks for the secret ("what is the passphrase?"). Reward
                    = +1 if the model's answer matches, else 0.

Because the cross-round trace is cut (context_window.py) and the secret is random per rollout:
  - the ONLY way to score in recall rounds is to have written the secret to the notepad in round 0;
  - the secret cannot be memorised into the weights (it changes every rollout) — the model can only
    learn the transferable SKILL of "write the key fact down, then read it back".

This wraps directly into the EnvironmentAdapter interface (no CLBench dependency).
"""

from __future__ import annotations

import random
import string
from typing import Any, Dict, List, Optional, Tuple

from eval_protocol.mcp.adapter import EnvironmentAdapter

# Keys the model must remember a random CODE for. Many facts + hard codes => the base model's casual
# note-taking is insufficient, leaving headroom for RFT to teach thorough/accurate recording.
KEYS = ["RED", "BLUE", "GREEN", "GOLD", "SILVER", "BLACK", "WHITE", "VIOLET"]
DEFAULT_NUM_FACTS = 6
DEFAULT_ROUNDS = 8          # 1 LEARN + 7 RECALL
NOTEPAD_MAX_CHARS = 2000


def _code(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(4))


def _render_notepad(notepad: str) -> str:
    return f"--- NOTEPAD ---\n{notepad if notepad else '(empty — you have written nothing down yet)'}\n--- END NOTEPAD ---"


class MemoryEnv:
    """N-round learn-and-recall episode. Round 0 reveals a secret; later rounds test recall."""

    def __init__(self, rounds: int = DEFAULT_ROUNDS, num_facts: int = DEFAULT_NUM_FACTS):
        self.rounds = rounds
        self.num_facts = min(num_facts, len(KEYS))
        self.rng: Optional[random.Random] = None
        self.facts: Dict[str, str] = {}
        self.ask_key = ""
        self.round_idx = 0
        self.done = False
        self.notepad = ""
        self.pending_notepad: Optional[str] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.rng = random.Random(seed if seed is not None else random.randint(0, 1_000_000))
        keys = KEYS[: self.num_facts]
        self.facts = {k: _code(self.rng) for k in keys}
        self.round_idx = 0
        self.done = False
        self.notepad = ""
        self.pending_notepad = None
        facts_str = "; ".join(f"{k} = {v}" for k, v in self.facts.items())
        # DISCOVERY framing: present the codes plainly, with NO hint to remember/write/be-tested.
        # The base model mostly won't record them -> low recall -> real headroom for RL to DISCOVER
        # that writing to the notepad earns reward (learning to USE memory, not following instructions).
        prompt = (
            f"=== ROUND 1 of {self.rounds} ===\n"
            f"CODES: {facts_str}\n"
            f"Call `respond` with answer='ok'."
        )
        return {"prompt": prompt, "instance_complete": False}, {}

    def _ask(self) -> str:
        self.ask_key = self.rng.choice(list(self.facts.keys()))
        return self.ask_key

    def step(self, answer: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        np_update = self.pending_notepad
        self.pending_notepad = None
        if np_update:
            self.notepad = str(np_update)[:NOTEPAD_MAX_CHARS]

        ans = (answer or "").strip().upper()
        reward = 0.0
        is_recall = self.round_idx >= 1
        if is_recall and self.ask_key:
            reward = 1.0 if self.facts[self.ask_key] in ans else 0.0

        completed_round = self.round_idx
        self.round_idx += 1
        self.done = self.round_idx >= self.rounds

        if not self.done:
            asked = self._ask()
            prompt = (
                f"=== NEW HAND (ROUND {self.round_idx + 1} of {self.rounds}, RECALL) — you CANNOT see "
                f"earlier rounds; your ONLY memory is the notepad below ===\n"
                f"{_render_notepad(self.notepad)}\n\n"
                f"QUESTION: what is the secret code for {asked}? Call `respond` with just the code."
            )
        else:
            prompt = "=== EPISODE COMPLETE ==="

        info = {
            "instance_complete": True,
            "round_index": completed_round,
            "is_recall": is_recall,
            "hand_reward": reward,
            "notepad_len": len(self.notepad),
        }
        return {"prompt": prompt, "instance_complete": True}, reward, self.done, False, info


class MemoryAdapter(EnvironmentAdapter):
    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> MemoryEnv:
        cfg = config or {}
        return MemoryEnv(rounds=int(cfg.get("rounds", DEFAULT_ROUNDS)),
                         num_facts=int(cfg.get("num_facts", DEFAULT_NUM_FACTS)))

    def create_environment_with_seed(self, config=None, seed=None):
        env = self.create_environment(config)
        obs, info = env.reset(seed=seed)
        return env, obs, info

    def reset_environment(self, env: MemoryEnv, seed: Optional[int] = None):
        return env.reset(seed=seed)

    def step_environment(self, env: MemoryEnv, action: Any):
        return env.step(action)

    def close_environment(self, env: MemoryEnv) -> None:
        pass

    def get_default_config(self) -> Dict[str, Any]:
        return {"rounds": DEFAULT_ROUNDS}
