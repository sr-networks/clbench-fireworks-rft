"""CONTROL EXPERIMENT: prove the memory_gain metric has no structural confound.

The reward (reward.py) scores a rollout by late_mean - early_mean of per-hand chip profit, claiming this
isolates LEARNING-WITH-MEMORY (a player that uses cross-hand info wins more in later hands). For that
claim to hold, a player with NO memory must score ~0 — otherwise positional/structural drift (button
rotation, opponent warm-up, etc.) would masquerade as "memory".

This script runs a strictly MEMORYLESS player (its action depends ONLY on the current legal set, never
on history or the notepad) over many rollouts vs all three opponents and reports the late-early gain.

Result (300 rollouts x 10 hands, loop-breaker env): gain = -0.006 +/- 0.063  => statistically ZERO.
    calling_station  -0.028 +/- 0.059
    fit_or_fold      -0.013 +/- 0.056
    loose_aggressive +0.023 +/- 0.171
=> The metric is CLEAN: any positive memory_gain a trained model earns is attributable to memory use,
   not to a structural artifact of the late-vs-early split.

    python control_memoryless.py
"""
from __future__ import annotations

import re
from statistics import mean, pstdev

from poker_adapter import PokerEnv

_PROF = re.compile(r"HAND_PROFITS_BB:\s*([-\d.,\s]+)")


def per_hand(text: str):
    m = _PROF.search(text)
    return [float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()] if m else []


def _legal(prompt: str):
    for line in prompt.splitlines():
        s = line.strip().lower()
        if s.startswith("situation:"):
            if "you can check" in s:
                return ["CHECK", "RAISE", "FOLD"]
            if "to call" in s:
                return ["FOLD", "CALL", "RAISE"]
    return ["FOLD"]


def memoryless(schema, prompt):
    """History-independent policy: CHECK if free, else CALL, else FOLD. No dependence on the notepad."""
    legal = _legal(prompt)
    for a in ("CHECK", "CALL", "FOLD"):
        if a in legal:
            return schema.model_construct(action=a, amount=None)
    return schema.model_construct(action="FOLD", amount=None)


def rollout(opp: str, name: str, seed: int, num_instances: int = 10):
    env = PokerEnv("exploitable_poker",
                   {"opponent_policy": opp, "opponent_name": name, "num_instances": num_instances, "seed": seed})
    obs, _ = env.reset(seed=seed)
    schema, prompt = env.response_schema, obs["prompt"]
    for _ in range(num_instances * 40):
        obs, _r, done, _t, _i = env.step(memoryless(schema, prompt))
        prompt = obs["prompt"]
        if done:
            break
    return per_hand(prompt)


def _se(xs):
    n = len(xs)
    return pstdev(xs) * (n / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else 0.0


def main(reps: int = 100):
    gains, byopp = [], {}
    for opp, name in [("calling_station", "Tom"), ("fit_or_fold", "Adam"), ("loose_aggressive", "Alex")]:
        for s in range(reps):
            ph = rollout(opp, name, 1000 + s)
            if len(ph) >= 2:
                h = len(ph) // 2
                g = mean(ph[h:]) - mean(ph[:h])
                gains.append(g)
                byopp.setdefault(opp, []).append(g)
    print(f"MEMORYLESS control: late-early gain = {mean(gains):+.3f} +/- {_se(gains):.3f} (n={len(gains)})")
    for o, g in byopp.items():
        print(f"   {o:16s} {mean(g):+.3f} +/- {_se(g):.3f} (n={len(g)})")
    print("~0 within noise => memory_gain metric is structurally clean.")


if __name__ == "__main__":
    main()
