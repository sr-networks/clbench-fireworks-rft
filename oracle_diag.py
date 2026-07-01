"""Experiment 1 — ORACLE-MEMORY HEADROOM (base model only, no adapter).

Question: can the base 1.7B convert a CORRECT opponent read into more chips? We replay the SAME
seeded hands (common random numbers) under two notepad conditions and compare chip outcomes:

  EMPTY  : notepad stays empty (the model gets no read).
  ORACLE : notepad is pre-filled with the correct counter-strategy for this opponent (and frozen —
           the model cannot overwrite it).

If ORACLE >> EMPTY -> memory has real headroom; training to USE memory is worthwhile.
If ORACLE ~ EMPTY  -> the model can't exploit even when handed the answer -> poker is skill-bound
                      and memory training is hopeless (switch task).

We test all three opponents because their optimal counters are OPPOSITE (value-bet vs bluff vs
float), which is the basis for the weight-proof randomized-opponent training task (Experiment 2).

    python oracle_diag.py --seeds 8 --hands 5 --out /tmp/oracle.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_eval import BASE, POKER_TOOL, SYS, parse_action  # reuse harness  # noqa: E402
from poker_adapter import PokerEnv  # noqa: E402

ORACLE = {
    "calling_station": (
        "OPPONENT READ: CALLING STATION. It calls almost any bet, never folds, never raises. "
        "COUNTER: bet/raise for value with any made hand or strong draw; NEVER bluff (it won't "
        "fold); fold your trash cheaply; do not pay off when it suddenly raises."
    ),
    "fit_or_fold": (
        "OPPONENT READ: FIT-OR-FOLD. It continues only with strong hands and folds everything else. "
        "COUNTER: bet/raise relentlessly to make it fold; bluff often, especially on scary boards; "
        "but if it ever calls or raises twice, it is strong — give up."
    ),
    "loose_aggressive": (
        "OPPONENT READ: LOOSE-AGGRESSIVE. It raises wide preflop and c-bets the flop a lot, but "
        "GIVES UP on the turn without a strong hand. COUNTER: call flop c-bets with any pair or "
        "draw, then BET the turn when it checks to you to steal; fold to turn/river aggression "
        "unless you have top pair or better."
    ),
}


@torch.no_grad()
def gen(model, tok, messages) -> str:
    ids = tok.apply_chat_template(messages, tools=POKER_TOOL, add_generation_prompt=True,
                                  return_tensors="pt", enable_thinking=False).to(model.device)
    out = model.generate(ids, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def play(model, tok, opponent, seed, hands, oracle_text):
    """Play a seeded multi-hand episode with a FROZEN notepad (oracle_text or '')."""
    env = PokerEnv("exploitable_poker", {"num_instances": hands, "opponent_policy": opponent, "seed": seed})
    obs, _ = env.reset(seed=seed)
    env.notepad = oracle_text or ""          # frozen read; model writes are ignored below
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": obs["prompt"]}]
    per_hand = []
    for _ in range(hands * 12):
        text = gen(model, tok, messages)
        act = parse_action(text)
        messages.append({"role": "assistant", "content": text})
        amt = act["amount"] if isinstance(act["amount"], int) and act["amount"] >= 0 else None
        try:
            action_obj = env.response_schema(thinking=act["thinking"][:120], action=act["action"], amount=amt)
        except Exception:
            action_obj = env.response_schema(thinking="", action="FOLD", amount=None)
        env.pending_notepad = None           # FREEZE notepad: ignore the model's own writes
        obs, reward, done, _, info = env.step(action_obj)
        env.notepad = oracle_text or ""       # re-assert in case step touched it
        if info["instance_complete"]:
            per_hand.append(float(info["hand_reward"]))
        messages.append({"role": "user", "content": obs["prompt"]})
        # keep context bounded: system + last new-hand boundary onward
        if "=== NEW HAND" in obs["prompt"]:
            messages = [{"role": "system", "content": SYS}, {"role": "user", "content": obs["prompt"]}]
        if done or len(per_hand) >= hands:
            break
    return per_hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--hands", type=int, default=5)
    ap.add_argument("--opponents", default="calling_station,fit_or_fold,loose_aggressive")
    ap.add_argument("--out", default="/tmp/oracle.json")
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to(dev).eval()

    out = {}
    for opp in a.opponents.split(","):
        rows = {"empty": [], "oracle": []}
        for s in range(a.seeds):
            for cond, text in (("empty", ""), ("oracle", ORACLE[opp])):
                ph = play(model, tok, opp, seed=100 + s, hands=a.hands, oracle_text=text)
                rows[cond].extend(ph)
            em = sum(rows["empty"]) / max(len(rows["empty"]), 1)
            orc = sum(rows["oracle"]) / max(len(rows["oracle"]), 1)
            print(f"[{opp}] seed {s+1}/{a.seeds}  empty={em:+.2f}  oracle={orc:+.2f}  gain={orc-em:+.2f}",
                  flush=True)
        out[opp] = rows
    # summary
    print("\n=== ORACLE HEADROOM SUMMARY (mean chips/hand) ===")
    for opp, rows in out.items():
        em = sum(rows["empty"]) / max(len(rows["empty"]), 1)
        orc = sum(rows["oracle"]) / max(len(rows["oracle"]), 1)
        print(f"  {opp:16s}  empty={em:+.3f}  oracle={orc:+.3f}  GAIN={orc-em:+.3f}  (n={len(rows['empty'])})")
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
