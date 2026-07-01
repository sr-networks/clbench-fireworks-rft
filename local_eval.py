"""Local CLBench-style eval: play a long poker sequence with a local model (base or +LoRA),
in STATEFUL (memory carries across hands) vs STATELESS (memory reset each hand) mode, and record
per-hand chip reward. Reproduces the continual-learning-bench "stateful vs stateless" comparison.

    python local_eval.py --hands 60 --seed 0 [--lora <adapter_dir>] --out base.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poker_adapter import PokerEnv, _render_notepad  # noqa: E402

BASE = "Qwen/Qwen3-1.7B"

POKER_TOOL = [{
    "type": "function",
    "function": {
        "name": "poker_act",
        "description": "Take a poker action and optionally update your notepad memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["FOLD", "CALL", "CHECK", "RAISE"]},
                "amount": {"type": "integer", "description": "total chips to raise to (RAISE only)"},
                "thinking": {"type": "string"},
                "notepad_update": {"type": "string", "description": "full updated notepad (memory across hands)"},
            },
            "required": ["action"],
        },
    },
}]

SYS = (
    "You play a SERIES of heads-up Texas Hold'em hands vs the SAME fixed opponent. Its strategy "
    "never changes, so learn it early and exploit it later. Each turn call poker_act with a legal "
    "action. Use notepad_update to record what you learn about the opponent — it is your memory "
    "across hands."
)

_TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_action(text: str):
    m = _TC.search(text)
    blob = m.group(1) if m else (text[text.find("{"): text.rfind("}") + 1] if "{" in text else "")
    try:
        d = json.loads(blob)
        args = d.get("arguments", d)
        return {
            "action": str(args.get("action", "FOLD")).upper(),
            "amount": args.get("amount"),
            "thinking": args.get("thinking", ""),
            "notepad_update": args.get("notepad_update", ""),
        }
    except Exception:
        return {"action": "FOLD", "amount": None, "thinking": "", "notepad_update": ""}


@torch.no_grad()
def gen(model, tok, messages) -> str:
    ids = tok.apply_chat_template(messages, tools=POKER_TOOL, add_generation_prompt=True,
                                  return_tensors="pt", enable_thinking=False).to(model.device)
    out = model.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def run_sequence(model, tok, n_hands: int, seed: int, stateful: bool):
    env = PokerEnv("exploitable_poker", {"num_instances": n_hands, "opponent_policy": "calling_station", "seed": seed})
    obs, _ = env.reset(seed=seed)
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": obs["prompt"]}]
    per_hand = []
    for _ in range(n_hands * 12):
        text = gen(model, tok, messages)
        act = parse_action(text)
        messages.append({"role": "assistant", "content": text})
        schema = env.response_schema
        amt = act["amount"] if isinstance(act["amount"], int) and act["amount"] >= 0 else None
        try:
            action_obj = schema(thinking=act["thinking"][:200], action=act["action"], amount=amt)
        except Exception:
            action_obj = schema(thinking="", action="FOLD", amount=None)
        env.pending_notepad = (act["notepad_update"] or "").strip() or None if stateful else None
        obs, reward, done, _, info = env.step(action_obj)
        if info["instance_complete"]:
            per_hand.append(float(info["hand_reward"]))
        if stateful:
            messages.append({"role": "user", "content": obs["prompt"]})
        else:
            # STATELESS: drop all history + notepad; the model sees only the new hand
            env.notepad = ""
            messages = [{"role": "system", "content": SYS}, {"role": "user", "content": obs["prompt"]}]
        if done:
            break
    return per_hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to(dev).eval()
    label = "base"
    if a.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.lora).to(dev).eval()
        label = "finetuned"

    res = {}
    for stateful in (True, False):
        mode = "stateful" if stateful else "stateless"
        print(f"running {label} / {mode} ({a.hands} hands)...", flush=True)
        res[mode] = run_sequence(model, tok, a.hands, a.seed, stateful)
        print(f"  mean reward = {sum(res[mode])/max(len(res[mode]),1):.2f} over {len(res[mode])} hands")
    json.dump({"label": label, "hands": a.hands, "seed": a.seed, **res}, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
