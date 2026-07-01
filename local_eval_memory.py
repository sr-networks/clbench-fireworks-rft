"""Local stateful-vs-stateless eval for the learn-and-recall MEMORY task.

This is the goal-relevant curve: "good at learning WITH memory" = (with-memory - without-memory).

Each rollout is N rounds (1 LEARN that reveals random codes, then RECALL rounds asking for a code).
The cross-round trace is CUT — the model's only memory is the notepad. So:
  - STATEFUL  : notepad persists across rounds -> the model CAN write codes in round 0 and read
                them back -> recall reward depends on whether it learned to USE the notepad.
  - STATELESS : notepad wiped every round -> the model never has the codes at recall time -> ~0.

Plotting per-round recall reward for base vs finetuned, stateful vs stateless, reproduces the
continual-learning "stateful system vs stateless baseline" comparison on a clean, weight-proof task.

    python local_eval_memory.py --episodes 40 [--lora <dir>] --out mem_base.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_adapter import MemoryEnv  # noqa: E402

BASE = "Qwen/Qwen3-1.7B"

RESPOND_TOOL = [{
    "type": "function",
    "function": {
        "name": "respond",
        "description": "Reply to the environment and optionally update your notepad memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "your answer for this round"},
                "notepad_update": {"type": "string",
                                   "description": "full updated notepad — your ONLY memory across rounds"},
            },
            "required": ["answer"],
        },
    },
}]

SYS = (
    "You go through a SERIES of rounds. Early rounds may reveal information; later rounds test "
    "whether you can recall it. You CANNOT see earlier rounds — your ONLY memory is the notepad. "
    "Each turn call `respond`. Use notepad_update to carry anything you may need later."
)

_TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse(text: str):
    m = _TC.search(text)
    blob = m.group(1) if m else (text[text.find("{"): text.rfind("}") + 1] if "{" in text else "")
    try:
        d = json.loads(blob)
        args = d.get("arguments", d)
        return str(args.get("answer", "")), str(args.get("notepad_update", "") or "")
    except Exception:
        return "", ""


@torch.no_grad()
def gen(model, tok, messages) -> str:
    ids = tok.apply_chat_template(messages, tools=RESPOND_TOOL, add_generation_prompt=True,
                                  return_tensors="pt", enable_thinking=False).to(model.device)
    out = model.generate(ids, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def run_episode(model, tok, seed: int, stateful: bool, rounds: int):
    """Returns per-recall-round reward list for one episode."""
    env = MemoryEnv(rounds=rounds)
    obs, _ = env.reset(seed=seed)
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": obs["prompt"]}]
    rewards = []
    for _ in range(rounds):
        text = gen(model, tok, messages)
        answer, notepad_update = parse(text)
        messages.append({"role": "assistant", "content": text})
        env.pending_notepad = notepad_update.strip() or None if stateful else None
        obs, reward, done, _, info = env.step(answer)
        if info["is_recall"]:
            rewards.append(float(reward))
        if stateful:
            # context cut: only system + the new prompt (which embeds the notepad) survive
            messages = [{"role": "system", "content": SYS}, {"role": "user", "content": obs["prompt"]}]
        else:
            env.notepad = ""
            messages = [{"role": "system", "content": SYS}, {"role": "user", "content": obs["prompt"]}]
        if done:
            break
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=8)
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
        print(f"running {label} / {mode} ({a.episodes} episodes x {a.rounds} rounds)...", flush=True)
        # mean recall accuracy per episode -> one point per episode
        per_ep = []
        for e in range(a.episodes):
            r = run_episode(model, tok, seed=1000 + e, stateful=stateful, rounds=a.rounds)
            per_ep.append(sum(r) / max(len(r), 1))
            if (e + 1) % 5 == 0:
                print(f"  ep {e+1}/{a.episodes}  running mean={sum(per_ep)/len(per_ep):.3f}", flush=True)
        res[mode] = per_ep
        print(f"  {mode} mean recall acc = {sum(per_ep)/max(len(per_ep),1):.3f}")
    json.dump({"label": label, "episodes": a.episodes, "rounds": a.rounds, **res}, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
