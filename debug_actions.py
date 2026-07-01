"""Fast probe: for each decision in a couple of hands, print the situation + the base model's
greedy action under EMPTY vs ORACLE notepad. Reveals whether the read influences behavior at all."""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_eval import BASE, POKER_TOOL, SYS, parse_action
from poker_adapter import PokerEnv
from oracle_diag import ORACLE


@torch.no_grad()
def gen(model, tok, messages):
    ids = tok.apply_chat_template(messages, tools=POKER_TOOL, add_generation_prompt=True,
                                  return_tensors="pt", enable_thinking=False).to(model.device)
    out = model.generate(ids, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def sit_of(prompt):
    for ln in prompt.splitlines():
        if ln.strip().startswith("Situation:"):
            return ln.strip()[:90]
    return "(no situation line)"


def run(model, tok, opp, seed, hands, oracle_text, tag):
    env = PokerEnv("exploitable_poker", {"num_instances": hands, "opponent_policy": opp, "seed": seed})
    obs, _ = env.reset(seed=seed)
    env.notepad = oracle_text or ""
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": obs["prompt"]}]
    acts = []
    for _ in range(hands * 12):
        text = gen(model, tok, messages)
        act = parse_action(text)
        acts.append((sit_of(obs["prompt"]), act["action"], act["amount"]))
        messages.append({"role": "assistant", "content": text})
        amt = act["amount"] if isinstance(act["amount"], int) and act["amount"] >= 0 else None
        try:
            ao = env.response_schema(thinking="", action=act["action"], amount=amt)
        except Exception:
            ao = env.response_schema(thinking="", action="FOLD", amount=None)
        env.pending_notepad = None
        obs, reward, done, _, info = env.step(ao)
        env.notepad = oracle_text or ""
        messages.append({"role": "user", "content": obs["prompt"]})
        if "=== NEW HAND" in obs["prompt"]:
            messages = [{"role": "system", "content": SYS}, {"role": "user", "content": obs["prompt"]}]
        if done or env.instance_idx >= hands:
            break
    return acts


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to(dev).eval()
    opp = "calling_station"
    e = run(model, tok, opp, seed=100, hands=2, oracle_text="", tag="EMPTY")
    o = run(model, tok, opp, seed=100, hands=2, oracle_text=ORACLE[opp], tag="ORACLE")
    print(f"\n=== {opp}, seed 100 — decision-by-decision ===")
    print(f"{'#':>2}  {'SITUATION':<92} {'EMPTY':>14} {'ORACLE':>14}")
    for i in range(max(len(e), len(o))):
        s = e[i][0] if i < len(e) else (o[i][0] if i < len(o) else "")
        ea = f"{e[i][1]}:{e[i][2]}" if i < len(e) else "-"
        oa = f"{o[i][1]}:{o[i][2]}" if i < len(o) else "-"
        flag = "  <-- differs" if ea != oa else ""
        print(f"{i:>2}  {s:<92} {ea:>14} {oa:>14}{flag}")


if __name__ == "__main__":
    main()
