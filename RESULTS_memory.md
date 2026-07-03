# RL can train *memory recall* into a small LLM — without teaching it knowledge or task skill

**Headline result:** GRPO reproducibly improves Qwen3-1.7B's *memory recall* — the fraction of its own
remembered state it carries forward each step — from ~0.89 to ~0.94 across two independent runs, on task
content that **never repeats** (so nothing can be memorized into the weights), against a scrambled-memory
control that stayed **flat** (so non-memory skill gains are excluded). The effect is small at the score
level and does **not** reach the ceiling that one paragraph of explicit instruction installs for free:
prompting remains the stronger, cheaper lever at this model size; RL moves the behavior in the same
direction, slowly.

**Model:** Qwen3-1.7B (Fireworks RFT, GRPO, LoRA). **Task:** CLBench-derived blind-spectrum monitoring —
12 scans of one hidden radio band (~11–14 transmitters × 8 MHz in 180 MHz); each scan shows only ~3
currently-active transmitters (noisy: p_miss 0.15, p_fa 0.2); the agent must report ALL occupied regions,
so dormant transmitters must be *remembered*. **Score:** per-scan occupied-IoU (memoryless ≈ 0.16).
**Memory channel:** each scan echoes the model's *own previous report* back ("YOUR RUNNING LIST"); context
is windowed to the current scan, so this self-authored register is the only cross-scan information.
**Reward:** 3 × mean occ. `memory_gain` (late−early occ) is measured, never rewarded (rewarding a delta
invites sandbagging).

## Methodological core

Mean-performance RL can "succeed" at a memory task by other means: memorizing episode content in the
weights (knowledge) or improving single-step mechanics (skill). Three devices close those doors:

1. **Never-repeating content ("epoch salt").** Bands are seeded per dataset-row (the 12 GRPO candidates of
   a row share one band, so advantages compare policy, not band luck) and re-salted **every epoch** (each
   RFT epoch runs in its own process; an import-time salt rotates the whole band set — verified in-cloud,
   one distinct salt per epoch). No band is ever seen twice; every epoch is automatically a fresh-band
   evaluation. (This closed a real seam: all earlier runs had unknowingly reused the same 48 bands.)
2. **Scrambled-memory control arm.** An identical training run in which the echoed list's *content* is
   replaced per scan by random in-band frequencies (same count, format, instruction). Training effects
   that survive scrambling are non-memory skill; effects that require real content are memory.
3. **Behavioral instrumentation.** Beyond the score, we measure the behavior itself from traces:
   **carry-rate** = fraction of the echoed list preserved in the next report (recall), report size, and
   the within-group correlation between carry and reward. Instrument noise-ordering (empirical):
   carry-rate < occ < memory_gain.

## Experiments (all: 48 rows × 12 candidates × 12 scans, 12 epochs, lr 1e-4, temp 1.2, 4096 tok, brief thinking)

| run | arm | prompt | echo | outcome |
|---|---|---|---|---|
| `tnfxdqkv` | probe | explicit | real | lr≈0 base: occ 0.452/0.474 on two fresh band sets → band-noise ±0.02 |
| `fva3tx6z` | **A** | explicit | real | occ **0.472 flat** — *saturated* (see below) |
| `geote9qj` | **B** | explicit | **scrambled** | occ **0.356 flat**, gain flat — zero training effect |
| `dtbn6lhm` | **C** | **weak** | real | occ 0.403→0.439 (slope +0.0037/ep, R²=0.65); **carry 0.888→0.938** |
| `c4jk2e4z` | **C′** | weak | real | occ 0.401→0.425 (slope +0.0019/ep); **carry 0.900→0.935** |
| (local) | oracle | — | — | accumulate-all-detections ceiling: occ **0.447 ± 0.042** |

The **weak** prompt describes the task and the running list but does not teach accumulation; the
**explicit** prompt adds one paragraph ("dormant transmitters still occupy the band — report every
transmitter you have ever seen").

## Findings

### 1. RL trains memory recall (C + C′, the headline)
From the weak-prompt base, GRPO improved recall in both independent runs — **carry 0.888→0.938 and
0.900→0.935** (converging endpoints ≈0.94; the worst-decile candidates improved most), report size 12.0→14.5
/ 12.9→13.5, occ rising with same-sign slopes (+0.0037, +0.0019/ep; both runs end at their maximum). On
never-repeating bands, with the scrambled arm flat, the improvement can only be *policy-level memory use*.
Honest weights on the claim: the score-level effect is small (+0.012–0.022 occ by halves, vs band-noise
±0.02 — it clears the floor via slope consistency and the behavioral measure, not by magnitude); the
`memory_gain` rise seen in C did not replicate in C′ (it is the noisiest instrument); and 12 epochs of RL
did not reach the instruction ceiling (carry 0.94 vs 0.967; occ 0.42–0.44 vs 0.472).

### 2. The pre-registered causal chain, measured at every link
Within a GRPO candidate group (same band, same scans): candidates differ behaviorally (sd(carry) 0.073 in
C), reward ranks that behavior almost perfectly (**corr(carry, reward) = +0.75/+0.80** — the reward shape
*works*), and the policy moves along the paid gradient (carry rises). The one place the chain has slack is
band/detector noise diluting score expression — visible as C vs C′ magnitude differences.

### 3. Instruction dominates RL at this scale (A + the saturation diagnosis)
The explicit prompt alone installs occ 0.452–0.474 — **above every RL-trained level ever reached in this
project**, and above the naive accumulate-everything oracle (0.447): the instructed model already *curates*
(filters false alarms) rather than just accumulating. Its carry is 0.967 with reports at the 16-slot cap —
the behavior is ~saturated, and Arm A's 12 flat epochs reflect **no headroom**, not "RL can't learn"
(the tiny residual headroom, RL took: carry 0.967→0.972). Practical ordering at 1.7B:
**instruction ≫ RL ≫ nothing**, and RL's role below the instruction ceiling is to *converge toward* it.

### 4. RL preserves cheap memory, erodes expensive memory (the channel-dependence result)
With the env maintaining the memory channel (echo), 12 epochs of RL held the behavior rock-steady (A) or
improved it (C/C′). With memory as the agent's own job (earlier notepad-tools chapter: `notepad_read`/
`notepad_write`), the same optimizer **eroded** the prompt-installed behavior over 20 epochs (gain +0.055 →
−0.008, occ → memoryless floor) — the policy drifts to the simpler current-scan strategy when memory upkeep
costs actions. And from a *vague* prompt the 1.7B never explores notepad use at all (exploration failure:
zero accumulating rollouts for GRPO to reinforce, across two rollout interfaces and thinking on/off).

## What this adds up to

On a small model, with content memorization made impossible and skill drift controlled at zero:
- **Memory *use* is trainable by RL** — measured behaviorally, replicated — but slowly, and only within the
  envelope that instruction defines. If you can write the instruction, write the instruction.
- **Memory *maintenance* is not trainable at this scale** — RL actively destroys it when it costs actions.
- **Reward design was not the bottleneck** (corr +0.8 with the target behavior); **content isolation was** —
  without fresh-band salting, every apparent training gain is confounded by content repetition (our own
  earlier "positive" runs included).

## Limitations
- One model (1.7B), one task family, one optimizer (GRPO via Fireworks RFT); lr/temperature/group-size not
  swept (a sweep is justified now only for pushing magnitude, since signal and headroom are verified).
- Arm A/B/C band sets are drawn independently per epoch (the harness exposes no epoch id for paired salts);
  the ±0.02 probe noise floor bounds the resulting comparison noise.
- occ ceilings are noise-limited (~0.45–0.50 with curation), not 1.0; all effect sizes must be read against
  that compressed range.
- The full-history ("ICL-on") variant of the old positive result was never re-tested under fresh-band rigor:
  confounded, unrefuted.
- `memory_gain` replicated in only one of two C-runs; the claim rests on carry-rate + occ slope, not on gain.

## Reproduce
`make_spectrum_dataset.py --scaffold|--scaffold-weak --think --prefix <ns>` → `firectl create dataset` →
upload `test_spectrum_scaffold.py` / `test_spectrum_scramble.py` (they set `SPECTRUM_SCAFFOLD` /
`SPECTRUM_SCRAMBLE` / `SPECTRUM_EPOCH_SALT` before import) → `firectl create reinforcement-fine-tuning-job
… --epochs 12 --learning-rate 1e-4 --max-output-tokens 4096 --response-candidates-count 12`. Inspect any
epoch's rollouts: `firectl download dataset rft-evalv3-<job>-epoch-<n> --output-dir traces && python
view_trace.py traces/dataset/*/eval_results_dataset.jsonl`.
