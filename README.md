# Blind-Spectrum Memory Training on Fireworks RFT

Training *memory use* into a small open model (**Qwen3-1.7B**) with **Fireworks Reinforcement Fine-Tuning**
(GRPO, via [`eval-protocol`](https://evalprotocol.io)), on a task derived from CLBench's
`blind_spectrum_monitoring`.

**The question:** can RL make a model *better at using memory* — recalling information from early in a
multi-step task to do better later — rather than baking task skill into the weights? We measure it as
**`memory_gain` = late-half − early-half performance**, and require it to rise *under training* while the
scenario is **randomized per rollout** (so there is nothing to memorize in the weights).

> **Status: COMPLETE — see [`RESULTS_memory.md`](RESULTS_memory.md) for the final, controlled result
> (2026-07-03).** Headline: **GRPO reproducibly improves the 1.7B's memory *recall*** (carry-rate
> 0.89 → 0.94 in two independent runs) on **never-repeating content** (knowledge-baking impossible by
> construction) with a **scrambled-memory control flat** (skill drift excluded) — while one paragraph of
> explicit instruction still installs a *stronger* policy than 12 epochs of RL reach, and agent-managed
> notepad memory *erodes* under prolonged RL. Sections below describe the task and infra; historical
> results (pre fresh-band rigor) are marked as such.

---

## The task

Each rollout is a series of **12 scans** of ONE fixed but unknown band of ~11–14 transmitters. Each scan
shows only the ~3 currently-active transmitters' noisy peaks; the rest are **dormant** (invisible that
scan). The agent reports the persistent occupied regions. A memoryless agent (reports only the current
peaks) covers ~3/12 of the band; an agent that **accumulates across scans** covers more — so coverage
rising late-vs-early *is* memory.

- **Reward = per-scan occupied-spectrum IoU** (`spectrum_reward.py`), scored in the env vs the hidden
  ground truth. Memoryless ≈ 0.16, full-memory → ~1.0 — a 4× signal that *requires* recalling dormant
  transmitters. (The bench's native **available-IoU is a dead end**: memoryless already ≈ 0.47, so it
  barely rewards memory.)
- **Proof metric = `memory_gain` = late-half occ − early-half occ**, tracked but **never rewarded**
  (rewarding it invites sandbagging — tanking early scans to inflate the delta).

---

## Final results — **[`RESULTS_memory.md`](RESULTS_memory.md)** (the definitive account)

The controlled experiment (fresh bands every epoch via **epoch-salting**, so no band is ever seen twice;
**scrambled-memory control arm**; behavioral **carry-rate** instrumentation):

| arm | prompt | memory echo | outcome |
|---|---|---|---|
| A `fva3tx6z` | explicit | real | occ **0.472 flat** — behavior *saturated* by instruction (carry 0.967, above the naive-oracle 0.447) |
| B `geote9qj` | explicit | **scrambled** | **flat** (occ 0.356, 12 epochs) — zero non-memory skill drift |
| C `dtbn6lhm` | weak | real | **rises**: occ 0.403→0.439 (R²=0.65), **carry 0.888→0.938** |
| C′ `c4jk2e4z` | weak | real | **replicates the mechanism**: occ slope same sign, **carry 0.900→0.935** |

**Findings:** (1) RL trains memory *recall* — replicated behaviorally, knowledge excluded by construction,
skill controlled at zero; (2) the reward correctly ranks memory use within GRPO groups (corr ≈ +0.8), so
reward shape was never the bottleneck; (3) **instruction ≫ RL** at this scale — the explicit prompt installs
a stronger policy (0.472) than RL reaches from a weak base in 12 epochs (0.42–0.44); (4) RL **preserves**
env-maintained memory but **erodes** agent-managed (notepad-tool) memory over long training — the
channel-cost dependence result.

*Historical note:* earlier positive readings (`btalo63n`, `dmzj2mz8`, behavioral tables in
`RUNS_spectrum.md`) predate the fresh-band rigor — they reused the same 48 bands across all epochs and are
confounded by content repetition; keep them as history, cite `RESULTS_memory.md` as the result.

---

## The GRPO fix that made *anything* train (the hard-won one)

GRPO normalizes advantages *within a prompt's candidate group* — those candidates must face the **same**
task. The env originally drew a fresh random band per rollout, so a row's 12 candidates each got a
*different* band → advantage was band-luck, not policy → **zero gradient, frozen policy at any learning
rate** (six flat runs). Fix: seed the band **deterministically from the per-row `session_id`**, so a row's
candidates share one band while different rows give different bands. This single change turned flat runs
into training. (See `_band_seed` in `spectrum_mcp.py` / `spectrum_turn_processor.py`.)

---

## Why the setup diverges from CLBench (deliberate)

Four choices exist specifically to *isolate and reward memory*:
1. **Reward = occupied-IoU**, not the bench's available-IoU (memory-insensitive).
2. **ICL-off windowing** (`spectrum_context_window.py`) — the model sees only `[system] + [current scan]`,
   so its only cross-scan memory is the notepad/scaffold, not re-read history.
3. **Forced dormancy** (`n_active=3` of ~12) — most transmitters dormant each scan, so memory is *required*.
4. **`memory_gain`** (late−early) as the proof metric.

---

## Two rollout interfaces (both cloud-runnable, no hosting)

- **McpGym** (`test_spectrum_rft.py` + `spectrum_mcp.py`) — the proven path. Scans arrive as **tool
  results** (agent calls `submit_report`, gets the next scan back).
- **Custom `SpectrumTurnRolloutProcessor`** (`test_spectrum_turn.py` + `spectrum_turn_processor.py`) —
  **bench-shaped**. Scans arrive as **`user` messages**, the agent replies with **`assistant`** tool calls,
  `submit_report` acks. Confirmed running in-cloud: the cloud runs whatever `rollout_processor` the uploaded
  `@evaluation_test` names, via pytest — no MCP server, no hosted endpoint.

The message-*role* of the scan (tool vs user) is the only difference between the two; it doesn't change what
the model learns. The custom processor exists to match CLBench's `user`/`assistant` turn structure.

---

## Repo layout

**Current (spectrum) working set:**

| file | role |
|---|---|
| `spectrum_adapter.py` | env: per-row deterministic band, scan advance, occ-IoU scoring, notepad state, windowing marker |
| `spectrum_mcp.py` | McpGym server + tools: `notepad_read`, `notepad_write`, `submit_report` |
| `spectrum_server.py` | MCP server launcher (subprocess for the McpGym rollout) |
| `spectrum_reward.py` | occ-IoU reward + `memory_gain` / `mean_occ` / … metrics (parses per-scan `SCAN_OCC`) |
| `spectrum_context_window.py` | ICL-off windowing (monkeypatches the policy to window model input to the current scan) |
| `spectrum_turn_processor.py` | custom `RolloutProcessor` — bench-shaped `user`/`assistant` rollout, notepad tools inline |
| `test_spectrum_rft.py` | evaluator entry (McpGym path) |
| `test_spectrum_turn.py` | evaluator entry (custom-processor path) |
| `make_spectrum_dataset.py` | dataset generator (system prompt + N rows; the band comes from the env per-row seed) |
| `eval_bench90.sh` · `EVAL_bench90.md` | run a trained model on the **official** 90-instance bench schedule |
| `RUNS_spectrum.md` | detailed run-by-run results log |
| `spectrum_templates/` | vendored CLBench Jinja templates (scan rendering) |
| `spectrum_np48.jsonl`, `spectrum24*.jsonl`, `spectrum48.jsonl` | datasets |

**Earlier pivots (kept for reference, not the current path):** `poker_*.py` / `poker_*.jsonl` (exploitable-
poker memory env — a 1.7B never learned the notepad, pivoted away) and `memory_*.py` (a synthetic
learn-and-recall probe). Plus utilities: `plot_eval.py`, `compare_epochs.py`, `oracle_diag.py`, etc.

---

## How to run

```bash
# 1. generate a dataset (N rows; the band is drawn per-row in the env, so rows are interchangeable)
python make_spectrum_dataset.py --n 48 --out spectrum_np48.jsonl

# 2. register the dataset + upload the evaluator
firectl create dataset clbench-spectrum-np spectrum_np48.jsonl
python -m eval_protocol upload --entry test_spectrum_rft.py::test_spectrum_rft --force --yes
#    (or test_spectrum_turn.py::test_spectrum_turn for the bench-shaped user/assistant interface)

# 3. launch the RFT job (Qwen3-1.7B, GRPO)
firectl create reinforcement-fine-tuning-job \
  --base-model accounts/fireworks/models/qwen3-1p7b \
  --dataset clbench-spectrum-np \
  --evaluator accounts/<acct>/evaluators/test-spectrum-rft-test-spectrum-rft \
  --output-model clbench-spectrum-np \
  --epochs 10 --learning-rate 5e-5 --temperature 1.2 \
  --max-output-tokens 1024 --response-candidates-count 12 \
  --max-concurrent-rollouts 96

# 4. watch memory_gain / mean_occ / scans_completed per epoch (dashboard, or poll the job's outputMetrics)

# 5. (optional) evaluate a trained model on the OFFICIAL 90-instance bench schedule
./eval_bench90.sh "<litellm-model-id>" icl_notepad     # see EVAL_bench90.md
```

Local `pytest test_spectrum_rft.py` needs a callable model; Qwen3 models are **not serverless** on the free
tier, so end-to-end validation happens in the cloud RFT run (the mechanics are unit-tested locally).

---

## Differences from CLBench `blind_spectrum_monitoring`

Aligned: `user`/`assistant` turn structure (custom processor), scan generation + Jinja rendering (bench's own
task + templates), detector noise (`p_miss=0.15`, `p_false_alarm=0.2`).

| Aspect | CLBench `default` | Ours | Type |
|---|---|---|---|
| Reward metric | available-IoU | **occupied-IoU** (+ `memory_gain`) | deliberate — available-IoU is memory-insensitive |
| Context regime | full history (`icl`) | **ICL-off, windowed to current scan** | deliberate — notepad is the *only* memory |
| Dormancy | `n_active` 2/3/4 by stage (of 13) | fixed **`n_active=3`** (of 11–14) | deliberate — force memory |
| Notepad | `notepad_update` **field** (`icl_notepad`) | `notepad_read`/`notepad_write` **tools** | structural |
| Scenario/data | 3 curated variants, **frozen corpus** | **random band per row**, seeded from `row_id` | structural — GRPO needs a comparable per-row band group |
| Episode | 90 scans, 3 stages (Wide→Mixed→Full), 5 permuted runs | 12 scans, 1 stationary band | structural |
| Band | 168 MHz, 13 ch (W=15 wide + narrow) | 180 MHz, 11–14 ch, fixed 8 MHz width | structural |
| Action | full `ScanReport` (center + variable bw) | `submit_report(center_freqs)`, fixed 8 MHz, ≤16 regions | structural — simplified for a weak model |
| Thinking | agent's choice | **OFF (`/no_think`)** | Qwen3-1.7B workaround (thinking blows the token budget) |
| Purpose | evaluation (held-out) | **RFT training** (GRPO, many epochs) | structural |

---

## Limitations & honest caveats

- **1.7B ceiling.** Agent-controlled notepad memory doesn't train at this size; the working results all rely
  on the environment helping maintain the memory (scaffold) or on full-context replay. A 4B+ base is the
  likely unlock (capacity-blocked when tried).
- **Not the official bench number.** We use random per-row bands + occ-IoU + ICL-off, not the frozen
  90-instance corpus + available-IoU. `eval_bench90.sh` bridges to the official schedule — but note
  available-IoU *under-shows* memory (see `EVAL_bench90.md`).
- **Noisy training curves** (small magnitude, non-monotonic). Read the **trend** — early-vs-late halves,
  slope, cross-run reproducibility, matched control — not peak epochs.
- **Capacity.** RFT jobs can stall at model-serving deployment creation during Fireworks capacity crunches,
  independent of the code (a known-good, unrelated job stalls identically). Diagnose with `percent` /
  `acceleratorSeconds` / a deployment check, and verify with an independent control job.

---

## Infra learnings (gotchas worth knowing)

- **Fireworks RFT runs `eval-protocol`** (confirmed: `eval-protocol-0.3.23` in the cloud streamlogs). It runs
  whatever `rollout_processor` the uploaded `@evaluation_test` names — including a **custom in-process one**
  (no MCP server, no hosting). The earlier belief that "the cloud won't drive a custom processor" was a
  capacity false-negative.
- **The in-training model lives in `config.completion_params`**, not `row.input_metadata` — a custom
  processor must read it there (this is what McpGym does; reading the wrong place → 404 retry-hang).
- **Fireworks rejects `tool_choice`** → set `litellm.drop_params = True` (or don't pass it).
- **`totalInputRequests` is unreliable** (stays 0 even for running jobs) — use `percent` /
  `acceleratorSeconds` / the streamlog instead.
- **Qwen3-1.7B thinking exhausts the token budget** before the tool call (`finish_reason=length`) → `/no_think`.
- **`RemoteRolloutProcessor`** also yields `user`/`assistant`, but needs a *hosted* HTTP server; the custom
  in-process processor is the hosting-free way to the same interface.
