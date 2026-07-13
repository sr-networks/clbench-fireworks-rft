"""dbx engine: episode runner + scripted policies for the database_exploration memory arm.

Wraps the OFFICIAL CLBench task class (src.tasks.database_exploration) with the same memory
scaffold as the spectrum arm:
  - context is WIPED between questions; the model's NOTEPAD is the only persistence
  - within a question: mini-conversation (question -> query results accumulate -> answer)
  - one tool: act{action: QUERY|ANSWER, content, notepad_update?}
  - after each answer the engine emits a QSTAT line (parsed later by the reward):
      QSTAT pos=<k> grp=<g> correct=<0|1> nq=<queries used> ev=<result|notepad|prophecy|none>

Evidence (ev) — the anti-answer-baking gate. For an answer to count as an informed
one-shot, its VALUE must be traceable to this episode:
  result   – value appears in the last query result of this question (honest compute)
  notepad  – value was in the notepad at question start AND appeared in some EARLIER
             query result this episode (legit cached discovery)
  prophecy – value was in the notepad but never appeared in any earlier result: it was
             written from the weights, not from observation (baked answers smuggled
             through the notepad). The reward zeroes + penalizes these.
  none     – value appears nowhere the model looked (guess or direct weight recall)

Episode spec: (variant, seed) -> products_v{variant}.db + a seeded permutation of a
subset of questions_pool_v{variant}.json (budget queries/question configurable).
Variants permute category->suffix mapping AND quirk bundles, so schema facts are not
weight-bakeable across rows (see make_dbx_variants.py).

Scripted policies (red-team + floors, no model needed):
  - explorer: memoryless competent agent — rediscovers tables/quirks EVERY question,
    then computes with template-aware SQL. Calibrates honest memoryless nq + budget.
  - explorer_notepad: same agent but with persistent discovered facts (oracle notepad) —
    proves nq<=1 is achievable WITH memory (currency ceiling).
  - guesser: answers a constant instantly (nq=0) — guessability floor for the currency.
"""
import json
import os
import random
import re
import tempfile
import urllib.request
from pathlib import Path

from src.registry import get_task_class
from src.interface import Response
from src.tasks.database_exploration.task import DatabaseAction

HERE = Path(__file__).parent

_ACTIVE_POOL = None   # pool of the variant of the CURRENT episode (for scripted policies)

SYSTEM_PROMPT = """You are a database analyst answering a SERIES of questions about ONE \
unknown SQLite database, shown ONE AT A TIME. You interact through the `act` tool.

Between questions your conversation is WIPED. The ONLY thing that persists is YOUR \
NOTEPAD, shown to you at the start of every question. The database itself never changes.

At every question, do exactly this:
1. Read your notepad as memory.
2. If the notepad already has the facts you need (which tables belong to which product \
category; how prices, timestamps and 'verified' are encoded), go STRAIGHT to one final \
computing query — do not re-explore.
3. If facts are missing, discover them with a few TARGETED queries (e.g. \
SELECT main_cat, prc FROM items_g1 LIMIT 2) — the query budget is small, so be frugal. \
Do not re-derive anything already in the notepad.
4. ANSWER with the exact value from your query result: a bare number or short text, \
nothing else.
5. ALWAYS include notepad_update with the COMPLETE updated fact sheet — old facts you \
leave out of notepad_update are forgotten forever.

Notepad format:
TABLE MAP: <category> = items_gX + fdbk_gX (+ attrs_gX / taxn_gX if they exist)
UNITS per category: prices in <dollars|cents>; timestamps as <epoch-ms|epoch-s|ISO text>; \
verified as <0/1 int|'true'/'false' text>
FACTS: <anything else worth keeping>

Tool actions:
- act(action="QUERY", content="<sql>") — run one exploratory SQL statement \
(SELECT / WITH / PRAGMA / .tables / .schema <table>).
- act(action="ANSWER", content="<value>") — submit the final answer: a raw number or \
short text, nothing else.
- Pass notepad_update="..." with any action to REPLACE your notepad text (it is the \
only thing that survives the wipe)."""

TOOLS = [{"type": "function", "function": {
    "name": "act",
    "description": "Take one action: QUERY to run SQL, or ANSWER to submit the final answer.",
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["QUERY", "ANSWER"]},
        "content": {"type": "string",
                    "description": "SQL statement (QUERY) or the final answer value (ANSWER)."},
        "notepad_update": {"type": "string",
                           "description": "Optional: replace your persistent notepad with this text."},
    }, "required": ["action", "content"]},
}}]

NOTEPAD_MAX = 1500


def load_pool(variant: int = 0):
    return json.loads((HERE / "dbx_data" / f"questions_pool_v{variant}.json").read_text())


def make_episode(seed: int, variant: int = 0, n_questions: int = 15, budget: int = 8):
    """Deterministic episode: seeded, group-balanced permutation of the variant's pool."""
    global _ACTIVE_POOL
    pool = _ACTIVE_POOL = load_pool(variant)
    rng = random.Random(seed)
    by_g = {}
    for q in pool:
        by_g.setdefault(q["group"], []).append(q)
    picked = []
    per_g = n_questions // 3
    for g, qs in sorted(by_g.items()):
        picked += rng.sample(qs, per_g)
    while len(picked) < n_questions:
        rest = [q for q in pool if q not in picked]
        picked.append(rng.choice(rest))
    rng.shuffle(picked)
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(picked, tf)
    tf.close()
    task = get_task_class("database_exploration")(
        db_path=str(HERE / "dbx_data" / f"products_v{variant}.db"), questions_path=tf.name,
        num_questions=n_questions, max_queries_per_question=budget)
    task.build_canonical_run_state()
    os.unlink(tf.name)
    return task, picked


def _scrub(text: str) -> str:
    """Neutralize QSTAT-lookalikes in MODEL-CONTROLLABLE text (query results echo DB
    output — SELECT 'QSTAT ...' would forge reward lines; same for the notepad echo).
    Genuine QSTAT lines are emitted by the engine/processor only, after the wipe."""
    return (text or "").replace("QSTAT", "QSTAT_")


def notepad_block(notepad: str) -> str:
    return ("\n\n=== YOUR NOTEPAD (persists across questions; everything else was wiped) ===\n"
            + (_scrub(notepad) if notepad.strip() else "(empty)"))


def _num_tokens(text: str) -> set:
    return set(re.findall(r"-?\d+(?:\.\d+)?", (text or "").replace(",", "")))


def _val_in(ans: str, tokens: set, raw_text: str = "") -> bool:
    """Does the answered value appear among the numeric tokens (float-tolerant)?
    The answered value = FIRST number in the answer text (the bench's own parsing —
    models write things like '[ANSWER] 1013' or '42 reviews'); pure-text answers
    fall back to a substring check."""
    a = (ans or "").strip().replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", a)
    if not nums:
        return bool(a) and a.lower() in (raw_text or "").lower()
    av = float(nums[0])
    for t in tokens:
        try:
            if abs(float(t) - av) <= max(1e-9, abs(av) * 1e-6):
                return True
        except ValueError:
            pass
    return False


def run_episode(policy, seed: int, variant: int = 0, n_questions: int = 15,
                budget: int = 8, verbose: bool = False):
    """policy(messages, notepad) -> (action, content, notepad_update|None).
    Returns per-question stats + final task metrics."""
    task, picked = make_episode(seed, variant, n_questions, budget)
    notepad = ""
    stats = []
    seen_tokens: set = set()   # every numeric token observed in ANY query result so far
    seen_text: list = []       # raw observations (for non-numeric provenance)
    q = task.build_current_query()
    while True:
        pos = q.metadata["question_num"]
        grp = picked[pos - 1]["group"]
        notepad_start = notepad                 # what the model actually READ
        prior_tokens = set(seen_tokens)         # provenance snapshot BEFORE this question
        prior_text = "\n".join(seen_text)
        last_obs = ""
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q.prompt + notepad_block(notepad)}]
        nq = 0
        while True:
            action, content, np_up = policy(messages, notepad)
            if np_up is not None and np_up.strip():
                notepad = np_up[:NOTEPAD_MAX]
            sr = task.step(Response(action=DatabaseAction(action=action, content=content), metadata={}))
            obs = sr.observation.content
            if action == "QUERY" and not sr.observation.instance_complete:
                nq += 1
                last_obs = obs
                seen_tokens |= _num_tokens(obs)
                seen_text.append(obs)
                messages.append({"role": "assistant", "content": f"[{action}] {content}"})
                messages.append({"role": "user", "content": _scrub(obs)})
                continue
            # instance complete (ANSWER, or budget exceeded during QUERY)
            oc = sr.instance_outcome
            correct = bool(oc.success) if oc else False
            used = task._question_history[-1]["num_queries"]
            in_result = _val_in(content, _num_tokens(last_obs), last_obs)
            in_np = _val_in(content, _num_tokens(notepad_start), notepad_start)
            has_prov = in_np and _val_in(content, prior_tokens, prior_text)
            ev = ("result" if in_result else
                  "notepad" if has_prov else
                  "prophecy" if in_np else "none")
            seen_tokens |= _num_tokens(obs)   # feedback (incl. revealed GT) is observed too
            seen_text.append(obs)
            qstat = f"QSTAT pos={pos} grp={grp} correct={int(correct)} nq={used} ev={ev}"
            stats.append(dict(pos=pos, grp=grp, correct=correct, nq=used, ev=ev, qstat=qstat))
            if verbose:
                print(qstat, "|", picked[pos - 1]["question"][:60])
            break
        if sr.done:
            break
        q = sr.next_query
    result = task.evaluate()
    return stats, result.metrics, notepad


# ---------------- scripted policies ----------------

class Explorer:
    """Competent agent that knows the QUESTION TEMPLATES (it can write the right SQL once it
    knows table names + quirks) but must DISCOVER the schema facts by querying.
    remember=False: rediscovers everything each question (memoryless floor).
    remember=True : discovered facts persist across questions (oracle-notepad ceiling)."""

    def __init__(self, remember: bool = False):
        self.remember = remember
        self.facts: set = set()   # {'layout', 'map', 'quirk_g1', ...} — persists iff remember
        self._qkey = None
        self._plan: list = []
        self._answering = False

    def __call__(self, messages, notepad):
        user0 = next(m["content"] for m in messages if m["role"] == "user")
        qkey = user0.split("===")[0]
        if qkey != self._qkey:                       # new question -> build the query plan
            self._qkey = qkey
            if not self.remember:
                self.facts = set()
            qrow = self._pool_row(user0)
            g = qrow["group"]
            plan = []
            if "layout" not in self.facts:
                plan.append(".tables"); self.facts.add("layout")
            if "map" not in self.facts:
                # probe main_cat table by table until the category is found (deterministic order)
                probes = {"g1": 1, "g2": 2, "g3": 3}[g]
                for i in range(probes):
                    plan.append(f"SELECT main_cat FROM items_{['g1','g2','g3'][i]} LIMIT 1")
                self.facts.add("map")
            needs_quirk = any(w in qrow["question"].lower()
                              for w in ("price", "dollar", "year", "verified"))
            if needs_quirk and f"quirk_{g}" not in self.facts:
                plan.append(f"SELECT prc, ts, vrf FROM items_{g} JOIN fdbk_{g} USING(ref_id) LIMIT 2")
                self.facts.add(f"quirk_{g}")
            plan.append(qrow["sql"])                 # the compute query (counted honestly)
            self._plan = plan
            self._answering = False
        if self._plan:
            return "QUERY", self._plan.pop(0), None
        # plan exhausted -> the last observation holds the computed value
        last = messages[-1]["content"]
        lines = [l for l in last.splitlines() if l.strip()]
        val = lines[-1].split(" | ")[0].strip() if lines else ""
        return "ANSWER", val, None

    def _pool_row(self, user0):
        for q in (_ACTIVE_POOL or load_pool()):
            if q["question"] in user0:
                return q
        return None


def guesser(messages, notepad):
    return "ANSWER", "100", None


# ---------------- model-backed policy ----------------

class ModelPolicy:
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 2048):
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.api_key = os.environ["FIREWORKS_API_KEY"]

    def __call__(self, messages, notepad):
        body = json.dumps({"model": self.model, "messages": messages, "tools": TOOLS,
                           "temperature": self.temperature, "max_tokens": self.max_tokens}).encode()
        req = urllib.request.Request("https://api.fireworks.ai/inference/v1/chat/completions",
                                     data=body, headers={"Authorization": f"Bearer {self.api_key}",
                                                         "Content-Type": "application/json"})
        for _ in range(5):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    msg = json.loads(r.read())["choices"][0]["message"]
                break
            except Exception:
                continue
        else:
            return "ANSWER", "", None
        for tc in (msg.get("tool_calls") or []):
            try:
                args = json.loads(tc["function"]["arguments"])
                return (args.get("action", "ANSWER"), str(args.get("content", "")),
                        args.get("notepad_update"))
            except Exception:
                pass
        # fallback: try to parse a bare-text action
        txt = msg.get("content") or ""
        m = re.search(r"(QUERY|ANSWER)[:\s]+(.*)", txt)
        if m:
            return m.group(1), m.group(2).strip(), None
        return "ANSWER", txt.strip()[:80], None
