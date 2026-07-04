#!/usr/bin/env python3
"""Build a self-contained HTML trace browser for any RFT epoch — the web-UI Traces tab, but local.

    firectl download dataset rft-evalv3-<job>-epoch-<n> --output-dir tr
    python traces_html.py tr/dataset/*/eval_results_dataset.jsonl -o traces_ep<n>.html
    open traces_ep<n>.html

Sortable summary row per rollout (row_id, score, key metrics from the eval reason), click to expand the
full conversation with roles color-coded and tool calls/results pretty-printed.
"""

import argparse
import glob
import html
import json

E = html.escape


def render_msg(m):
    role = m.get("role", "?")
    content = (m.get("content") or "").strip()
    out = []
    if content:
        cls = {"system": "sys", "user": "usr", "assistant": "ast", "tool": "tool"}.get(role, "")
        out.append(f'<div class="msg {cls}"><b>[{role}]</b> {E(content)}</div>')
    for tc in (m.get("tool_calls") or []):
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        args = fn.get("arguments") or ""
        try:
            args = json.dumps(json.loads(args), indent=1)
        except Exception:
            pass
        out.append(f'<div class="msg call"><b>[{role} → {E(str(fn.get("name")))}]</b> <pre>{E(args)}</pre></div>')
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", default="traces.html")
    ap.add_argument("--max", type=int, default=400, help="max rollouts to include")
    a = ap.parse_args()
    rows = []
    for pat in a.files:
        for f in glob.glob(pat):
            rows += [json.loads(l) for l in open(f) if l.strip()]
    rows = rows[: a.max]
    items = []
    for i, r in enumerate(rows):
        er = r.get("evaluation_result") or {}
        rid = (r.get("input_metadata") or {}).get("row_id", "?")
        score = er.get("score")
        reason = er.get("reason", "")
        body = "".join(render_msg(m) for m in r.get("messages", []))
        items.append(
            f'<details><summary><code>[{i:>3}] {E(str(rid))}</code> · score {score if score is None else round(score,3)}'
            f' · <span class="rsn">{E(reason)}</span></summary><div class="conv">{body}</div></details>'
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>traces ({len(rows)} rollouts)</title><style>
 body {{ font: 14px/1.5 -apple-system, sans-serif; margin: 24px; background:#fafafa; }}
 details {{ background:#fff; border:1px solid #e2e2e2; border-radius:6px; margin:6px 0; padding:6px 12px; }}
 summary {{ cursor:pointer; }} .rsn {{ color:#666; font-size:12.5px; }}
 .conv {{ margin-top:8px; }}
 .msg {{ padding:6px 10px; margin:4px 0; border-radius:5px; white-space:pre-wrap; font-size:13px; }}
 .sys {{ background:#f3eefc; }} .usr {{ background:#e8f2fb; }} .ast {{ background:#eef9ee; }}
 .tool {{ background:#fdf6e3; }} .call {{ background:#e6f4ea; border-left:3px solid #1a7a2e; }}
 pre {{ margin:4px 0 0; font-size:12px; white-space:pre-wrap; }}
</style></head><body><h2>Rollout traces — {len(rows)} rollouts</h2>{"".join(items)}</body></html>"""
    open(a.out, "w").write(page)
    print(f"wrote {a.out} ({len(page)//1024} KB, {len(rows)} rollouts)")


if __name__ == "__main__":
    main()
