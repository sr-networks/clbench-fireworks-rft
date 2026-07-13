"""RAW rollout transcript: every message of the same exemplary episode (xoi922eh ep4, median complete,
row 188 = canon-full_grid_active-2), verbatim and in order — system / user / assistant (thinking as-is) /
tool_calls (function name + arguments JSON) / tool result. No parsing, no panels, no diffs."""
import json
import html as H
from glob import glob

SCRATCH = ("/private/tmp/claude-501/-Users-sten-Documents-Coding-fireworks/"
           "bc95af7b-ea9a-424a-82d1-2001ebd855ce/scratchpad")
REPO = "/Users/sten/Documents/Coding/fireworks/clbench-fireworks-rft"
PICK = 188  # same rollout as the annotated page (median complete episode by mean dorm)

f = glob(f"{SCRATCH}/dormc_xoi922eh_ep4/dataset/*/eval_results_dataset.jsonl")[0]
row = [json.loads(l) for l in open(f)][PICK]
meta = row["input_metadata"]
ev = row["evaluation_result"]
rollout_id = row.get("execution_metadata", {}).get("rollout_id", "?")

COLORS = {"system": "#f4f0fa", "user": "#eef6ff", "assistant": "#f0fff0", "tool": "#fff8e6"}

parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>RAW rollout — xoi922eh ep4 ({H.escape(meta['row_id'])})</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 1020px; margin: 24px auto;
        padding: 0 16px; color: #1a1a1a; }}
 .m {{ border: 1px solid #ccc; border-radius: 6px; margin: 10px 0; }}
 .role {{ font-family: ui-monospace, Menlo, monospace; font-size: 0.8em; font-weight: 700;
          padding: 4px 10px; border-bottom: 1px solid #ccc; }}
 pre {{ margin: 0; padding: 8px 10px; white-space: pre-wrap; word-break: break-word;
        font-family: ui-monospace, Menlo, monospace; font-size: 0.78em; line-height: 1.4; }}
 .tc {{ border-top: 1px dashed #999; }}
 .hdr {{ background: #eee; border-radius: 6px; padding: 10px 14px; font-size: 0.9em; }}
 .runid {{ background: #ddd; border-radius: 4px; padding: 1px 6px;
           font-family: ui-monospace, monospace; font-size: 0.85em; }}
</style></head><body>
<h1 style="font-size:1.25em">RAW rollout transcript</h1>
<div class="hdr">
run <span class="runid">xoi922eh</span> (dormc2) · epoch-4 eval · task
<span class="runid">{H.escape(meta['row_id'])}</span> · rollout <span class="runid">{H.escape(rollout_id)}</span>
· temperature 1.2 · same episode as the <a href="ROLLOUT_xoi922eh_ep4_example.html">annotated page</a>
(median complete episode, mean dorm 0.491).<br>
episode result: <span style="font-family:monospace; font-size:0.85em">{H.escape(ev.get('reason',''))}</span>
</div>
"""]

for i, m in enumerate(row["messages"]):
    role = m.get("role", "?")
    bg = COLORS.get(role, "#fff")
    parts.append(f'<div class="m"><div class="role" style="background:{bg}">[{i}] {H.escape(role)}</div>')
    content = m.get("content") or ""
    if content:
        parts.append(f"<pre>{H.escape(content)}</pre>")
    for tc in (m.get("tool_calls") or []):
        fn = tc.get("function", {})
        try:
            args = json.dumps(json.loads(fn.get("arguments", "")), indent=2)
        except Exception:
            args = fn.get("arguments", "")
        parts.append(f'<pre class="tc">tool_call: {H.escape(fn.get("name", "?"))}'
                     f'  (id {H.escape(tc.get("id", ""))})\n{H.escape(args)}</pre>')
    if not content and not m.get("tool_calls"):
        parts.append("<pre>(empty)</pre>")
    parts.append("</div>")

parts.append("""<p><small>Source: Fireworks eval dataset <span class="runid">rft-evalv3-xoi922eh-epoch-4</span>,
row 188. Generator <span class="runid">live_assets/render_rollout_raw.py</span>.</small></p></body></html>""")

out = f"{REPO}/ROLLOUT_xoi922eh_ep4_raw.html"
open(out, "w").write("\n".join(parts))
print("written", out, f"({len(row['messages'])} messages)")
