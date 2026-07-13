"""Render ONE exemplary notepad rollout as a standalone review HTML page.
Selection rule (stated on the page): among the 30-scan COMPLETE episodes of xoi922eh epoch-4 (the clean
dormc winner), pick the MEDIAN by mean SCAN_DORM — a representative rollout, not the best one.
Per scan we show: the peaks the model SAW, the notepad it READ, its thinking trace (collapsed), the report
it SUBMITTED, the notepad it WROTE (line-diffed vs what it read), and the per-scan scores."""
import json
import re
import html as H
from glob import glob
from statistics import mean, median

SCRATCH = ("/private/tmp/claude-501/-Users-sten-Documents-Coding-fireworks/"
           "bc95af7b-ea9a-424a-82d1-2001ebd855ce/scratchpad")
REPO = "/Users/sten/Documents/Coding/fireworks/clbench-fireworks-rft"
OCC = re.compile(r"SCAN_OCC:\s*([0-9.]+)")
DORM = re.compile(r"SCAN_DORM:\s*([0-9.]+)")
PEAKLINE = re.compile(r"- peak_id: \S+ \| freq: ([0-9.]+) MHz \| power: (-?[0-9.]+) dBm \| width: ([0-9.]+) MHz")

f = glob(f"{SCRATCH}/dormc_xoi922eh_ep4/dataset/*/eval_results_dataset.jsonl")[0]
rows = [json.loads(l) for l in open(f)]


def series(row, pat):
    txt = "\n".join((m.get("content") or "") for m in row["messages"] if isinstance(m, dict))
    return [float(x) for x in pat.findall(txt)]


# ---- pick the median complete episode by mean dorm ----
complete = []
for i, r in enumerate(rows):
    occ = series(r, OCC)
    if len(occ) >= 30:
        d = series(r, DORM)
        complete.append((mean(d) if d else 0.0, i))
complete.sort()
med_dorm, pick = complete[len(complete) // 2]
dorms_all = [c[0] for c in complete]
row = rows[pick]
print(f"complete episodes: {len(complete)}; dorm range {dorms_all[0]:.3f}–{dorms_all[-1]:.3f}, "
      f"median {median(dorms_all):.3f}; picked row {pick} (dorm {med_dorm:.3f}) "
      f"row_id={row['input_metadata']['row_id']}")

ev = row["evaluation_result"]
meta = row["input_metadata"]
rollout_id = row.get("execution_metadata", {}).get("rollout_id", "?")

# ---- walk scans ----
ms = row["messages"]
system_msg = ms[0]["content"]
scans = []  # dicts: peaks, notepad_read, think, answer_text, report(cfs,bws), notepad_wrote, toolline
i = 1
while i + 2 < len(ms) + 1 and i + 1 < len(ms):
    if ms[i]["role"] != "user":
        break
    user = ms[i]["content"] or ""
    asst = ms[i + 1] if i + 1 < len(ms) else {}
    tool = ms[i + 2] if i + 2 < len(ms) else {}
    peaks = PEAKLINE.findall(user)
    npad = ""
    if "=== YOUR NOTEPAD ===" in user:
        npad = user.split("=== YOUR NOTEPAD ===", 1)[1]
        npad = npad.split("(You cannot see", 1)[0].strip()
    content = asst.get("content") or ""
    think = ""
    rest = content
    if "<think>" in content:
        think = content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        rest = content.split("</think>", 1)[1].strip() if "</think>" in content else ""
    cfs, bws, wrote = [], [], ""
    for tc in (asst.get("tool_calls") or []):
        try:
            args = json.loads(tc["function"]["arguments"])
            cfs = args.get("center_freqs", [])
            bws = args.get("bandwidths", [])
            wrote = args.get("notepad_update") or ""
        except Exception:
            pass
    toolline = (tool.get("content") or "").replace("ok\n", "").strip() if tool.get("role") == "tool" else ""
    scans.append(dict(peaks=peaks, npad=npad, think=think, rest=rest, cfs=cfs, bws=bws,
                      wrote=wrote, toolline=toolline))
    i += 3

occs = series(row, OCC)
dorms = series(row, DORM)


def diff_notepad(read, wrote):
    """Line-diff the written notepad vs the read one: green = added, red-strike = dropped."""
    rl = [l for l in read.splitlines() if l.strip()]
    wl = [l for l in wrote.splitlines() if l.strip()]
    rs, ws = set(rl), set(wl)
    out = []
    for l in wl:
        cls = "npadd" if l not in rs else ""
        out.append(f'<span class="{cls}">{H.escape(l)}</span>' if cls else H.escape(l))
    for l in rl:
        if l not in ws:
            out.append(f'<span class="npdrop">{H.escape(l)}</span>')
    return "\n".join(out)


parts = []
parts.append(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Exemplary notepad rollout — xoi922eh ep4 ({meta['row_id']})</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 1060px; margin: 24px auto;
        padding: 0 16px; color: #1a1a1a; line-height: 1.45; }}
 h1 {{ font-size: 1.35em; }} h2 {{ font-size: 1.1em; margin-top: 1.6em; }}
 .banner {{ background: #eef6ff; border: 1px solid #b8d4f0; border-radius: 8px; padding: 12px 16px; }}
 .mono {{ font-family: ui-monospace, Menlo, monospace; font-size: 0.86em; }}
 .scan {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 14px; margin: 14px 0; }}
 .scanhead {{ font-weight: 700; }}
 .metrics {{ font-family: ui-monospace, Menlo, monospace; font-size: 0.8em; color: #555;
             background: #f6f6f6; padding: 4px 8px; border-radius: 5px; display: inline-block; }}
 .npad {{ font-family: ui-monospace, Menlo, monospace; font-size: 0.82em; white-space: pre-wrap;
          background: #fffbe6; border: 1px solid #e6d98c; border-radius: 6px; padding: 8px 10px; }}
 .npwrote {{ background: #f0fff0; border-color: #9cd49c; }}
 .npadd {{ background: #c9f7c9; font-weight: 600; }}
 .npdrop {{ background: #ffd9d9; text-decoration: line-through; color: #a33; }}
 .peaks {{ font-family: ui-monospace, Menlo, monospace; font-size: 0.8em; color: #333; }}
 details {{ margin: 6px 0; }} summary {{ cursor: pointer; color: #2757a8; font-size: 0.9em; }}
 .think {{ font-family: ui-monospace, Menlo, monospace; font-size: 0.78em; white-space: pre-wrap;
           background: #f4f0fa; border: 1px solid #d5c8ec; border-radius: 6px; padding: 8px 10px;
           max-height: 340px; overflow-y: auto; }}
 table {{ border-collapse: collapse; font-size: 0.85em; }}
 td, th {{ border: 1px solid #ccc; padding: 3px 9px; text-align: left; }}
 .good {{ color: #1a7f37; font-weight: 600; }}
 .runid {{ background: #eee; border-radius: 4px; padding: 1px 6px; font-family: ui-monospace, monospace;
           font-size: 0.8em; }}
</style></head><body>
<h1>One exemplary notepad rollout, end to end</h1>
<div class="banner">
<b>What this page is:</b> a single complete 30-scan episode played by the RL-trained notepad model, shown
turn by turn for review — what the model <i>saw</i>, the notepad it <i>read</i>, what it <i>thought</i>
(collapsed), the report it <i>submitted</i>, and the notepad it <i>wrote back</i> (diffed:
<span class="npadd">&nbsp;added&nbsp;</span> / <span class="npdrop">&nbsp;dropped&nbsp;</span>).<br><br>
<b>Provenance:</b> run <span class="runid">xoi922eh</span> (dormc2, the clean completion-guard winner),
epoch-4 eval, task <span class="runid">{H.escape(meta['row_id'])}</span>, rollout
<span class="runid">{H.escape(rollout_id)}</span>, temperature 1.2.<br>
<b>Selection rule (no cherry-picking):</b> of the {len(complete)} complete 30-scan episodes in this epoch's
192 rollouts, this is the <b>MEDIAN by memory score</b> (mean per-scan dorm {med_dorm:.3f}; run range
{dorms_all[0]:.3f}–{dorms_all[-1]:.3f}) — a typical rollout, not the best one.<br>
<b>Episode result:</b> {H.escape(ev.get('reason', ''))}
</div>

<h2>Score track (per scan)</h2>
<p class="mono">SCAN_OCC (report vs true transmitter set) and SCAN_DORM (coverage of recallable channels —
seen earlier, invisible now; earnable only via the notepad):</p>
<table><tr><th>scan</th>{''.join(f'<td>{i+1}</td>' for i in range(len(occs)))}</tr>
<tr><th>occ</th>{''.join(f'<td>{v:.2f}</td>' for v in occs)}</tr>
<tr><th>dorm</th><td>—</td>{''.join(f'<td>{v:.2f}</td>' for v in dorms)}</tr></table>

<details><summary>system prompt (the rules the model plays by)</summary>
<div class="think">{H.escape(system_msg)}</div></details>
""")

for si, s in enumerate(scans, 1):
    peaks_html = "<br>".join(
        f"{fq} MHz &nbsp; {pw} dBm &nbsp; width {wd} MHz" for fq, pw, wd in s["peaks"]) or "(none)"
    report_html = ", ".join(f"{c}/{b}" for c, b in zip(s["cfs"], s["bws"])) or "(empty report)"
    parts.append(f"""
<div class="scan">
<div class="scanhead">Scan {si} of {len(scans)}</div>
<div class="metrics">{H.escape(s['toolline'])}</div>
<table style="margin-top:8px; width:100%; border:none"><tr>
<td style="border:none; vertical-align:top; width:33%">
  <b>saw</b> ({len(s['peaks'])} peaks)<br><span class="peaks">{peaks_html}</span></td>
<td style="border:none; vertical-align:top; width:33%">
  <b>read (notepad in)</b><div class="npad">{H.escape(s['npad']) or '(empty)'}</div></td>
<td style="border:none; vertical-align:top; width:34%">
  <b>wrote (notepad out, diffed)</b><div class="npad npwrote">{diff_notepad(s['npad'], s['wrote']) or '(no update)'}</div></td>
</tr></table>
<b>submitted report</b> <span class="mono">(center_freq/bandwidth MHz)</span>:
<span class="mono">{H.escape(report_html)}</span>
<details><summary>thinking trace ({len(s['think'])} chars)</summary>
<div class="think">{H.escape(s['think']) or '(empty)'}</div></details>
</div>""")

parts.append("""
<p><small>Generated from the Fireworks eval dataset <span class="mono">rft-evalv3-xoi922eh-epoch-4</span>;
generator script <span class="mono">live_assets/render_rollout.py</span>. Back to the
<a href="LIVE_dormant_arm.html">live lab notebook</a>.</small></p>
</body></html>""")

out = f"{REPO}/ROLLOUT_xoi922eh_ep4_example.html"
open(out, "w").write("\n".join(parts))
print("written", out, f"({len(scans)} scans)")
