"""Render a run into a self-contained Markdown report.

A run's node outputs already hold everything a reader wants — an LLM narrative, a
list of findings, a summary object — but in the trace UI they're raw payloads and
nothing is downloadable. This turns any run into one Markdown document you can read,
download, and share, with no per-graph knowledge: it walks node outputs generically,
surfaces prose where a node wrote prose, and tables where a node wrote records.

Public-safe: no domain vocabulary, no assumptions about graph shape.
"""
from __future__ import annotations
import json
import re
from typing import Any

# keys whose values read as prose (an LLM narrative, a written summary) rather than data
_NARRATIVE_KEYS = ("summary", "narrative", "report", "text", "message", "content",
                   "recommendation", "analysis", "explanation", "assessment")
_MAX_STR = 12000        # per prose block
_MAX_ROWS = 60          # per table
_MAX_COLS = 10


def _maybe_json(v: Any) -> Any:
    """A node output is often a JSON-encoded string (double-encoded state). Decode it."""
    if isinstance(v, str):
        s = v.strip()
        if s and s[0] in "{[":
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def _salvage(s: str) -> str:
    """Recover a narrative from a JSON string that won't parse — e.g. a report clipped
    mid-object by an older/lower capture cap. Pull the longest narrative-key value,
    tolerating a missing closing quote at the truncation point."""
    best = ""
    for key in _NARRATIVE_KEYS:
        m = re.search(r'"' + key + r'"\s*:\s*"', s)
        if not m:
            continue
        rest, out, i = s[m.end():], [], 0
        while i < len(rest):
            ch = rest[i]
            if ch == "\\" and i + 1 < len(rest):
                out.append(rest[i:i + 2]); i += 2; continue
            if ch == '"':
                break
            out.append(ch); i += 1
        cand = "".join(out)
        try:
            cand = json.loads('"' + cand + '"')
        except Exception:
            pass
        if len(cand) > len(best):
            best = cand
    return best


def _fmt_ms(ms) -> str:
    if ms is None:
        return ""
    ms = float(ms)
    return f"{ms/1000:.2f}s" if ms >= 1000 else f"{int(ms)}ms"


def _table(rows: list[dict]) -> list[str]:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    cols = cols[:_MAX_COLS]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join([" --- "] * len(cols)) + "|"]
    for r in rows[:_MAX_ROWS]:
        cells = []
        for c in cols:
            val = r.get(c, "")
            if isinstance(val, (dict, list)):
                val = json.dumps(val, default=str)
            cells.append(str(val).replace("\n", " ").replace("|", "\\|")[:200])
        out.append("| " + " | ".join(cells) + " |")
    if len(rows) > _MAX_ROWS:
        out.append(f"\n_… {len(rows) - _MAX_ROWS} more rows omitted_")
    return out


def _render(v: Any, depth: int = 0) -> list[str]:
    v = _maybe_json(v)
    if v is None or v == "" or v == [] or v == {}:
        return []
    if isinstance(v, str):
        # a JSON-ish string that didn't decode (clipped by an older capture cap) —
        # pull the narrative out rather than dumping raw escaped JSON on the reader
        if v.lstrip()[:1] in "{[":
            salv = _salvage(v)
            if salv:
                return [salv[:_MAX_STR] + "\n\n_(recovered from a truncated capture — re-run for the full report)_"]
        return [v[:_MAX_STR] + ("\n\n_… truncated_" if len(v) > _MAX_STR else "")]
    if isinstance(v, (int, float, bool)):
        return [str(v)]
    if isinstance(v, list):
        if v and all(isinstance(x, dict) for x in v):
            return _table(v)
        out = []
        for x in v:
            x = _maybe_json(x)
            if isinstance(x, (dict, list)):
                out += _render(x, depth + 1)
            else:
                out.append(f"- {x}")
        return out
    if isinstance(v, dict):
        out = []
        # prose fields first, rendered as prose (not key/value)
        for k in list(v.keys()):
            if k.lower() in _NARRATIVE_KEYS and isinstance(_maybe_json(v[k]), str):
                out += _render(v[k], depth + 1)
                out.append("")
        for k, val in v.items():
            if k.lower() in _NARRATIVE_KEYS and isinstance(_maybe_json(val), str):
                continue
            body = _render(val, depth + 1)
            if not body:
                continue
            label = k.replace("_", " ")
            if len(body) == 1 and len(body[0]) < 80 and not body[0].startswith(("|", "-")):
                out.append(f"**{label}:** {body[0]}")
            else:
                out.append(f"**{label}:**")
                out += body
                out.append("")
        return out
    return [str(v)]


def _find_headline(run: dict) -> list[str]:
    """The most report-like narrative in the run — surfaced up top so the report leads
    with the answer, not the trace. Deepest narrative string wins (usually the review)."""
    best = ""
    for s in run.get("spans", []):
        if s.get("type") != "node":
            continue
        o = _maybe_json(s.get("output"))
        stack = [o]
        while stack:
            cur = _maybe_json(stack.pop())
            if isinstance(cur, dict):
                for k, val in cur.items():
                    if k.lower() in _NARRATIVE_KEYS and isinstance(_maybe_json(val), str):
                        cand = _maybe_json(val)
                        if len(cand) > len(best):
                            best = cand
                    else:
                        stack.append(val)
            elif isinstance(cur, list):
                stack.extend(cur)
            elif isinstance(cur, str) and cur.lstrip()[:1] in "{[":
                cand = _salvage(cur)          # narrative trapped in an unparseable (clipped) JSON string
                if len(cand) > len(best):
                    best = cand
    return [best[:_MAX_STR]] if best else []


def render_run_report(run: dict) -> str:
    g = run.get("graph") or "run"
    rid = run.get("id", "")
    L = [f"# {g} — run report", ""]

    started = run.get("started_ms")
    when = ""
    if started:
        import datetime
        when = datetime.datetime.fromtimestamp(started / 1000).strftime("%Y-%m-%d %H:%M")
    meta = [f"**Run:** `{rid}`", f"**Status:** {run.get('status', '?')}"]
    if when:
        meta.append(f"**When:** {when}")
    if run.get("duration_ms") is not None:
        meta.append(f"**Duration:** {_fmt_ms(run.get('duration_ms'))}")
    if run.get("node_count"):
        meta.append(f"**Nodes:** {run['node_count']}")
    if run.get("cost_usd") is not None:
        meta.append(f"**Cost:** ${run['cost_usd']:.5f}")
    if run.get("tags"):
        meta.append("**Tags:** " + ", ".join(str(t) for t in run["tags"]))
    L.append("  ·  ".join(meta))
    L.append("")

    headline = _find_headline(run)
    if headline:
        L += ["## Summary", "", headline[0], ""]

    if run.get("error"):
        L += ["## Error", "", "```", str(run["error"])[:4000], "```", ""]

    if run.get("input") not in (None, "", {}):
        L += ["## Input", "", "```json",
              json.dumps(run["input"], indent=2, default=str)[:2000], "```", ""]

    nodes = [s for s in run.get("spans", []) if s.get("type") == "node" and not s.get("parent_id")]
    if nodes:
        L += ["## Steps", ""]
        for s in nodes:
            head = f"### {s.get('name', 'node')}"
            if s.get("dur_ms") is not None:
                head += f"  ·  {_fmt_ms(s.get('dur_ms'))}"
            if s.get("status") == "error":
                head += "  ·  ⚠ error"
            L.append(head)
            L.append("")
            body = _render(s.get("output"))
            L += body if body else ["_no output captured_"]
            L.append("")

    L += ["---", f"_Generated by Windhover from run {rid}._"]
    return "\n".join(L)
