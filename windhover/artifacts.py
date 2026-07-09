"""Run artifacts — files a graph wrote, detected from recorded outputs.

Nodes that generate reports/charts/exports typically return the file path in
their output (e.g. {"docs": ["/out/report.docx"]}). This module finds those
paths so the UI can list, preview, and download them.

Security model: the server NEVER reads an arbitrary path. The allowlist is
re-derived from the run's stored input/outputs on every request, so only files
the traced code itself recorded can be fetched — and only through the same
token gate as the rest of /api.
"""
from __future__ import annotations

import os
import re

# extensions worth surfacing, with how the browser can handle them
_INLINE_DOC = {"html": "text/html", "htm": "text/html", "pdf": "application/pdf"}
_INLINE_IMG = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "gif": "image/gif", "webp": "image/webp", "svg": "image/svg+xml"}
_INLINE_TEXT = {"py": "text/plain", "md": "text/plain", "txt": "text/plain",
                "json": "application/json", "csv": "text/csv", "tsv": "text/tab-separated-values",
                "log": "text/plain", "yaml": "text/plain", "yml": "text/plain",
                "toml": "text/plain", "xml": "text/xml", "sql": "text/plain"}
_DOWNLOAD = {"docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
             "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
             "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
             "doc": "application/msword", "xls": "application/vnd.ms-excel",
             "zip": "application/zip", "parquet": "application/octet-stream",
             "db": "application/octet-stream", "sqlite": "application/octet-stream"}

_EXTS = {**_INLINE_DOC, **_INLINE_IMG, **_INLINE_TEXT, **_DOWNLOAD}

# absolute unix, ~/, or windows-drive path ending in a known extension
_PATH_RE = re.compile(
    r"^(?:/|~/|[A-Za-z]:[\\/])[^\n\r\t*?<>|\"']{1,500}\.(%s)$" % "|".join(_EXTS),
    re.IGNORECASE,
)

MAX_ARTIFACTS = 50


def classify(path: str) -> dict:
    """kind + mime + whether the browser can render it inline."""
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in _INLINE_DOC:
        kind = "pdf" if ext == "pdf" else "html"
    elif ext in _INLINE_IMG:
        kind = "image"
    elif ext in _INLINE_TEXT:
        kind = "text"
    elif ext in ("docx", "doc"):
        kind = "word"          # client renders via vendored mammoth
    elif ext in ("xlsx", "xls"):
        kind = "sheet"         # client renders via vendored SheetJS
    else:
        kind = "file"
    # `inline` drives the SERVER's content-disposition: office formats are still
    # served as downloads; the client fetches their bytes separately to preview.
    return {"ext": ext, "mime": _EXTS.get(ext, "application/octet-stream"),
            "kind": kind, "inline": ext not in _DOWNLOAD}


def extract_paths(obj) -> list[str]:
    """Collect file-path-looking strings from arbitrary recorded JSON, in order."""
    out: list[str] = []
    seen: set[str] = set()

    def walk(v):
        if len(out) >= MAX_ARTIFACTS:
            return
        if isinstance(v, str):
            s = v.strip()
            if len(s) <= 500 and _PATH_RE.match(s) and s not in seen:
                seen.add(s)
                out.append(s)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(obj)
    return out


def run_artifacts(run: dict) -> list[dict]:
    """All artifacts recorded by a run: detected paths + on-disk status."""
    sources = [run.get("input")]
    for s in run.get("spans", []):
        sources.append(s.get("output"))
    paths = extract_paths(sources)
    arts = []
    for p in paths:
        real = os.path.expanduser(p)
        info = classify(p)
        try:
            st = os.stat(real)
            info.update(exists=True, size=st.st_size, mtime_ms=int(st.st_mtime * 1000))
        except OSError:
            info.update(exists=False, size=None, mtime_ms=None)
        info.update(path=p, name=os.path.basename(p))
        arts.append(info)
    return arts


def resolve(run: dict, path: str) -> str | None:
    """Return the real filesystem path ONLY if `path` is recorded by this run
    and exists — anything else is refused."""
    if path not in {a["path"] for a in run_artifacts(run)}:
        return None
    real = os.path.expanduser(path)
    return real if os.path.isfile(real) else None
