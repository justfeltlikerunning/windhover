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

import json
import os
import re

# extensions worth surfacing, with how the browser can handle them
_INLINE_DOC = {"html": "text/html", "htm": "text/html", "pdf": "application/pdf"}
_INLINE_IMG = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "gif": "image/gif", "webp": "image/webp", "svg": "image/svg+xml",
               "bmp": "image/bmp", "ico": "image/x-icon", "avif": "image/avif",
               "tif": "image/tiff", "tiff": "image/tiff"}
_INLINE_TEXT = {"py": "text/plain", "txt": "text/plain",
                "json": "application/json", "csv": "text/csv", "tsv": "text/tab-separated-values",
                "log": "text/plain", "yaml": "text/plain", "yml": "text/plain",
                "toml": "text/plain", "xml": "text/xml", "sql": "text/plain",
                "ndjson": "application/x-ndjson", "jsonl": "application/x-ndjson",
                "ini": "text/plain", "cfg": "text/plain", "conf": "text/plain",
                "rst": "text/plain", "tex": "text/plain", "sh": "text/plain",
                "js": "text/plain", "ts": "text/plain", "css": "text/plain",
                "diff": "text/plain", "patch": "text/plain"}
_MARKDOWN = {"md": "text/markdown", "markdown": "text/markdown"}
_MEDIA_VIDEO = {"mp4": "video/mp4", "webm": "video/webm", "mov": "video/quicktime",
                "m4v": "video/mp4", "ogv": "video/ogg"}
_MEDIA_AUDIO = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg",
                "m4a": "audio/mp4", "flac": "audio/flac", "aac": "audio/aac"}
_OFFICE_WORD = {"docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
_OFFICE_SHEET = {"xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                 "xls": "application/vnd.ms-excel"}
_OFFICE_SLIDES = {"pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation"}
# detected + listed, but download-only (no in-browser render)
_DOWNLOAD = {"doc": "application/msword", "ppt": "application/vnd.ms-powerpoint",
             "zip": "application/zip", "tar": "application/x-tar", "gz": "application/gzip",
             "tgz": "application/gzip", "bz2": "application/x-bzip2", "xz": "application/x-xz",
             "7z": "application/x-7z-compressed", "rar": "application/vnd.rar",
             "parquet": "application/octet-stream", "npy": "application/octet-stream",
             "npz": "application/octet-stream", "pkl": "application/octet-stream",
             "h5": "application/octet-stream", "onnx": "application/octet-stream",
             "pt": "application/octet-stream", "ckpt": "application/octet-stream",
             "bin": "application/octet-stream", "db": "application/octet-stream",
             "sqlite": "application/octet-stream"}

_EXTS = {**_INLINE_DOC, **_INLINE_IMG, **_INLINE_TEXT, **_MARKDOWN,
         **_MEDIA_VIDEO, **_MEDIA_AUDIO, **_OFFICE_WORD, **_OFFICE_SHEET,
         **_OFFICE_SLIDES, **_DOWNLOAD}
# formats the client renders even though the server serves them as a download
_CLIENT_RENDER = set(_OFFICE_WORD) | set(_OFFICE_SHEET) | set(_OFFICE_SLIDES)

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
    elif ext in _MARKDOWN:
        kind = "markdown"      # client renders via vendored marked
    elif ext in _INLINE_TEXT:
        kind = "text"
    elif ext in _MEDIA_VIDEO:
        kind = "video"         # native <video>
    elif ext in _MEDIA_AUDIO:
        kind = "audio"         # native <audio>
    elif ext in _OFFICE_WORD:
        kind = "word"          # client renders via vendored mammoth (.docx only)
    elif ext in _OFFICE_SHEET:
        kind = "sheet"         # client renders via vendored SheetJS
    elif ext in _OFFICE_SLIDES:
        kind = "slides"        # client renders a slide outline via vendored jszip
    else:
        kind = "file"          # download-only
    # `inline` drives the SERVER's content-disposition. Office formats are served
    # as downloads (client fetches their bytes separately to render); media and
    # everything text/image/pdf are served inline so the browser can stream them.
    return {"ext": ext, "mime": _EXTS.get(ext, "application/octet-stream"),
            "kind": kind, "inline": ext not in _DOWNLOAD and ext not in _CLIENT_RENDER}


def extract_paths(obj) -> list[str]:
    """Collect file-path-looking strings from arbitrary recorded JSON, in order."""
    out: list[str] = []
    seen: set[str] = set()

    def walk(v):
        if len(out) >= MAX_ARTIFACTS:
            return
        if isinstance(v, str):
            s = v.strip()
            # a node may return its output as a JSON string (e.g. a serialized dict of paths);
            # decode and recurse so those paths are still discovered
            if s[:1] in "{[":
                try:
                    walk(json.loads(s))
                    return
                except Exception:
                    pass
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
