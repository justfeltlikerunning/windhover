"""Windhover tracer — one LangChain callback, two sinks.

`SpanBuilder` turns LangGraph/LangChain callbacks into runs + span trees and
hands each event to a `sink`. In-process runs use a DB sink; external apps use
`WindhoverTracer` (HTTP sink) — identical span logic either way.

Concurrency-safe: every ROOT execution gets its own run context, so one tracer
instance handles parallel invokes, `.batch()`, and even bare `llm.invoke()`
calls (an LLM/tool/retriever with no parent opens an implicit run).

Callback realities handled:
  * LangGraph exposes the node name only on `on_chain_start`
    (metadata.langgraph_node) → we map langchain run_id -> our span.
  * Token usage lives in `llm_output.token_usage` (OpenAI-style) or a
    generation's `usage_metadata` — both read, incl. cache/reasoning details.
  * GraphInterrupt raised through on_chain_error is a PAUSE, not a failure.
  * Parent linkage uses LangChain's parent_run_id.
Best-effort: a sink error never propagates into the user's graph.
"""
from __future__ import annotations
import json, time, uuid
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - allows import without langchain
    BaseCallbackHandler = object  # type: ignore

_PRICING: Optional[dict] = None


def _load_pricing() -> dict:
    global _PRICING
    if _PRICING is None:
        try:
            p = Path(__file__).parent / "pricing.json"
            _PRICING = {k: v for k, v in json.loads(p.read_text()).items()
                        if not k.startswith("_")}
        except Exception:
            _PRICING = {}
    return _PRICING


def cost_of(model: Optional[str], pt: Optional[int], ct: Optional[int]) -> Optional[float]:
    if not model or (pt is None and ct is None):
        return None
    table = _load_pricing()
    best = max((k for k in table if model.startswith(k)), key=len, default=None)
    if not best:
        return None
    rate = table[best]
    return round((pt or 0) / 1e6 * rate["input"] + (ct or 0) / 1e6 * rate["output"], 6)


def _err_text(error: Any) -> str:
    """Full traceback when we have one — the UI maps 'File …, line N' frames
    onto node source so you can see exactly where a run broke."""
    try:
        if isinstance(error, BaseException) and error.__traceback__ is not None:
            import traceback
            return "".join(traceback.format_exception(
                type(error), error, error.__traceback__))[-4000:]
    except Exception:
        pass
    return str(error)


def _jsonable(o: Any):
    """LangChain messages serialize as {'role','content'} so the UI can render
    conversations; everything else falls back to str()."""
    content = getattr(o, "content", None)
    role = getattr(o, "type", None)
    if content is not None and isinstance(role, str):
        d = {"role": role, "content": content if isinstance(content, (str, list)) else str(content)}
        tc = getattr(o, "tool_calls", None)
        if tc:
            d["tool_calls"] = [{"name": c.get("name"), "args": c.get("args")}
                               if isinstance(c, dict) else str(c) for c in tc]
        return d
    return str(o)


def _trunc(v: Any, n: int = 4000) -> Any:
    s = json.dumps(v, default=_jsonable)
    return json.loads(s) if len(s) <= n else s[:n] + "…"


def _model_name(serialized: dict, kw: dict) -> str:
    inv = kw.get("invocation_params") or {}
    for src in (inv, (serialized or {}).get("kwargs", {}), serialized or {}):
        for key in ("model", "model_name", "model_id", "deployment_name"):
            if src.get(key):
                return str(src[key])
    ids = (serialized or {}).get("id") or []
    return ids[-1] if ids else "llm"


def _usage(response: Any) -> tuple[Optional[int], Optional[int], Optional[dict]]:
    """(prompt, completion, detail) — detail carries cache_read/cache_creation/
    reasoning token counts when the provider reports them."""
    out = getattr(response, "llm_output", None) or {}
    for key in ("token_usage", "usage"):
        u = out.get(key) if isinstance(out, dict) else None
        if u:
            return (u.get("prompt_tokens") or u.get("input_tokens"),
                    u.get("completion_tokens") or u.get("output_tokens"), None)
    try:
        for gens in response.generations:
            for g in gens:
                um = getattr(getattr(g, "message", None), "usage_metadata", None)
                if um:
                    detail = {}
                    for side in ("input_token_details", "output_token_details"):
                        for k, v in (um.get(side) or {}).items():
                            if v:
                                detail[f"{side.split('_')[0]}_{k}"] = v
                    return um.get("input_tokens"), um.get("output_tokens"), detail or None
    except Exception:
        pass
    return None, None, None


def _gen_text(response: Any) -> str:
    try:
        parts = []
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                text = getattr(g, "text", "") or str(getattr(msg, "content", "") or "")
                if not text and msg is not None:
                    # structured-output / function-calling: content is empty,
                    # the payload lives in the tool calls
                    tc = (getattr(msg, "tool_calls", None) or
                          (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls"))
                    if tc:
                        text = json.dumps(tc, default=str)
                parts.append(text)
        return "\n".join(p for p in parts if p)
    except Exception:
        return ""


class SpanBuilder(BaseCallbackHandler):
    _INTERNAL_TAG_PREFIXES = ("graph:", "langsmith:", "seq:", "langgraph_")

    def __init__(self, sink: Callable[[dict], None], run_name: str = "external",
                 session: Optional[str] = None, tags: Optional[list] = None):
        self.sink = sink
        self.run_name = run_name
        self.session = session
        self.tags = tags
        self.run_id = uuid.uuid4().hex[:12]     # id of the FIRST run this tracer opens
        self._closed = 0                         # runs completed (cleanup bookkeeping)
        self._runs: dict = {}                    # lc root id -> run ctx
        self._root_of: dict = {}                 # lc run id -> lc root id
        self._open: dict = {}                    # lc run id -> pending span info
        self._span_of: dict = {}                 # lc run id -> our span id

    # -- infra ---------------------------------------------------------------
    def _emit(self, ev: dict) -> None:
        try:
            self.sink(ev)
        except Exception:
            pass  # observability must never break the graph

    def _resolve(self, run_id, parent_run_id):
        """(lc_root, ctx-or-None) for an event; registers run_id under its root."""
        root = None
        if parent_run_id is not None:
            root = self._root_of.get(parent_run_id)
        if root is None:
            root = self._root_of.get(run_id) or (run_id if parent_run_id is None else parent_run_id)
        self._root_of[run_id] = root
        ctx = self._runs.get(root)
        if ctx is not None:
            ctx["members"].add(run_id)
        return root, ctx

    def _open_ctx(self, root, inputs, metadata=None, lc_tags=None):
        md = metadata or {}
        name = str(md.get("windhover_run_name") or self.run_name)
        session = str(md["windhover_session"]) if md.get("windhover_session") else self.session
        user_tags = [t for t in (lc_tags or [])
                     if not str(t).startswith(self._INTERNAL_TAG_PREFIXES)]
        extra = md.get("windhover_tags") or []
        tags = list(dict.fromkeys([*(self.tags or []), *map(str, extra), *map(str, user_tags)])) or None
        ctx = {"id": self.run_id if not self._closed and not self._runs else uuid.uuid4().hex[:12],
               "t0": time.time(), "seq": 0, "interrupted": False, "members": {root}}
        self._runs[root] = ctx
        self._emit({"kind": "run_open", "run_id": ctx["id"], "graph": name,
                    "input": _trunc(inputs), "started_ms": int(ctx["t0"] * 1000),
                    "session": session, "tags": tags,
                    "thread_id": md.get("thread_id")})
        return ctx

    def _finish_span(self, info: dict, *, output=None, status="ok", error=None,
                     model=None, pt=None, ct=None, usage_detail=None):
        ctx = self._runs.get(info["root"])
        if ctx is None:
            return
        now = time.time()
        seq = info.get("seq")
        if seq is None:
            seq = ctx["seq"]; ctx["seq"] += 1
        self._emit({"kind": "span", "id": info["span_id"], "run_id": ctx["id"],
                    "parent_id": self._span_of.get(info.get("parent")),
                    "seq": seq, "type": info["type"], "name": info["name"],
                    "status": status, "started_ms": int(info["t"] * 1000),
                    "ended_ms": int(now * 1000),
                    "offset_ms": int((info["t"] - ctx["t0"]) * 1000),
                    "dur_ms": int((now - info["t"]) * 1000),
                    "input": info.get("input"), "output": output, "model": model,
                    "prompt_tokens": pt, "completion_tokens": ct,
                    "cost_usd": cost_of(model, pt, ct), "error": error,
                    "retries": info.get("retries"),
                    "ttft_ms": int(info["ttft"] * 1000) if info.get("ttft") is not None else None,
                    "usage_detail": usage_detail, "params": info.get("params")})

    def _mark_interrupted(self, ctx, payload):
        if ctx is None or ctx["interrupted"]:
            return
        ctx["interrupted"] = True
        now = time.time()
        self._emit({"kind": "span", "id": uuid.uuid4().hex[:12], "run_id": ctx["id"],
                    "parent_id": None, "seq": ctx["seq"], "type": "interrupt",
                    "name": "interrupt", "status": "ok",
                    "started_ms": int(now * 1000), "ended_ms": int(now * 1000),
                    "offset_ms": int((now - ctx["t0"]) * 1000), "dur_ms": 0,
                    "input": None, "output": _trunc(payload) if payload else None,
                    "model": None, "prompt_tokens": None, "completion_tokens": None,
                    "cost_usd": None, "error": None})
        ctx["seq"] += 1

    def _close_ctx(self, root, status, error=None):
        ctx = self._runs.pop(root, None)
        if ctx is None:
            return
        self._emit({"kind": "run_close", "run_id": ctx["id"], "status": status,
                    "ended_ms": int(time.time() * 1000), "error": error})
        # long-lived tracers (production apps) must not grow without bound
        self._closed += 1
        for m in ctx.get("members", ()):
            self._root_of.pop(m, None)
            self._span_of.pop(m, None)
            self._open.pop(m, None)

    # -- chains / nodes --------------------------------------------------------
    def on_chain_start(self, serialized, inputs, *, run_id=None, parent_run_id=None,
                       metadata=None, tags=None, **kw):
        root, ctx = self._resolve(run_id, parent_run_id)
        if ctx is None:
            ctx = self._open_ctx(root, inputs, metadata, tags)
        node = (metadata or {}).get("langgraph_node")
        if node and run_id != root:
            sid = uuid.uuid4().hex[:12]
            self._span_of[run_id] = sid
            self._open[run_id] = {"span_id": sid, "type": "node", "name": node,
                                  "t": time.time(), "parent": parent_run_id,
                                  "input": None, "root": root}

    def on_chain_end(self, outputs, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if info:
            self._finish_span(info, output=_trunc(outputs))
            return
        if run_id in self._runs:
            ctx = self._runs[run_id]
            dyn = isinstance(outputs, dict) and "__interrupt__" in outputs
            if dyn:
                self._mark_interrupted(ctx, outputs.get("__interrupt__"))
            self._close_ctx(run_id, "interrupted" if ctx["interrupted"] else "done")

    def on_chain_error(self, error, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        # LangGraph signals a human-in-the-loop pause by raising GraphInterrupt
        # through the node — that's a pause, not a failure.
        if type(error).__name__ in ("GraphInterrupt", "NodeInterrupt"):
            payload = []
            try:
                for i in (error.args[0] if error.args else []):
                    payload.append(getattr(i, "value", str(i)))
            except Exception:
                pass
            root = info["root"] if info else self._root_of.get(run_id, run_id)
            if info:
                self._finish_span(info, status="interrupted",
                                  output=_trunc(payload) if payload else None)
            self._mark_interrupted(self._runs.get(root), payload)
            return
        err = _err_text(error)
        if info:
            self._finish_span(info, status="error", error=err)
        if run_id in self._runs:
            self._close_ctx(run_id, "error", error=err)

    # -- streaming / retries / custom events ----------------------------------
    def on_llm_new_token(self, token, *, run_id=None, **kw):
        # first token stamps TTFT; the growing text flushes as a partial span
        # every ~0.5s so a live-tailed drawer shows the model typing
        info = self._open.get(run_id)
        if info is None:
            return
        now = time.time()
        if "ttft" not in info:
            info["ttft"] = now - info["t"]
        info["buf"] = (info.get("buf") or "") + (token or "")
        ctx = self._runs.get(info["root"])
        if ctx is not None and now - info.get("flushed", 0) >= 0.5 and info.get("buf"):
            info["flushed"] = now
            self._emit({"kind": "span", "id": info["span_id"], "run_id": ctx["id"],
                        "parent_id": self._span_of.get(info.get("parent")),
                        "seq": info["seq"], "type": "llm", "name": info["name"],
                        "status": "running", "started_ms": int(info["t"] * 1000),
                        "ended_ms": None,
                        "offset_ms": int((info["t"] - ctx["t0"]) * 1000),
                        "dur_ms": int((now - info["t"]) * 1000),
                        "input": info.get("input"), "output": info["buf"][-4000:],
                        "model": info["name"], "prompt_tokens": None,
                        "completion_tokens": None, "cost_usd": None, "error": None,
                        "retries": info.get("retries"),
                        "ttft_ms": int(info["ttft"] * 1000),
                        "usage_detail": None, "params": info.get("params")})

    def on_retry(self, retry_state, *, run_id=None, **kw):
        info = self._open.get(run_id)
        if info is not None:
            info["retries"] = getattr(retry_state, "attempt_number", None) or \
                              (info.get("retries") or 0) + 1

    def on_custom_event(self, name, data, *, run_id=None, **kw):
        root = self._root_of.get(run_id)
        ctx = self._runs.get(root) if root else None
        if ctx is None:
            return
        now = time.time()
        self._emit({"kind": "span", "id": uuid.uuid4().hex[:12], "run_id": ctx["id"],
                    "parent_id": self._span_of.get(run_id),
                    "seq": ctx["seq"], "type": "event", "name": str(name),
                    "status": "ok", "started_ms": int(now * 1000),
                    "ended_ms": int(now * 1000),
                    "offset_ms": int((now - ctx["t0"]) * 1000), "dur_ms": 0,
                    "input": None, "output": _trunc(data), "model": None,
                    "prompt_tokens": None, "completion_tokens": None,
                    "cost_usd": None, "error": None})
        ctx["seq"] += 1

    # -- LLMs ------------------------------------------------------------------
    @staticmethod
    def _llm_params(kw) -> Optional[dict]:
        inv = kw.get("invocation_params") or {}
        out = {}
        for k in ("temperature", "max_tokens", "max_completion_tokens", "stream", "top_p"):
            if inv.get(k) is not None:
                out[k] = inv[k]
        tools = []
        for tl in (inv.get("tools") or []):
            fn = tl.get("function", tl) if isinstance(tl, dict) else {}
            if isinstance(fn, dict) and fn.get("name"):
                tools.append(fn["name"])
        if tools:
            out["tools_offered"] = tools
        return out or None

    def _leaf_start(self, span_type, name, payload, run_id, parent_run_id, params=None):
        root, ctx = self._resolve(run_id, parent_run_id)
        if ctx is None:  # bare llm/tool/retriever usage -> implicit run
            ctx = self._open_ctx(root, payload)
        sid = uuid.uuid4().hex[:12]
        self._span_of[run_id] = sid
        seq = ctx["seq"]; ctx["seq"] += 1
        self._open[run_id] = {"span_id": sid, "type": span_type, "name": name,
                              "t": time.time(), "parent": parent_run_id,
                              "input": _trunc(payload), "root": root, "seq": seq,
                              "params": params}

    def _leaf_close_if_root(self, run_id, status="done", error=None):
        if run_id in self._runs:  # this leaf WAS the root (bare usage)
            self._close_ctx(run_id, status, error=error)

    def on_llm_start(self, serialized, prompts, *, run_id=None, parent_run_id=None, **kw):
        self._leaf_start("llm", _model_name(serialized, kw), prompts,
                         run_id, parent_run_id, self._llm_params(kw))

    def on_chat_model_start(self, serialized, messages, *, run_id=None, parent_run_id=None, **kw):
        flat = [[_jsonable(m) for m in conv] for conv in messages]
        self._leaf_start("llm", _model_name(serialized, kw), flat,
                         run_id, parent_run_id, self._llm_params(kw))

    def on_llm_end(self, response, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if not info:
            return
        pt, ct, detail = _usage(response)
        self._finish_span(info, output=_gen_text(response)[:4000],
                          model=info["name"], pt=pt, ct=ct, usage_detail=detail)
        self._leaf_close_if_root(run_id)

    def on_llm_error(self, error, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if info:
            err = _err_text(error)
            self._finish_span(info, status="error", error=err, model=info["name"])
            self._leaf_close_if_root(run_id, "error", err)

    # -- retrievers (LangChain RAG) ---------------------------------------------
    def on_retriever_start(self, serialized, query, *, run_id=None, parent_run_id=None, **kw):
        self._leaf_start("retriever", (serialized or {}).get("name") or "retriever",
                         query, run_id, parent_run_id)

    def on_retriever_end(self, documents, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if not info:
            return
        docs = list(documents or [])
        preview = [{"content": str(getattr(d, "page_content", d))[:300],
                    "metadata": _trunc(getattr(d, "metadata", None), 500)}
                   for d in docs[:8]]
        self._finish_span(info, output=_trunc({"count": len(docs), "documents": preview}))
        self._leaf_close_if_root(run_id)

    def on_retriever_error(self, error, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if info:
            err = _err_text(error)
            self._finish_span(info, status="error", error=err)
            self._leaf_close_if_root(run_id, "error", err)

    # -- tools -------------------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id=None, parent_run_id=None, **kw):
        self._leaf_start("tool", (serialized or {}).get("name", "tool"),
                         input_str, run_id, parent_run_id)

    def on_tool_end(self, output, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if info:
            self._finish_span(info, output=_trunc(str(output)[:4000]))
            self._leaf_close_if_root(run_id)

    def on_tool_error(self, error, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if info:
            err = _err_text(error)
            self._finish_span(info, status="error", error=err)
            self._leaf_close_if_root(run_id, "error", err)


def apply_to_store(store, ev: dict, source: str = "ui") -> None:
    """Route one tracer event into the store. Shared by the local sink and the
    HTTP ingest endpoint so both persist identically. (Events carry `run_id`;
    the runs table keys on `id` — translate here.)"""
    kind = ev.get("kind")
    if kind == "run_open":
        store.open_run({"id": ev["run_id"], "graph": ev.get("graph"), "source": source,
                        "session": ev.get("session"), "tags": ev.get("tags"),
                        "input": ev.get("input"), "started_ms": ev["started_ms"],
                        "thread_id": ev.get("thread_id")})
    elif kind == "span":
        store.add_span(ev)
    elif kind == "run_close":
        store.close_run(ev["run_id"], ev.get("status", "done"),
                        ev.get("ended_ms"), ev.get("error"))


def db_sink(store) -> Callable[[dict], None]:
    return lambda ev: apply_to_store(store, ev, source="ui")


def http_sink(base_url: str, token: Optional[str] = None,
              max_queue: int = 2000) -> Callable[[dict], None]:
    """Non-blocking: events go onto a bounded queue drained by a daemon thread,
    so a slow or unreachable Windhover host NEVER adds latency to the traced
    app. On overflow the oldest events are dropped (observability is
    best-effort; the app comes first)."""
    import queue as _q, threading as _t, urllib.request
    url = base_url.rstrip("/") + "/api/ingest"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    buf: "_q.Queue" = _q.Queue(maxsize=max_queue)

    def _drain():
        while True:
            ev = buf.get()
            try:
                req = urllib.request.Request(url, data=json.dumps(ev).encode(),
                                             headers=headers)
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass  # drop — never retry-storm a down collector

    _t.Thread(target=_drain, daemon=True).start()

    def sink(ev: dict) -> None:
        try:
            buf.put_nowait(ev)
        except _q.Full:
            try:
                buf.get_nowait()          # shed oldest, keep newest
                buf.put_nowait(ev)
            except Exception:
                pass
    return sink


class WindhoverTracer(SpanBuilder):
    """Drop into any app: config={"callbacks": [WindhoverTracer("http://host:8090")]}.
    Non-blocking; pass token= when the collector sets WINDHOVER_TOKEN."""
    def __init__(self, base_url: str, name: str = "external",
                 session: Optional[str] = None, tags: Optional[list] = None,
                 token: Optional[str] = None):
        super().__init__(http_sink(base_url, token=token),
                         run_name=name, session=session, tags=tags)
