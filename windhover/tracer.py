"""Windhover tracer — one LangChain callback, two sinks.

`SpanBuilder` turns LangGraph/LangChain callbacks into a run + span tree and hands
each event to a `sink`. In-process runs use a DB sink; external apps use
`WindhoverTracer` (HTTP sink) — identical span logic, so the trace looks the same
wherever it came from.

Callback realities handled:
  * LangGraph exposes the node name only on `on_chain_start` (metadata.langgraph_node),
    never on end → we map langchain run_id -> our span.
  * Token usage lives in either `llm_output.token_usage` (OpenAI-style) or a
    generation's `usage_metadata` (input_tokens/output_tokens) — we read both.
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


def _trunc(v: Any, n: int = 4000) -> Any:
    s = json.dumps(v, default=str)
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
    # OpenAI-style aggregate
    out = getattr(response, "llm_output", None) or {}
    for key in ("token_usage", "usage"):
        u = out.get(key) if isinstance(out, dict) else None
        if u:
            return (u.get("prompt_tokens") or u.get("input_tokens"),
                    u.get("completion_tokens") or u.get("output_tokens"), None)
    # per-generation usage_metadata (chat models)
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
    def __init__(self, sink: Callable[[dict], None], run_name: str = "external",
                 session: Optional[str] = None, tags: Optional[list] = None):
        self.sink = sink
        self.run_name = run_name
        self.session = session
        self.tags = tags
        self.run_id = uuid.uuid4().hex[:12]
        self._root = None            # langchain run_id of the graph root
        self._t0: Optional[float] = None
        self._open: dict[str, dict] = {}   # langchain run_id -> pending span
        self._span_of: dict[str, str] = {} # langchain run_id -> our span id (for parent links)
        self._seq = 0
        self._interrupted = False

    # -- infra --
    def _emit(self, ev: dict) -> None:
        try:
            self.sink(ev)
        except Exception:
            pass  # observability must never break the graph

    def _rel(self, t: float) -> int:
        return int((t - (self._t0 or t)) * 1000)

    def _finish_span(self, info: dict, *, output=None, status="ok", error=None,
                     model=None, pt=None, ct=None, usage_detail=None):
        now = time.time()
        sid = info["span_id"]
        seq = info.get("seq")
        if seq is None:
            seq = self._seq
            self._seq += 1
        self._emit({"kind": "span", "id": sid, "run_id": self.run_id,
                    "parent_id": self._span_of.get(info.get("parent")),
                    "seq": seq, "type": info["type"], "name": info["name"],
                    "status": status, "started_ms": int(info["t"] * 1000),
                    "ended_ms": int(now * 1000), "offset_ms": self._rel(info["t"]),
                    "dur_ms": int((now - info["t"]) * 1000),
                    "input": info.get("input"), "output": output, "model": model,
                    "prompt_tokens": pt, "completion_tokens": ct,
                    "cost_usd": cost_of(model, pt, ct), "error": error,
                    "retries": info.get("retries"),
                    "ttft_ms": int(info["ttft"] * 1000) if info.get("ttft") is not None else None,
                    "usage_detail": usage_detail, "params": info.get("params")})

    # -- chains / nodes --
    _INTERNAL_TAG_PREFIXES = ("graph:", "langsmith:", "seq:", "langgraph_")

    def on_chain_start(self, serialized, inputs, *, run_id=None, parent_run_id=None,
                       metadata=None, tags=None, **kw):
        node = (metadata or {}).get("langgraph_node")
        if parent_run_id is None and self._root is None:
            self._root = run_id
            self._t0 = time.time()
            # standard LangChain idioms work from ANY app, no constructor needed:
            #   config={"metadata": {"windhover_session": ..., "windhover_tags": [...]},
            #           "tags": [...]}
            md = metadata or {}
            if md.get("windhover_run_name"):
                self.run_name = str(md["windhover_run_name"])
            if md.get("windhover_session"):
                self.session = str(md["windhover_session"])
            user_tags = [t for t in (tags or [])
                         if not str(t).startswith(self._INTERNAL_TAG_PREFIXES)]
            extra = md.get("windhover_tags") or []
            merged = list(dict.fromkeys(
                [*(self.tags or []), *map(str, extra), *map(str, user_tags)]))
            self.tags = merged or None
            self._emit({"kind": "run_open", "run_id": self.run_id, "graph": self.run_name,
                        "input": _trunc(inputs), "started_ms": int(self._t0 * 1000),
                        "session": self.session, "tags": self.tags,
                        "thread_id": md.get("thread_id")})
        if node:
            sid = uuid.uuid4().hex[:12]
            self._span_of[run_id] = sid
            self._open[run_id] = {"span_id": sid, "type": "node", "name": node,
                                  "t": time.time(), "parent": parent_run_id,
                                  "input": None}

    def on_chain_end(self, outputs, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if info:
            self._finish_span(info, output=_trunc(outputs))
        elif run_id == self._root:
            # LangGraph human-in-the-loop: a paused graph surfaces __interrupt__
            # in its final output — that's a pause awaiting Command(resume=...),
            # not a completion.
            interrupted = self._interrupted or (
                isinstance(outputs, dict) and "__interrupt__" in outputs)
            if interrupted and not self._interrupted:
                # marker not yet emitted by the GraphInterrupt path
                self._interrupted = True
                self._finish_span(
                    {"span_id": uuid.uuid4().hex[:12], "type": "interrupt",
                     "name": "interrupt", "t": time.time(), "parent": None,
                     "input": None},
                    output=_trunc(outputs.get("__interrupt__")))
            self._emit({"kind": "run_close", "run_id": self.run_id,
                        "status": "interrupted" if interrupted else "done",
                        "ended_ms": int(time.time() * 1000)})

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
            if info:
                self._finish_span(info, status="interrupted",
                                  output=_trunc(payload) if payload else None)
            if not self._interrupted:  # one marker span per pause
                self._interrupted = True
                now = time.time()
                self._emit({"kind": "span", "id": uuid.uuid4().hex[:12],
                            "run_id": self.run_id, "parent_id": None,
                            "seq": self._seq, "type": "interrupt", "name": "interrupt",
                            "status": "ok", "started_ms": int(now * 1000),
                            "ended_ms": int(now * 1000), "offset_ms": self._rel(now),
                            "dur_ms": 0, "input": None,
                            "output": _trunc(payload) if payload else None,
                            "model": None, "prompt_tokens": None,
                            "completion_tokens": None, "cost_usd": None, "error": None})
                self._seq += 1
            return
        err = _err_text(error)
        if info:
            self._finish_span(info, status="error", error=err)
        if run_id == self._root:
            self._emit({"kind": "run_close", "run_id": self.run_id, "status": "error",
                        "ended_ms": int(time.time() * 1000), "error": err})

    # -- LLMs --
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

    def _llm_start(self, serialized, prompt, run_id, parent_run_id, kw):
        sid = uuid.uuid4().hex[:12]
        self._span_of[run_id] = sid
        seq = self._seq
        self._seq += 1  # reserved now so streaming partials keep a stable order
        self._open[run_id] = {"span_id": sid, "type": "llm",
                              "name": _model_name(serialized, kw), "t": time.time(),
                              "parent": parent_run_id, "input": _trunc(prompt),
                              "seq": seq, "params": self._llm_params(kw)}

    def on_llm_start(self, serialized, prompts, *, run_id=None, parent_run_id=None, **kw):
        self._llm_start(serialized, prompts, run_id, parent_run_id, kw)

    def on_chat_model_start(self, serialized, messages, *, run_id=None, parent_run_id=None, **kw):
        flat = [[getattr(m, "content", str(m)) for m in conv] for conv in messages]
        self._llm_start(serialized, flat, run_id, parent_run_id, kw)

    def on_llm_end(self, response, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if not info:
            return
        pt, ct, detail = _usage(response)
        self._finish_span(info, output=_gen_text(response)[:4000],
                          model=info["name"], pt=pt, ct=ct, usage_detail=detail)

    def on_llm_error(self, error, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if info:
            self._finish_span(info, status="error", error=_err_text(error), model=info["name"])

    def on_llm_new_token(self, token, *, run_id=None, **kw):
        # streaming: first token stamps time-to-first-token; the growing text is
        # flushed as a partial span every ~0.5s so a live-tailed drawer shows
        # the model typing
        info = self._open.get(run_id)
        if info is None:
            return
        now = time.time()
        if "ttft" not in info:
            info["ttft"] = now - info["t"]
        info["buf"] = (info.get("buf") or "") + (token or "")
        if now - info.get("flushed", 0) >= 0.5 and info.get("buf"):
            info["flushed"] = now
            self._emit({"kind": "span", "id": info["span_id"], "run_id": self.run_id,
                        "parent_id": self._span_of.get(info.get("parent")),
                        "seq": info["seq"], "type": "llm", "name": info["name"],
                        "status": "running", "started_ms": int(info["t"] * 1000),
                        "ended_ms": None, "offset_ms": self._rel(info["t"]),
                        "dur_ms": int((now - info["t"]) * 1000),
                        "input": info.get("input"), "output": info["buf"][-4000:],
                        "model": info["name"], "prompt_tokens": None,
                        "completion_tokens": None, "cost_usd": None, "error": None,
                        "retries": info.get("retries"),
                        "ttft_ms": int(info["ttft"] * 1000),
                        "usage_detail": None, "params": info.get("params")})

    # -- retries (tenacity, e.g. rate limits) --
    def on_retry(self, retry_state, *, run_id=None, **kw):
        info = self._open.get(run_id)
        if info is not None:
            info["retries"] = getattr(retry_state, "attempt_number", None) or \
                              (info.get("retries") or 0) + 1

    # -- custom events (langchain dispatch_custom_event) --
    def on_custom_event(self, name, data, *, run_id=None, **kw):
        now = time.time()
        self._emit({"kind": "span", "id": uuid.uuid4().hex[:12], "run_id": self.run_id,
                    "parent_id": self._span_of.get(run_id),
                    "seq": self._seq, "type": "event", "name": str(name), "status": "ok",
                    "started_ms": int(now * 1000), "ended_ms": int(now * 1000),
                    "offset_ms": self._rel(now), "dur_ms": 0,
                    "input": None, "output": _trunc(data), "model": None,
                    "prompt_tokens": None, "completion_tokens": None,
                    "cost_usd": None, "error": None})
        self._seq += 1

    # -- retrievers (LangChain RAG) --
    def on_retriever_start(self, serialized, query, *, run_id=None, parent_run_id=None, **kw):
        sid = uuid.uuid4().hex[:12]
        self._span_of[run_id] = sid
        self._open[run_id] = {"span_id": sid, "type": "retriever",
                              "name": (serialized or {}).get("name") or "retriever",
                              "t": time.time(), "parent": parent_run_id,
                              "input": _trunc(query)}

    def on_retriever_end(self, documents, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if not info:
            return
        docs = list(documents or [])
        preview = [{"content": str(getattr(d, "page_content", d))[:300],
                    "metadata": _trunc(getattr(d, "metadata", None), 500)}
                   for d in docs[:8]]
        self._finish_span(info, output=_trunc({"count": len(docs), "documents": preview}))

    def on_retriever_error(self, error, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if info:
            self._finish_span(info, status="error", error=_err_text(error))

    # -- tools --
    def on_tool_start(self, serialized, input_str, *, run_id=None, parent_run_id=None, **kw):
        sid = uuid.uuid4().hex[:12]
        self._span_of[run_id] = sid
        self._open[run_id] = {"span_id": sid, "type": "tool",
                              "name": (serialized or {}).get("name", "tool"),
                              "t": time.time(), "parent": parent_run_id,
                              "input": _trunc(input_str)}

    def on_tool_end(self, output, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if info:
            self._finish_span(info, output=_trunc(str(output)[:4000]))

    def on_tool_error(self, error, *, run_id=None, **kw):
        info = self._open.pop(run_id, None)
        if info:
            self._finish_span(info, status="error", error=_err_text(error))


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


def http_sink(base_url: str) -> Callable[[dict], None]:
    import urllib.request
    url = base_url.rstrip("/") + "/api/ingest"

    def sink(ev: dict) -> None:
        req = urllib.request.Request(url, data=json.dumps(ev).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3)
    return sink


class WindhoverTracer(SpanBuilder):
    """Drop into any app: config={"callbacks": [WindhoverTracer("http://host:8090")]}."""
    def __init__(self, base_url: str, name: str = "external",
                 session: Optional[str] = None, tags: Optional[list] = None):
        super().__init__(http_sink(base_url), run_name=name, session=session, tags=tags)
