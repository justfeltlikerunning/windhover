"""Windhover HTTP server — observes a LangGraph graph and any app that traces to it.

Runs execute on a worker thread with a DB-sink tracer, so a full span tree
(nodes -> LLM/tool children, tokens, cost) persists even if the browser leaves.
A background watcher re-extracts topology in a subprocess when the graph's source
changes and pushes it to the UI (the "living graph"). Runs use the imported graph
(restart to run new code); the *view* always reflects current-on-disk topology.
"""
from __future__ import annotations
import os, sys, json, time, hashlib, threading, queue, subprocess, importlib, traceback
import uuid as uuid_mod
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Config
from .store import Store
from .tracer import SpanBuilder, db_sink, apply_to_store, _trunc
from . import artifacts, push

cfg = Config.from_env()
store = Store(cfg.db_path)
STATIC = Path(__file__).parent / "static"

GRAPHS: dict = {}          # name -> compiled graph
if cfg.graphs:
    sys.path.insert(0, cfg.graph_dir)
    for _name, _ref in cfg.graphs:
        try:
            _m, _a = _ref.split(":")
            GRAPHS[_name] = getattr(importlib.import_module(_m), _a)
        except Exception as _e:
            print(f"[windhover] failed to import graph {_name} ({_ref}): {_e}")


def _default_name() -> str:
    return next(iter(GRAPHS), "")


def _graph_for(name=None):
    """Resolve a graph by registry name; no name -> the first graph."""
    if name:
        return GRAPHS.get(name)
    return next(iter(GRAPHS.values()), None)

app = FastAPI(title="Windhover")


def _auth_ok(token: str, path: str, auth_header: str, query_token: str) -> bool:
    """True when the request may proceed. Only /api is gated; static/UI stay
    open (they contain no data — every payload comes through /api)."""
    if not token or not path.startswith("/api"):
        return True
    supplied = (auth_header or "").strip()
    if supplied.lower().startswith("bearer "):
        supplied = supplied[7:].strip()
    return supplied == token or (query_token or "") == token


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    if not _auth_ok(cfg.token, request.url.path,
                    request.headers.get("authorization", ""),
                    request.query_params.get("token", "")):
        return JSONResponse({"error": "unauthorized — supply WINDHOVER_TOKEN as "
                            "Bearer header or ?token="}, 401)
    return await call_next(request)


# ---- topology manager (subprocess extraction, mtime-cached) ---------------
class Topo:
    def __init__(self, name: str, ref: str):
        self.name, self.ref = name, ref
        self.lock = threading.Lock()
        self.sig = None
        self.data = {"topology": {"nodes": [], "edges": []}, "schema": {}, "sources": {}}
        self.hash = ""
        self.version = 0

    def _signature(self) -> float:
        try:
            return max((p.stat().st_mtime for p in Path(cfg.graph_dir).glob("*.py")),
                       default=0.0)
        except Exception:
            return 0.0

    def refresh(self, force=False) -> None:
        sig = self._signature()
        with self.lock:
            if not force and sig == self.sig:
                return
            self.sig = sig
        try:
            out = subprocess.run(
                [sys.executable, "-m", "windhover.extract", self.ref, cfg.graph_dir],
                capture_output=True, text=True, timeout=20,
                cwd=str(Path(__file__).parent.parent))
            data = json.loads(out.stdout)
        except Exception:
            return
        h = hashlib.sha1(json.dumps(data["topology"], sort_keys=True).encode()).hexdigest()[:12]
        with self.lock:
            changed = h != self.hash
            self.data, self.hash = data, h
            if changed:
                self.version += 1

    def get(self) -> dict:
        with self.lock:
            return {**self.data["topology"], "graph": self.name, "hash": self.hash,
                    "version": self.version, "xray": self.data.get("topology_xray")}

    def schema(self) -> dict:
        with self.lock:
            return self.data.get("schema", {})

    def sources(self) -> dict:
        with self.lock:
            return self.data.get("sources", {})


TOPOS: dict = {name: Topo(name, ref) for name, ref in cfg.graphs if name in GRAPHS}
for _t in TOPOS.values():
    _t.refresh(force=True)


def _topo_for(name=None):
    if name:
        return TOPOS.get(name)
    return next(iter(TOPOS.values()), None)


def _watch_loop():
    while cfg.watch and TOPOS:
        time.sleep(2)
        for tp in TOPOS.values():
            try:
                tp.refresh()
            except Exception:
                pass


if cfg.watch and TOPOS:
    threading.Thread(target=_watch_loop, daemon=True).start()


def _retention_loop():
    while True:
        try:
            res = store.prune(cfg.retention_days)
            if res["pruned_runs"]:
                print(f"[windhover] retention: pruned {res['pruned_runs']} runs "
                      f"older than {cfg.retention_days}d")
        except Exception as e:
            print(f"[windhover] retention error: {e}")
        time.sleep(6 * 3600)


if cfg.retention_days > 0:
    threading.Thread(target=_retention_loop, daemon=True).start()


def _fire_webhook(summary: dict) -> None:
    """POST a compact alert when a run needs attention. Fire-and-forget.
    Per-graph overrides (WINDHOVER_WEBHOOK "name=url" entries) win over the
    default URL; a graph with neither stays silent."""
    if summary.get("status") not in ("error", "interrupted"):
        return
    def _post():
        try:
            import urllib.request
            run = store.run_detail(summary["id"]) or summary
            hook = cfg.webhook_map.get(run.get("graph") or _default_name()) or cfg.webhook
            if not hook:
                return
            body = {
                "source": "windhover", "graph": run.get("graph") or _default_name(),
                "run_id": summary["id"], "status": summary["status"],
                "duration_ms": summary.get("duration_ms"),
                "session": run.get("session"), "thread_id": run.get("thread_id"),
                "error": (str(summary.get("error") or "").strip().splitlines() or [None])[-1],
                "text": f"[windhover] run {summary['id']} {summary['status']}"
                        f" ({run.get('graph') or _default_name()})",
            }
            req = urllib.request.Request(
                hook, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
    threading.Thread(target=_post, daemon=True).start()


def _push_alert(summary: dict) -> None:
    """Send a Web Push notification when a run needs attention. Fire-and-forget."""
    if summary.get("status") not in ("error", "interrupted"):
        return
    run = store.run_detail(summary["id"]) or summary
    graph = run.get("graph") or _default_name()
    err = (str(summary.get("error") or "").strip().splitlines() or [""])[-1]
    title = f"Windhover — run {summary['status']}"
    body = f"{graph} · {summary['id']}" + (f"\n{err}" if err else "")
    tag, url = summary["id"], f"/#run={summary['id']}"
    if summary["status"] == "interrupted":
        try:
            n = store.awaiting_count()
        except Exception:
            n = 1
        if n > 1:  # several await — land on the fleet queue, not one run
            title = f"Windhover — {n} runs awaiting approval"
            body = f"latest: {graph} · {summary['id']}"
            tag, url = "windhover-awaiting", "/#fleet"
    push.send_to_all(store, cfg, {
        "title": title, "body": body, "tag": tag, "url": url,
        "status": summary["status"],
    })


def _on_run_closed(summary: dict) -> None:
    if cfg.webhook or cfg.webhook_map:
        _fire_webhook(summary)
    if cfg.push_enabled:
        _push_alert(summary)


if cfg.webhook or cfg.webhook_map or cfg.push_enabled:
    store.on_run_closed = _on_run_closed


def _digest_loop() -> None:
    """Daily summary push at WINDHOVER_DIGEST (local HH:MM). Quiet days skip."""
    hh, mm = (int(x) for x in cfg.digest.split(":"))
    while True:
        now = time.localtime()
        target = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hh, mm, 0,
                              0, 0, -1))
        if target <= time.time():
            target += 86400
        time.sleep(max(1, target - time.time()))
        try:
            payload = push.digest_summary(store.overview(days=1,
                                          serving=tuple(GRAPHS.keys())))
            if payload:
                push.send_to_all(store, cfg, payload)
        except Exception as e:
            print(f"[digest] {e}")
        time.sleep(61)  # step past the minute so one schedule fires once


if cfg.digest and cfg.push_enabled:
    try:
        int(cfg.digest.split(":")[0]); int(cfg.digest.split(":")[1])
        threading.Thread(target=_digest_loop, daemon=True).start()
    except Exception:
        print(f"[digest] invalid WINDHOVER_DIGEST={cfg.digest!r} — expected HH:MM")


def _template(schema: dict) -> dict:
    out = {}
    for k, p in (schema.get("properties") or {}).items():
        out[k] = {"array": [], "object": {}, "string": "", "integer": 0,
                  "number": 0, "boolean": False}.get(p.get("type"))
    return out


# ---- endpoints ------------------------------------------------------------
@app.get("/api/graphs")
def api_graphs():
    return JSONResponse({"graphs": list(GRAPHS.keys()), "default": _default_name(),
                         "version": __version__})


@app.get("/api/graph")
def api_graph(graph: str = None):
    tp = _topo_for(graph)
    if tp is None:
        return JSONResponse({"nodes": [], "edges": [], "graph": "", "hash": "",
                             "version": 0, "xray": None})
    return JSONResponse(tp.get())


@app.get("/api/schema")
def api_schema(graph: str = None):
    tp = _topo_for(graph)
    if tp is None:
        return JSONResponse({"schema": {}, "template": {},
                             "context_schema": {}, "context_template": {}})
    s = tp.schema()
    with tp.lock:
        ctx = tp.data.get("context_schema") or {}
    return JSONResponse({"schema": s, "template": _template(s),
                         "context_schema": ctx, "context_template": _template(ctx)})


def _sse(ev: str, data: dict) -> str:
    return f"event: {ev}\ndata: {json.dumps(data)}\n\n"


def _pending_next(gr, config) -> list:
    """Nodes still pending on the LATEST checkpoint of this thread. Must query
    by thread only — carrying a checkpoint_id (fork/resume-from) would read the
    HISTORICAL checkpoint, whose next-nodes are always non-empty."""
    try:
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if not thread_id:
            return []
        st = gr.get_state({"configurable": {"thread_id": thread_id}})
        return list(st.next or [])
    except Exception:
        return []


def _stream_execution(gr, graph_input, config, tracer, stream_kwargs=None):
    """Shared SSE executor for /api/run and /api/threads/…/resume. Detects both
    dynamic interrupts (__interrupt__ updates) and static breakpoints (stream
    ends with pending next-nodes) and corrects the run status accordingly."""
    run_id = tracer.run_id
    seq_extra = [0]
    q: "queue.Queue" = queue.Queue()

    def worker():
        try:
            interrupted = False
            for mode, chunk in gr.stream(graph_input, config=config,
                                            stream_mode=["updates", "custom"],
                                            **(stream_kwargs or {})):
                if mode == "custom":
                    # get_stream_writer() output from inside a node -> live progress
                    q.put(("progress", {"data": _trunc(chunk, 600)}))
                    continue
                cached = bool((chunk.get("__metadata__") or {}).get("cached")) \
                    if isinstance(chunk, dict) else False
                for node in chunk:
                    if node == "__metadata__":
                        continue
                    if node == "__interrupt__":
                        interrupted = True
                        q.put(("interrupt", {"run_id": run_id}))
                        continue
                    if cached:
                        # cache hits fire NO callbacks — synthesize the span so
                        # the trace shows the node ran (from cache)
                        now = int(time.time() * 1000)
                        store.add_span({"id": uuid_mod.uuid4().hex[:12],
                                        "run_id": run_id, "parent_id": None,
                                        "seq": 10_000 + seq_extra[0], "type": "node",
                                        "name": node, "status": "ok",
                                        "started_ms": now, "ended_ms": now,
                                        "offset_ms": None, "dur_ms": 0,
                                        "output": _trunc(chunk.get(node), 2000),
                                        "params": {"cached": True}})
                        seq_extra[0] += 1
                    q.put(("node", {"node": node, "cached": cached or None}))
            if not interrupted and getattr(gr, "checkpointer", None) is not None:
                nxt = _pending_next(gr, config)
                if nxt:
                    interrupted = True
                    q.put(("interrupt", {"run_id": run_id, "next": nxt}))
            if interrupted:
                # correct the status ONLY if the tracer didn't already mark it
                # (dynamic interrupts close as interrupted themselves — a second
                # close here would double-fire the webhook)
                row = store.run_detail(run_id)
                if row and row.get("status") != "interrupted":
                    store.close_run(run_id, "interrupted", int(time.time() * 1000))
            q.put(("interrupted" if interrupted else "done", {"run_id": run_id}))
        except Exception as e:
            # tracer already recorded the error span/close; surface to the client too
            q.put(("error", {"message": str(e), "trace": traceback.format_exc()[-600:]}))
        q.put((None, None))

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        yield _sse("start", {"run_id": run_id})
        while True:
            ev, data = q.get()
            if ev is None:
                break
            yield _sse(ev, data)
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/run")
async def api_run(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        # non-dict graph inputs (e.g. functional-API entrypoints taking a
        # scalar) carry no _directives — run them as-is on the default graph
        gr = _graph_for(_default_name())
        if gr is None:
            return JSONResponse({"error": "no local graph"}, 400)
        tracer = SpanBuilder(db_sink(store), run_name=_default_name())
        config = {"callbacks": [tracer]}
        if getattr(gr, "checkpointer", None) is not None:
            config["configurable"] = {"thread_id": tracer.run_id}
        return _stream_execution(gr, payload, config, tracer)
    gname = payload.pop("_graph", None) or _default_name()
    gr = _graph_for(gname)
    if gr is None:
        return JSONResponse({"error": "no local graph (WINDHOVER_GRAPH unset "
                            "or unknown graph name)"}, 400)
    session = payload.pop("_session", None)
    tags = payload.pop("_tags", None)
    thread = payload.pop("_thread", None)
    pause_before = payload.pop("_interrupt_before", None)
    pause_after = payload.pop("_interrupt_after", None)
    extra_conf = payload.pop("_configurable", None)
    tracer = SpanBuilder(db_sink(store), run_name=gname, session=session, tags=tags)
    config = {"callbacks": [tracer]}
    if thread or getattr(gr, "checkpointer", None) is not None:
        # a checkpointed graph needs a thread; default to the run id so
        # time-travel works out of the box
        config["configurable"] = {"thread_id": thread or tracer.run_id}
    if isinstance(extra_conf, dict) and extra_conf:
        config.setdefault("configurable", {}).update(extra_conf)
    sk = {}
    if pause_before:
        sk["interrupt_before"] = [str(n) for n in pause_before]
    if pause_after:
        sk["interrupt_after"] = [str(n) for n in pause_after]
    return _stream_execution(gr, payload, config, tracer, sk)


@app.post("/api/threads/{thread_id}/resume")
async def api_thread_resume(request: Request, thread_id: str, graph: str = None):
    """Human-in-the-loop: answer an interrupt (Command(resume=…)), redirect
    (Command(goto=…)), continue past a static breakpoint (no body), or fork
    from an earlier checkpoint (checkpoint_id)."""
    gr = _graph_for(graph)
    if gr is None or getattr(gr, "checkpointer", None) is None:
        return JSONResponse({"error": "no local graph with a checkpointer"}, 400)
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        from langgraph.types import Command
    except Exception:
        return JSONResponse({"error": "langgraph.types.Command unavailable"}, 500)
    if "value" in body:
        graph_input = Command(resume=body["value"])
    elif body.get("goto"):
        graph_input = Command(goto=str(body["goto"]), update=body.get("update"))
    else:
        graph_input = None  # plain continue (static breakpoint / after state edit)
    tracer = SpanBuilder(db_sink(store), run_name=graph or _default_name(),
                         session=body.get("_session"),
                         tags=(body.get("_tags") or []) + ["resume"])
    configurable = {"thread_id": thread_id}
    if body.get("checkpoint_id"):
        configurable["checkpoint_id"] = str(body["checkpoint_id"])
    config = {"callbacks": [tracer], "configurable": configurable}
    return _stream_execution(gr, graph_input, config, tracer)


@app.post("/api/threads/{thread_id}/state")
async def api_thread_update_state(request: Request, thread_id: str, graph: str = None):
    """Human-in-the-loop: edit state at the current (or a given) checkpoint —
    LangGraph's update_state. Follow with …/resume to continue on the edit."""
    gr = _graph_for(graph)
    if gr is None or getattr(gr, "checkpointer", None) is None:
        return JSONResponse({"error": "no local graph with a checkpointer"}, 400)
    body = await request.json()
    values = body.get("values")
    if not isinstance(values, dict):
        return JSONResponse({"error": "requires values (object of state keys)"}, 400)
    configurable = {"thread_id": thread_id}
    if body.get("checkpoint_id"):
        configurable["checkpoint_id"] = str(body["checkpoint_id"])
    try:
        new_cfg = gr.update_state({"configurable": configurable}, values,
                                  as_node=body.get("as_node"))
        return JSONResponse({"ok": True, "checkpoint_id":
                             (new_cfg.get("configurable") or {}).get("checkpoint_id")})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 400)


@app.post("/api/ingest")
async def api_ingest(request: Request):
    ev = await request.json()
    apply_to_store(store, ev, source="ingest")
    return {"ok": True, "run_id": ev.get("run_id")}


def _run_filters(limit: int, offset: int, q, status, graph, session, tag,
                 bookmarked, since_ms, until_ms) -> dict:
    return dict(limit=max(1, min(limit, 500)), offset=max(0, offset),
                q=q or None, status=status or None, graph=graph or None,
                session=session or None, tag=tag or None,
                bookmarked=bool(bookmarked) or None,
                since_ms=since_ms, until_ms=until_ms)


@app.get("/api/runs")
def api_runs(limit: int = 50, offset: int = 0, q: str = None, status: str = None,
             graph: str = None, session: str = None, tag: str = None,
             bookmarked: int = 0, since_ms: int = None, until_ms: int = None):
    return JSONResponse(store.runs(**_run_filters(
        limit, offset, q, status, graph, session, tag, bookmarked, since_ms, until_ms)))


@app.get("/api/sessions")
def api_sessions(limit: int = 100, graph: str = None):
    return JSONResponse(store.sessions(limit=limit, graph=graph or None))


@app.get("/api/runs/{run_id}")
def api_run_detail(run_id: str):
    d = store.run_detail(run_id)
    return JSONResponse(d) if d else JSONResponse({"error": "not found"}, 404)


@app.patch("/api/runs/{run_id}")
async def api_run_patch(request: Request, run_id: str):
    body = await request.json()
    tags = body.get("tags")
    if tags is not None and not isinstance(tags, list):
        return JSONResponse({"error": "tags must be a list"}, 400)
    ok = store.update_run_meta(run_id, tags=tags, bookmarked=body.get("bookmarked"))
    return JSONResponse({"ok": ok}) if ok else JSONResponse({"error": "not found"}, 404)


@app.post("/api/runs/{run_id}/scores")
async def api_score_add(request: Request, run_id: str):
    body = await request.json()
    try:
        name, value = str(body["name"]).strip(), float(body["value"])
    except (KeyError, TypeError, ValueError):
        return JSONResponse({"error": "requires name (str) and value (number)"}, 400)
    import math
    if not name or not math.isfinite(value):
        return JSONResponse({"error": "name must be non-empty and value finite "
                            "(NaN/Infinity would corrupt API JSON)"}, 400)
    sc = store.add_score(run_id, name, value, comment=body.get("comment"),
                         source=body.get("source", "api"))
    return JSONResponse(sc) if sc else JSONResponse({"error": "run not found"}, 404)


@app.delete("/api/scores/{score_id}")
def api_score_delete(score_id: str):
    return JSONResponse({"ok": store.delete_score(score_id)})


@app.get("/api/export")
def api_export(format: str = "json", limit: int = 10000, q: str = None,
               status: str = None, graph: str = None, session: str = None,
               tag: str = None, bookmarked: int = 0,
               since_ms: int = None, until_ms: int = None):
    data = store.runs(**{**_run_filters(limit, 0, q, status, graph, session, tag,
                                        bookmarked, since_ms, until_ms),
                         "limit": max(1, min(limit, 100_000))})["runs"]
    if format == "csv":
        import csv, io
        cols = ["id", "graph", "source", "status", "session", "tags", "started_ms",
                "duration_ms", "node_count", "llm_calls", "prompt_tokens",
                "completion_tokens", "total_tokens", "cost_usd", "error"]
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(cols)
        for r in data:
            w.writerow([json.dumps(r[k]) if isinstance(r.get(k), (list, dict))
                        else r.get(k) for k in cols])
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=windhover-runs.csv"})
    return JSONResponse(data)


@app.get("/api/nodes/{name}")
def api_node(name: str, limit: int = 25):
    return JSONResponse(store.node_history(name, limit))


@app.get("/api/nodes/{name}/source")
def api_node_source(name: str, graph: str = None):
    tp = _topo_for(graph)
    src = (tp.sources() if tp else {}).get(name)
    if src:
        return JSONResponse(src)
    return JSONResponse({"error": "no source available for this node "
                        "(external-only run, or a runnable inspect can't trace)"}, 404)


@app.get("/api/threads/{thread_id}/history")
def api_thread_history(thread_id: str, limit: int = 80, graph: str = None):
    """Time-travel: LangGraph checkpoint history for a thread (local graph
    with a checkpointer only)."""
    gr = _graph_for(graph)
    if gr is None or getattr(gr, "checkpointer", None) is None:
        return JSONResponse({"error": "no local graph with a checkpointer"}, 404)
    steps = []
    try:
        for st in gr.get_state_history({"configurable": {"thread_id": thread_id}}):
            md = st.metadata or {}
            steps.append({
                "checkpoint_id": ((st.config or {}).get("configurable") or {}).get("checkpoint_id"),
                "step": md.get("step"),
                "source": md.get("source"),
                "writes": _trunc(md.get("writes"), 2000) if md.get("writes") is not None else None,
                "next": list(st.next or []),
                "values": _trunc(st.values, 3000),
                "created_at": str(getattr(st, "created_at", "") or "") or None,
            })
            if len(steps) >= limit:
                break
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)
    return JSONResponse({"thread_id": thread_id, "steps": steps})


@app.get("/api/memory/namespaces")
def api_memory_namespaces(graph: str = None):
    """LangGraph long-term memory (Store) browser — namespaces."""
    gr = _graph_for(graph)
    st = getattr(gr, "store", None) if gr is not None else None
    if st is None:
        return JSONResponse({"error": "no local graph with a store"}, 404)
    try:
        return JSONResponse({"namespaces": [list(ns) for ns in st.list_namespaces()]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.get("/api/memory/items")
def api_memory_items(namespace: str, query: str = None, limit: int = 50,
                     graph: str = None):
    """Items in one namespace (dot-separated); optional semantic/text query."""
    gr = _graph_for(graph)
    st = getattr(gr, "store", None) if gr is not None else None
    if st is None:
        return JSONResponse({"error": "no local graph with a store"}, 404)
    try:
        ns = tuple(namespace.split(".")) if namespace else ()
        items = st.search(ns, query=query, limit=max(1, min(limit, 200)))
        return JSONResponse({"namespace": namespace, "items": [
            {"key": i.key, "value": _trunc(i.value, 3000),
             "created_at": str(getattr(i, "created_at", "") or "") or None,
             "updated_at": str(getattr(i, "updated_at", "") or "") or None,
             "score": getattr(i, "score", None)} for i in items]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.get("/api/datasets")
def api_datasets():
    return JSONResponse([{k: d[k] for k in ("id", "name", "n_items", "created_ms")}
                         for d in store.datasets()])


@app.post("/api/datasets")
async def api_dataset_add(request: Request):
    body = await request.json()
    name, items = body.get("name"), body.get("items")
    if not name or not isinstance(items, list) or not items:
        return JSONResponse({"error": "requires name and a non-empty items list "
                            "[{input: {...}, expected?: ...}]"}, 400)
    for it in items:
        if not isinstance(it, dict) or "input" not in it:
            return JSONResponse({"error": "every item needs an input object"}, 400)
    ds = store.add_dataset(str(name), items)
    return JSONResponse({"id": ds["id"], "name": ds["name"], "n_items": len(items)})


@app.delete("/api/datasets/{ds_id}")
def api_dataset_delete(ds_id: str):
    return JSONResponse({"ok": store.delete_dataset(ds_id)})


def _expected_score(expected, output) -> float:
    """Naive match: exact for scalars, substring against the JSON otherwise."""
    try:
        if isinstance(expected, (int, float, bool)):
            return 1.0 if any(v == expected for v in
                              (output.values() if isinstance(output, dict) else [output])) else 0.0
        return 1.0 if str(expected) in json.dumps(output, default=str) else 0.0
    except Exception:
        return 0.0


@app.post("/api/datasets/{ds_id}/run")
def api_dataset_run(ds_id: str, graph: str = None):
    """Batch-eval: run the local graph over every dataset item; expected values
    become an expected_match score on each run. Fire-and-forget worker."""
    gname = graph or _default_name()
    gr = _graph_for(gname)
    if gr is None:
        return JSONResponse({"error": "no local graph (WINDHOVER_GRAPH unset)"}, 400)
    ds = store.dataset(ds_id)
    if not ds:
        return JSONResponse({"error": "dataset not found"}, 404)
    session = f"eval:{ds['name']}:{int(time.time())}"

    def worker():
        for i, item in enumerate(ds["items"]):
            tracer = SpanBuilder(db_sink(store), run_name=gname,
                                 session=session, tags=[f"dataset:{ds['name']}"])
            config = {"callbacks": [tracer]}
            if getattr(gr, "checkpointer", None) is not None:
                config["configurable"] = {"thread_id": tracer.run_id}
            try:
                out = gr.invoke(dict(item["input"]), config=config)
            except Exception:
                continue  # tracer already recorded the error run
            if "expected" in item:
                store.add_score(tracer.run_id, "expected_match",
                                _expected_score(item["expected"], out),
                                comment=f"item {i}", source="dataset")

    threading.Thread(target=worker, daemon=True).start()
    return JSONResponse({"ok": True, "session": session, "items": len(ds["items"])})


@app.get("/api/stats")
def api_stats(days: int = 30, graph: str = None):
    return JSONResponse(store.stats(days=max(1, min(days, 365)), graph=graph or None))


@app.get("/api/overview")
def api_overview(days: int = 7):
    """Fleet view: per-graph health + runs awaiting a human, across ALL graphs."""
    return JSONResponse(store.overview(days=max(1, min(days, 90)),
                                       serving=tuple(GRAPHS.keys())))


@app.get("/api/runs/{run_id}/artifacts")
def api_artifacts(run_id: str):
    """Files this run's code recorded in its outputs (reports, charts, exports)."""
    run = store.run_detail(run_id)
    if not run:
        return JSONResponse({"error": "unknown run"}, 404)
    return JSONResponse({"artifacts": artifacts.run_artifacts(run)})


@app.get("/api/runs/{run_id}/artifact")
def api_artifact(run_id: str, path: str):
    """Serve one artifact — only paths recorded by THIS run are resolvable."""
    run = store.run_detail(run_id)
    if not run:
        return JSONResponse({"error": "unknown run"}, 404)
    real = artifacts.resolve(run, path)
    if not real:
        return JSONResponse({"error": "not an artifact of this run (or missing on this host)"}, 404)
    info = artifacts.classify(path)
    headers = {"X-Content-Type-Options": "nosniff"}
    if not info["inline"]:
        headers["Content-Disposition"] = f'attachment; filename="{os.path.basename(real)}"'
    return FileResponse(real, media_type=info["mime"], headers=headers)


@app.get("/api/events")
async def api_events(request: Request):
    async def gen():
        import asyncio
        seen = -1
        def _ver():
            return sum(tp.version for tp in TOPOS.values())
        yield _sse("hello", {"version": _ver()})
        while True:
            if await request.is_disconnected():
                break
            if _ver() != seen:
                seen = _ver()
                yield _sse("topology", {"version": seen})
            else:
                yield ": ping\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/push/config")
def push_config():
    """Client bootstrap: is push on, and the VAPID app-server key to subscribe with."""
    return JSONResponse({"enabled": cfg.push_enabled and push.available(),
                         "publicKey": cfg.vapid_public})


@app.post("/api/push/subscribe")
async def push_subscribe(request: Request):
    body = await request.json()
    sub = body.get("subscription") or body
    if not isinstance(sub, dict) or not sub.get("endpoint"):
        return JSONResponse({"error": "missing subscription.endpoint"}, 400)
    store.add_push_subscription(sub, label=str(body.get("label", "")))
    return JSONResponse({"ok": True})


@app.post("/api/push/unsubscribe")
async def push_unsubscribe(request: Request):
    body = await request.json()
    endpoint = (body.get("endpoint") or (body.get("subscription") or {}).get("endpoint"))
    if endpoint:
        store.remove_push_subscription(endpoint)
    return JSONResponse({"ok": True})


@app.post("/api/push/test")
def push_test():
    if not (cfg.push_enabled and push.available()):
        return JSONResponse({"error": "web push not configured"}, 400)
    n = len(store.push_subscriptions())
    push.send_to_all(store, cfg, {
        "title": "Windhover", "body": "Test alert — notifications are working.",
        "tag": "windhover-test", "url": "/",
    })
    return JSONResponse({"ok": True, "subscriptions": n})


@app.get("/sw.js")
def service_worker():
    # served from root so its scope covers the whole app
    return FileResponse(STATIC / "sw.js", media_type="application/javascript",
                        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})


@app.get("/manifest.json")
def manifest():
    return FileResponse(STATIC / "manifest.json")


@app.get("/")
def index():
    # no-cache so UI upgrades take effect on next load (vendor assets still cache)
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
