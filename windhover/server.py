"""Windhover HTTP server — observes a LangGraph graph and any app that traces to it.

Runs execute on a worker thread with a DB-sink tracer, so a full span tree
(nodes -> LLM/tool children, tokens, cost) persists even if the browser leaves.
A background watcher re-extracts topology in a subprocess when the graph's source
changes and pushes it to the UI (the "living graph"). Runs use the imported graph
(restart to run new code); the *view* always reflects current-on-disk topology.
"""
from __future__ import annotations
import sys, json, time, hashlib, threading, queue, subprocess, importlib, traceback
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .store import Store
from .tracer import SpanBuilder, db_sink, apply_to_store

cfg = Config.from_env()
store = Store(cfg.db_path)
STATIC = Path(__file__).parent / "static"

graph = None
if cfg.graph_ref:
    sys.path.insert(0, cfg.graph_dir)
    _m, _a = cfg.graph_ref.split(":")
    graph = getattr(importlib.import_module(_m), _a)

app = FastAPI(title="Windhover")


# ---- topology manager (subprocess extraction, mtime-cached) ---------------
class Topo:
    def __init__(self):
        self.lock = threading.Lock()
        self.sig = None
        self.data = {"topology": {"nodes": [], "edges": []}, "schema": {}, "sources": {}}
        self.hash = ""
        self.version = 0

    def _signature(self) -> float:
        if not cfg.graph_ref:
            return 0.0
        try:
            return max((p.stat().st_mtime for p in Path(cfg.graph_dir).glob("*.py")),
                       default=0.0)
        except Exception:
            return 0.0

    def refresh(self, force=False) -> None:
        if not cfg.graph_ref:
            return
        sig = self._signature()
        with self.lock:
            if not force and sig == self.sig:
                return
            self.sig = sig
        try:
            out = subprocess.run(
                [sys.executable, "-m", "windhover.extract", cfg.graph_ref, cfg.graph_dir],
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
            return {**self.data["topology"], "graph": cfg.graph_ref, "hash": self.hash,
                    "version": self.version, "xray": self.data.get("topology_xray")}

    def schema(self) -> dict:
        with self.lock:
            return self.data.get("schema", {})

    def sources(self) -> dict:
        with self.lock:
            return self.data.get("sources", {})


TOPO = Topo()
TOPO.refresh(force=True)


def _watch_loop():
    while cfg.watch and cfg.graph_ref:
        time.sleep(2)
        try:
            TOPO.refresh()
        except Exception:
            pass


if cfg.watch and cfg.graph_ref:
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


def _template(schema: dict) -> dict:
    out = {}
    for k, p in (schema.get("properties") or {}).items():
        out[k] = {"array": [], "object": {}, "string": "", "integer": 0,
                  "number": 0, "boolean": False}.get(p.get("type"))
    return out


# ---- endpoints ------------------------------------------------------------
@app.get("/api/graph")
def api_graph():
    return JSONResponse(TOPO.get())


@app.get("/api/schema")
def api_schema():
    s = TOPO.schema()
    return JSONResponse({"schema": s, "template": _template(s)})


def _sse(ev: str, data: dict) -> str:
    return f"event: {ev}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/run")
async def api_run(request: Request):
    if graph is None:
        return JSONResponse({"error": "no local graph (WINDHOVER_GRAPH unset)"}, 400)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    session = payload.pop("_session", None)
    tags = payload.pop("_tags", None)
    tracer = SpanBuilder(db_sink(store), run_name=cfg.graph_ref, session=session, tags=tags)
    run_id = tracer.run_id
    q: "queue.Queue" = queue.Queue()

    def worker():
        try:
            interrupted = False
            for update in graph.stream(payload, config={"callbacks": [tracer]},
                                       stream_mode="updates"):
                for node in update:
                    if node == "__interrupt__":
                        interrupted = True
                        q.put(("interrupt", {"run_id": run_id}))
                    else:
                        q.put(("node", {"node": node}))
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
def api_sessions(limit: int = 100):
    return JSONResponse(store.sessions(limit=limit))


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
    if not name:
        return JSONResponse({"error": "name must be non-empty"}, 400)
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
def api_node_source(name: str):
    src = TOPO.sources().get(name)
    if src:
        return JSONResponse(src)
    return JSONResponse({"error": "no source available for this node "
                        "(external-only run, or a runnable inspect can't trace)"}, 404)


@app.get("/api/stats")
def api_stats(days: int = 30):
    return JSONResponse(store.stats(days=max(1, min(days, 365))))


@app.get("/api/events")
async def api_events(request: Request):
    async def gen():
        import asyncio
        seen = -1
        yield _sse("hello", {"version": TOPO.version})
        while True:
            if await request.is_disconnected():
                break
            if TOPO.version != seen:
                seen = TOPO.version
                yield _sse("topology", {"version": seen})
            else:
                yield ": ping\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(gen(), media_type="text/event-stream")


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
