<p align="center"><img src="https://raw.githubusercontent.com/justfeltlikerunning/windhover/main/docs/logo.svg" width="112" alt="Windhover — a hovering kestrel"></p>
<h1 align="center">Windhover</h1>
<p align="center">
  <a href="https://pypi.org/project/windhover/"><img src="https://img.shields.io/pypi/v/windhover" alt="PyPI"></a>
  <a href="https://github.com/justfeltlikerunning/windhover/actions/workflows/ci.yml"><img src="https://github.com/justfeltlikerunning/windhover/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/license-MIT-8E2434" alt="MIT">
</p>

> *Windhover* — the old poetic name for the kestrel, the falcon that hangs motionless
> in the wind, watching everything below. This tool does the same for your agent graphs.

**Self-hosted, mobile-friendly observability for [LangGraph](https://github.com/langchain-ai/langgraph).**
Trace depth like LangSmith (LLM prompts, tokens, cost, latency — plus retrievers and
human-in-the-loop interrupts), run history, a timing waterfall, per-node stats, error
forensics down to the throwing source line — and a **living graph view** that auto-updates
when your code's topology changes. Point it at any compiled graph, or trace runs in from
your own app. No LangSmith account, no cloud tunnel, no fragile websocket. HTTP + SSE, MIT.

> Nothing about your graph's domain is baked in. Topology, the input form, and run
> outputs all come from the graph itself. Windhover observes — it never edits your graph.

| Living graph (parallel fan-out) | Trace drawer — retrievers, LLM calls, cost, state |
|---|---|
| ![Graph view](https://raw.githubusercontent.com/justfeltlikerunning/windhover/main/docs/graph.png) | ![Trace drawer](https://raw.githubusercontent.com/justfeltlikerunning/windhover/main/docs/trace.png) |

| Runs — search, tags, sessions, interrupts | Dashboards — per-day, per-model |
|---|---|
| ![Runs table](https://raw.githubusercontent.com/justfeltlikerunning/windhover/main/docs/runs.png) | ![Stats](https://raw.githubusercontent.com/justfeltlikerunning/windhover/main/docs/stats.png) |

## Quick start
```bash
pip install windhover langgraph
WINDHOVER_GRAPH=windhover.demo_graph:graph windhover   # -> :8090
```
Open `http://<host>:8090`. **New run** (input pre-filled from the graph's schema) →
watch it execute → **Runs** for history, span trees, and replay → **Stats** for cost/latency.
Edit the graph file while it runs and the canvas updates itself.

Your own graph: `WINDHOVER_GRAPH="myapp.graphs:g" WINDHOVER_GRAPH_DIR=/path python -m windhover.server`

Several graphs behind one URL: `WINDHOVER_GRAPH="checkout=app.flows:checkout,support=app.flows:support"`
(a selector appears in the top bar; `langgraph.json` projects serve all their graphs automatically).

## Trace runs from any app
```python
from windhover import WindhoverTracer
graph.invoke(input, config={"callbacks": [WindhoverTracer("http://HOST:8090")]})
```
Node spans, LLM calls (model/prompt/response/tokens/cost), and tools show up in **Runs** —
wherever your app runs. Non-blocking, best-effort; never raises into your graph.

Sessions and tags use standard LangChain config — no Windhover imports needed beyond the tracer:
```python
graph.invoke(input, config={
    "callbacks": [WindhoverTracer("http://HOST:8090")],
    "metadata": {"windhover_session": "chat-42", "windhover_tags": ["prod"]},
    "tags": ["also-captured"],          # langgraph-internal tags are filtered out
})
```

## Features
- **Any graph** — topology from `graph.get_graph()`; input form from its state schema.
- **Full trace tree** — nodes → nested LLM / tool / **retriever** spans: prompts, responses,
  tokens, cost, latency, retrieved documents with their metadata.
- **Clickable graph** — tap a node for health, latency, wiring, its **source code**, and recent executions with payloads.
- **Error forensics** — failed runs show the full traceback; the failing node turns red on the
  graph, and the node's source renders with the **throwing line highlighted**.
- **Human-in-the-loop console** — a paused graph shows an amber **interrupted** status with the
  question it's asking; answer it (`Command(resume=…)`), redirect it (`Command(goto=…)`), set
  static breakpoints per run (`interrupt_before`), **edit state** at any checkpoint
  (`update_state`), or **fork** a thread from any historical checkpoint — all from the UI,
  all pure LangGraph primitives.
- **State evolution** — every trace shows which state keys each node wrote, in order.
- **X-ray** — graphs with subgraphs get a canvas toggle that expands composite nodes
  (`get_graph(xray=True)`).
- **Search & filters** — full-text over prompts/payloads/errors (FTS5, LIKE fallback),
  status/tag/session filters, bookmarks, pagination, CSV/JSON export.
- **Sessions** — group runs into threads/batches; roll-up tokens, cost, errors.
- **Scores** — attach numeric evals to runs (API or UI): eval harnesses, LLM-as-judge, human review.
- **Live tail** — open a running run and watch spans arrive — including **the model typing**
  (streamed tokens flush into the span twice a second); nodes push progress via
  `get_stream_writer()`.
- **Call configs** — every LLM span records temperature/max-tokens/stream **and the tools the
  model was offered**; conditional-edge branch labels and `add_node(metadata=…)` render on the
  graph and node pane; graphs with a context schema get a runtime-context box on New run.
- **Custom events** — `dispatch_custom_event("name", {...})` anywhere in your app lands as an
  event marker in the trace, parented to the node that fired it.
- **Retries + TTFT** — tenacity retries badge the span (`↻2`); streaming LLM calls record
  time-to-first-token; cache-read / reasoning token details show on the model line.
- **Memory browser** — graphs compiled with a LangGraph `Store` get a Memory view: browse
  namespaces and search long-term memory items.
- **Time-travel** — checkpointed graphs get a per-thread checkpoint browser: state, writes,
  and next-nodes at every superstep (`get_state_history`).
- **Run diff** — compare any two runs node-by-node: identical vs differing outputs,
  duration and token deltas.
- **Datasets / batch eval** — store golden input sets, run the graph over them, and get an
  `expected_match` score per item (see Datasets on the Stats page).
- **Run history + replay** — SQLite; runs persist even if the browser closes (worker thread).
- **Living graph** — file watcher re-extracts topology in a subprocess and pushes it to the UI.
- **Dashboards** — runs/tokens per day, per-model usage and latency, per-node latency, error rate.
- **Multi-graph** — serve every graph in your project behind one URL; a top-bar selector
  scopes all views (runs, sessions, stats included — metrics never mix graphs).
- **Alerts** — `WINDHOVER_WEBHOOK` pushes a JSON summary when a run errors or pauses.
- **Never slows your app** — the remote tracer is non-blocking (bounded queue; drops
  rather than delays when the collector is down).
- **Mobile-first PWA**, light/dark. Fully local (FastAPI + Cytoscape.js).

## Datasets API
```bash
curl -X POST :8090/api/datasets -H 'Content-Type: application/json' -d '{
  "name": "golden", "items": [
    {"input": {"n": 2},  "expected": 6},
    {"input": {"n": 40}, "expected": "big"}]}'
curl -X POST :8090/api/datasets/golden/run   # -> runs land in an eval:golden:<ts> session
```

## Scores API
```bash
curl -X POST :8090/api/runs/RUN_ID/scores -H 'Content-Type: application/json' \
     -d '{"name": "accuracy", "value": 0.92, "comment": "vs golden set"}'
```

## Config (env)
`WINDHOVER_GRAPH` (module:attr; unset = ingest-only) · `WINDHOVER_GRAPH_DIR` · `WINDHOVER_DB`
· `WINDHOVER_HOST`/`WINDHOVER_PORT` (0.0.0.0/8090) · `WINDHOVER_WATCH` (1) · `WINDHOVER_PRICING`
· `WINDHOVER_RETENTION_DAYS` (0 = keep forever; else prune older runs on startup + every 6h)
· `WINDHOVER_TOKEN` (set to require `Authorization: Bearer <token>` — or `?token=` — on all
`/api` routes; the UI prompts once and remembers it).
Edit `windhover/pricing.json` for your models' $/1M rates (unknown model → cost null).

## Docs
**[The guide](docs/GUIDE.md)** covers every feature with how-tos and the fine print —
including the most important nuance: *panels light up based on what your graph supports*
(Memory needs a `store=`, time-travel/HITL need a `checkpointer=`, X-ray needs subgraphs,
live typing needs `streaming=True`). An absent tab isn't a bug — it's a graph without that
capability. [SPEC.md](SPEC.md) has the architecture.

## Notes
Runs use the imported graph (restart to run new code); the *view* always reflects
current-on-disk topology. All frontend assets are vendored — no CDN, works fully offline.
Deep links: `#runs`, `#sessions`, `#stats`, `#run=<id>`.

## License
MIT.
