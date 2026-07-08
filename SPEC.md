# Windhover — engineering spec (current as of v0.14)

Self-hosted LangGraph/LangChain observability + human-in-the-loop console.
Code is the only source of truth; Windhover observes and, for HITL, drives the
graph **only through native LangGraph primitives** (interrupt/resume, goto,
breakpoints, update_state, checkpoint forking). No visual editing, ever.

## Design principles
1. **Observe, never define.** Topology, schemas, source, memory — all derive from
   the user's compiled graph. Features appear only when the graph supports them.
2. **One tracer, two sinks.** `SpanBuilder` handles every LangChain callback and
   feeds either the local DB sink or the HTTP ingest sink — identical traces
   wherever the run happened.
3. **Runs are durable records.** Execution happens off the request thread; a run
   persists even if the browser disconnects. Resumes are NEW runs on the same
   thread — traces are immutable.
4. **Degrade gracefully.** No graph → ingest-only collector. No FTS5/json1 →
   LIKE fallbacks. Unknown model → cost null. Interrupt → pause, not error.
5. **Real package.** pip-installable (`windhover`), CI-tested 3.10–3.12, all
   frontend assets vendored (offline-capable), MIT.

## Architecture
- **FastAPI + SSE** (no websockets). Single-file UI (`static/index.html`) +
  vendored cytoscape/dagre/fonts.
- **Multi-graph registry**: `WINDHOVER_GRAPH` comma list (`name=module:attr`)
  or every graph in `langgraph.json`; per-graph `Topo` watchers; `graph=`
  scopes graph/schema/source/memory/threads/stats/sessions/dataset-run;
  `/api/run` takes `_graph`; run rows store the registry name.
- **Topology**: subprocess (`windhover.extract`) re-imports the graph on file
  mtime change → nodes/edges (+labels/metadata), input & context schemas,
  per-node source (inspect, unwrapped), x-ray variant; sha1-hash change bumps a
  version pushed over `/api/events` SSE.
- **Tracer** (`windhover.tracer.SpanBuilder`): concurrency-safe per-root run
  contexts (parallel invokes = separate runs; bare llm/tool/retriever calls
  open implicit runs; bookkeeping purged per run — no growth in long-lived
  apps). Captures chains→node spans, LLM (params, tools offered,
  tokens+cache/reasoning details, TTFT, streaming partials ~0.5 s), tools,
  retrievers, retries, custom events, GraphInterrupt→interrupted (a pause,
  never an error), messages as {role, content, tool_calls},
  session/tags/run_name/thread_id via config metadata. The HTTP sink is
  non-blocking (bounded queue + drain thread, sheds oldest). Best-effort:
  never raises into — or slows — the user's graph.
- **Store** (`windhover.store`, SQLite WAL, schema v7): runs / spans (tree) /
  scores / datasets (+span_fts). Feature-detects FTS5 & json1 at startup;
  idempotent column migrations.
- **HITL** (`/api/threads/*`): thin wrappers over `Command(resume|goto)`,
  `interrupt_before/after`, `update_state`, `get_state_history`, checkpoint-id
  forking. Requires the local graph to have a checkpointer. Pending-node
  detection queries by thread only (a checkpoint_id would read history).
- **Cache visibility**: LangGraph `CachePolicy` hits fire no callbacks; local
  runs synthesize the node span with `params.cached`.
- **Auth**: optional `WINDHOVER_TOKEN` Bearer/?token= gate on `/api` only.
- **Alerts**: `WINDHOVER_WEBHOOK` POSTs a summary on error/interrupted runs
  (store.on_run_closed hook, fire-and-forget, deduped).

## Data model (schema v7)
`runs`: id · graph · source(ui|ingest) · status(running|done|error|interrupted) ·
session · tags(json) · thread_id · input(json) · error · timing · aggregates
(node_count, llm_calls, tokens, cost_usd) · bookmarked.
`spans`: id · run_id · parent_id · seq · type(node|llm|tool|retriever|event|
interrupt) · name · status(ok|error|interrupted|running←streaming partial) ·
timing/offset/dur · input/output(json) · model · tokens · cost_usd · error
(full traceback) · retries · ttft_ms · usage_detail(json) · params(json).
`scores`: id · run_id · name · value · comment · source.
`datasets`: id · name · items(json).

## API surface
Observe: `/api/graph` `/api/schema` `/api/runs[…filters]` `/api/runs/{id}`
`/api/nodes/{name}[/source]` `/api/sessions` `/api/stats` `/api/export`
`/api/memory/namespaces` `/api/memory/items` `/api/events` (SSE).
Act: `/api/run` (SSE; `_session/_tags/_thread/_interrupt_before/_interrupt_after/
_configurable`), `/api/ingest`, `PATCH /api/runs/{id}`, scores CRUD, datasets
CRUD + `/run`, `/api/threads/{id}/history|resume|state`.

User-facing behavior, nuances, and how-tos: see `docs/GUIDE.md`.
