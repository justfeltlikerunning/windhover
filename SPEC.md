# Windhover v0.3 — engineering spec

Self-hosted LangGraph observability. Code is the only source of truth; Windhover
observes. Goal: Langfuse-class trace depth (LLM-level: prompts, tokens, cost,
latency) + a living graph view that auto-updates when the code's topology changes.
No visual editing, ever.

## Design principles
1. **Observe, never define.** Topology always derives from the compiled graph.
2. **The tracer is the single source of span data** — used identically for local
   runs (in-process → DB sink) and external apps (HTTP → ingest). One code path.
3. **Runs are durable records.** Execution happens off the request thread; a run
   persists even if the browser disconnects.
4. **Degrade gracefully.** No graph configured → ingest-only collector. Unknown
   model → cost null (not a crash). Missing token usage → tokens null.
5. **Real package.** pip-installable, typed, smoke-tested. No CDN at runtime once
   vendored (roadmap). MIT.

## Data model (SQLite, WAL)
`schema_meta(version)`. Migrate v2 `events` → v3 node spans best-effort.

`runs`
  id TEXT PK · graph · source(ui|ingest) · status(running|done|error) · session ·
  tags(json) · input(json) · error · started_ms · ended_ms · duration_ms ·
  node_count · llm_calls · prompt_tokens · completion_tokens · total_tokens · cost_usd

`spans`  (a tree: node spans are top-level; llm/tool spans nest under a node)
  id TEXT PK · run_id · parent_id(nullable) · seq · type(node|llm|tool) · name ·
  status(ok|error) · started_ms · ended_ms · offset_ms(rel to run) · dur_ms ·
  input(json) · output(json) · model · prompt_tokens · completion_tokens ·
  cost_usd · error
  INDEX(run_id, seq)

## Tracer contract (LangChain callback → sink)
`SpanBuilder(sink, run_name, session=None, tags=None)` emits to `sink(event)`:
  - `{"kind":"run_open", run_id, graph, input, started_ms, session, tags}`
  - `{"kind":"span", ...span row...}`  (node, llm, tool)
  - `{"kind":"run_close", run_id, status, ended_ms, error}`
Sinks: `db_sink(store)` (in-process), `http_sink(base_url)` (WindhoverTracer).

Callback facts (LangGraph, verified):
  - node name only on `on_chain_start` via `metadata.langgraph_node` → map run_id→span.
  - llm span: `on_chat_model_start`/`on_llm_start` (model, prompt) →
    `on_llm_end` (LLMResult: `llm_output.token_usage` OR generation
    `usage_metadata`; response text). parent_run_id = enclosing node's run_id.
  - tool span: `on_tool_start`/`on_tool_end`.
  - root = first `on_chain_start` with `parent_run_id is None`.

## Cost
`pricing.json`: `{ "<model-prefix>": {"input": $/1M, "output": $/1M} }`.
Longest-prefix match on model name; miss → null. Local bench/subscription models
have no public price → null by design (matches canary's "est_cost null").

## HTTP API
GET  `/api/graph`                topology {nodes,edges,graph,hash}
GET  `/api/schema`               input json-schema + blank template
POST `/api/run`                  run local graph; SSE (start/node/done/error) + persist spans
POST `/api/ingest`               external tracer events (run_open|span|run_close)
GET  `/api/runs?limit=&session=` history
GET  `/api/runs/{id}`            run + span tree
GET  `/api/stats`               aggregates (counts, cost/tokens over time, per-node latency, error rate)
GET  `/api/events`               SSE: topology-changed pushes (living graph)

## Living graph
Background watcher polls mtime of graph module + `WINDHOVER_GRAPH_DIR/**.py` (~2s).
On change → re-extract topology in a **subprocess** (avoids importlib.reload
fragility) → hash → if changed, broadcast `topology` on `/api/events`. UI (EventSource)
re-fetches `/api/graph`, re-layouts, toasts "graph changed".

## Frontend (single file, mobile-first PWA)
Views: **Graph** (topology + live run + live auto-refresh), **Runs** (history w/
cost+tokens → detail = span tree: node → nested LLM calls w/ model/tokens/cost/
prompt/response, timing waterfall, replay), **Stats** (strip: runs, cost, tokens,
error-rate, avg node latency).

## Out of scope (roadmap)
Vendored deps offline · subgraph drill-in · conditional-edge labels · auth ·
retention/pruning · export.

## Layout
windhover/{__init__,config,store,tracer,server,demo_graph}.py · pricing.json ·
static/{index.html,manifest.json,icon.svg} · tests/test_smoke.py · pyproject.toml ·
LICENSE · README.md · SPEC.md
