# Windhover guide — features, how-tos, and the fine print

Everything Windhover shows comes from your graph. That has one important consequence,
worth understanding before anything else:

## Feature availability — panels light up when your graph supports them

Windhover detects what your graph provides and shows only the views that apply.
An "absent" tab is not a bug — it means the graph doesn't carry that capability yet.

| You see…                         | …when your graph has                                        |
|----------------------------------|-------------------------------------------------------------|
| Graph / Runs / Stats / Sessions  | always                                                       |
| Node source panels               | a local graph (`WINDHOVER_GRAPH`) whose nodes are plain Python (inspectable) |
| **Memory** tab                   | `compile(store=…)` — any LangGraph `BaseStore`               |
| **time-travel**, thread chips    | `compile(checkpointer=…)`                                    |
| Resume / breakpoints / state edit / fork | a checkpointer (they all operate on threads)          |
| **X-ray** toggle                 | subgraphs (`get_graph(xray=True)` differs)                   |
| Progress toasts during a run     | nodes that call `get_stream_writer()`                        |
| "Model typing" live output       | an LLM constructed with `streaming=True`                     |
| TTFT on LLM spans                | streaming calls (first-token time only exists when tokens stream) |
| Edge labels on the canvas        | conditional edges whose branch names differ from their targets |
| Node metadata in the node pane   | `add_node("x", fn, metadata={…})`                            |
| Runtime-context box on New run   | a graph compiled with a `context_schema`                     |

Minimal fully-featured compile:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

graph = builder.compile(checkpointer=MemorySaver(), store=InMemoryStore())
```

### Persistence caveat — read this one

`MemorySaver` and `InMemoryStore` live **inside the server process**. Restart the
server and threads, checkpoints, and memory items are gone (runs and spans are NOT —
they live in Windhover's own SQLite). For durable threads/memory use LangGraph's
persistent backends:

```python
from langgraph.checkpoint.sqlite import SqliteSaver          # pip install langgraph-checkpoint-sqlite
graph = builder.compile(checkpointer=SqliteSaver.from_conn_string("checkpoints.db"))
# or PostgresSaver / PostgresStore from langgraph-checkpoint-postgres
```

Also: runs recorded **before** you added a checkpointer have no thread id, so they
never grow time-travel buttons retroactively. Same for any field added by an upgrade —
old rows simply don't have it.

## How-to: observe a local graph

```bash
pip install windhover langgraph
WINDHOVER_GRAPH="myapp.graphs:graph" WINDHOVER_GRAPH_DIR=/path/to/app windhover
```

Projects with a **`langgraph.json`** (the LangGraph Studio/CLI convention) need no flags —
run `windhover` in the project directory and every graph the file defines is served.

**Multiple graphs**: `WINDHOVER_GRAPH="checkout=app.flows:checkout,support=app.flows:support"`
puts a graph selector in the top bar. The selector scopes **everything** — canvas, New run,
node source, Memory, Runs, Sessions, Stats (so per-node latency and model usage never mix
graphs). The Runs page keeps an "All graphs" override in its own filter; human-in-the-loop
actions always follow the graph the run belongs to.

- The canvas always reflects **current-on-disk** topology (a subprocess re-extracts on
  file change). Runs, however, execute the graph **imported at startup** — restart the
  server to run new code. The "Graph definition changed" toast is the tell.
- **New run** builds its input form from your graph's own state schema. Blank template
  fields mean your state type isn't introspectable (e.g. `dict`) — you can still paste
  any JSON.

## How-to: trace from your own app (no local graph needed)

```python
from windhover import WindhoverTracer

graph.invoke(inputs, config={
    "callbacks": [WindhoverTracer("http://observability-host:8090")],
    "metadata": {
        "windhover_session":  "checkout-7431",   # groups runs into a session
        "windhover_tags":     ["prod", "eu"],    # filterable tags
        "windhover_run_name": "checkout-flow",   # display name in Runs
    },
})
```

- Everything is standard LangChain config — no other Windhover imports.
- The tracer is best-effort and non-blocking: if the Windhover host is down, your
  graph runs normally and the trace is simply lost.
- External runs show `ingest` as their source. They have full traces, sessions,
  scores, search — but **no source panels, HITL, or time-travel** (those need the
  graph running inside the Windhover server).

## How-to: human-in-the-loop

All five controls are plain LangGraph primitives — Windhover adds no magic, only UI.

1. **Ask a human mid-run** — in a node:
   ```python
   from langgraph.types import interrupt
   def payout(state):
       ok = interrupt({"question": f"approve ${state['amount']}?"})
       if not ok:
           return {"status": "rejected"}
       ...
   ```
   The run turns amber **interrupted**; its drawer shows the question with a respond
   box. Answers parse as JSON when possible (`true`, `42`, `{"a":1}`), else as text.
2. **Static breakpoints** — pick "pause before" chips on New run (`interrupt_before`).
   Handy for inspecting state before an expensive node. Resume continues.
3. **Redirect** — the goto picker sends execution to any node (`Command(goto=…)`).
4. **Edit state** — in time-travel, ✎ on any checkpoint patches values via
   `update_state`; then Resume continues *on the edited state*.
5. **Fork** — ⑂ re-runs from any historical checkpoint, branching the thread.

Fine print:
- Resuming records a **new run** on the same thread, tagged `resume` — traces stay
  immutable; the thread ties them together.
- A pause is **not** an error: LangGraph delivers interrupts as a `GraphInterrupt`
  exception internally; Windhover records the node span as `interrupted`, not failed.
- Answering an interrupt re-executes the interrupted node **from its start** (that's
  LangGraph's contract) — keep side effects before an `interrupt()` idempotent.

## How-to: long-term memory (the Memory tab)

```python
from langgraph.config import get_store

def remember(state):
    store = get_store()
    store.put(("users", state["user_id"]), "preferences", {"tone": "brief"})
```

The Memory tab lists namespaces and their items with search. **It populates only when
your nodes actually write to the store** — an empty tab on a fresh graph is expected.
Windhover is read-only here: it browses, it never writes or deletes memories.

## How-to: datasets & batch eval

```bash
curl -X POST :8090/api/datasets -H 'Content-Type: application/json' -d '{
  "name": "golden",
  "items": [
    {"input": {"question": "capital of France?"}, "expected": "Paris"},
    {"input": {"question": "2+2?"},               "expected": 4}
  ]}'
```

"Run eval" (Stats page) executes the local graph per item; each run lands in an
`eval:golden:<timestamp>` session with an `expected_match` score: exact match for
numbers/booleans, substring-of-output-JSON for strings. It's a smoke-level matcher —
for semantic grading, run your own judge and `POST /api/runs/{id}/scores`.

## More things the audit covered

- **Concurrency & batch** — one tracer instance safely handles parallel `.invoke()`s:
  every root execution becomes its own run. (`graph.batch()` shares a single LangChain
  root, so a batch records as one run — use separate invokes when you want separate runs.)
- **Bare LangChain (no graph)** — `llm.invoke(..., config={"callbacks": [tracer]})` with
  no chain around it opens an implicit run: Windhover works for plain pipelines too.
- **Functional API** — `@entrypoint`/`@task` graphs trace fully (tasks appear as node
  spans). The canvas shows only the entrypoint — tasks are dynamic calls, not static
  topology; the trace is where their structure lives.
- **Node caching** — LangGraph `CachePolicy` cache hits fire **no callbacks**; for local
  runs Windhover synthesizes the span with a `cached` marker so the trace stays complete.
  External-tracer apps can't see cache hits at all (there's nothing to observe).
- **Conversations** — payloads shaped like message lists (`role`/`content`) render as a
  chat transcript instead of raw JSON — LangChain messages are captured in that shape
  automatically, tool calls included.

## How-to: everything else, briefly

- **Custom events**: `from langchain_core.callbacks import dispatch_custom_event;
  dispatch_custom_event("cache-refreshed", {"rows": 1200})` inside any node/tool →
  an event marker in the trace, parented where it fired.
- **Progress**: `from langgraph.config import get_stream_writer;
  get_stream_writer()({"step": "embedding", "pct": 40})` → live toasts during a run.
  (Writer output only flows in streaming executions — Windhover's local runs always
  stream, but your own `.invoke()` elsewhere won't emit it.)
- **Errors**: open a failed run → full traceback; the failing node is red on replay;
  "view source" highlights the throwing line inside the node's own code.
- **Compare**: open a run → `compare` → open a second run → `compare` again: node-by-node
  output diff with duration/token deltas.
- **Search**: full-text over prompts/payloads/errors. Uses SQLite FTS5 when available,
  transparently falls back to LIKE scans on minimal SQLite builds (slower, same results).
- **Cost**: longest-prefix match against `windhover/pricing.json` ($/1M tokens). Unknown
  model → cost shows `—`, never a guess. Cached/reasoning token counts display on the
  model line but aren't priced separately.
- **Structured output**: function-calling responses have empty text content; Windhover
  shows the tool-call JSON as the LLM output instead of an empty box.
- **Auth**: set `WINDHOVER_TOKEN` to require `Authorization: Bearer …` (or `?token=`)
  on `/api`. The UI prompts once and remembers. The HTML shell itself stays public —
  it contains no data. External tracers pass it as `WindhoverTracer(url, token="…")`.
  **Set the token before exposing Windhover beyond localhost** — the HITL endpoints can
  resume and edit graph state.
- **Alerts**: `WINDHOVER_WEBHOOK=https://…` POSTs a compact JSON summary whenever a run
  errors or pauses on an interrupt (`{source, graph, run_id, status, error, text, …}`) —
  point it at a Slack/Discord webhook or your own receiver. Fire-and-forget.
- **Retention**: `WINDHOVER_RETENTION_DAYS=30` prunes old runs on startup and every 6h.
  Default keeps everything.
- **Deep links**: `#runs`, `#sessions`, `#stats`, `#run=<id>`.

## Running it for real

- **The tracer never slows your app.** Events go onto a bounded in-memory queue drained
  by a background thread; if the Windhover host is slow or down, events are dropped —
  your pipeline's latency is unchanged. (Sheds oldest-first at 2,000 queued events.)
- **Systemd** (adjust paths):
  ```ini
  [Unit]
  Description=Windhover
  After=network.target

  [Service]
  WorkingDirectory=/opt/myapp
  Environment=WINDHOVER_GRAPH=myapp.graphs:graph
  Environment=WINDHOVER_DB=/var/lib/windhover/windhover.db
  Environment=WINDHOVER_TOKEN=change-me
  Environment=WINDHOVER_RETENTION_DAYS=30
  ExecStart=/opt/myapp/.venv/bin/windhover
  Restart=always
  TimeoutStopSec=10

  [Install]
  WantedBy=multi-user.target
  ```
  `TimeoutStopSec` matters: open SSE connections otherwise stretch restarts.
- **Reverse proxy**: plain HTTP + SSE — any proxy works; disable response buffering for
  `/api/events` and `/api/run` (nginx: `proxy_buffering off;`).
- **Backups**: runs live in one SQLite file (`WINDHOVER_DB`); copy it (plus `-wal`) or
  rely on `/api/export` for tabular run data.

## Try it all on the demo graph

`WINDHOVER_GRAPH=windhover.demo_graph:graph windhover`, then:

- `{"n": 4}` — normal run: parallel fan-out, state evolution, memory write.
- `{"n": -3}` — error forensics: red node, traceback, highlighted source line.
- `{"n": 200}` — HITL: pauses asking "grow 200 → 600?"; answer `true` or `false`
  in the drawer.
- After a few runs: Memory tab (`demo/summaries`), time-travel on any run's thread,
  compare two runs with different `n`.
