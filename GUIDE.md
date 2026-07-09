# Windhover guide - features, how-tos, and the fine print

Everything Windhover shows comes from your graph. That has one important consequence,
worth understanding before anything else:

## Feature availability - panels light up when your graph supports them

Windhover detects what your graph provides and shows only the views that apply.
An "absent" tab is not a bug - it means the graph doesn't carry that capability yet.

| You see…                         | …when your graph has                                        |
|----------------------------------|-------------------------------------------------------------|
| Graph / Runs / Stats / Sessions  | always                                                       |
| Node source panels               | a local graph (`WINDHOVER_GRAPH`) whose nodes are plain Python (inspectable) |
| **Memory** tab                   | `compile(store=…)` - any LangGraph `BaseStore`               |
| **time-travel**, thread chips    | `compile(checkpointer=…)`                                    |
| Resume / breakpoints / state edit / fork | a checkpointer (they all operate on threads)          |
| **X-ray** toggle                 | subgraphs (`get_graph(xray=True)` differs)                   |
| Progress toasts during a run     | nodes that call `get_stream_writer()`                        |
| "Model typing" live output       | an LLM constructed with `streaming=True`                     |
| TTFT on LLM spans                | streaming calls (first-token time only exists when tokens stream) |
| Edge labels on the canvas        | conditional edges whose branch names differ from their targets |
| Node metadata in the node pane   | `add_node("x", fn, metadata={…})`                            |
| Runtime-context box on New run   | a graph compiled with a `context_schema`                     |
| **Fleet** view (cross-graph)     | more than one graph served (hidden in single-graph mode)     |
| **Artifacts** on runs/nodes      | node outputs that record absolute file paths (see how-to)    |
| 🔔 alerts button / Web Push      | `WINDHOVER_VAPID_PUBLIC`/`_PRIVATE` set, served over HTTPS   |

Minimal fully-featured compile:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

graph = builder.compile(checkpointer=MemorySaver(), store=InMemoryStore())
```

### Persistence caveat - read this one

`MemorySaver` and `InMemoryStore` live **inside the server process**. Restart the
server and threads, checkpoints, and memory items are gone (runs and spans are NOT -
they live in Windhover's own SQLite). For durable threads/memory use LangGraph's
persistent backends:

```python
from langgraph.checkpoint.sqlite import SqliteSaver          # pip install langgraph-checkpoint-sqlite
graph = builder.compile(checkpointer=SqliteSaver.from_conn_string("checkpoints.db"))
# or PostgresSaver / PostgresStore from langgraph-checkpoint-postgres
```

Also: runs recorded **before** you added a checkpointer have no thread id, so they
never grow time-travel buttons retroactively. Same for any field added by an upgrade -
old rows simply don't have it.

## How-to: observe a local graph

```bash
pip install windhover langgraph
WINDHOVER_GRAPH="myapp.graphs:graph" WINDHOVER_GRAPH_DIR=/path/to/app windhover
```

Projects with a **`langgraph.json`** (the LangGraph Studio/CLI convention) need no flags -
run `windhover` in the project directory and every graph the file defines is served.

- The canvas always reflects **current-on-disk** topology (a subprocess re-extracts on
  file change). Runs, however, execute the graph **imported at startup**: restart the
  server to run new code. The "Graph definition changed" toast is the tell.
- **New run** builds its input form from your graph's own state schema. Blank template
  fields mean your state type isn't introspectable (e.g. `dict`) - you can still paste
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

- Everything is standard LangChain config - no other Windhover imports.
- The tracer is best-effort and non-blocking: if the Windhover host is down, your
  graph runs normally and the trace is simply lost.
- External runs show `ingest` as their source. They have full traces, sessions,
  scores, search - but **no source panels, HITL, or time-travel** (those need the
  graph running inside the Windhover server).

## How-to: human-in-the-loop

All five controls are plain LangGraph primitives - Windhover adds no magic, only UI.

1. **Ask a human mid-run**: in a node:
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
2. **Static breakpoints**: pick "pause before" chips on New run (`interrupt_before`).
   Handy for inspecting state before an expensive node. Resume continues.
3. **Redirect**: the goto picker sends execution to any node (`Command(goto=…)`).
4. **Edit state**: in time-travel, ✎ on any checkpoint patches values via
   `update_state`; then Resume continues *on the edited state*.
5. **Fork**: ⑂ re-runs from any historical checkpoint, branching the thread.

Fine print:
- Resuming records a **new run** on the same thread, tagged `resume` - traces stay
  immutable; the thread ties them together.
- A pause is **not** an error: LangGraph delivers interrupts as a `GraphInterrupt`
  exception internally; Windhover records the node span as `interrupted`, not failed.
- Answering an interrupt re-executes the interrupted node **from its start** (that's
  LangGraph's contract) - keep side effects before an `interrupt()` idempotent.

## How-to: long-term memory (the Memory tab)

```python
from langgraph.config import get_store

def remember(state):
    store = get_store()
    store.put(("users", state["user_id"]), "preferences", {"tone": "brief"})
```

The Memory tab lists namespaces and their items with search. **It populates only when
your nodes actually write to the store** - an empty tab on a fresh graph is expected.
Windhover is read-only here: it browses, it never writes or deletes memories.

## How-to: artifacts - preview and download files your graph writes

There is no artifact API to call. The contract is simply: **return the file's
absolute path in the node's output state**, and Windhover surfaces it.

```python
def report(state):
    path = write_report(state)              # "/srv/out/report_2026-07-08.docx"
    return {"docs": [path]}                 # any key, any nesting - this is enough
```

- The run drawer grows an **artifacts** section (all files across the run) and each
  node execution in the node pane shows chips under its payload.
- **Inline preview**: HTML (sandboxed iframe - scripts never execute), PDF, images/SVG,
  Python (highlighted), CSV/TSV (as a table), JSON/markdown/text. **Download-only**:
  Word/Excel/zip - browsers can't render those; Windhover doesn't pretend otherwise.
- Rules: absolute paths (`/…`, `~/…`, or `C:\…`) with a recognized extension. Relative
  paths and URLs are deliberately ignored so ordinary strings never false-positive.
- **Locality**: the file must exist on the Windhover server's host. Runs traced in from
  another machine list their files flagged `missing here` instead of erroring.
- **Security**: the server only serves paths recorded in that run's own stored outputs -
  the allowlist is re-derived from the run on every request, so it can never read
  arbitrary files - and it sits behind the same `/api` token gate.

## How-to: the Fleet view (multiple graphs, one glance)

Serving more than one graph makes **Fleet** the landing page (single-graph
instances never see it):

- **Needs attention**: every run across all graphs that's paused on an interrupt
  (its question inline, a resume box right on the row - blank continues past a
  breakpoint) or still running. An interrupted run whose thread already has a newer
  run counts as *handled* and drops off - resumes create new runs, so the queue
  never accumulates stale entries.
- **Per-graph cards**: last-run status, 7-day run/error counts with a daily
  sparkline (errors in red), and the three most recent runs. Cards for graphs that
  only ingested traces (or were renamed) are flagged `not serving` - clicking one
  opens their run history, since there's no live topology to show.
- The top-bar graph selector hides here (Fleet is cross-graph by definition) and
  returns on scoped views. Deep link: `#fleet`. Script access: `GET /api/overview`.

## How-to: phone alerts (Web Push)

Windhover can push run alerts to any installed PWA - iOS 16.4+, Android, desktop.

```bash
pip install windhover[push]
python -c "from py_vapid import Vapid01; v=Vapid01(); v.generate_keys(); \
           print(v.private_pem().decode())"          # or any VAPID keygen
export WINDHOVER_VAPID_PUBLIC=…  WINDHOVER_VAPID_PRIVATE=…
export WINDHOVER_VAPID_SUBJECT=mailto:you@yourdomain.com
```

- **HTTPS is mandatory**: browsers only allow push from a secure origin, and Apple
  rejects VAPID contacts on fake domains (`.local` → 403). A reverse proxy with a
  real certificate (Caddy, Tailscale `serve`, etc.) in front of Windhover is enough.
- On the device: open the HTTPS URL → add to home screen → open the installed app →
  tap the **🔔** (iOS requires the tap to come from an installed PWA). A test push
  confirms delivery. The bell is stateful: filled = on, slashed = blocked in OS settings.
- Alerts fire on **error** and **interrupted** runs - the same events as
  `WINDHOVER_WEBHOOK`. Tapping one deep-links to the run (or to the Fleet queue when
  several runs are awaiting approval). Expired subscriptions are pruned automatically.
- `WINDHOVER_DIGEST=07:30` adds one daily summary push (runs/errors/awaiting across
  all graphs). Quiet days send nothing.
- Webhooks can route per graph: `WINDHOVER_WEBHOOK="https://hooks.example/default,billing=https://hooks.example/billing"`.

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
numbers/booleans, substring-of-output-JSON for strings. It's a smoke-level matcher -
for semantic grading, run your own judge and `POST /api/runs/{id}/scores`.

## More things the audit covered

- **Concurrency & batch**: one tracer instance safely handles parallel `.invoke()`s:
  every root execution becomes its own run. (`graph.batch()` shares a single LangChain
  root, so a batch records as one run - use separate invokes when you want separate runs.)
- **Bare LangChain (no graph)**: `llm.invoke(..., config={"callbacks": [tracer]})` with
  no chain around it opens an implicit run: Windhover works for plain pipelines too.
- **Functional API**: `@entrypoint`/`@task` graphs trace fully (tasks appear as node
  spans). The canvas shows only the entrypoint - tasks are dynamic calls, not static
  topology; the trace is where their structure lives.
- **Node caching**: LangGraph `CachePolicy` cache hits fire **no callbacks**; for local
  runs Windhover synthesizes the span with a `cached` marker so the trace stays complete.
  External-tracer apps can't see cache hits at all (there's nothing to observe).
- **Conversations**: payloads shaped like message lists (`role`/`content`) render as a
  chat transcript instead of raw JSON - LangChain messages are captured in that shape
  automatically, tool calls included.

## How-to: everything else, briefly

- **Custom events**: `from langchain_core.callbacks import dispatch_custom_event;
  dispatch_custom_event("cache-refreshed", {"rows": 1200})` inside any node/tool →
  an event marker in the trace, parented where it fired.
- **Progress**: `from langgraph.config import get_stream_writer;
  get_stream_writer()({"step": "embedding", "pct": 40})` → live toasts during a run.
  (Writer output only flows in streaming executions - Windhover's local runs always
  stream, but your own `.invoke()` elsewhere won't emit it.)
- **Errors**: open a failed run → full traceback; the failing node is red on replay;
  "view source" highlights the throwing line inside the node's own code.
- **Compare**: open a run → `compare` → open a second run → `compare` again: node-by-node
  output diff with duration/token deltas.
- **Search**: full-text over prompts/payloads/errors. Uses SQLite FTS5 when available,
  transparently falls back to LIKE scans on minimal SQLite builds (slower, same results).
- **Cost**: longest-prefix match against `windhover/pricing.json` ($/1M tokens). Unknown
  model → cost shows `-`, never a guess. Cached/reasoning token counts display on the
  model line but aren't priced separately.
- **Structured output**: function-calling responses have empty text content; Windhover
  shows the tool-call JSON as the LLM output instead of an empty box.
- **Auth**: set `WINDHOVER_TOKEN` to require `Authorization: Bearer …` (or `?token=`)
  on `/api`. The UI prompts once and remembers. The HTML shell itself stays public -
  it contains no data.
- **Retention**: `WINDHOVER_RETENTION_DAYS=30` prunes old runs on startup and every 6h.
  Default keeps everything.
- **Deep links**: `#fleet`, `#runs`, `#sessions`, `#stats`, `#run=<id>`.
- **Mobile**: pull down on any list view to refresh (the graph view pans instead);
  returning the PWA to the foreground refetches the current view automatically.

## Try it all on the demo graph

`WINDHOVER_GRAPH=windhover.demo_graph:graph windhover`, then:

- `{"n": 4}` - normal run: parallel fan-out, state evolution, memory write.
- `{"n": -3}` - error forensics: red node, traceback, highlighted source line.
- `{"n": 200}` - HITL: pauses asking "grow 200 → 600?"; answer `true` or `false`
  in the drawer.
- After a few runs: Memory tab (`demo/summaries`), time-travel on any run's thread,
  compare two runs with different `n`.
