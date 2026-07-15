"""Windhover smoke tests — store roundtrip, cost math, tracer against a real
LangGraph run (incl. a fake LLM node to exercise LLM-span capture). Run:

    python -m pytest tests/ -q      (or: python tests/test_smoke.py)
"""
import json, os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from windhover.store import Store
from windhover.tracer import SpanBuilder, db_sink, cost_of


def test_cost():
    assert cost_of("gpt-4o", 1_000_000, 1_000_000) == 12.5      # 2.5 + 10
    assert cost_of("gpt-4o-2024-11-20", 1_000_000, 0) == 2.5    # prefix match
    assert cost_of("bench/fable", 1000, 1000) is None           # unknown -> None
    assert cost_of(None, 1, 1) is None


def test_store_roundtrip():
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    s.open_run({"id": "r1", "graph": "g", "source": "ui", "started_ms": 1000})
    s.add_span({"id": "n1", "run_id": "r1", "type": "node", "name": "a",
                "seq": 0, "offset_ms": 0, "dur_ms": 10, "output": {"x": 1}})
    s.add_span({"id": "l1", "run_id": "r1", "parent_id": "n1", "type": "llm",
                "name": "gpt-4o", "seq": 1, "dur_ms": 5,
                "prompt_tokens": 1_000_000, "completion_tokens": 0, "cost_usd": 2.5})
    s.close_run("r1", "done", 1050)
    d = s.run_detail("r1")
    assert d["status"] == "done" and d["duration_ms"] == 50
    assert d["node_count"] == 1 and d["llm_calls"] == 1
    assert abs(d["cost_usd"] - 2.5) < 1e-9
    assert len(d["spans"]) == 2
    llm = [x for x in d["spans"] if x["type"] == "llm"][0]
    assert llm["parent_id"] == "n1"
    print("store roundtrip OK")


def test_tracer_local_graph():
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)

    class St(TypedDict):
        x: int
    def a(st): return {"x": st["x"] + 1}
    def b(st): return {"x": st["x"] * 10}
    g = StateGraph(St)
    g.add_node("inc", a); g.add_node("mul", b)
    g.add_edge(START, "inc"); g.add_edge("inc", "mul"); g.add_edge("mul", END)
    app = g.compile()

    tracer = SpanBuilder(db_sink(s), run_name="test")
    out = app.invoke({"x": 4}, config={"callbacks": [tracer]})
    assert out["x"] == 50
    time.sleep(.1)
    d = s.run_detail(tracer.run_id)
    assert d is not None and d["status"] == "done"
    names = [sp["name"] for sp in d["spans"] if sp["type"] == "node"]
    assert "inc" in names and "mul" in names
    print("tracer local-graph OK — nodes:", names)


def test_search_filters_scores_sessions():
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    now = int(time.time() * 1000)
    for i, (sess, tag) in enumerate([("chat-1", "alpha"), ("chat-1", "beta"), (None, "alpha")]):
        rid = f"r{i}"
        s.open_run({"id": rid, "graph": "g", "source": "ui", "session": sess,
                    "tags": [tag], "started_ms": now - i * 1000})
        s.add_span({"id": f"n{i}", "run_id": rid, "type": "node", "name": "work",
                    "seq": 0, "dur_ms": 5, "output": {"text": f"needle-{i} haystack"}})
        s.close_run(rid, "done" if i else "error", now - i * 1000 + 50)
    # payload search (FTS5 or LIKE fallback — same contract either way)
    hit = s.runs(q=f"needle-1")
    assert hit["total"] == 1 and hit["runs"][0]["id"] == "r1"
    assert s.runs(status="error")["total"] == 1
    assert s.runs(tag="alpha")["total"] == 2
    assert s.runs(session="chat-1")["total"] == 2
    ss = s.sessions()
    assert len(ss) == 1 and ss[0]["session"] == "chat-1" and ss[0]["runs"] == 2
    assert ss[0]["errors"] == 1
    # bookmarks
    assert s.update_run_meta("r0", bookmarked=True)
    assert s.runs(bookmarked=True)["total"] == 1
    assert not s.update_run_meta("nope", bookmarked=True)
    # scores: add, aggregate onto runs page, embed in detail, delete
    sc = s.add_score("r0", "accuracy", 0.9, comment="ok", source="test")
    assert sc is not None
    assert s.add_score("missing-run", "x", 1.0) is None
    assert s.runs(bookmarked=True)["runs"][0]["scores"] == {"accuracy": 0.9}
    assert s.run_detail("r0")["scores"][0]["name"] == "accuracy"
    assert s.delete_score(sc["id"])
    # pagination
    pg = s.runs(limit=2, offset=2)
    assert pg["total"] == 3 and len(pg["runs"]) == 1
    print("search/filters/scores/sessions OK (fts=%s json1=%s)" % (s.has_fts, s.has_json1))


def test_prune_and_stats():
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    old = int((time.time() - 40 * 86400) * 1000)
    now = int(time.time() * 1000)
    s.open_run({"id": "old1", "graph": "g", "started_ms": old})
    s.close_run("old1", "done", old + 10)
    s.open_run({"id": "new1", "graph": "g", "started_ms": now})
    s.add_span({"id": "L1", "run_id": "new1", "type": "llm", "name": "m", "seq": 0,
                "dur_ms": 3, "model": "gpt-4o", "prompt_tokens": 10, "completion_tokens": 5})
    s.close_run("new1", "done", now + 10)
    st = s.stats(days=30)
    assert st["models"] and st["models"][0]["model"] == "gpt-4o"
    assert any(d["runs"] for d in st["daily"])
    res = s.prune(30)
    assert res["pruned_runs"] == 1
    assert s.run_detail("old1") is None and s.run_detail("new1") is not None
    print("prune/stats OK")


def test_tracer_metadata_session_tags():
    """The generic wire-level idiom: config metadata/tags from ANY LangChain app."""
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    tr = SpanBuilder(db_sink(s), run_name="t")
    tr.on_chain_start({}, {"x": 1}, run_id="root1", parent_run_id=None,
                      metadata={"windhover_session": "sess-42", "windhover_tags": ["custom"]},
                      tags=["graph:step:1", "langsmith:hidden", "mytag"])
    tr.on_chain_end({}, run_id="root1")
    d = s.run_detail(tr.run_id)
    assert d is not None and d["session"] == "sess-42"
    assert set(d["tags"]) == {"custom", "mytag"}   # internal langgraph tags filtered
    print("tracer metadata session/tags OK")


def _tiny_graph(fail=False):
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END

    class St(TypedDict):
        x: int

    def boom(st):
        raise RuntimeError("kaboom from boom()")

    def inc(st):
        return {"x": st["x"] + 1}

    g = StateGraph(St)
    g.add_node("inc", boom if fail else inc)
    g.add_edge(START, "inc"); g.add_edge("inc", END)
    return g.compile()


def test_source_extraction():
    from windhover.extract import sources
    app = _tiny_graph()
    src = sources(app)
    assert "inc" in src, f"no source for inc: {list(src)}"
    s = src["inc"]
    assert "def inc(st):" in s["code"]
    assert s["file"].endswith(".py") and s["line_start"] > 0
    assert s["line_end"] >= s["line_start"]
    print("source extraction OK —", s["file"].split("/")[-1], f"L{s['line_start']}")


def test_error_traceback_capture():
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    app = _tiny_graph(fail=True)
    tracer = SpanBuilder(db_sink(s), run_name="failing")
    try:
        app.invoke({"x": 1}, config={"callbacks": [tracer]})
        assert False, "graph should have raised"
    except RuntimeError:
        pass
    time.sleep(.1)
    d = s.run_detail(tracer.run_id)
    assert d is not None and d["status"] == "error"
    err_spans = [sp for sp in d["spans"] if sp["status"] == "error" and sp["type"] == "node"]
    assert err_spans, "no error node span recorded"
    err = err_spans[0]["error"]
    # full traceback with file+line so the UI can highlight the source line
    assert "Traceback" in err and 'File "' in err and "kaboom from boom()" in err
    assert d["error"] and "kaboom" in d["error"]
    print("error traceback capture OK")


def test_retriever_spans():
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    tr = SpanBuilder(db_sink(s), run_name="rag")
    tr.on_chain_start({}, {"q": "hi"}, run_id="root", parent_run_id=None)

    class Doc:
        def __init__(self, c): self.page_content, self.metadata = c, {"src": "kb.md"}
    tr.on_retriever_start({"name": "VectorStoreRetriever"}, "pricing policy",
                          run_id="ret1", parent_run_id="root")
    tr.on_retriever_end([Doc("alpha"), Doc("beta")], run_id="ret1")
    tr.on_chain_end({}, run_id="root")
    d = s.run_detail(tr.run_id)
    ret = [sp for sp in d["spans"] if sp["type"] == "retriever"]
    assert ret and ret[0]["name"] == "VectorStoreRetriever"
    assert ret[0]["input"] == "pricing policy"
    assert ret[0]["output"]["count"] == 2
    assert ret[0]["output"]["documents"][0]["content"] == "alpha"
    print("retriever spans OK")


def test_interrupt_status():
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    tr = SpanBuilder(db_sink(s), run_name="hitl")
    tr.on_chain_start({}, {"x": 1}, run_id="root", parent_run_id=None)
    tr.on_chain_end({"__interrupt__": [{"value": "approve the wire transfer?"}]}, run_id="root")
    d = s.run_detail(tr.run_id)
    assert d["status"] == "interrupted", d["status"]
    ints = [sp for sp in d["spans"] if sp["type"] == "interrupt"]
    assert ints and "approve the wire transfer?" in json.dumps(ints[0]["output"])
    print("interrupt status OK")


def test_xray_topology():
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from windhover.extract import topology

    class St(TypedDict):
        x: int
    child = StateGraph(St)
    child.add_node("inner_a", lambda st: {"x": st["x"] + 1})
    child.add_node("inner_b", lambda st: {"x": st["x"] * 2})
    child.add_edge(START, "inner_a"); child.add_edge("inner_a", "inner_b")
    child.add_edge("inner_b", END)
    parent = StateGraph(St)
    parent.add_node("child", child.compile())
    parent.add_edge(START, "child"); parent.add_edge("child", END)
    app = parent.compile()
    plain, xray = topology(app), topology(app, xray=True)
    assert len(xray["nodes"]) > len(plain["nodes"]), (len(plain["nodes"]), len(xray["nodes"]))
    assert any("inner_a" in n["id"] for n in xray["nodes"])
    print("xray topology OK —", len(plain["nodes"]), "->", len(xray["nodes"]), "nodes")


def test_auth_check():
    os.environ.setdefault("WINDHOVER_DB", tempfile.mktemp(suffix=".db"))
    from windhover.server import _auth_ok
    assert _auth_ok("", "/api/runs", "", "")                     # no token configured -> open
    assert _auth_ok("s3cret", "/", "", "")                       # UI shell never gated
    assert _auth_ok("s3cret", "/static/x.js", "", "")
    assert not _auth_ok("s3cret", "/api/runs", "", "")           # gated without creds
    assert _auth_ok("s3cret", "/api/runs", "Bearer s3cret", "")
    assert _auth_ok("s3cret", "/api/runs", "bearer  s3cret", "") # case/space tolerant
    assert _auth_ok("s3cret", "/api/events", "", "s3cret")       # query token (SSE)
    assert not _auth_ok("s3cret", "/api/runs", "Bearer wrong", "")
    print("auth check OK")


def test_thread_capture_and_history():
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)

    class St(TypedDict):
        x: int
    g = StateGraph(St)
    g.add_node("inc", lambda st: {"x": st["x"] + 1})
    g.add_edge(START, "inc"); g.add_edge("inc", END)
    app = g.compile(checkpointer=MemorySaver())
    tr = SpanBuilder(db_sink(s), run_name="tt")
    app.invoke({"x": 1}, config={"callbacks": [tr],
                                 "configurable": {"thread_id": "thread-9"}})
    time.sleep(.1)
    d = s.run_detail(tr.run_id)
    assert d["thread_id"] == "thread-9", d["thread_id"]
    steps = list(app.get_state_history({"configurable": {"thread_id": "thread-9"}}))
    assert len(steps) >= 2   # input checkpoint + node step
    print("thread capture + history OK —", len(steps), "checkpoints")


def test_datasets_and_scoring():
    os.environ.setdefault("WINDHOVER_DB", tempfile.mktemp(suffix=".db"))
    from windhover.server import _expected_score
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    ds = s.add_dataset("golden", [{"input": {"n": 2}, "expected": 6},
                                  {"input": {"n": 5}}])
    assert s.datasets()[0]["n_items"] == 2
    assert s.dataset("golden")["id"] == ds["id"]
    assert _expected_score(6, {"n": 6}) == 1.0
    assert _expected_score(7, {"n": 6}) == 0.0
    assert _expected_score("flattened", {"answer": "tiers were flattened in Q3"}) == 1.0
    assert _expected_score("missing", {"answer": "nope"}) == 0.0
    assert s.delete_dataset("golden") and not s.datasets()
    print("datasets + scoring OK")


def test_ttft_retry_custom_event_usage_detail():
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    tr = SpanBuilder(db_sink(s), run_name="deep")
    tr.on_chain_start({}, {"q": 1}, run_id="root", parent_run_id=None)
    tr.on_chat_model_start({"kwargs": {"model": "gpt-4o"}}, [[]],
                           run_id="llm1", parent_run_id="root")
    time.sleep(0.02)
    tr.on_llm_new_token("Hel", run_id="llm1")          # -> ttft stamped
    tr.on_llm_new_token("lo", run_id="llm1")           # ignored (only first counts)

    class RS:  # tenacity RetryCallState stand-in
        attempt_number = 3
    tr.on_retry(RS(), run_id="llm1")

    class Msg:
        content = "Hello"
        usage_metadata = {"input_tokens": 900, "output_tokens": 40,
                          "input_token_details": {"cache_read": 700},
                          "output_token_details": {"reasoning": 12}}
    class Gen:
        text = "Hello"; message = Msg()
    class Resp:
        llm_output = None; generations = [[Gen()]]
    tr.on_llm_end(Resp(), run_id="llm1")
    tr.on_custom_event("checkpoint-saved", {"rows": 42}, run_id="root")
    tr.on_chain_end({}, run_id="root")

    d = s.run_detail(tr.run_id)
    llm = [x for x in d["spans"] if x["type"] == "llm"][0]
    assert llm["ttft_ms"] is not None and llm["ttft_ms"] >= 15
    assert llm["retries"] == 3
    assert llm["usage_detail"] == {"input_cache_read": 700, "output_reasoning": 12}
    assert llm["prompt_tokens"] == 900
    ev = [x for x in d["spans"] if x["type"] == "event"]
    assert ev and ev[0]["name"] == "checkpoint-saved" and ev[0]["output"] == {"rows": 42}
    print("ttft/retry/custom-event/usage-detail OK")


def test_demo_memory_and_events_end_to_end():
    """Demo graph writes long-term memory + dispatches a custom event — both
    must be observable."""
    import importlib
    import windhover.demo_graph as dg
    importlib.reload(dg)  # fresh InMemoryStore per test
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    tr = SpanBuilder(db_sink(s), run_name="demo")
    dg.graph.invoke({"n": 7}, config={"callbacks": [tr],
                                      "configurable": {"thread_id": "mem-t"}})
    time.sleep(.1)
    d = s.run_detail(tr.run_id)
    ev = [x for x in d["spans"] if x["type"] == "event"]
    assert ev and ev[0]["name"] == "summary-ready", [x["type"] for x in d["spans"]]
    assert dg.graph.store is not None
    items = dg.graph.store.search(("demo", "summaries"))
    assert items and items[0].value["n"] == 21   # n=7 -> grow x3
    assert ("demo", "summaries") in dg.graph.store.list_namespaces()
    print("demo memory + custom event end-to-end OK")


def test_hitl_interrupt_resume():
    """Dynamic interrupt pauses (status interrupted); Command(resume) finishes."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command, interrupt
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)

    class St(TypedDict):
        x: int
    def gate(st):
        ok = interrupt({"question": f"allow x={st['x']}?"})
        return {"x": st["x"] if ok else 0}
    g = StateGraph(St)
    g.add_node("gate", gate)
    g.add_edge(START, "gate"); g.add_edge("gate", END)
    app = g.compile(checkpointer=MemorySaver())
    cfgc = {"configurable": {"thread_id": "hitl-1"}}

    tr1 = SpanBuilder(db_sink(s), run_name="hitl")
    out = app.invoke({"x": 5}, config={"callbacks": [tr1], **cfgc})
    assert "__interrupt__" in out
    time.sleep(.05)
    assert s.run_detail(tr1.run_id)["status"] == "interrupted"

    tr2 = SpanBuilder(db_sink(s), run_name="hitl")
    out2 = app.invoke(Command(resume=True), config={"callbacks": [tr2], **cfgc})
    assert out2["x"] == 5
    time.sleep(.05)
    assert s.run_detail(tr2.run_id)["status"] == "done"
    print("HITL interrupt/resume OK")


def test_static_breakpoint_and_state_edit():
    """interrupt_before pauses with pending next; update_state edit sticks."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)

    class St(TypedDict):
        x: int
    g = StateGraph(St)
    g.add_node("inc", lambda st: {"x": st["x"] + 1})
    g.add_node("mul", lambda st: {"x": st["x"] * 10})
    g.add_edge(START, "inc"); g.add_edge("inc", "mul"); g.add_edge("mul", END)
    app = g.compile(checkpointer=MemorySaver())
    cfgc = {"configurable": {"thread_id": "bp-1"}}

    tr = SpanBuilder(db_sink(s), run_name="bp")
    for _ in app.stream({"x": 1}, config={"callbacks": [tr], **cfgc},
                        stream_mode="updates", interrupt_before=["mul"]):
        pass
    st = app.get_state(cfgc)
    assert st.next == ("mul",), st.next          # paused before mul
    app.update_state(cfgc, {"x": 100})           # human edits the state
    out = app.invoke(None, config=cfgc)          # continue
    assert out["x"] == 1000, out                 # 100 * 10 — edit took effect
    print("static breakpoint + state edit OK")


def test_streaming_partials_params_tools():
    """Token stream flushes partial spans (live 'model typing'); invocation
    params + offered tools captured per LLM span."""
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    tr = SpanBuilder(db_sink(s), run_name="stream")
    tr.on_chain_start({}, {"q": 1}, run_id="root", parent_run_id=None)
    tr.on_chat_model_start(
        {"kwargs": {"model": "fable-fast"}}, [[]], run_id="L", parent_run_id="root",
        invocation_params={"model": "fable-fast", "temperature": 0.1, "stream": True,
                           "tools": [{"function": {"name": "get_weather"}}]})
    tr.on_llm_new_token("Hello ", run_id="L")
    tr._open["L"]["flushed"] = 0            # force the throttle window open
    tr.on_llm_new_token("wind", run_id="L")  # -> partial flush
    mid = s.run_detail(tr.run_id)
    llm_mid = [x for x in mid["spans"] if x["type"] == "llm"][0]
    assert llm_mid["status"] == "running" and "Hello wind" in llm_mid["output"]

    class Msg:
        content = "Hello windhover"
        usage_metadata = {"input_tokens": 10, "output_tokens": 3}
    class Gen:
        text = "Hello windhover"; message = Msg()
    class Resp:
        llm_output = None; generations = [[Gen()]]
    tr.on_llm_end(Resp(), run_id="L")
    tr.on_chain_end({}, run_id="root")
    d = s.run_detail(tr.run_id)
    llm = [x for x in d["spans"] if x["type"] == "llm"][0]
    assert llm["status"] == "ok" and llm["output"] == "Hello windhover"
    assert llm["params"]["temperature"] == 0.1
    assert llm["params"]["tools_offered"] == ["get_weather"]
    assert len([x for x in d["spans"] if x["type"] == "llm"]) == 1  # partial replaced
    print("streaming partials + params/tools OK")


def test_edge_labels_and_node_metadata():
    from typing import TypedDict, Literal
    from langgraph.graph import StateGraph, START, END
    from windhover.extract import topology

    class St(TypedDict):
        x: int
    def route(st) -> Literal["yes", "no"]:
        return "yes"
    g = StateGraph(St)
    g.add_node("decide", lambda st: {"x": st["x"]},
               metadata={"owner": "team-a", "doc": "routing"})
    g.add_node("approve", lambda st: {"x": 1})
    g.add_node("reject", lambda st: {"x": 0})
    g.add_edge(START, "decide")
    g.add_conditional_edges("decide", route, {"yes": "approve", "no": "reject"})
    g.add_edge("approve", END); g.add_edge("reject", END)
    topo = topology(g.compile())
    decide = next(n for n in topo["nodes"] if n["id"] == "decide")
    assert decide["metadata"] == {"owner": "team-a", "doc": "routing"}
    labels = {(e["source"], e["target"]): e["label"] for e in topo["edges"]}
    assert labels[("decide", "approve")] == "yes"
    assert labels[("decide", "reject")] == "no"
    assert labels[("__start__", "decide")] is None
    print("edge labels + node metadata OK")


def test_multi_root_and_bare_llm():
    """One tracer, many executions: batch/concurrent invokes each get their own
    run; a bare llm call with no chain opens an implicit run."""
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    tr = SpanBuilder(db_sink(s), run_name="multi")
    for i in (1, 2):
        tr.on_chain_start({}, {"a": i}, run_id=f"r{i}", parent_run_id=None)
        tr.on_chain_start({}, {}, run_id=f"n{i}", parent_run_id=f"r{i}",
                          metadata={"langgraph_node": "work"})
        tr.on_chain_end({"ok": i}, run_id=f"n{i}")
        tr.on_chain_end({"ok": i}, run_id=f"r{i}")
    runs = s.runs()["runs"]
    assert len(runs) == 2 and all(r["status"] == "done" and r["node_count"] == 1
                                  for r in runs)
    assert len({r["id"] for r in runs}) == 2

    tr2 = SpanBuilder(db_sink(s), run_name="bare")
    tr2.on_chat_model_start({"kwargs": {"model": "gpt-4o"}}, [[]],
                            run_id="L", parent_run_id=None)
    class M:
        content = "hi"; usage_metadata = {"input_tokens": 5, "output_tokens": 2}
    class G:
        text = "hi"; message = M()
    class R:
        llm_output = None; generations = [[G()]]
    tr2.on_llm_end(R(), run_id="L")
    bare = [r for r in s.runs()["runs"] if r["graph"] == "bare"]
    assert bare and bare[0]["status"] == "done" and bare[0]["llm_calls"] == 1
    print("multi-root + bare-llm OK")


def test_functional_api_tracing():
    """@entrypoint/@task graphs trace like node graphs."""
    from langgraph.func import entrypoint, task
    from langgraph.checkpoint.memory import MemorySaver
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)

    @task
    def double(x: int) -> int:
        return x * 2

    @entrypoint(checkpointer=MemorySaver())
    def flow(x: int) -> int:
        return double(x).result() + 1

    tr = SpanBuilder(db_sink(s), run_name="func")
    out = flow.invoke(3, config={"callbacks": [tr],
                                 "configurable": {"thread_id": "fx"}})
    assert out == 7
    time.sleep(.1)
    d = s.run_detail(tr.run_id)
    assert d["status"] == "done"
    names = {sp["name"] for sp in d["spans"] if sp["type"] == "node"}
    assert "double" in names, names
    print("functional API tracing OK —", sorted(names))


def test_message_serialization():
    from windhover.tracer import _trunc
    class FakeMsg:
        type = "human"; content = "hello there"; tool_calls = None
    v = _trunc({"messages": [FakeMsg()]})
    assert v == {"messages": [{"role": "human", "content": "hello there"}]}, v
    print("message serialization OK")


def test_langgraph_json_discovery():
    import json as _json
    d = tempfile.mkdtemp()
    gdir = os.path.join(d, "src"); os.makedirs(gdir)
    open(os.path.join(gdir, "app.py"), "w").write("graph = None\nother = None\n")
    open(os.path.join(d, "langgraph.json"), "w").write(
        _json.dumps({"graphs": {"main": "./src/app.py:graph",
                                "side": "./src/app.py:other"}}))
    os.environ.pop("WINDHOVER_GRAPH", None)
    os.environ["WINDHOVER_GRAPH_DIR"] = d
    try:
        from windhover.config import Config
        cfg = Config.from_env()
        assert cfg.graphs == (("main", "app:graph"), ("side", "app:other")), cfg.graphs
        assert cfg.graph_ref == "app:graph"          # back-compat: first graph
        assert cfg.graph_dir == gdir, cfg.graph_dir
    finally:
        os.environ.pop("WINDHOVER_GRAPH_DIR", None)
    print("langgraph.json multi-discovery OK")


def test_env_multi_graph_parsing():
    from windhover.config import Config
    os.environ["WINDHOVER_GRAPH"] = "alpha=m1:g1, m2:g2 ,beta=m3:g3"
    try:
        cfg = Config.from_env()
        assert cfg.graphs == (("alpha", "m1:g1"), ("m2:g2", "m2:g2"),
                              ("beta", "m3:g3")), cfg.graphs
    finally:
        os.environ.pop("WINDHOVER_GRAPH", None)
    print("env multi-graph parsing OK")


def test_graph_scoped_stats_and_sessions():
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    now = int(time.time() * 1000)
    for i, g in enumerate(("alpha", "alpha", "beta")):
        rid = f"g{i}"
        s.open_run({"id": rid, "graph": g, "session": f"sess-{g}", "started_ms": now - i})
        s.add_span({"id": f"s{i}", "run_id": rid, "type": "node", "name": "shared_name",
                    "seq": 0, "dur_ms": 100 * (i + 1)})
        s.add_span({"id": f"l{i}", "run_id": rid, "type": "llm", "name": "m", "seq": 1,
                    "model": f"model-{g}", "dur_ms": 5, "prompt_tokens": 10,
                    "completion_tokens": 1})
        s.close_run(rid, "done", now + 10)
    st_a = s.stats(graph="alpha")
    assert st_a["totals"]["runs"] == 2
    assert st_a["per_node"][0]["n"] == 2            # only alpha's spans, not beta's
    assert [m["model"] for m in st_a["models"]] == ["model-alpha"]
    st_all = s.stats()
    assert st_all["totals"]["runs"] == 3
    # cross-graph per_node groups by (graph, name): shared_name stays TWO rows
    shared = [r for r in st_all["per_node"] if r["name"] == "shared_name"]
    assert len(shared) == 2 and {r["graph"] for r in shared} == {"alpha", "beta"}
    assert {r["n"] for r in shared} == {2, 1}
    ses_b = s.sessions(graph="beta")
    assert [x["session"] for x in ses_b] == ["sess-beta"]
    all_ses = s.sessions()
    assert len(all_ses) == 2
    # each session reports the graphs it touched
    by = {x["session"]: x["graphs"] for x in all_ses}
    assert by["sess-alpha"] == ["alpha"] and by["sess-beta"] == ["beta"]
    print("graph-scoped stats/sessions OK")


def test_graph_group_prefix_scope():
    """Path-style graph names form a subject tree; a "prefix/*" filter scopes
    runs/sessions/stats to that subtree at any depth — and never to lookalike
    names (opsy must not match ops/*)."""
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    now = int(time.time() * 1000)
    graphs = ["ops/etl/nightly", "ops/etl/backfill", "ops/monitor/heartbeat",
              "support/triage", "chat", "opsy"]
    for i, g in enumerate(graphs):
        rid = f"p{i}"
        s.open_run({"id": rid, "graph": g, "session": f"s-{g}", "started_ms": now - i})
        s.add_span({"id": f"sp{i}", "run_id": rid, "type": "node", "name": "step",
                    "seq": 0, "dur_ms": 10})
        s.close_run(rid, "done", now + 1)
    assert s.runs(graph="ops/*")["total"] == 3
    assert s.runs(graph="ops/etl/*")["total"] == 2
    assert s.runs(graph="ops/etl/nightly")["total"] == 1
    assert s.runs(graph="opsy")["total"] == 1          # exact match untouched
    assert {x["session"] for x in s.sessions(graph="ops/*")} == \
        {"s-ops/etl/nightly", "s-ops/etl/backfill", "s-ops/monitor/heartbeat"}
    st = s.stats(graph="ops/etl/*")
    assert st["totals"]["runs"] == 2
    # subject scope groups per-node rows per graph (same-named nodes never merge)
    assert {r["graph"] for r in st["per_node"]} == {"ops/etl/nightly", "ops/etl/backfill"}
    assert all(r["n"] == 1 for r in st["per_node"])
    assert any(d["runs"] == 2 for d in st["daily"])
    # exact single-graph stats keep the flat per-node shape
    st1 = s.stats(graph="ops/etl/nightly")
    assert st1["per_node"] and "graph" not in st1["per_node"][0]
    print("graph group prefix scope OK")


def test_env_graph_names_with_slashes():
    from windhover.config import Config
    os.environ["WINDHOVER_GRAPH"] = "ops/etl/nightly=m1:g1,support/triage=m2:g2"
    try:
        cfg = Config.from_env()
        assert cfg.graphs == (("ops/etl/nightly", "m1:g1"),
                              ("support/triage", "m2:g2")), cfg.graphs
    finally:
        os.environ.pop("WINDHOVER_GRAPH", None)
    print("env graph names with slashes OK")


def test_nonblocking_sink_and_webhook_hook():
    # sink to an unreachable host must return instantly and never raise
    from windhover.tracer import http_sink
    sink = http_sink("http://10.255.255.1:9", max_queue=10)
    t0 = time.time()
    for i in range(50):
        sink({"kind": "span", "i": i})
    assert time.time() - t0 < 0.5, "sink blocked the caller"

    # store hook fires with the closing status (webhook wiring point)
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    fired = []
    s.on_run_closed = lambda summary: fired.append(summary)
    s.open_run({"id": "w1", "graph": "g", "started_ms": 1000})
    s.close_run("w1", "error", 1500, error="Boom: it broke")
    assert fired and fired[0]["status"] == "error" and "Boom" in fired[0]["error"]
    print("non-blocking sink + webhook hook OK")


def test_fork_not_marked_interrupted():
    """Pending-next check must query by thread only — a config carrying
    checkpoint_id reads the historical checkpoint and falsely flags forks."""
    os.environ.setdefault("WINDHOVER_DB", tempfile.mktemp(suffix=".db"))
    from windhover.server import _pending_next
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver

    class St(TypedDict):
        x: int
    g = StateGraph(St)
    g.add_node("inc", lambda st: {"x": st["x"] + 1})
    g.add_node("mul", lambda st: {"x": st["x"] * 10})
    g.add_edge(START, "inc"); g.add_edge("inc", "mul"); g.add_edge("mul", END)
    app = g.compile(checkpointer=MemorySaver())
    cfgc = {"configurable": {"thread_id": "fork-t"}}
    app.invoke({"x": 1}, config=cfgc)                       # completes
    hist = list(app.get_state_history(cfgc))
    early = next(s for s in hist if s.next)                 # historical, pending
    cid = early.config["configurable"]["checkpoint_id"]
    fork_cfg = {"configurable": {"thread_id": "fork-t", "checkpoint_id": cid}}
    assert app.get_state(fork_cfg).next, "sanity: historical checkpoint has next"
    assert _pending_next(app, fork_cfg) == [], "must ignore checkpoint_id"
    print("fork-not-interrupted OK")


def test_score_rejects_nonfinite():
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    s.open_run({"id": "sf1", "graph": "g", "started_ms": 1})
    s.close_run("sf1", "done", 2)
    assert s.add_score("sf1", "ok", 0.5) is not None
    assert s.add_score("sf1", "bad", float("inf")) is None
    assert s.add_score("sf1", "bad", float("nan")) is None
    # the runs API payload must stay JSON-parseable
    import json as _json
    _json.loads(_json.dumps(s.runs()["runs"][0], allow_nan=False))
    print("non-finite score rejection OK")


def test_tracer_cleanup_no_leak():
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    tr = SpanBuilder(db_sink(s), run_name="leak")
    for i in range(20):
        tr.on_chain_start({}, {"i": i}, run_id=f"r{i}", parent_run_id=None)
        tr.on_chain_start({}, {}, run_id=f"n{i}", parent_run_id=f"r{i}",
                          metadata={"langgraph_node": "w"})
        tr.on_chain_end({}, run_id=f"n{i}")
        tr.on_chain_end({}, run_id=f"r{i}")
    assert s.runs(limit=100)["total"] == 20
    assert not tr._runs and not tr._root_of and not tr._span_of and not tr._open, (
        len(tr._runs), len(tr._root_of), len(tr._span_of), len(tr._open))
    print("tracer cleanup (no leak) OK")


if __name__ == "__main__":
    test_cost(); print("cost OK")
    test_store_roundtrip()
    test_tracer_local_graph()
    test_search_filters_scores_sessions()
    test_prune_and_stats()
    test_tracer_metadata_session_tags()
    test_source_extraction()
    test_error_traceback_capture()
    test_retriever_spans()
    test_interrupt_status()
    test_xray_topology()
    test_auth_check()
    test_thread_capture_and_history()
    test_datasets_and_scoring()
    test_ttft_retry_custom_event_usage_detail()
    test_demo_memory_and_events_end_to_end()
    test_hitl_interrupt_resume()
    test_static_breakpoint_and_state_edit()
    test_streaming_partials_params_tools()
    test_edge_labels_and_node_metadata()
    test_multi_root_and_bare_llm()
    test_functional_api_tracing()
    test_message_serialization()
    test_langgraph_json_discovery()
    test_env_multi_graph_parsing()
    test_graph_scoped_stats_and_sessions()
    test_nonblocking_sink_and_webhook_hook()
    test_fork_not_marked_interrupted()
    test_score_rejects_nonfinite()
    test_tracer_cleanup_no_leak()
    print("ALL SMOKE TESTS PASSED")


def test_overview_fleet():
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    now = int(time.time() * 1000)
    # alpha: 4 done + 1 interrupted (awaiting approval, with a question payload)
    for i in range(4):
        rid = f"a{i}"
        s.open_run({"id": rid, "graph": "alpha", "started_ms": now - 1000 * (i + 2)})
        s.close_run(rid, "done", now - 1000 * (i + 1))
    s.open_run({"id": "a-wait", "graph": "alpha", "thread_id": "t1",
                "started_ms": now - 500})
    s.add_span({"id": "iv1", "run_id": "a-wait", "type": "interrupt", "name": "interrupt",
                "seq": 0, "output": {"question": "Approve refund #841?"}})
    s.close_run("a-wait", "interrupted", now - 400)
    # beta: 1 error, 1 running
    s.open_run({"id": "b0", "graph": "beta", "started_ms": now - 3000})
    s.close_run("b0", "error", now - 2900, error="boom")
    s.open_run({"id": "b-run", "graph": "beta", "started_ms": now - 100})
    # gamma: ingest-only graph with an old run (outside the 7d window)
    s.open_run({"id": "c0", "graph": "gamma", "started_ms": now - 30 * 86400_000})
    s.close_run("c0", "done", now - 30 * 86400_000 + 5)
    # delta: an interrupted run that was ALREADY resumed (newer run, same thread)
    # — handled, must NOT appear in attention
    s.open_run({"id": "d-old-wait", "graph": "delta", "thread_id": "t9",
                "started_ms": now - 9000})
    s.close_run("d-old-wait", "interrupted", now - 8900)
    s.open_run({"id": "d-resume", "graph": "delta", "thread_id": "t9",
                "started_ms": now - 8000})
    s.close_run("d-resume", "done", now - 7900)

    ov = s.overview(days=7, recent_n=3, serving=("alpha", "beta"))
    by = {g["name"]: g for g in ov["graphs"]}
    # every graph appears; serving flags honest
    assert by["alpha"]["serving"] and by["beta"]["serving"] and not by["gamma"]["serving"]
    # counts inside the window
    assert by["alpha"]["runs_7d"] == 5 and by["alpha"]["errors_7d"] == 0
    # resumed interrupt is handled — excluded from attention and rollups
    assert "d-old-wait" not in [a["id"] for a in ov["attention"]]
    assert by["delta"]["interrupted_now"] == 0
    # daily sparkline buckets: one slot per day in the window, sums match counts
    for g in ov["graphs"]:
        assert len(g["daily"]) == 7
    assert sum(d["runs"] for d in by["alpha"]["daily"]) == 5
    assert sum(d["errors"] for d in by["beta"]["daily"]) == 1
    assert sum(d["runs"] for d in by["gamma"]["daily"]) == 0   # old run outside window
    assert by["beta"]["errors_7d"] == 1
    assert by["gamma"]["runs_7d"] == 0          # old run outside window
    # attention: newest first, both statuses, question surfaced, waiting clock
    att = ov["attention"]
    assert [a["id"] for a in att] == ["b-run", "a-wait"]
    aw = att[1]
    assert aw["status"] == "interrupted" and "Approve refund #841?" in aw["interrupt_summary"]
    assert aw["waiting_ms"] >= 0 and aw["thread_id"] == "t1"
    # per-graph rollups + recent capped at 3, newest first, last_run = newest
    assert by["alpha"]["interrupted_now"] == 1 and by["beta"]["running_now"] == 1
    assert len(by["alpha"]["recent"]) == 3
    assert by["alpha"]["recent"][0]["id"] == "a-wait" == by["alpha"]["last_run"]["id"]
    # gamma still lists its ancient run in recent (review, not amnesia)
    assert by["gamma"]["recent"][0]["id"] == "c0"
    print("overview fleet OK")


def test_awaiting_count_and_webhook_parse_and_digest():
    # awaiting_count mirrors the overview attention rule
    p = tempfile.mktemp(suffix=".db")
    s = Store(p)
    now = int(time.time() * 1000)
    s.open_run({"id": "w1", "graph": "g", "thread_id": "ta", "started_ms": now - 100})
    s.close_run("w1", "interrupted", now - 90)
    s.open_run({"id": "w2", "graph": "g", "thread_id": "tb", "started_ms": now - 80})
    s.close_run("w2", "interrupted", now - 70)
    assert s.awaiting_count() == 2
    s.open_run({"id": "w2r", "graph": "g", "thread_id": "tb", "started_ms": now - 50})
    s.close_run("w2r", "done", now - 40)
    assert s.awaiting_count() == 1          # tb was resumed -> handled

    # WINDHOVER_WEBHOOK parsing: default + per-graph, query strings survive
    from windhover.config import Config
    d, m = Config._parse_webhooks("https://hooks.example/a?x=1")
    assert d == "https://hooks.example/a?x=1" and m == {}
    d, m = Config._parse_webhooks(
        "https://hooks.example/default, billing=https://hooks.example/billing?k=v ,other=https://h.example/o")
    assert d == "https://hooks.example/default"
    assert m == {"billing": "https://hooks.example/billing?k=v", "other": "https://h.example/o"}
    d, m = Config._parse_webhooks("")
    assert d == "" and m == {}

    # digest builder: quiet day -> None; busy day -> counts in the body
    from windhover.push import digest_summary
    assert digest_summary({"graphs": [{"name": "g", "runs_7d": 0, "errors_7d": 0}],
                           "attention": []}) is None
    msg = digest_summary({
        "graphs": [{"name": "a", "runs_7d": 5, "errors_7d": 1},
                   {"name": "b", "runs_7d": 2, "errors_7d": 0}],
        "attention": [{"status": "interrupted"}, {"status": "running"}]})
    assert msg["tag"] == "windhover-digest" and msg["url"] == "/#fleet"
    assert "7 runs" in msg["body"] and "1 error" in msg["body"]
    assert "1 awaiting approval" in msg["body"]
    print("awaiting/webhook-parse/digest OK")


def test_artifacts_extract_classify_resolve():
    from windhover.artifacts import extract_paths, classify, run_artifacts, resolve
    # extraction: nested structures, dedupe, extension filter, non-paths ignored
    obj = {"docs": ["/out/report.docx", "/out/report.docx"],
           "chart": {"png": "/tmp/chart.png"},
           "notes": ["not/a/path.docx", "no extension /tmp/file", "plain text"],
           "win": r"C:\out\sheet.xlsx", "home": "~/data/results.csv",
           "url": "https://example.com/page.html"}
    got = extract_paths(obj)
    assert got == ["/out/report.docx", "/tmp/chart.png", r"C:\out\sheet.xlsx",
                   "~/data/results.csv"], got
    # classification: inline vs download
    assert classify("/a/b.html")["kind"] == "html" and classify("/a/b.html")["inline"]
    assert classify("/a/b.pdf")["inline"] and classify("/a/b.py")["kind"] == "text"
    assert classify("/a/b.docx")["inline"] is False
    assert classify("/a/b.PNG")["kind"] == "image"
    # extended coverage: office/media/markdown render; legacy/binary download-only
    assert classify("/a/b.md")["kind"] == "markdown"
    assert classify("/a/b.mp4")["kind"] == "video" and classify("/a/b.mp4")["inline"]
    assert classify("/a/b.mp3")["kind"] == "audio"
    assert classify("/a/b.xlsx")["kind"] == "sheet" and classify("/a/b.xls")["kind"] == "sheet"
    assert classify("/a/b.pptx")["kind"] == "slides" and classify("/a/b.pptx")["inline"] is False
    assert classify("/a/b.doc")["kind"] == "file"      # legacy Word: no mammoth render
    assert classify("/a/b.ppt")["kind"] == "file" and classify("/a/b.zip")["kind"] == "file"
    assert classify("/a/model.pt")["kind"] == "file"   # ML artifact, download-only

    # run_artifacts + resolve against real files
    d = tempfile.mkdtemp()
    real = os.path.join(d, "report.html")
    open(real, "w").write("<h1>hi</h1>")
    run = {"input": {"x": 1},
           "spans": [{"output": {"docs": [real, os.path.join(d, "gone.pdf")]}}]}
    arts = run_artifacts(run)
    assert [a["exists"] for a in arts] == [True, False]
    assert arts[0]["size"] == 11 and arts[0]["name"] == "report.html"
    # resolve: recorded+exists -> real path; recorded+missing -> None; unrecorded -> None
    assert resolve(run, real) == real
    assert resolve(run, os.path.join(d, "gone.pdf")) is None
    assert resolve(run, "/etc/passwd") is None          # never, even though it exists
    print("artifacts extract/classify/resolve OK")


def test_version_single_source():
    # __version__ must come from installed dist metadata (or the dev fallback),
    # never a hand-maintained string — the 0.32.0 release shipped with three
    # divergent versions (pyproject 0.32.0 / __init__ 0.3.0 / UI footer v0.31).
    from importlib.metadata import PackageNotFoundError, version
    import windhover
    try:
        expected = version("windhover")
    except PackageNotFoundError:
        expected = "0.0.0.dev0"
    assert windhover.__version__ == expected
    html = open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "windhover", "static", "index.html")).read()
    import re
    assert not re.search(r'id="ver">v?\d', html), "footer version must not be hardcoded"


def test_pricing_env_override(tmp_path, monkeypatch):
    # WINDHOVER_PRICING points cost tracking at a deployment's own rate table
    import windhover.tracer as tr
    p = tmp_path / "pricing.json"
    p.write_text('{"my-model": {"input": 1.0, "output": 2.0}, "_note": "meta"}')
    monkeypatch.setenv("WINDHOVER_PRICING", str(p))
    tr._PRICING = None
    try:
        assert set(tr._load_pricing()) == {"my-model"}   # env table wins, _-keys dropped
    finally:
        tr._PRICING = None
