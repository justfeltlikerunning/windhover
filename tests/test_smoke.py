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
    import json as _json, importlib
    d = tempfile.mkdtemp()
    gdir = os.path.join(d, "src"); os.makedirs(gdir)
    open(os.path.join(gdir, "app.py"), "w").write("graph = None\n")
    open(os.path.join(d, "langgraph.json"), "w").write(
        _json.dumps({"graphs": {"main": "./src/app.py:graph"}}))
    os.environ.pop("WINDHOVER_GRAPH", None)
    os.environ["WINDHOVER_GRAPH_DIR"] = d
    try:
        from windhover.config import Config
        cfg = Config.from_env()
        assert cfg.graph_ref == "app:graph", cfg.graph_ref
        assert cfg.graph_dir == gdir, cfg.graph_dir
    finally:
        os.environ.pop("WINDHOVER_GRAPH_DIR", None)
    print("langgraph.json discovery OK")


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
    print("ALL SMOKE TESTS PASSED")
