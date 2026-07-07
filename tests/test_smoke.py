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
    print("ALL SMOKE TESTS PASSED")
