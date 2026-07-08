"""Extract a compiled graph's topology, input schema, and per-node source as
JSON. Run as a subprocess so the live watcher always sees current-on-disk code
without importlib.reload fragility.

    python -m windhover.extract "module:attr" "/graph/dir"
"""
import sys, json, importlib, inspect, functools


def topology(graph, xray: bool = False) -> dict:
    g = graph.get_graph(xray=True) if xray else graph.get_graph()
    nodes = [{"id": nid, "label": str(getattr(n, "name", nid)).replace("__", ""),
              "terminal": nid.endswith(("__start__", "__end__")),
              "metadata": getattr(n, "metadata", None) or None}
             for nid, n in g.nodes.items()]
    edges = [{"id": f"e{i}", "source": e.source, "target": e.target,
              "conditional": bool(getattr(e, "conditional", False)),
              "label": str(e.data) if getattr(e, "data", None) not in (None, e.target) else None}
             for i, e in enumerate(g.edges)]
    return {"nodes": nodes, "edges": edges}


def input_schema(graph) -> dict:
    for meth in ("get_input_jsonschema", "get_input_schema"):
        try:
            s = getattr(graph, meth)()
            return s if isinstance(s, dict) else s.model_json_schema()
        except Exception:
            continue
    return {}


def context_schema(graph) -> dict:
    """Runtime context/config schema (langgraph >=1.0 name, older fallback)."""
    for meth in ("get_context_jsonschema", "get_config_jsonschema"):
        try:
            s = getattr(graph, meth)()
            if isinstance(s, dict) and s.get("properties"):
                return s
        except Exception:
            continue
    return {}


def _unwrap(fn):
    """Peel partials, decorators, and bound methods down to the user function."""
    seen = set()
    while id(fn) not in seen:
        seen.add(id(fn))
        if isinstance(fn, functools.partial):
            fn = fn.func
        elif hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        elif inspect.ismethod(fn):
            fn = fn.__func__
        else:
            break
    return fn


def _node_callable(graph, name):
    """Best-effort: the user callable backing a LangGraph node. The compiled
    graph keeps the builder's node specs; RunnableCallable exposes .func/.afunc."""
    candidates = (
        lambda: graph.builder.nodes[name].runnable.func,
        lambda: graph.builder.nodes[name].runnable.afunc,
        lambda: graph.builder.nodes[name].runnable,
    )
    for get in candidates:
        try:
            fn = get()
        except Exception:
            continue
        if fn is not None:
            return _unwrap(fn)
    return None


def sources(graph) -> dict:
    """{node: {file, line_start, line_end, code}} for every node we can trace
    to real source. Missing nodes simply aren't in the map — never an error."""
    out = {}
    builder = getattr(graph, "builder", None)
    for name in (getattr(builder, "nodes", None) or {}):
        fn = _node_callable(graph, name)
        if fn is None:
            continue
        for target in (fn, type(fn)):  # class-based runnables: show the class
            try:
                lines, start = inspect.getsourcelines(target)
                out[name] = {"file": inspect.getsourcefile(target) or "",
                             "line_start": start,
                             "line_end": start + len(lines) - 1,
                             "code": "".join(lines)[:20000]}
                break
            except Exception:
                continue
    return out


def load(ref: str, dir_: str):
    sys.path.insert(0, dir_)
    mod, attr = ref.split(":")
    return getattr(importlib.import_module(mod), attr)


if __name__ == "__main__":
    ref, dir_ = sys.argv[1], sys.argv[2]
    g = load(ref, dir_)
    out = {"topology": topology(g), "schema": input_schema(g), "sources": sources(g),
           "context_schema": context_schema(g)}
    try:  # subgraph x-ray view, only when it actually differs
        tx = topology(g, xray=True)
        if tx != out["topology"]:
            out["topology_xray"] = tx
    except Exception:
        pass
    print(json.dumps(out))
