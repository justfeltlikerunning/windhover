"""Extract a compiled graph's topology as JSON. Run as a subprocess so the live
watcher always sees current-on-disk code without importlib.reload fragility.

    python -m windhover.extract "module:attr" "/graph/dir"
"""
import sys, json, importlib


def topology(graph) -> dict:
    g = graph.get_graph()
    nodes = [{"id": nid, "label": str(getattr(n, "name", nid)).replace("__", ""),
              "terminal": nid in ("__start__", "__end__")}
             for nid, n in g.nodes.items()]
    edges = [{"id": f"e{i}", "source": e.source, "target": e.target,
              "conditional": bool(getattr(e, "conditional", False))}
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


def load(ref: str, dir_: str):
    sys.path.insert(0, dir_)
    mod, attr = ref.split(":")
    return getattr(importlib.import_module(mod), attr)


if __name__ == "__main__":
    ref, dir_ = sys.argv[1], sys.argv[2]
    g = load(ref, dir_)
    print(json.dumps({"topology": topology(g), "schema": input_schema(g)}))
