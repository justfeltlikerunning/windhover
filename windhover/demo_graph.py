"""Self-contained demo graph so Windhover runs out of the box:

    WINDHOVER_GRAPH=windhover.demo_graph:graph python -m windhover.server

No external services. Edit this file while Windhover runs to watch the topology
update itself in the UI. Includes a parallel fan-out (grow -> parity/sign/
magnitude -> summarize) so the graph view and trace show concurrent branches;
`notes` uses an additive reducer so the branches can write it concurrently.
"""
import operator
import time
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    n: int
    notes: Annotated[list, operator.add]


def seed(s):
    time.sleep(.2)
    n = s.get("n", 1)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n} — the demo guard tripped")
    return {"n": n}
def grow(s):
    time.sleep(.3)
    grown = s["n"] * 3
    if grown > 500:  # human-in-the-loop: big growth needs approval
        from langgraph.types import interrupt
        approved = interrupt({"question": f"grow {s['n']} -> {grown}? (true/false)"})
        if not approved:
            return {"n": s["n"]}
    return {"n": grown}
def parity(s):     time.sleep(.4);  return {"notes": ["even" if s["n"] % 2 == 0 else "odd"]}
def sign(s):       time.sleep(.25); return {"notes": ["positive" if s["n"] > 0 else "non-positive"]}
def magnitude(s):  time.sleep(.35); return {"notes": ["big" if abs(s["n"]) > 100 else "small"]}
def summarize(s):
    time.sleep(.1)
    try:  # progress + custom events + long-term memory, all observable in Windhover
        from langgraph.config import get_stream_writer, get_store
        writer = get_stream_writer()
        writer({"stage": "summarize", "pct": 50})
        from langchain_core.callbacks import dispatch_custom_event
        dispatch_custom_event("summary-ready", {"n": s["n"], "parts": len(s["notes"])})
        store = get_store()
        if store is not None:
            store.put(("demo", "summaries"), f"n-{s['n']}",
                      {"n": s["n"], "notes": s["notes"]})
    except Exception:
        pass
    time.sleep(.1)
    return {"notes": [f"n={s['n']}: " + ", ".join(s["notes"])]}


_g = StateGraph(State)
for name, fn in [("seed", seed), ("grow", grow), ("parity", parity),
                 ("sign", sign), ("magnitude", magnitude), ("summarize", summarize)]:
    _g.add_node(name, fn)
_g.add_edge(START, "seed")
_g.add_edge("seed", "grow")
_g.add_edge("grow", "parity")
_g.add_edge("grow", "sign")
_g.add_edge("grow", "magnitude")
_g.add_edge("parity", "summarize")
_g.add_edge("sign", "summarize")
_g.add_edge("magnitude", "summarize")
_g.add_edge("summarize", END)

try:  # checkpointer + store make Time-travel and Memory demoable
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore
    graph = _g.compile(checkpointer=MemorySaver(), store=InMemoryStore())
except Exception:
    graph = _g.compile()
