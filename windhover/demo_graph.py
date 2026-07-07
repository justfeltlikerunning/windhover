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
def grow(s):       time.sleep(.3);  return {"n": s["n"] * 3}
def parity(s):     time.sleep(.4);  return {"notes": ["even" if s["n"] % 2 == 0 else "odd"]}
def sign(s):       time.sleep(.25); return {"notes": ["positive" if s["n"] > 0 else "non-positive"]}
def magnitude(s):  time.sleep(.35); return {"notes": ["big" if abs(s["n"]) > 100 else "small"]}
def summarize(s):  time.sleep(.2);  return {"notes": [f"n={s['n']}: " + ", ".join(s["notes"])]}


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

graph = _g.compile()
