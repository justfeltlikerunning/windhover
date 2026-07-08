"""Second demo graph — a tiny retrieval pipeline, so multi-graph serving is
demoable out of the box:

    WINDHOVER_GRAPH="numbers=windhover.demo_graph:graph,rag=windhover.demo_rag:graph" windhover
"""
import operator
import time
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

_DOCS = [
    {"id": "handbook#12", "text": "Refunds are processed within five business days."},
    {"id": "handbook#31", "text": "Enterprise plans include priority support."},
    {"id": "faq#4", "text": "You can export your data as CSV at any time."},
]


class State(TypedDict):
    question: str
    hits: Annotated[list, operator.add]
    answer: str


def retrieve(s):
    time.sleep(.15)
    q = (s.get("question") or "").lower()
    hits = [d for d in _DOCS if any(w in d["text"].lower() for w in q.split())] or _DOCS[:1]
    return {"hits": hits}


def grade(s):
    time.sleep(.1)
    return {"hits": []}  # additive reducer: nothing new, grading is a pass-through demo


def answer(s):
    time.sleep(.2)
    src = s["hits"][0] if s["hits"] else {"id": "none", "text": ""}
    return {"answer": f"Per {src['id']}: {src['text']}"}


_g = StateGraph(State)
_g.add_node("retrieve", retrieve)
_g.add_node("grade", grade)
_g.add_node("answer", answer)
_g.add_edge(START, "retrieve")
_g.add_edge("retrieve", "grade")
_g.add_edge("grade", "answer")
_g.add_edge("answer", END)

try:
    from langgraph.checkpoint.memory import MemorySaver
    graph = _g.compile(checkpointer=MemorySaver())
except Exception:
    graph = _g.compile()
