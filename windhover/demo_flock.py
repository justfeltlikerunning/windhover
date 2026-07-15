"""Demo flock — six small graphs to demo subject grouping (path-style names).

Graph names with "/" form a subject tree of ANY depth — pure convention, no
config. Groups appear in the graph selector, as collapsible Fleet sections
with rolled-up stats, and as "subject/*" scopes on runs/sessions/stats.

    WINDHOVER_GRAPH="support/triage=windhover.demo_flock:triage,\
support/billing/refunds=windhover.demo_flock:refunds,\
support/billing/invoices=windhover.demo_flock:invoices,\
ops/etl/nightly=windhover.demo_flock:nightly,\
ops/etl/backfill=windhover.demo_flock:backfill,\
ops/monitor/heartbeat=windhover.demo_flock:heartbeat,\
research/rag=windhover.demo_rag:graph,\
playground=windhover.demo_graph:graph" windhover

Or the langgraph.json equivalent — names there group the same way:

    {"graphs": {"support/triage": "./flock.py:triage", ...}}

No external services; every node is a small pure function.
"""
import time
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END


def _compile(g):
    try:
        from langgraph.checkpoint.memory import MemorySaver
        return g.compile(checkpointer=MemorySaver())
    except Exception:
        return g.compile()


# ---- support/triage — conditional routing ---------------------------------
class TriageState(TypedDict):
    message: str
    category: str
    reply: str


def classify(s):
    time.sleep(.15)
    m = (s.get("message") or "").lower()
    cat = "billing" if any(w in m for w in ("refund", "invoice", "charge")) \
        else "bug" if any(w in m for w in ("error", "crash", "broken")) \
        else "general"
    return {"category": cat}


def route(s) -> Literal["escalate", "respond"]:
    return "escalate" if s["category"] == "bug" else "respond"


def respond(s):
    time.sleep(.2)
    return {"reply": f"[{s['category']}] Thanks — here's what to do next."}


def escalate(s):
    time.sleep(.1)
    return {"reply": "Routed to an engineer with full context."}


_t = StateGraph(TriageState)
_t.add_node("classify", classify)
_t.add_node("respond", respond)
_t.add_node("escalate", escalate)
_t.add_edge(START, "classify")
_t.add_conditional_edges("classify", route, {"respond": "respond", "escalate": "escalate"})
_t.add_edge("respond", END)
_t.add_edge("escalate", END)
triage = _compile(_t)


# ---- support/billing/refunds — human-in-the-loop approval -----------------
class RefundState(TypedDict):
    order_id: str
    amount: float
    approved: bool
    result: str


def validate(s):
    time.sleep(.15)
    if not s.get("order_id"):
        raise ValueError("refund request has no order_id")
    return {"amount": float(s.get("amount") or 0)}


def check_policy(s):
    time.sleep(.2)
    if s["amount"] > 100:  # big refunds wait for a human
        from langgraph.types import interrupt
        ok = interrupt({"question": f"Refund ${s['amount']:.2f} on order "
                                    f"{s['order_id']}? (true/false)"})
        return {"approved": bool(ok)}
    return {"approved": True}


def issue(s):
    time.sleep(.15)
    return {"result": (f"refunded ${s['amount']:.2f} to {s['order_id']}"
                       if s["approved"] else "refund declined by reviewer")}


_r = StateGraph(RefundState)
_r.add_node("validate", validate)
_r.add_node("check_policy", check_policy)
_r.add_node("issue", issue)
_r.add_edge(START, "validate")
_r.add_edge("validate", "check_policy")
_r.add_edge("check_policy", "issue")
_r.add_edge("issue", END)
refunds = _compile(_r)


# ---- support/billing/invoices ----------------------------------------------
class InvoiceState(TypedDict):
    customer: str
    lines: list
    total: float
    document: str


def fetch_lines(s):
    time.sleep(.15)
    return {"lines": s.get("lines") or [{"item": "starter plan", "usd": 29.0}]}


def total(s):
    time.sleep(.1)
    return {"total": round(sum(float(l.get("usd") or 0) for l in s["lines"]), 2)}


def format_doc(s):
    time.sleep(.15)
    return {"document": f"INVOICE {s.get('customer') or 'unknown'} — "
                        f"{len(s['lines'])} lines — ${s['total']:.2f}"}


_i = StateGraph(InvoiceState)
_i.add_node("fetch_lines", fetch_lines)
_i.add_node("total", total)
_i.add_node("format_doc", format_doc)
_i.add_edge(START, "fetch_lines")
_i.add_edge("fetch_lines", "total")
_i.add_edge("total", "format_doc")
_i.add_edge("format_doc", END)
invoices = _compile(_i)


# ---- ops/etl/nightly --------------------------------------------------------
class EtlState(TypedDict):
    source: str
    rows: int
    clean: int
    loaded: int


def extract(s):
    time.sleep(.2)
    return {"rows": int(s.get("rows") or 1200)}


def transform(s):
    time.sleep(.3)
    return {"clean": int(s["rows"] * 0.97)}


def load(s):
    time.sleep(.2)
    if s.get("source") == "flaky":
        raise RuntimeError("warehouse rejected the batch (demo failure)")
    return {"loaded": s["clean"]}


_n = StateGraph(EtlState)
_n.add_node("extract", extract)
_n.add_node("transform", transform)
_n.add_node("load", load)
_n.add_edge(START, "extract")
_n.add_edge("extract", "transform")
_n.add_edge("transform", "load")
_n.add_edge("load", END)
nightly = _compile(_n)


# ---- ops/etl/backfill -------------------------------------------------------
class BackfillState(TypedDict):
    days: int
    chunks: int
    loaded: int


def plan(s):
    time.sleep(.15)
    return {"days": int(s.get("days") or 7)}


def chunk(s):
    time.sleep(.2)
    return {"chunks": max(1, s["days"] // 2)}


def load_chunks(s):
    time.sleep(.05 * s["chunks"])
    return {"loaded": s["chunks"]}


_b = StateGraph(BackfillState)
_b.add_node("plan", plan)
_b.add_node("chunk", chunk)
_b.add_node("load_chunks", load_chunks)
_b.add_edge(START, "plan")
_b.add_edge("plan", "chunk")
_b.add_edge("chunk", "load_chunks")
_b.add_edge("load_chunks", END)
backfill = _compile(_b)


# ---- ops/monitor/heartbeat ---------------------------------------------------
class BeatState(TypedDict):
    target: str
    up: bool
    verdict: str


def ping(s):
    time.sleep(.1)
    if s.get("target") == "down.example":
        raise ConnectionError("heartbeat target unreachable (demo failure)")
    return {"up": True}


def assess(s):
    time.sleep(.1)
    return {"verdict": f"{s.get('target') or 'service'} healthy"}


_h = StateGraph(BeatState)
_h.add_node("ping", ping)
_h.add_node("assess", assess)
_h.add_edge(START, "ping")
_h.add_edge("ping", "assess")
_h.add_edge("assess", END)
heartbeat = _compile(_h)
