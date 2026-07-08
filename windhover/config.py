"""Windhover configuration — one place, read from the environment."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    graph_ref: str            # "module:attr" of a compiled graph ("" = ingest-only)
    graph_dir: str            # import path for the graph module
    db_path: str
    host: str
    port: int
    watch: bool               # live-topology file watcher
    pricing_path: str
    retention_days: int       # 0 = keep runs forever
    token: str                # WINDHOVER_TOKEN: require Bearer/query token on /api ("" = open)
    webhook: str              # WINDHOVER_WEBHOOK: POST run summaries on error/interrupted ("" = off)

    @staticmethod
    def _discover() -> tuple[str, str]:
        """No WINDHOVER_GRAPH? Look for a langgraph.json (LangGraph's standard
        project file) in WINDHOVER_GRAPH_DIR / cwd and use its first graph.
        Returns (graph_ref, graph_dir) or ("", dir)."""
        import json as _json
        base = os.environ.get("WINDHOVER_GRAPH_DIR", os.getcwd())
        cfg_path = os.path.join(base, "langgraph.json")
        try:
            graphs = _json.loads(open(cfg_path).read()).get("graphs") or {}
            for _name, ref in graphs.items():
                path, _, attr = str(ref).partition(":")
                if not attr:
                    continue
                if path.endswith(".py"):
                    full = os.path.normpath(os.path.join(base, path))
                    return (f"{os.path.splitext(os.path.basename(full))[0]}:{attr}",
                            os.path.dirname(full))
                return f"{path}:{attr}", base   # already module:attr
        except Exception:
            pass
        return "", base

    @classmethod
    def from_env(cls) -> "Config":
        graph_ref = os.environ.get("WINDHOVER_GRAPH", "")
        graph_dir = os.environ.get("WINDHOVER_GRAPH_DIR", os.getcwd())
        if not graph_ref:
            graph_ref, graph_dir = cls._discover()
        return cls(
            graph_ref=graph_ref,
            graph_dir=graph_dir,
            db_path=os.environ.get("WINDHOVER_DB", str(PKG_DIR.parent / "windhover.db")),
            host=os.environ.get("WINDHOVER_HOST", "0.0.0.0"),
            port=int(os.environ.get("WINDHOVER_PORT", "8090")),
            watch=os.environ.get("WINDHOVER_WATCH", "1") not in ("0", "false", "no"),
            pricing_path=os.environ.get("WINDHOVER_PRICING", str(PKG_DIR / "pricing.json")),
            retention_days=int(os.environ.get("WINDHOVER_RETENTION_DAYS", "0") or 0),
            token=os.environ.get("WINDHOVER_TOKEN", ""),
            webhook=os.environ.get("WINDHOVER_WEBHOOK", ""),
        )
