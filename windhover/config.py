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

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            graph_ref=os.environ.get("WINDHOVER_GRAPH", ""),
            graph_dir=os.environ.get("WINDHOVER_GRAPH_DIR", os.getcwd()),
            db_path=os.environ.get("WINDHOVER_DB", str(PKG_DIR.parent / "windhover.db")),
            host=os.environ.get("WINDHOVER_HOST", "0.0.0.0"),
            port=int(os.environ.get("WINDHOVER_PORT", "8090")),
            watch=os.environ.get("WINDHOVER_WATCH", "1") not in ("0", "false", "no"),
            pricing_path=os.environ.get("WINDHOVER_PRICING", str(PKG_DIR / "pricing.json")),
            retention_days=int(os.environ.get("WINDHOVER_RETENTION_DAYS", "0") or 0),
        )
