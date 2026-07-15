"""Windhover configuration — one place, read from the environment.

Multi-graph: WINDHOVER_GRAPH accepts a comma-separated list of graphs, each
"module:attr" or "name=module:attr". With no env set, ALL graphs from a
langgraph.json (the LangGraph Studio/CLI project file) are served.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    graphs: tuple           # ((name, "module:attr"), …); () = ingest-only
    graph_dir: str          # import path for the graph modules
    db_path: str
    host: str
    port: int
    watch: bool             # live-topology file watcher
    pricing_path: str
    retention_days: int     # 0 = keep runs forever
    token: str              # WINDHOVER_TOKEN: require Bearer/query token on /api ("" = open)
    webhook: str            # WINDHOVER_WEBHOOK: POST run summaries on error/interrupted ("" = off)
    vapid_public: str       # WINDHOVER_VAPID_PUBLIC: base64url app-server key ("" = web push off)
    vapid_private: str      # WINDHOVER_VAPID_PRIVATE: base64url private key
    vapid_subject: str      # WINDHOVER_VAPID_SUBJECT: contact (mailto:/https URL) sent to the push service
    digest: str             # WINDHOVER_DIGEST: "HH:MM" local time for a daily summary push ("" = off)
    webhook_map: dict       # per-graph webhook overrides from WINDHOVER_WEBHOOK "name=url" entries
    alert_on: tuple         # WINDHOVER_ALERT_ON: which run outcomes fire push/webhook
    report_dir: str         # WINDHOVER_REPORT_DIR: write each alerting run's .md report here ("" = off)

    @property
    def alerts_report(self) -> bool:
        return "done" in self.alert_on or "tagged" in self.alert_on

    @property
    def push_enabled(self) -> bool:
        return bool(self.vapid_public and self.vapid_private)

    @property
    def graph_ref(self) -> str:
        """First graph's ref — kept for single-graph callers/back-compat."""
        return self.graphs[0][1] if self.graphs else ""

    @staticmethod
    def _parse_webhooks(raw: str) -> tuple:
        """WINDHOVER_WEBHOOK: a URL, or a comma list mixing a default URL with
        per-graph "name=url" overrides. '=' only counts before '://' so query
        strings in plain URLs stay intact. Returns (default_url, {name: url})."""
        default, per = "", {}
        for part in (raw or "").split(","):
            part = part.strip()
            if not part:
                continue
            scheme = part.find("://")
            eq = part.find("=")
            if eq != -1 and (scheme == -1 or eq < scheme):
                name, _, url = part.partition("=")
                if name.strip() and url.strip():
                    per[name.strip()] = url.strip()
            else:
                default = part
        return default, per

    @staticmethod
    def _parse_env_graphs(raw: str) -> tuple:
        out = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part.split(":")[0]:          # "name=module:attr"
                name, _, ref = part.partition("=")
                out.append((name.strip(), ref.strip()))
            else:
                out.append((part, part))            # name defaults to the ref
        return tuple(out)

    @staticmethod
    def _discover() -> tuple:
        """No WINDHOVER_GRAPH? Serve every graph a langgraph.json defines.
        Returns ((name, ref), …), graph_dir."""
        import json as _json
        base = os.environ.get("WINDHOVER_GRAPH_DIR", os.getcwd())
        try:
            graphs = _json.loads(open(os.path.join(base, "langgraph.json")).read()).get("graphs") or {}
        except Exception:
            return (), base
        out, gdir = [], base
        for name, ref in graphs.items():
            path, _, attr = str(ref).partition(":")
            if not attr:
                continue
            if path.endswith(".py"):
                full = os.path.normpath(os.path.join(base, path))
                gdir = os.path.dirname(full)        # langgraph convention: shared src dir
                out.append((str(name), f"{os.path.splitext(os.path.basename(full))[0]}:{attr}"))
            else:
                out.append((str(name), f"{path}:{attr}"))
        return tuple(out), gdir

    @classmethod
    def from_env(cls) -> "Config":
        raw = os.environ.get("WINDHOVER_GRAPH", "")
        graph_dir = os.environ.get("WINDHOVER_GRAPH_DIR", os.getcwd())
        if raw:
            graphs = cls._parse_env_graphs(raw)
        else:
            graphs, graph_dir = cls._discover()
        webhook, webhook_map = cls._parse_webhooks(os.environ.get("WINDHOVER_WEBHOOK", ""))
        return cls(
            graphs=graphs,
            graph_dir=graph_dir,
            db_path=os.environ.get("WINDHOVER_DB", str(PKG_DIR.parent / "windhover.db")),
            host=os.environ.get("WINDHOVER_HOST", "0.0.0.0"),
            port=int(os.environ.get("WINDHOVER_PORT", "8090")),
            watch=os.environ.get("WINDHOVER_WATCH", "1") not in ("0", "false", "no"),
            pricing_path=os.environ.get("WINDHOVER_PRICING", str(PKG_DIR / "pricing.json")),
            retention_days=int(os.environ.get("WINDHOVER_RETENTION_DAYS", "0") or 0),
            token=os.environ.get("WINDHOVER_TOKEN", ""),
            webhook=webhook,
            vapid_public=os.environ.get("WINDHOVER_VAPID_PUBLIC", ""),
            vapid_private=os.environ.get("WINDHOVER_VAPID_PRIVATE", ""),
            vapid_subject=os.environ.get("WINDHOVER_VAPID_SUBJECT", "mailto:windhover@localhost"),
            digest=os.environ.get("WINDHOVER_DIGEST", ""),
            webhook_map=webhook_map,
            alert_on=tuple(s.strip().lower() for s in
                           os.environ.get("WINDHOVER_ALERT_ON", "error,interrupted").split(",") if s.strip()),
            report_dir=os.environ.get("WINDHOVER_REPORT_DIR", ""),
        )
