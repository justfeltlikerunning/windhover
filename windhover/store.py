"""Windhover data store — SQLite (WAL), a run + span-tree model.

Runs hold aggregate totals; spans form a tree (node spans top-level, LLM/tool
spans nest under a node). Thread-safe via a module lock; each op uses a short-
lived connection so worker threads and the request thread never share one.

Search uses FTS5 over span payloads when the host SQLite has it, and degrades
to LIKE scans when it doesn't; tag filtering likewise prefers json1 and falls
back to a substring match. Feature detection happens once at startup so the
same code runs on any Python/SQLite build.
"""
from __future__ import annotations
import json, os, sqlite3, threading, time, uuid
from typing import Any, Optional

SCHEMA_VERSION = 7
_lock = threading.Lock()


def _fts_match(q: str) -> str:
    """Turn free text into a safe FTS5 MATCH string (quoted AND terms)."""
    return " ".join('"%s"' % t.replace('"', '""') for t in q.split())


class Store:
    def __init__(self, path: str):
        self.path = path
        self.has_fts = False
        self.has_json1 = False
        self.on_run_closed = None   # optional hook(run_dict) — used for webhooks
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=5000")
        return c

    def _init(self) -> None:
        with _lock, self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS schema_meta(version INTEGER);
            CREATE TABLE IF NOT EXISTS runs(
              id TEXT PRIMARY KEY, graph TEXT, source TEXT, status TEXT,
              session TEXT, tags TEXT, input TEXT, error TEXT,
              started_ms INTEGER, ended_ms INTEGER, duration_ms INTEGER,
              node_count INTEGER DEFAULT 0, llm_calls INTEGER DEFAULT 0,
              prompt_tokens INTEGER, completion_tokens INTEGER,
              total_tokens INTEGER, cost_usd REAL);
            CREATE TABLE IF NOT EXISTS spans(
              id TEXT PRIMARY KEY, run_id TEXT, parent_id TEXT, seq INTEGER,
              type TEXT, name TEXT, status TEXT,
              started_ms INTEGER, ended_ms INTEGER, offset_ms INTEGER, dur_ms INTEGER,
              input TEXT, output TEXT, model TEXT,
              prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL, error TEXT);
            CREATE TABLE IF NOT EXISTS scores(
              id TEXT PRIMARY KEY, run_id TEXT, name TEXT, value REAL,
              comment TEXT, source TEXT, created_ms INTEGER);
            CREATE TABLE IF NOT EXISTS datasets(
              id TEXT PRIMARY KEY, name TEXT UNIQUE, items TEXT, created_ms INTEGER);
            CREATE INDEX IF NOT EXISTS ix_spans_run ON spans(run_id, seq);
            CREATE INDEX IF NOT EXISTS ix_runs_started ON runs(started_ms DESC);
            CREATE INDEX IF NOT EXISTS ix_runs_session ON runs(session);
            CREATE INDEX IF NOT EXISTS ix_scores_run ON scores(run_id);
            """)
            cols = [r[1] for r in c.execute("PRAGMA table_info(runs)").fetchall()]
            if "bookmarked" not in cols:
                c.execute("ALTER TABLE runs ADD COLUMN bookmarked INTEGER DEFAULT 0")
            if "thread_id" not in cols:
                c.execute("ALTER TABLE runs ADD COLUMN thread_id TEXT")
            scols = [r[1] for r in c.execute("PRAGMA table_info(spans)").fetchall()]
            for col, typ in (("retries", "INTEGER"), ("ttft_ms", "INTEGER"),
                             ("usage_detail", "TEXT"), ("params", "TEXT")):
                if col not in scols:
                    c.execute(f"ALTER TABLE spans ADD COLUMN {col} {typ}")
            try:
                c.execute("SELECT count(*) FROM json_each('[1]')")
                self.has_json1 = True
            except sqlite3.OperationalError:
                self.has_json1 = False
            try:
                c.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS span_fts
                             USING fts5(text, span_id UNINDEXED, run_id UNINDEXED)""")
                self.has_fts = True
                # one-time backfill (first boot after upgrade)
                n_fts = c.execute("SELECT count(*) FROM span_fts").fetchone()[0]
                n_spans = c.execute("SELECT count(*) FROM spans").fetchone()[0]
                if n_fts == 0 and n_spans > 0:
                    for s in c.execute("SELECT id,run_id,name,model,error,input,output FROM spans"):
                        c.execute("INSERT INTO span_fts(text,span_id,run_id) VALUES(?,?,?)",
                                  (self._span_text(dict(s)), s["id"], s["run_id"]))
            except sqlite3.OperationalError:
                self.has_fts = False
            row = c.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
            if row is None:
                self._migrate_v2(c)
                c.execute("INSERT INTO schema_meta(version) VALUES(?)", (SCHEMA_VERSION,))
            elif row["version"] < SCHEMA_VERSION:
                c.execute("UPDATE schema_meta SET version=?", (SCHEMA_VERSION,))

    @staticmethod
    def _span_text(s: dict) -> str:
        parts = [s.get("name"), s.get("model"), s.get("error")]
        for f in ("input", "output"):
            v = s.get(f)
            if v is not None:
                parts.append(v if isinstance(v, str) else json.dumps(v, default=str))
        return " ".join(p for p in parts if p)

    def _migrate_v2(self, c: sqlite3.Connection) -> None:
        """Best-effort: fold legacy v2 flat `events` into node spans (dev data)."""
        try:
            has = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'").fetchone()
            if not has:
                return
            for e in c.execute("SELECT run_id,seq,node,summary,offset_ms,dur_ms FROM events").fetchall():
                c.execute("""INSERT OR IGNORE INTO spans
                  (id,run_id,parent_id,seq,type,name,status,offset_ms,dur_ms,output)
                  VALUES(?,?,?,?,?,?,?,?,?,?)""",
                  (f"{e['run_id']}-{e['seq']}", e["run_id"], None, e["seq"], "node",
                   e["node"], "ok", e["offset_ms"], e["dur_ms"], e["summary"]))
            c.execute("ALTER TABLE events RENAME TO events_legacy_v2")
        except Exception:
            pass  # migration is a courtesy; never block startup

    # ---- writes -----------------------------------------------------------
    def open_run(self, run: dict) -> None:
        with _lock, self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO runs
              (id,graph,source,status,session,tags,input,started_ms,thread_id)
              VALUES(?,?,?,?,?,?,?,?,?)""",
              (run["id"], run.get("graph"), run.get("source", "ui"), "running",
               run.get("session"), json.dumps(run.get("tags")),
               json.dumps(run.get("input"), default=str), run["started_ms"],
               run.get("thread_id")))

    def add_span(self, s: dict) -> None:
        with _lock, self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO spans
              (id,run_id,parent_id,seq,type,name,status,started_ms,ended_ms,offset_ms,
               dur_ms,input,output,model,prompt_tokens,completion_tokens,cost_usd,error,
               retries,ttft_ms,usage_detail,params)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (s["id"], s["run_id"], s.get("parent_id"), s.get("seq", 0), s["type"],
               s.get("name"), s.get("status", "ok"), s.get("started_ms"), s.get("ended_ms"),
               s.get("offset_ms"), s.get("dur_ms"),
               json.dumps(s.get("input"), default=str) if s.get("input") is not None else None,
               json.dumps(s.get("output"), default=str) if s.get("output") is not None else None,
               s.get("model"), s.get("prompt_tokens"), s.get("completion_tokens"),
               s.get("cost_usd"), s.get("error"),
               s.get("retries"), s.get("ttft_ms"),
               json.dumps(s.get("usage_detail")) if s.get("usage_detail") else None,
               json.dumps(s.get("params")) if s.get("params") else None))
            if self.has_fts:
                c.execute("DELETE FROM span_fts WHERE span_id=?", (s["id"],))
                c.execute("INSERT INTO span_fts(text,span_id,run_id) VALUES(?,?,?)",
                          (self._span_text(s), s["id"], s["run_id"]))

    def close_run(self, run_id: str, status: str, ended_ms: int, error: Optional[str] = None) -> None:
        with _lock, self._conn() as c:
            agg = c.execute("""SELECT
                COUNT(*) FILTER (WHERE type='node') nc,
                COUNT(*) FILTER (WHERE type='llm') lc,
                SUM(prompt_tokens) pt, SUM(completion_tokens) ct, SUM(cost_usd) cost
                FROM spans WHERE run_id=?""", (run_id,)).fetchone()
            st = c.execute("SELECT started_ms FROM runs WHERE id=?", (run_id,)).fetchone()
            started = st["started_ms"] if st else ended_ms
            pt, ct = agg["pt"], agg["ct"]
            c.execute("""UPDATE runs SET status=?,ended_ms=?,duration_ms=?,error=?,
                node_count=?,llm_calls=?,prompt_tokens=?,completion_tokens=?,total_tokens=?,cost_usd=?
                WHERE id=?""",
                (status, ended_ms, ended_ms - started, error, agg["nc"], agg["lc"],
                 pt, ct, (pt or 0) + (ct or 0) if (pt or ct) else None, agg["cost"], run_id))
        if self.on_run_closed is not None:
            try:
                self.on_run_closed({"id": run_id, "status": status, "error": error,
                                    "duration_ms": ended_ms - started})
            except Exception:
                pass

    def update_run_meta(self, run_id: str, tags: Optional[list] = None,
                        bookmarked: Optional[bool] = None) -> bool:
        sets, args = [], []
        if tags is not None:
            sets.append("tags=?"); args.append(json.dumps([str(t) for t in tags]))
        if bookmarked is not None:
            sets.append("bookmarked=?"); args.append(1 if bookmarked else 0)
        if not sets:
            return False
        args.append(run_id)
        with _lock, self._conn() as c:
            cur = c.execute(f"UPDATE runs SET {','.join(sets)} WHERE id=?", args)
            return cur.rowcount > 0

    def add_score(self, run_id: str, name: str, value: float,
                  comment: Optional[str] = None, source: str = "api") -> Optional[dict]:
        import math
        if not math.isfinite(float(value)):
            return None  # NaN/Infinity would render the runs API unparseable
        with _lock, self._conn() as c:
            if not c.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone():
                return None
            sc = {"id": uuid.uuid4().hex[:12], "run_id": run_id, "name": str(name),
                  "value": float(value), "comment": comment, "source": source,
                  "created_ms": int(time.time() * 1000)}
            c.execute("""INSERT INTO scores(id,run_id,name,value,comment,source,created_ms)
                         VALUES(?,?,?,?,?,?,?)""",
                      (sc["id"], sc["run_id"], sc["name"], sc["value"],
                       sc["comment"], sc["source"], sc["created_ms"]))
            return sc

    def delete_score(self, score_id: str) -> bool:
        with _lock, self._conn() as c:
            return c.execute("DELETE FROM scores WHERE id=?", (score_id,)).rowcount > 0

    # ---- datasets -----------------------------------------------------------
    def add_dataset(self, name: str, items: list) -> dict:
        ds = {"id": uuid.uuid4().hex[:12], "name": str(name),
              "items": items, "created_ms": int(time.time() * 1000)}
        with _lock, self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO datasets(id,name,items,created_ms)
                         VALUES(?,?,?,?)""",
                      (ds["id"], ds["name"], json.dumps(items, default=str), ds["created_ms"]))
        return ds

    def datasets(self) -> list[dict]:
        with _lock, self._conn() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT id,name,items,created_ms FROM datasets ORDER BY created_ms DESC").fetchall()]
        for r in rows:
            r["items"] = json.loads(r["items"]) if r.get("items") else []
            r["n_items"] = len(r["items"])
        return rows

    def dataset(self, ds_id: str):
        for d in self.datasets():
            if d["id"] == ds_id or d["name"] == ds_id:
                return d
        return None

    def delete_dataset(self, ds_id: str) -> bool:
        with _lock, self._conn() as c:
            return c.execute("DELETE FROM datasets WHERE id=? OR name=?",
                             (ds_id, ds_id)).rowcount > 0

    def prune(self, days: int) -> dict:
        """Delete runs (and their spans/scores/index rows) older than `days`."""
        cutoff = int((time.time() - days * 86400) * 1000)
        with _lock, self._conn() as c:
            ids = [r["id"] for r in
                   c.execute("SELECT id FROM runs WHERE started_ms < ?", (cutoff,)).fetchall()]
            for chunk in (ids[i:i + 500] for i in range(0, len(ids), 500)):
                q = ",".join("?" * len(chunk))
                if self.has_fts:
                    c.execute(f"DELETE FROM span_fts WHERE run_id IN ({q})", chunk)
                c.execute(f"DELETE FROM spans WHERE run_id IN ({q})", chunk)
                c.execute(f"DELETE FROM scores WHERE run_id IN ({q})", chunk)
                c.execute(f"DELETE FROM runs WHERE id IN ({q})", chunk)
        return {"pruned_runs": len(ids), "cutoff_ms": cutoff}

    # ---- reads ------------------------------------------------------------
    def runs(self, limit: int = 50, offset: int = 0, q: Optional[str] = None,
             status: Optional[str] = None, graph: Optional[str] = None,
             session: Optional[str] = None, tag: Optional[str] = None,
             bookmarked: Optional[bool] = None,
             since_ms: Optional[int] = None, until_ms: Optional[int] = None) -> dict:
        where, args = [], []
        if q:
            if self.has_fts:
                where.append("id IN (SELECT DISTINCT run_id FROM span_fts WHERE span_fts MATCH ?)")
                args.append(_fts_match(q))
            else:
                like = f"%{q}%"
                where.append("""id IN (SELECT DISTINCT run_id FROM spans
                                WHERE input LIKE ? OR output LIKE ? OR name LIKE ? OR error LIKE ?)""")
                args += [like, like, like, like]
        if status:
            where.append("status=?"); args.append(status)
        if graph:
            where.append("graph=?"); args.append(graph)
        if session:
            where.append("session=?"); args.append(session)
        if tag:
            if self.has_json1:
                where.append("""EXISTS (SELECT 1 FROM json_each(COALESCE(runs.tags,'[]')) je
                                WHERE je.value=?)""")
                args.append(tag)
            else:
                where.append("COALESCE(runs.tags,'') LIKE ?")
                args.append(f'%"{tag}"%')
        if bookmarked:
            where.append("bookmarked=1")
        if since_ms is not None:
            where.append("started_ms>=?"); args.append(since_ms)
        if until_ms is not None:
            where.append("started_ms<=?"); args.append(until_ms)
        cond = (" WHERE " + " AND ".join(where)) if where else ""
        with _lock, self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM runs{cond}", args).fetchone()[0]
            rows = [dict(r) for r in c.execute(
                f"SELECT * FROM runs{cond} ORDER BY started_ms DESC LIMIT ? OFFSET ?",
                [*args, limit, offset]).fetchall()]
            if rows:
                ph = ",".join("?" * len(rows))
                sc: dict[str, dict] = {}
                for s in c.execute(
                        f"SELECT run_id,name,AVG(value) v FROM scores WHERE run_id IN ({ph}) "
                        "GROUP BY run_id,name", [r["id"] for r in rows]).fetchall():
                    sc.setdefault(s["run_id"], {})[s["name"]] = round(s["v"], 4)
                for r in rows:
                    r["tags"] = json.loads(r["tags"]) if r.get("tags") else None
                    r["scores"] = sc.get(r["id"]) or None
        return {"runs": rows, "total": total, "limit": limit, "offset": offset}

    def sessions(self, limit: int = 100, graph: Optional[str] = None) -> list[dict]:
        cond, args = "", []
        if graph:
            cond = " AND graph=?"; args.append(graph)
        with _lock, self._conn() as c:
            return [dict(r) for r in c.execute(f"""SELECT session,
                COUNT(*) runs, COUNT(*) FILTER (WHERE status='error') errors,
                MIN(started_ms) first_ms, MAX(started_ms) last_ms,
                SUM(total_tokens) tokens, SUM(cost_usd) cost, SUM(duration_ms) duration_ms
                FROM runs WHERE session IS NOT NULL AND session!=''{cond}
                GROUP BY session ORDER BY last_ms DESC LIMIT ?""",
                (*args, limit)).fetchall()]

    def run_detail(self, run_id: str) -> Optional[dict]:
        with _lock, self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not r:
                return None
            spans = [dict(s) for s in c.execute(
                "SELECT * FROM spans WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()]
            scores = [dict(s) for s in c.execute(
                "SELECT * FROM scores WHERE run_id=? ORDER BY created_ms", (run_id,)).fetchall()]
        d = dict(r)
        for k in ("input", "tags"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        for s in spans:
            for k in ("input", "output", "usage_detail", "params"):
                s[k] = json.loads(s[k]) if s.get(k) else None
        d["spans"] = spans
        d["scores"] = scores
        return d

    def node_history(self, name: str, limit: int = 25) -> dict:
        """Aggregate + recent executions of one named node/span across all runs,
        each with its payloads and direct child spans (LLM/tool calls inside it)."""
        with _lock, self._conn() as c:
            agg = c.execute("""SELECT COUNT(*) n, AVG(dur_ms) avg_ms,
                MIN(dur_ms) min_ms, MAX(dur_ms) max_ms,
                COUNT(*) FILTER (WHERE status='error') errors,
                SUM(COALESCE(prompt_tokens,0)+COALESCE(completion_tokens,0)) tokens,
                SUM(cost_usd) cost
                FROM spans WHERE name=?""", (name,)).fetchone()
            rows = [dict(r) for r in c.execute("""SELECT s.*, r.graph, r.source
                FROM spans s LEFT JOIN runs r ON r.id=s.run_id
                WHERE s.name=? ORDER BY s.started_ms DESC LIMIT ?""",
                (name, limit)).fetchall()]
            kids: dict[str, list] = {}
            if rows:
                q = ",".join("?" * len(rows))
                for k in c.execute(
                        f"SELECT * FROM spans WHERE parent_id IN ({q}) ORDER BY seq",
                        [r["id"] for r in rows]).fetchall():
                    kids.setdefault(k["parent_id"], []).append(dict(k))
        for r in rows:
            r["children"] = kids.get(r["id"], [])
            for s in (r, *r["children"]):
                for f in ("input", "output"):
                    s[f] = json.loads(s[f]) if s.get(f) else None
        return {"name": name, "summary": dict(agg), "recent": rows}

    def stats(self, days: int = 30, graph: Optional[str] = None) -> dict:
        cutoff = int((time.time() - days * 86400) * 1000)
        rcond, rargs = ("", [])
        scond, sargs = ("", [])
        if graph:
            rcond = " WHERE graph=?"; rargs = [graph]
            scond = " AND runs.graph=?"; sargs = [graph]
        with _lock, self._conn() as c:
            tot = c.execute(f"""SELECT COUNT(*) runs,
                COUNT(*) FILTER (WHERE status='error') errors,
                SUM(total_tokens) tokens, SUM(cost_usd) cost,
                SUM(llm_calls) llm_calls FROM runs{rcond}""", rargs).fetchone()
            per_node = c.execute(f"""SELECT spans.name, COUNT(*) n,
                AVG(spans.dur_ms) avg_ms, SUM(spans.cost_usd) cost
                FROM spans JOIN runs ON runs.id = spans.run_id
                WHERE spans.type='node'{scond}
                GROUP BY spans.name ORDER BY avg_ms DESC LIMIT 20""", sargs).fetchall()
            models = c.execute(f"""SELECT spans.model, COUNT(*) calls,
                SUM(spans.prompt_tokens) prompt_tokens,
                SUM(spans.completion_tokens) completion_tokens,
                SUM(spans.cost_usd) cost, AVG(spans.dur_ms) avg_ms
                FROM spans JOIN runs ON runs.id = spans.run_id
                WHERE spans.type='llm' AND spans.model IS NOT NULL{scond}
                GROUP BY spans.model ORDER BY calls DESC LIMIT 20""", sargs).fetchall()
            daily = c.execute(f"""SELECT
                strftime('%Y-%m-%d', started_ms/1000, 'unixepoch') day,
                COUNT(*) runs, COUNT(*) FILTER (WHERE status='error') errors,
                SUM(total_tokens) tokens, SUM(cost_usd) cost
                FROM runs WHERE started_ms >= ?{' AND graph=?' if graph else ''}
                GROUP BY day ORDER BY day""", (cutoff, *rargs)).fetchall()
        try:
            db_bytes = os.path.getsize(self.path)
        except OSError:
            db_bytes = None
        return {"totals": {**dict(tot), "db_bytes": db_bytes},
                "per_node": [dict(r) for r in per_node],
                "models": [dict(r) for r in models],
                "daily": [dict(r) for r in daily],
                "days": days, "graph": graph}
