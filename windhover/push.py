"""Web Push (VAPID) delivery for alert notifications.

Optional feature: enabled only when WINDHOVER_VAPID_PUBLIC / _PRIVATE are set.
Sends to every stored browser PushSubscription and prunes ones the push service
reports as gone (404/410). Runs off the request/close path in a daemon thread so
delivery never blocks tracing.
"""
from __future__ import annotations

import json
import threading
import time

try:
    from pywebpush import WebPushException, webpush
    _AVAILABLE = True
except Exception:  # pragma: no cover - dependency optional
    _AVAILABLE = False


def available() -> bool:
    return _AVAILABLE


def _send_one(sub: dict, payload: str, cfg) -> int | None:
    """Return an HTTP status to prune on (404/410), or None to keep."""
    try:
        webpush(
            subscription_info=sub,
            data=payload,
            vapid_private_key=cfg.vapid_private,
            # Apple rejects the JWT unless sub is a valid mailto:/https: and exp < 24h.
            vapid_claims={"sub": cfg.vapid_subject, "exp": int(time.time()) + 12 * 3600},
            timeout=10,
        )
        return None
    except WebPushException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (404, 410):
            return status          # subscription expired — caller prunes it
        print(f"[push] delivery failed ({status}): {e}")
        return None
    except Exception as e:  # network/timeout — keep the sub, just log
        print(f"[push] delivery error: {e}")
        return None


def digest_summary(overview: dict) -> dict | None:
    """Build the daily-digest notification from an overview() snapshot.

    Returns None when there is nothing worth waking anyone for (no runs in
    the window and nothing awaiting) — a silent day stays silent.
    """
    graphs = [g for g in overview.get("graphs", []) if g.get("name")]
    runs = sum(g.get("runs_7d", 0) for g in graphs)      # caller passes days=1
    errors = sum(g.get("errors_7d", 0) for g in graphs)
    awaiting = sum(1 for a in overview.get("attention", [])
                   if a.get("status") == "interrupted")
    active = [g for g in graphs if g.get("runs_7d")]
    if runs == 0 and awaiting == 0:
        return None
    parts = [f"{runs} run{'s' if runs != 1 else ''} across "
             f"{len(active)} graph{'s' if len(active) != 1 else ''}"]
    if errors:
        parts.append(f"{errors} error{'s' if errors != 1 else ''}")
    if awaiting:
        parts.append(f"{awaiting} awaiting approval")
    return {"title": "Windhover — daily digest", "body": " · ".join(parts),
            "tag": "windhover-digest", "url": "/#fleet" if len(graphs) > 1 else "/"}


def send_to_all(store, cfg, payload: dict) -> None:
    """Fan a notification payload out to all subscriptions (fire-and-forget)."""
    if not (_AVAILABLE and cfg.push_enabled):
        return
    subs = store.push_subscriptions()
    if not subs:
        return
    data = json.dumps(payload, default=str)

    def _run():
        for sub in subs:
            gone = _send_one(sub, data, cfg)
            if gone is not None:
                try:
                    store.remove_push_subscription(sub["endpoint"])
                except Exception:
                    pass

    threading.Thread(target=_run, daemon=True).start()
