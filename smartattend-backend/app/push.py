"""
Web Push (VAPID) sender.

Fire-and-forget notifications to faculty/admin/HOD devices that have registered
a browser push subscription. Sending is done on a background thread pool so the
HTTP request that triggered a notification (a leave decision, a spoof attempt)
never blocks on push network I/O. Dead subscriptions (HTTP 404/410 from the push
service) are pruned automatically using a fresh DB session inside the worker.

If VAPID env vars are not set, every helper here is a safe no-op, so the app runs
fine locally / before keys are configured.
"""
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from . import models
from .database import SessionLocal

logger = logging.getLogger("smartattend.push")

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="push")

_ICON = "https://zaryabmalik314-code.github.io/icon-192x192.png"


def _vapid():
    """Returns (private_key, claims) or None if push is not configured."""
    private_key = os.getenv("VAPID_PRIVATE_KEY")
    subject = os.getenv("VAPID_SUBJECT", "mailto:zaryabmalik314@gmail.com")
    if not private_key:
        return None
    return private_key, {"sub": subject}


def vapid_public_key():
    """The application server key the browser needs to build a subscription."""
    return os.getenv("VAPID_PUBLIC_KEY")


def is_enabled():
    return bool(os.getenv("VAPID_PRIVATE_KEY") and os.getenv("VAPID_PUBLIC_KEY"))


def _send_one(sub_info, payload_json):
    """
    Blocking single send. Returns True to keep the subscription, False if it is
    dead and should be deleted. Imports pywebpush lazily so a missing dependency
    or unset keys degrades gracefully instead of crashing import.
    """
    cfg = _vapid()
    if not cfg:
        return True
    private_key, claims = cfg
    try:
        from pywebpush import webpush, WebPushException
    except Exception as e:  # pragma: no cover - dependency missing
        logger.warning("pywebpush unavailable: %s", e)
        return True

    try:
        webpush(
            subscription_info=sub_info,
            data=payload_json,
            vapid_private_key=private_key,
            vapid_claims=dict(claims),  # webpush mutates this dict, so pass a copy
            timeout=10,
        )
        return True
    except WebPushException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (404, 410):
            return False  # subscription expired — caller prunes it
        logger.warning("push send failed (status=%s): %s", status, e)
        return True
    except Exception as e:  # network/timeout — keep the sub, try again next time
        logger.warning("push send error: %s", e)
        return True


def _dispatch(subs, payload_json):
    """Runs on a worker thread. Sends to each sub, prunes dead ones."""
    dead_endpoints = []
    for s in subs:
        keep = _send_one(
            {"endpoint": s["endpoint"], "keys": {"p256dh": s["p256dh"], "auth": s["auth"]}},
            payload_json,
        )
        if not keep:
            dead_endpoints.append(s["endpoint"])

    if dead_endpoints:
        db = SessionLocal()
        try:
            db.query(models.PushSubscription).filter(
                models.PushSubscription.endpoint.in_(dead_endpoints)
            ).delete(synchronize_session=False)
            db.commit()
            logger.info("pruned %d dead push subscription(s)", len(dead_endpoints))
        except Exception as e:  # pragma: no cover
            logger.warning("failed to prune dead subscriptions: %s", e)
            db.rollback()
        finally:
            db.close()


def _fan_out(rows, title, body, url, tag):
    """Serialize the matched rows and hand off to the thread pool."""
    if not rows or not is_enabled():
        return
    subs = [{"endpoint": r.endpoint, "p256dh": r.p256dh, "auth": r.auth} for r in rows]
    payload_json = json.dumps({
        "title": title,
        "body": body,
        "url": url or "./index.html",
        "tag": tag,
        "icon": _ICON,
    })
    _executor.submit(_dispatch, subs, payload_json)


def push_to_subscribers(db, subscriber_type, subscriber_id, title, body, url=None, tag=None):
    rows = db.query(models.PushSubscription).filter(
        models.PushSubscription.subscriber_type == subscriber_type,
        models.PushSubscription.subscriber_id == subscriber_id,
    ).all()
    _fan_out(rows, title, body, url, tag)


def push_to_all_admins(db, title, body, url=None, tag=None):
    rows = db.query(models.PushSubscription).filter(
        models.PushSubscription.subscriber_type == "admin",
    ).all()
    _fan_out(rows, title, body, url, tag)


def push_to_department_hods(db, department, title, body, url=None, tag=None):
    """
    Push to HODs whose account department matches. subscriber_id for an HOD row
    is the HOD's id; we resolve which HOD ids belong to this department first.
    """
    hod_ids = [
        h.id for h in db.query(models.HOD).filter(models.HOD.department == department).all()
    ]
    if not hod_ids:
        return
    rows = db.query(models.PushSubscription).filter(
        models.PushSubscription.subscriber_type == "hod",
        models.PushSubscription.subscriber_id.in_(hod_ids),
    ).all()
    _fan_out(rows, title, body, url, tag)
