from fastapi import FastAPI, Depends, HTTPException, Header, WebSocket, WebSocketDisconnect, Request, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, func
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import asyncio
import json
import secrets
import os
import threading
import time

from . import models, schemas
from .database import engine, get_db, run_simple_migrations, SessionLocal
from .geofence import pick_best_reading, check_location, check_impossible_movement, check_gps_spoofing
from .face_verify import verify_face_from_frames, enroll_from_frames, embeddings_to_str
from .auth import hash_pin, verify_pin, hash_password, verify_password
from .ws_manager import manager, faculty_status_manager
from . import push

models.Base.metadata.create_all(bind=engine)
run_simple_migrations()


class RateLimiter:
    """Simple in-memory per-IP rate limiter (single-worker safe)."""

    def __init__(self, max_hits: int = 30, window_seconds: int = 60):
        self._max = max_hits
        self._window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, key: str):
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 10000:
                cutoff = now - self._window
                dead = [k for k, v in self._hits.items() if not v or v[-1] < cutoff]
                for k in dead:
                    del self._hits[k]
            bucket = self._hits[key]
            cutoff = now - self._window
            self._hits[key] = bucket = [t for t in bucket if t > cutoff]
            if len(bucket) >= self._max:
                raise HTTPException(status_code=429, detail="Too many requests — please slow down")
            bucket.append(now)


_public_limiter = RateLimiter(max_hits=30, window_seconds=60)
_enroll_limiter = RateLimiter(max_hits=5, window_seconds=300)

import ipaddress as _ipaddress

_CAMPUS_NETS: list = []
_raw = os.environ.get("CAMPUS_IPS", "")
if _raw.strip():
    for part in _raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            _CAMPUS_NETS.append(_ipaddress.ip_network(part, strict=False))
        except ValueError:
            pass


def _real_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_public(request: Request):
    _public_limiter.check(_real_ip(request))


def rate_limit_enroll(request: Request):
    _enroll_limiter.check(_real_ip(request))


app = FastAPI(title="SmartAttend API")


@app.get("/api/campus-network-check")
def campus_network_check(request: Request):
    if not _CAMPUS_NETS:
        return {"on_campus_network": False}
    ip_str = _real_ip(request)
    try:
        addr = _ipaddress.ip_address(ip_str)
        for net in _CAMPUS_NETS:
            if addr in net:
                return {"on_campus_network": True}
    except ValueError:
        pass
    return {"on_campus_network": False}


_scheduler = None


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    manager.set_loop(loop)
    faculty_status_manager.set_loop(loop)
    _start_checkout_reminder_scheduler()


_checkout_reminder_sent_today = False


def run_checkout_reminders():
    """
    Runs every 5 minutes. Reads the active TimeWindow end_time, adds 15
    minutes, and sends a push reminder to faculty who checked in but haven't
    checked out — but only once per day, and only after end_time + 15 min.
    """
    global _checkout_reminder_sent_today
    db = SessionLocal()
    try:
        local_now = datetime.utcnow() + timedelta(hours=PKT_OFFSET_HOURS)

        # Reset the daily flag at midnight
        if local_now.hour == 0 and local_now.minute < 5:
            _checkout_reminder_sent_today = False
            return
        if _checkout_reminder_sent_today:
            return

        window = get_active_time_window(db)
        end_time_str = window.end_time if window else "16:00"
        weekday = local_now.strftime("%A").lower()
        day_cfg = _parse_overrides(window).get(weekday, {}) if window else {}
        if day_cfg.get("off"):
            return
        end_time_str = day_cfg.get("end") or end_time_str
        try:
            eh, em = (int(p) for p in end_time_str.split(":"))
        except (ValueError, AttributeError):
            eh, em = 16, 0
        reminder_time = local_now.replace(hour=eh, minute=em, second=0, microsecond=0) + timedelta(minutes=15)
        if local_now < reminder_time:
            return

        local_day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start_utc = local_day_start - timedelta(hours=PKT_OFFSET_HOURS)

        checked_in = {
            row[0] for row in db.query(models.AttendanceRecord.faculty_id).filter(
                models.AttendanceRecord.record_type == "check_in",
                models.AttendanceRecord.status == "present",
                models.AttendanceRecord.timestamp >= day_start_utc,
            ).distinct().all()
        }
        checked_out = {
            row[0] for row in db.query(models.AttendanceRecord.faculty_id).filter(
                models.AttendanceRecord.record_type == "check_out",
                models.AttendanceRecord.timestamp >= day_start_utc,
            ).distinct().all()
        }
        pending = checked_in - checked_out
        for faculty_id in pending:
            push.push_to_subscribers(
                db, "faculty", faculty_id,
                title="Don't forget to check out",
                body="You checked in today but haven't checked out yet. Please mark your exit.",
                url="./index.html",
                tag="checkout-reminder",
            )
        _checkout_reminder_sent_today = True
        print(f"[checkout-reminder] {len(pending)} faculty reminded to check out (trigger: {end_time_str} + 15min)")
    except Exception as e:
        print(f"[checkout-reminder] job failed: {e}")
    finally:
        db.close()


def _start_checkout_reminder_scheduler():
    """Single in-process scheduler. Checks every 5 min, fires once daily after end_time + 15 min."""
    global _scheduler
    if _scheduler is not None:
        return
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger
    except Exception as e:
        print(f"[checkout-reminder] APScheduler unavailable, scheduler disabled: {e}")
        return
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        run_checkout_reminders,
        IntervalTrigger(minutes=5),
        id="checkout_reminders",
        replace_existing=True,
    )
    _scheduler.start()
    print("[checkout-reminder] scheduler started — checking every 5 min for end_time + 15min")


# Locked to actual known frontend origins — GitHub Pages (teacher app +
# admin dashboard, same host) and the Netlify fallback used during testing.
# Add any new real domain here before pointing a frontend at this backend.
ALLOWED_ORIGINS = [
    "https://zaryabmalik314-code.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://lgusmartattendance\.netlify\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_or_create_leave_balance(db: Session, faculty_id: int) -> models.LeaveBalance:
    balance = db.query(models.LeaveBalance).filter(models.LeaveBalance.faculty_id == faculty_id).first()
    if not balance:
        balance = models.LeaveBalance(faculty_id=faculty_id)
        db.add(balance)
        db.commit()
        db.refresh(balance)
    return balance


MAX_OFFLINE_QUEUE_HOURS = 24  # how far in the past a queued offline check-in's captured_at can be
CLOCK_SKEW_TOLERANCE_MINUTES = 2  # small allowance for device clock drift


def to_utc_iso(dt: datetime) -> str:
    """Serialize a stored (naive, but always-UTC) datetime with an explicit
    UTC offset, so clients don't misparse it as their own local time."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def resolve_record_timestamp(captured_at: Optional[datetime]) -> datetime:
    """
    Offline-queued check-ins send the ORIGINAL capture time (when the
    teacher actually scanned, before connectivity returned), so late-arrival
    and movement calculations stay fair — not penalized by sync delay.

    Validates captured_at against abuse/clock issues:
      - more than a couple minutes in the future -> untrusted, ignore it
      - more than MAX_OFFLINE_QUEUE_HOURS in the past -> too old to be a
        reasonable offline queue delay, ignore it
    Falls back to the server's current time in either case.
    """
    if captured_at is None:
        return datetime.utcnow()

    now = datetime.utcnow()
    if captured_at > now + timedelta(minutes=CLOCK_SKEW_TOLERANCE_MINUTES):
        return now
    if captured_at < now - timedelta(hours=MAX_OFFLINE_QUEUE_HOURS):
        return now
    return captured_at


def check_movement_against_last_record(db: Session, faculty_id: int, new_lat: float, new_lng: float, new_time: datetime) -> dict:
    """
    Looks up this faculty's most recent attendance record (check-in or
    check-out, any status) and checks whether traveling from there to the
    new location in the elapsed time was physically plausible. Returns the
    same shape as check_impossible_movement — {"flagged", "reason", "speed_kmh"}.
    If there's no prior record, nothing to compare against, so not flagged.
    """
    last_record = (
        db.query(models.AttendanceRecord)
        .filter(models.AttendanceRecord.faculty_id == faculty_id)
        .order_by(models.AttendanceRecord.timestamp.desc())
        .first()
    )
    if not last_record:
        return {"flagged": False, "reason": None, "speed_kmh": None}

    return check_impossible_movement(
        last_record.latitude, last_record.longitude, last_record.timestamp,
        new_lat, new_lng, new_time,
    )


# 30 days by default so admins/HODs aren't forced to log in again every day.
# Override with the env var if you want shorter-lived sessions.
ADMIN_SESSION_TTL_HOURS = int(os.getenv("ADMIN_SESSION_TTL_HOURS", "720"))
FACULTY_SESSION_TTL_HOURS = int(os.getenv("FACULTY_SESSION_TTL_HOURS", "720"))

# Late-arrival tracking — fixed daily start time for everyone (Pakistan local time).
# Server timestamps are stored in UTC, so we convert before comparing.
PKT_OFFSET_HOURS = 5  # Pakistan Standard Time is UTC+5, no daylight saving
EXPECTED_ARRIVAL_HOUR = 8   # 8:00 AM local — actual campus start time
EXPECTED_ARRIVAL_MINUTE = 0
LATE_GRACE_MINUTES = 10  # arriving up to 10 min after 8:00 doesn't count as "late"


def get_active_time_window(db: Session):
    """The single preset late-arrival calculation runs against, or None."""
    return db.query(models.TimeWindow).filter(models.TimeWindow.is_active == True).first()  # noqa: E712


def _parse_overrides(window) -> dict:
    if not window or not window.overrides:
        return {}
    try:
        parsed = json.loads(window.overrides)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def resolve_expected_start(local_time: datetime, db: Session, department: Optional[str] = None):
    """
    Works out the expected arrival time for the local (PKT) day this
    check-in falls on.

    Returns (hour, minute, grace_minutes) — or None if it's a non-working
    day, meaning no late minutes should ever accrue. A day is non-working
    if the active preset marks that weekday {"off": true}, or if the date
    is in the holiday calendar (university-wide, or for this department).
    """
    date_str = local_time.strftime("%Y-%m-%d")
    holiday = (
        db.query(models.Holiday)
        .filter(
            models.Holiday.date == date_str,
            or_(models.Holiday.department.is_(None), models.Holiday.department == department),
        )
        .first()
    )
    if holiday:
        return None

    window = get_active_time_window(db)
    if not window:
        # No preset configured — fall back to the original fixed schedule so
        # behaviour is unchanged for anyone who hasn't set one up yet.
        return (EXPECTED_ARRIVAL_HOUR, EXPECTED_ARRIVAL_MINUTE, LATE_GRACE_MINUTES)

    weekday = local_time.strftime("%A").lower()
    day_cfg = _parse_overrides(window).get(weekday, {})
    if day_cfg.get("off"):
        return None

    start = day_cfg.get("start") or window.start_time
    try:
        hour, minute = (int(p) for p in start.split(":"))
    except (ValueError, AttributeError):
        hour, minute = EXPECTED_ARRIVAL_HOUR, EXPECTED_ARRIVAL_MINUTE
    return (hour, minute, window.grace_minutes)


def compute_late_minutes(utc_timestamp: datetime, db: Session, department: Optional[str] = None) -> int:
    """
    Returns how many minutes past the expected local arrival time (+ grace)
    this check-in was, or 0 if on time, early, or on a non-working day.

    The schedule comes from the active TimeWindow preset — including that
    weekday's override — rather than a hardcoded 8:00, and holidays and
    days marked "off" never accrue late minutes.
    """
    local_time = utc_timestamp + timedelta(hours=PKT_OFFSET_HOURS)
    resolved = resolve_expected_start(local_time, db, department)
    if resolved is None:
        return 0

    hour, minute, grace = resolved
    expected = local_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
    grace_deadline = expected + timedelta(minutes=grace)
    if local_time <= grace_deadline:
        return 0
    return int((local_time - expected).total_seconds() // 60)


def compute_duration_status(duration_minutes: int, db: Session) -> str:
    """
    Computes the attendance duration status based on hours worked and the
    active TimeWindow's configured working hours.
    Returns "full_day", "half_day", or "absent".
    """
    window = get_active_time_window(db)
    if window:
        try:
            sh, sm = (int(p) for p in window.start_time.split(":"))
            eh, em = (int(p) for p in window.end_time.split(":"))
            total_configured = (eh * 60 + em) - (sh * 60 + sm)
            if total_configured <= 0:
                total_configured = 480
        except (ValueError, AttributeError):
            total_configured = 480
    else:
        total_configured = 480  # default 8 hours

    half_threshold = total_configured // 2
    if duration_minutes >= total_configured:
        return "full_day"
    elif duration_minutes >= half_threshold:
        return "half_day"
    else:
        return "absent"


def update_checkin_duration(db: Session, faculty_id: int, checkout_time: datetime):
    """
    Called at check-out time. Finds today's first successful check-in for
    this faculty, computes the duration, and stores duration_minutes +
    duration_status on that check-in record.
    """
    local_co = checkout_time + timedelta(hours=PKT_OFFSET_HOURS)
    local_day_start = local_co.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start_utc = local_day_start - timedelta(hours=PKT_OFFSET_HOURS)

    checkin_record = (
        db.query(models.AttendanceRecord)
        .filter(
            models.AttendanceRecord.faculty_id == faculty_id,
            models.AttendanceRecord.record_type == "check_in",
            models.AttendanceRecord.status == "present",
            models.AttendanceRecord.timestamp >= day_start_utc,
        )
        .order_by(models.AttendanceRecord.timestamp.asc())
        .first()
    )
    if not checkin_record:
        return

    duration = int((checkout_time - checkin_record.timestamp).total_seconds() / 60)
    if duration < 0:
        return

    checkin_record.duration_minutes = duration
    checkin_record.duration_status = compute_duration_status(duration, db)


def get_current_admin(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)) -> models.Admin:
    """
    Protects admin-only endpoints. Expects header: Authorization: Bearer <token>
    Token comes from POST /api/admin/login.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    session = db.query(models.AdminSession).filter(models.AdminSession.token == token).first()
    if not session or session.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Invalid or expired admin session")

    admin = db.query(models.Admin).filter(models.Admin.id == session.admin_id).first()
    if not admin:
        raise HTTPException(status_code=401, detail="Admin account not found")
    return admin


def validate_admin_token(token: str, db: Session) -> Optional[models.Admin]:
    """Shared validation logic used by both the HTTP admin-auth dependency and the WebSocket endpoint."""
    session = db.query(models.AdminSession).filter(models.AdminSession.token == token).first()
    if not session or session.expires_at < datetime.utcnow():
        return None
    return db.query(models.Admin).filter(models.Admin.id == session.admin_id).first()


def validate_faculty_token(token: str, db: Session) -> Optional[models.Faculty]:
    session = db.query(models.FacultySession).filter(models.FacultySession.token == token).first()
    if not session or session.expires_at < datetime.utcnow():
        return None
    return db.query(models.Faculty).filter(models.Faculty.id == session.faculty_id).first()


def get_current_faculty(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)) -> models.Faculty:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    faculty = validate_faculty_token(token, db)
    if not faculty:
        raise HTTPException(status_code=401, detail="Invalid or expired faculty session")
    return faculty


def _extract_faculty_token(authorization: Optional[str], db: Session) -> Optional[models.Faculty]:
    """Try to extract a valid faculty from the Authorization header, or return None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ").strip()
    return validate_faculty_token(token, db)


@app.websocket("/ws/admin")
async def admin_websocket(websocket: WebSocket, token: str):
    """
    Admin dashboard connects here (after logging in) to receive live
    attendance events instead of needing to refresh the page. Pass the
    admin access token as a query param: wss://.../ws/admin?token=<token>
    """
    db = SessionLocal()
    try:
        admin = validate_admin_token(token, db)
    finally:
        db.close()

    if not admin:
        await websocket.accept()
        await websocket.close(code=4401)
        return

    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        manager.disconnect(websocket)


@app.websocket("/ws/hod")
async def hod_websocket(websocket: WebSocket, token: str):
    """
    HOD dashboard connects here to receive the same live broadcast stream
    as the admin dashboard. Both share the same ConnectionManager pool,
    so every event (check-in, new leave request, etc.) goes to all
    connected dashboards simultaneously regardless of account type.
    The HOD does its own client-side filtering by department so it only
    reacts to events relevant to its own faculty.
    """
    db = SessionLocal()
    try:
        session = db.query(models.HODSession).filter(models.HODSession.token == token).first()
        hod = db.query(models.HOD).filter(models.HOD.id == session.hod_id).first() if session and session.expires_at > datetime.utcnow() else None
    finally:
        db.close()

    if not hod:
        await websocket.accept()
        await websocket.close(code=4401)
        return

    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        manager.disconnect(websocket)


@app.websocket("/ws/faculty-status")
async def faculty_status_websocket(websocket: WebSocket, teacher_id: str):
    """Faculty connects after signup to receive live approval/rejection updates."""
    await faculty_status_manager.connect(teacher_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        faculty_status_manager.disconnect(teacher_id, websocket)


@app.get("/api/faculty/check-status/{teacher_id}")
def check_faculty_status(teacher_id: str, db: Session = Depends(get_db), _rl=Depends(rate_limit_public)):
    """Public endpoint — returns only the approval status (no sensitive data)."""
    faculty = db.query(models.Faculty).filter(models.Faculty.teacher_id == teacher_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="No account found for this Teacher ID")
    return {
        "status": faculty.approval_status,
        "review_note": faculty.review_note if faculty.approval_status == "rejected" else None,
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "smartattend-backend"}


@app.post("/api/faculty/enroll", response_model=schemas.FacultyOut)
def enroll_faculty(payload: schemas.FacultyEnrollRequest, db: Session = Depends(get_db), _rl=Depends(rate_limit_enroll)):
    if not payload.email.lower().endswith("@lgu.edu.pk"):
        raise HTTPException(status_code=400, detail="Only @lgu.edu.pk email addresses are allowed")

    existing = db.query(models.Faculty).filter(
        (models.Faculty.email == payload.email) | (models.Faculty.teacher_id == payload.teacher_id)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Faculty with this email or teacher ID already enrolled")

    enroll_result = enroll_from_frames(payload.face_images)
    if not enroll_result["embeddings"]:
        raise HTTPException(status_code=400, detail=f"No usable face detected in any submitted frame: {enroll_result['reason']}")

    faculty = models.Faculty(
        name=payload.name,
        email=payload.email,
        teacher_id=payload.teacher_id,
        department=payload.department,
        face_embeddings=embeddings_to_str(enroll_result["embeddings"]),
        face_photos=json.dumps(enroll_result["thumbnails"]),
        pin_hash=hash_pin(payload.pin),
        approval_status="pending",
    )
    db.add(faculty)
    db.commit()
    db.refresh(faculty)
    manager.broadcast_threadsafe({
        "event": "new_enrollment",
        "data": {"faculty_id": faculty.id, "faculty_name": faculty.name, "department": faculty.department}
    })
    return faculty


@app.get("/api/faculty", response_model=List[schemas.FacultyOut])
def list_faculty(
    approval_status: str = None,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    q = db.query(models.Faculty)
    if approval_status:
        q = q.filter(models.Faculty.approval_status == approval_status)
    return q.all()


@app.post("/api/faculty/{faculty_id}/approve", response_model=schemas.FacultyOut)
def approve_faculty(
    faculty_id: int,
    payload: schemas.ApprovalRequest,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """Admin-only — requires a valid admin session token."""
    if payload.approval_status not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="approval_status must be 'approved' or 'rejected'")

    faculty = db.query(models.Faculty).filter(models.Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    faculty.approval_status = payload.approval_status
    # A note only makes sense while something is still unresolved for the
    # faculty to act on. Clear it on approval so an old rejection reason
    # doesn't linger and confuse someone who is now fully approved.
    if payload.approval_status == "approved":
        faculty.review_note = None
    else:
        faculty.review_note = payload.note
    db.commit()
    db.refresh(faculty)
    manager.broadcast_threadsafe({
        "event": "approval_update",
        "data": {"faculty_id": faculty.id, "faculty_name": faculty.name, "status": faculty.approval_status}
    })
    faculty_status_manager.notify_threadsafe(faculty.teacher_id, {
        "event": "status_update",
        "status": faculty.approval_status,
        "review_note": faculty.review_note,
    })
    if payload.approval_status == "approved":
        push.push_to_subscribers(
            db, "faculty", faculty.id,
            "Account Approved",
            f"Welcome, {faculty.name}! Your SmartAttend account is now active — log in to start marking attendance.",
            url="./index.html", tag=f"approval-{faculty.id}",
        )
    elif payload.approval_status == "rejected":
        push.push_to_subscribers(
            db, "faculty", faculty.id,
            "Enrollment Update",
            payload.note or "Your enrollment request was not approved. Open the app for details.",
            url="./signup.html", tag=f"approval-{faculty.id}",
        )
    return faculty


@app.post("/api/faculty/{faculty_id}/deactivate", response_model=schemas.FacultyOut)
def deactivate_faculty(
    faculty_id: int,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """
    Admin-only offboarding. The account can no longer log in or check in,
    but every AttendanceRecord/LeaveRequest tied to them is left untouched —
    this only flips a flag, it never deletes history.
    """
    faculty = db.query(models.Faculty).filter(models.Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")
    faculty.is_active = False
    db.commit()
    db.refresh(faculty)
    return faculty


@app.post("/api/faculty/{faculty_id}/reactivate", response_model=schemas.FacultyOut)
def reactivate_faculty(
    faculty_id: int,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """Admin-only — undoes a deactivation if it was done by mistake."""
    faculty = db.query(models.Faculty).filter(models.Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")
    faculty.is_active = True
    db.commit()
    db.refresh(faculty)
    return faculty


@app.post("/api/attendance/manual", response_model=schemas.CheckInResponse)
def mark_manual_attendance(
    payload: schemas.ManualAttendanceRequest,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """
    Admin-only. Covers the case where a faculty member's phone isn't
    working and they called HR/admin to mark them present, or any other
    case needing a manual override. Skips GPS/face verification entirely —
    the admin session token itself is the authorization for this record.
    """
    faculty = db.query(models.Faculty).filter(models.Faculty.id == payload.faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    if faculty.approval_status != "approved":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot mark attendance — this faculty member is '{faculty.approval_status}', not approved yet.",
        )
    if not faculty.is_active:
        raise HTTPException(status_code=400, detail="Cannot mark attendance — this faculty account is deactivated.")

    # GPS/face columns are non-nullable on this table since every normal
    # record goes through real verification. A manual entry has neither, so
    # it uses sentinel values (0.0 / "manual_review") rather than loosening
    # those columns for every other record — the note+status makes it clear
    # this one was an admin override, not a real geofenced check-in.
    now = datetime.utcnow()
    record = models.AttendanceRecord(
        faculty_id=faculty.id,
        record_type=payload.type,
        status="present",
        timestamp=now,
        latitude=0.0,
        longitude=0.0,
        gps_accuracy=0.0,
        readings_used=0,
        wifi_ssid=None,
        face_match_score=0.0,
        face_verified="manual_review",
        notes=f"Manual entry by admin ({admin.email}). {payload.note or ''}".strip(),
    )
    db.add(record)

    if payload.type == "check_out":
        update_checkin_duration(db, faculty.id, now)

    db.commit()
    db.refresh(record)
    return schemas.CheckInResponse(
        status="present",
        record_id=record.id,
        face_match_score=None,
        gps_accuracy_used=None,
    )


@app.post("/api/auth/login", response_model=schemas.LoginResponse)
def login(payload: schemas.LoginRequest, request: Request, db: Session = Depends(get_db), _rl=Depends(rate_limit_public)):
    faculty = db.query(models.Faculty).filter(models.Faculty.teacher_id == payload.teacher_id).first()
    if not faculty or not verify_pin(payload.pin, faculty.pin_hash):
        return schemas.LoginResponse(status="invalid_credentials", faculty=None)

    if faculty.approval_status != "approved":
        return schemas.LoginResponse(status=faculty.approval_status, faculty=faculty)

    if not faculty.is_active:
        return schemas.LoginResponse(status="deactivated", faculty=faculty)

    forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (
        request.client.host if request.client else None
    )
    fp = payload.device_fingerprint

    if fp and faculty.last_device_fingerprint and fp != faculty.last_device_fingerprint:
        existing = db.query(models.DeviceSwitchRequest).filter(
            models.DeviceSwitchRequest.faculty_id == faculty.id,
            models.DeviceSwitchRequest.new_fingerprint == fp,
            models.DeviceSwitchRequest.status == "pending",
        ).first()
        if not existing:
            db.add(models.DeviceSwitchRequest(
                faculty_id=faculty.id, new_ip=client_ip, new_fingerprint=fp
            ))
            db.commit()
            manager.broadcast_threadsafe({
                "event": "device_switch_request",
                "data": {"faculty_id": faculty.id, "faculty_name": faculty.name}
            })
        return schemas.LoginResponse(status="device_switch_pending", faculty=faculty)

    if fp and not faculty.last_device_fingerprint:
        faculty.last_device_fingerprint = fp
    if client_ip and not faculty.last_device_ip:
        faculty.last_device_ip = client_ip

    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=FACULTY_SESSION_TTL_HOURS)
    db.add(models.FacultySession(token=token, faculty_id=faculty.id, expires_at=expires_at))
    db.commit()

    return schemas.LoginResponse(status="approved", faculty=faculty, access_token=token)


@app.post("/api/auth/re-enroll-face", response_model=schemas.ReEnrollFaceResponse)
def re_enroll_face(payload: schemas.ReEnrollFaceRequest, db: Session = Depends(get_db)):
    """
    Lets a teacher replace their stored face descriptor — either an
    already-approved teacher refreshing it (bad lighting, wants to redo
    it), or a rejected/flagged one retrying enrollment after reading the
    admin's note. Requires teacher_id + PIN, same as login, so a stranger
    can't overwrite someone else's biometric data.
    """
    faculty = db.query(models.Faculty).filter(models.Faculty.teacher_id == payload.teacher_id).first()
    if not faculty or not verify_pin(payload.pin, faculty.pin_hash):
        return schemas.ReEnrollFaceResponse(status="invalid_credentials", faculty=None)

    if faculty.approval_status not in ("approved", "rejected"):
        # "pending" already has a fresh enrollment sitting with the admin —
        # nothing useful to replace it with here.
        return schemas.ReEnrollFaceResponse(status="not_approved", faculty=faculty)

    enroll_result = enroll_from_frames(payload.face_images)
    if not enroll_result["embeddings"]:
        raise HTTPException(status_code=400, detail=f"No usable face detected in any submitted frame: {enroll_result['reason']}")

    was_rejected = faculty.approval_status == "rejected"
    faculty.face_embeddings = embeddings_to_str(enroll_result["embeddings"])
    faculty.face_photos = json.dumps(enroll_result["thumbnails"])
    if was_rejected:
        # Retrying after a rejection puts them back in the review queue —
        # the admin's earlier note has been addressed, so clear it rather
        # than leaving stale feedback attached to a fresh attempt.
        faculty.approval_status = "pending"
        faculty.review_note = None
    db.commit()
    db.refresh(faculty)

    return schemas.ReEnrollFaceResponse(status="resubmitted" if was_rejected else "ok", faculty=faculty)


class ResetPinVerifyRequest(BaseModel):
    teacher_id: str
    face_images: List[str]

class ResetPinVerifyResponse(BaseModel):
    verified: bool
    reason: Optional[str] = None

@app.post("/api/auth/reset-pin-verify")
def reset_pin_verify(payload: ResetPinVerifyRequest, db: Session = Depends(get_db)):
    faculty = db.query(models.Faculty).filter(models.Faculty.teacher_id == payload.teacher_id).first()
    if not faculty:
        return {"verified": False, "reason": "not_found"}
    if faculty.approval_status != "approved":
        return {"verified": False, "reason": "not_approved"}
    if not faculty.face_embeddings:
        return {"verified": False, "reason": "no_enrolled_face"}

    result = verify_face_from_frames(payload.face_images, faculty.face_embeddings)
    if result["verified"] == "pass":
        return {"verified": True, "reason": None}
    return {"verified": False, "reason": result.get("reason") or "face_mismatch"}


class ResetPinRequest(BaseModel):
    teacher_id: str
    face_images: List[str]
    new_pin: str

@app.post("/api/auth/reset-pin")
def reset_pin(payload: ResetPinRequest, db: Session = Depends(get_db)):
    faculty = db.query(models.Faculty).filter(models.Faculty.teacher_id == payload.teacher_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Teacher ID not found")
    if faculty.approval_status != "approved":
        raise HTTPException(status_code=403, detail="Account not approved")
    if not faculty.face_embeddings:
        raise HTTPException(status_code=400, detail="No enrolled face data")
    if not payload.new_pin or len(payload.new_pin) != 4 or not payload.new_pin.isdigit():
        raise HTTPException(status_code=400, detail="PIN must be exactly 4 digits")

    result = verify_face_from_frames(payload.face_images, faculty.face_embeddings)
    if result["verified"] != "pass":
        raise HTTPException(status_code=403, detail="Face verification failed")

    faculty.pin_hash = hash_pin(payload.new_pin)
    db.commit()
    return {"status": "ok"}


MAX_PHOTO_BASE64_CHARS = 500_000  # ~375KB binary — plenty for a resized profile pic, keeps DB rows small


@app.post("/api/faculty/upload-photo", response_model=schemas.UploadPhotoResponse)
def upload_photo(payload: schemas.UploadPhotoRequest, db: Session = Depends(get_db)):
    """
    Lets a teacher upload/replace their own profile picture, synced across
    devices via the backend instead of being stuck in one device's local
    storage. Requires teacher_id + PIN, same pattern as re-enroll-face.
    Frontend should resize/compress the image (e.g. to ~200x200) before
    sending, to keep payloads small.
    """
    faculty = db.query(models.Faculty).filter(models.Faculty.teacher_id == payload.teacher_id).first()
    if not faculty or not verify_pin(payload.pin, faculty.pin_hash):
        return schemas.UploadPhotoResponse(status="invalid_credentials", faculty=None)

    if faculty.approval_status != "approved":
        return schemas.UploadPhotoResponse(status="not_approved", faculty=faculty)

    if len(payload.photo_base64) > MAX_PHOTO_BASE64_CHARS:
        return schemas.UploadPhotoResponse(status="too_large", faculty=faculty)

    faculty.profile_photo = payload.photo_base64
    db.commit()
    db.refresh(faculty)

    return schemas.UploadPhotoResponse(status="ok", faculty=faculty)


ADMIN_BOOTSTRAP_SECRET = os.getenv("ADMIN_BOOTSTRAP_SECRET", "")


@app.post("/api/admin/bootstrap", response_model=schemas.AdminLoginResponse)
def bootstrap_admin(payload: schemas.AdminBootstrapRequest, request: Request, db: Session = Depends(get_db)):
    """
    Creates the FIRST admin account. Only works if no admin exists yet.
    Set ADMIN_BOOTSTRAP_SECRET env var and pass it as X-Bootstrap-Secret header.
    """
    existing_count = db.query(models.Admin).count()
    if existing_count > 0:
        raise HTTPException(status_code=400, detail="An admin already exists — use the dashboard login instead")

    if ADMIN_BOOTSTRAP_SECRET:
        provided = request.headers.get("x-bootstrap-secret", "")
        if provided != ADMIN_BOOTSTRAP_SECRET:
            raise HTTPException(status_code=403, detail="Invalid bootstrap secret")

    admin = models.Admin(
        email=payload.email,
        name=payload.name or payload.email,
        password_hash=hash_password(payload.password),
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)

    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=ADMIN_SESSION_TTL_HOURS)
    db.add(models.AdminSession(token=token, admin_id=admin.id, expires_at=expires_at))
    db.commit()

    return schemas.AdminLoginResponse(
        access_token=token,
        admin=schemas.AdminOut(name=admin.name, email=admin.email),
    )


@app.post("/api/admin/login", response_model=schemas.AdminLoginResponse)
def admin_login(payload: schemas.AdminLoginRequest, db: Session = Depends(get_db)):
    admin = db.query(models.Admin).filter(models.Admin.email == payload.email).first()
    if not admin or not verify_password(payload.password, admin.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=ADMIN_SESSION_TTL_HOURS)
    db.add(models.AdminSession(token=token, admin_id=admin.id, expires_at=expires_at))
    db.commit()

    return schemas.AdminLoginResponse(
        access_token=token,
        admin=schemas.AdminOut(name=admin.name, email=admin.email),
    )


HOD_SESSION_TTL_HOURS = int(os.getenv("HOD_SESSION_TTL_HOURS", "720"))  # 30 days, see ADMIN_SESSION_TTL_HOURS


def get_current_hod(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)) -> models.HOD:
    """
    Protects HOD-only endpoints. Same opaque-token pattern as
    get_current_admin, but checked against HODSession/HOD — the two
    account systems never share a session table, so an admin token can
    never accidentally pass as an HOD token or vice versa.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    session = db.query(models.HODSession).filter(models.HODSession.token == token).first()
    if not session or session.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Invalid or expired HOD session")

    hod = db.query(models.HOD).filter(models.HOD.id == session.hod_id).first()
    if not hod:
        raise HTTPException(status_code=401, detail="HOD account not found")
    return hod


@app.post("/api/hod/login", response_model=schemas.HODLoginResponse)
def hod_login(payload: schemas.HODLoginRequest, db: Session = Depends(get_db)):
    hod = db.query(models.HOD).filter(models.HOD.email == payload.email).first()
    if not hod or not verify_password(payload.password, hod.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=HOD_SESSION_TTL_HOURS)
    db.add(models.HODSession(token=token, hod_id=hod.id, expires_at=expires_at))
    db.commit()

    return schemas.HODLoginResponse(
        access_token=token,
        hod=schemas.HODOut(name=hod.name, email=hod.email, department=hod.department),
    )


@app.post("/api/admin/hods", response_model=schemas.HODOut)
def create_hod(
    payload: schemas.HODCreateRequest,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """Admin-only. One of these per department (18 total) — creates the login an HOD uses."""
    existing = db.query(models.HOD).filter(models.HOD.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An HOD with this email already exists")

    hod = models.HOD(
        email=payload.email,
        name=payload.name or payload.email,
        department=payload.department,
        password_hash=hash_password(payload.password),
    )
    db.add(hod)
    db.commit()
    db.refresh(hod)
    return schemas.HODOut(name=hod.name, email=hod.email, department=hod.department)


@app.get("/api/admin/hods", response_model=List[schemas.HODOut])
def list_hods(db: Session = Depends(get_db), admin: models.Admin = Depends(get_current_admin)):
    hods = db.query(models.HOD).all()
    return [schemas.HODOut(name=h.name, email=h.email, department=h.department) for h in hods]


@app.patch("/api/admin/hods/{email}", response_model=schemas.HODOut)
def update_hod(
    email: str,
    payload: schemas.HODUpdateRequest,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """Admin-only. Edit HOD name/department or reset their password."""
    hod = db.query(models.HOD).filter(models.HOD.email == email).first()
    if not hod:
        raise HTTPException(status_code=404, detail="HOD not found")
    if payload.name is not None:
        hod.name = payload.name
    if payload.department is not None:
        hod.department = payload.department
    if payload.password:
        hod.password_hash = hash_password(payload.password)
    db.commit()
    db.refresh(hod)
    return schemas.HODOut(name=hod.name, email=hod.email, department=hod.department)


@app.delete("/api/admin/hods/{email}", status_code=204)
def delete_hod(
    email: str,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """Admin-only. Permanently removes the HOD account and invalidates all their sessions."""
    hod = db.query(models.HOD).filter(models.HOD.email == email).first()
    if not hod:
        raise HTTPException(status_code=404, detail="HOD not found")
    db.query(models.HODSession).filter(models.HODSession.hod_id == hod.id).delete()
    db.delete(hod)
    db.commit()
    return None


@app.get("/api/hod/faculty", response_model=List[schemas.FacultyOut])
def hod_department_faculty(db: Session = Depends(get_db), hod: models.HOD = Depends(get_current_hod)):
    """
    Department-scoped by construction — an HOD's token only ever resolves
    to their own department, so there's no query parameter a client could
    tamper with to see another department's faculty.
    """
    faculty_rows = db.query(models.Faculty).filter(models.Faculty.department == hod.department).all()

    # PKT calendar-day boundary, same conversion used everywhere else in
    # this file — reused here rather than duplicated logic drifting apart.
    local_now = datetime.utcnow() + timedelta(hours=PKT_OFFSET_HOURS)
    local_today_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = local_today_start - timedelta(hours=PKT_OFFSET_HOURS)

    faculty_ids = [f.id for f in faculty_rows]
    present_today_ids = set()
    if faculty_ids:
        present_rows = (
            db.query(models.AttendanceRecord.faculty_id)
            .filter(
                models.AttendanceRecord.faculty_id.in_(faculty_ids),
                models.AttendanceRecord.timestamp >= today_start_utc,
                models.AttendanceRecord.status == "present",
            )
            .distinct()
            .all()
        )
        present_today_ids = {row[0] for row in present_rows}

    # Approved leave covering today. Without this the dashboard had no way to
    # tell "on leave" from "never showed up" and reported both as absent.
    #
    # start_date/end_date are stored as naive UTC (the client sends the local
    # PKT day boundaries as ISO and the create endpoint strips tzinfo), so they
    # compare directly against the same UTC window used for attendance above.
    # Half-open overlap: leave starts before tomorrow and ends at/after today.
    today_end_utc = today_start_utc + timedelta(days=1)
    on_leave_today_ids = set()
    if faculty_ids:
        leave_rows = (
            db.query(models.LeaveRequest.faculty_id)
            .filter(
                models.LeaveRequest.faculty_id.in_(faculty_ids),
                models.LeaveRequest.status == "approved",
                models.LeaveRequest.start_date < today_end_utc,
                models.LeaveRequest.end_date >= today_start_utc,
            )
            .distinct()
            .all()
        )
        on_leave_today_ids = {row[0] for row in leave_rows}

    out = []
    for f in faculty_rows:
        item = schemas.FacultyOut.model_validate(f)
        item.checked_in_today = f.id in present_today_ids
        item.on_leave_today = f.id in on_leave_today_ids
        out.append(item)
    return out


def _leave_request_out(r: models.LeaveRequest) -> schemas.LeaveRequestOut:
    out = schemas.LeaveRequestOut.model_validate(r)
    out.faculty_name = r.faculty.name if r.faculty else None
    out.department = r.faculty.department if r.faculty else None
    return out


@app.get("/api/hod/leave-requests", response_model=List[schemas.LeaveRequestOut])
def hod_leave_requests(db: Session = Depends(get_db), hod: models.HOD = Depends(get_current_hod)):
    rows = (
        db.query(models.LeaveRequest)
        .join(models.Faculty, models.LeaveRequest.faculty_id == models.Faculty.id)
        .filter(models.Faculty.department == hod.department)
        .order_by(models.LeaveRequest.created_at.desc())
        .all()
    )
    return [_leave_request_out(r) for r in rows]


def _notify_leave_decision(db: Session, req: models.LeaveRequest):
    """Push the leave approve/reject outcome to the faculty member's devices."""
    verb = "approved" if req.status == "approved" else "rejected"
    try:
        start = req.start_date.date().isoformat()
        end = req.end_date.date().isoformat()
        span = start if start == end else f"{start} to {end}"
    except Exception:
        span = "your requested dates"
    body = f"Your leave request ({span}) was {verb}."
    if req.decision_note:
        body += f" Note: {req.decision_note}"
    push.push_to_subscribers(
        db, "faculty", req.faculty_id,
        title=f"Leave {verb}",
        body=body,
        url="./index.html",
        tag=f"leave-{req.id}",
    )


@app.post("/api/hod/leave-requests/{request_id}", response_model=schemas.LeaveRequestOut)
def hod_decide_leave_request(
    request_id: int,
    payload: schemas.LeaveDecisionRequest,
    db: Session = Depends(get_db),
    hod: models.HOD = Depends(get_current_hod),
):
    req = (
        db.query(models.LeaveRequest)
        .join(models.Faculty, models.LeaveRequest.faculty_id == models.Faculty.id)
        .filter(models.LeaveRequest.id == request_id, models.Faculty.department == hod.department)
        .first()
    )
    if not req:
        raise HTTPException(status_code=404, detail="Leave request not found in your department")
    if req.status != "pending":
        raise HTTPException(status_code=400, detail=f"This request was already {req.status} — rejected decisions are final and pending ones can't be redecided twice")

    req.status = payload.status
    req.reviewed_by = hod.email
    req.reviewed_at = datetime.utcnow()
    req.decision_note = payload.note

    if payload.status == "approved":
        balance = get_or_create_leave_balance(db, req.faculty_id)
        # Count days by PKT-local calendar date. Timestamps are stored in UTC,
        # and the app sends the start at local 00:00 (which is the PREVIOUS day
        # in UTC), so diffing raw .date() overcounted a same-day leave as 2.
        _ls = (req.start_date + timedelta(hours=PKT_OFFSET_HOURS)).date()
        _le = (req.end_date + timedelta(hours=PKT_OFFSET_HOURS)).date()
        days = (_le - _ls).days + 1
        balance.casual_leave_used = min(balance.casual_leave_total, balance.casual_leave_used + max(days, 1))

    db.commit()
    db.refresh(req)
    manager.broadcast_threadsafe({
        "event": "leave_decided",
        "data": {"request_id": req.id, "status": req.status,
                  "faculty_id": req.faculty_id,
                  "department": req.faculty.department if req.faculty else None}
    })
    _notify_leave_decision(db, req)
    return _leave_request_out(req)


@app.get("/api/admin/leave-requests", response_model=List[schemas.LeaveRequestOut])
def admin_leave_requests(db: Session = Depends(get_db), admin: models.Admin = Depends(get_current_admin)):
    """
    Admin sees every department's leave requests, not just one — this is
    the fallback path for when an HOD is unavailable and someone still
    needs a decision made.
    """
    rows = db.query(models.LeaveRequest).order_by(models.LeaveRequest.created_at.desc()).all()
    return [_leave_request_out(r) for r in rows]


@app.post("/api/admin/leave-requests/{request_id}", response_model=schemas.LeaveRequestOut)
def admin_decide_leave_request(
    request_id: int,
    payload: schemas.LeaveDecisionRequest,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    req = db.query(models.LeaveRequest).filter(models.LeaveRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Leave request not found")
    if req.status != "pending":
        raise HTTPException(status_code=400, detail=f"This request was already {req.status} — rejected decisions are final")

    req.status = payload.status
    req.reviewed_by = admin.email
    req.reviewed_at = datetime.utcnow()
    req.decision_note = payload.note

    if payload.status == "approved":
        balance = get_or_create_leave_balance(db, req.faculty_id)
        # Count days by PKT-local calendar date. Timestamps are stored in UTC,
        # and the app sends the start at local 00:00 (which is the PREVIOUS day
        # in UTC), so diffing raw .date() overcounted a same-day leave as 2.
        _ls = (req.start_date + timedelta(hours=PKT_OFFSET_HOURS)).date()
        _le = (req.end_date + timedelta(hours=PKT_OFFSET_HOURS)).date()
        days = (_le - _ls).days + 1
        balance.casual_leave_used = min(balance.casual_leave_total, balance.casual_leave_used + max(days, 1))

    db.commit()
    db.refresh(req)
    manager.broadcast_threadsafe({
        "event": "leave_decided",
        "data": {"request_id": req.id, "status": req.status,
                  "faculty_id": req.faculty_id,
                  "department": req.faculty.department if req.faculty else None}
    })
    _notify_leave_decision(db, req)
    return _leave_request_out(req)


@app.get("/api/admin/device-switch-requests", response_model=List[schemas.DeviceSwitchRequestOut])
def admin_device_switch_requests(db: Session = Depends(get_db), admin: models.Admin = Depends(get_current_admin)):
    rows = db.query(models.DeviceSwitchRequest).order_by(models.DeviceSwitchRequest.created_at.desc()).all()
    out = []
    for r in rows:
        item = schemas.DeviceSwitchRequestOut.model_validate(r)
        item.faculty_name = r.faculty.name if r.faculty else None
        item.department = r.faculty.department if r.faculty else None
        out.append(item)
    return out


@app.post("/api/admin/device-switch-requests/{request_id}", response_model=schemas.DeviceSwitchRequestOut)
def admin_decide_device_switch(
    request_id: int,
    payload: schemas.DeviceSwitchDecisionRequest,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    req = db.query(models.DeviceSwitchRequest).filter(models.DeviceSwitchRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Device switch request not found")
    if req.status != "pending":
        raise HTTPException(status_code=400, detail=f"This request was already {req.status}")

    req.status = payload.status
    req.reviewed_at = datetime.utcnow()
    if payload.status == "approved" and req.faculty:
        if req.new_ip:
            req.faculty.last_device_ip = req.new_ip
        if req.new_fingerprint:
            req.faculty.last_device_fingerprint = req.new_fingerprint
    db.commit()
    db.refresh(req)

    item = schemas.DeviceSwitchRequestOut.model_validate(req)
    item.faculty_name = req.faculty.name if req.faculty else None
    item.department = req.faculty.department if req.faculty else None
    return item


@app.post("/api/admin/reset-device/{faculty_id}")
def admin_reset_device(
    faculty_id: int,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    faculty = db.query(models.Faculty).filter(models.Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")
    faculty.last_device_fingerprint = None
    faculty.last_device_ip = None
    db.commit()
    return {"status": "ok", "message": f"Device binding cleared for {faculty.name}"}


ADJUSTABLE_BALANCE_FIELDS = {
    "casual_leave_used",
    "casual_leave_total",
    "late_margin_used_minutes",
    "late_margin_total_minutes",
    "working_days_attended",
    "working_days_total",
}


@app.post("/api/admin/adjust-balance", response_model=schemas.LeaveBalanceOut)
def admin_adjust_balance(
    payload: schemas.AdjustBalanceRequest,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    if payload.field not in ADJUSTABLE_BALANCE_FIELDS:
        raise HTTPException(status_code=400, detail=f"Invalid field: {payload.field}")
    faculty = db.query(models.Faculty).filter(models.Faculty.id == payload.faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    b = get_or_create_leave_balance(db, payload.faculty_id)
    current = getattr(b, payload.field)
    new_val = max(0, current + payload.delta)
    setattr(b, payload.field, new_val)
    db.commit()
    db.refresh(b)

    return schemas.LeaveBalanceOut(
        faculty_id=b.faculty_id,
        semester_label=b.semester_label,
        casual_leave_total=b.casual_leave_total,
        casual_leave_used=b.casual_leave_used,
        casual_leave_remaining=b.casual_leave_total - b.casual_leave_used,
        working_days_total=b.working_days_total,
        working_days_attended=b.working_days_attended,
        working_days_remaining=max(0, b.working_days_total - b.working_days_attended),
        late_margin_total=b.late_margin_total_minutes,
        late_margin_used=b.late_margin_used_minutes,
        late_margin_remaining=max(0, b.late_margin_total_minutes - b.late_margin_used_minutes),
    )


@app.post("/api/admin/recalculate-balance", response_model=schemas.LeaveBalanceOut)
def admin_recalculate_balance(
    faculty_id: int,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    faculty = db.query(models.Faculty).filter(models.Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    b = get_or_create_leave_balance(db, faculty_id)

    check_ins = (
        db.query(models.AttendanceRecord)
        .filter(
            models.AttendanceRecord.faculty_id == faculty_id,
            models.AttendanceRecord.record_type == "check_in",
            models.AttendanceRecord.status == "present",
        )
        .all()
    )

    unique_days = set()
    total_late = 0
    for r in check_ins:
        if r.timestamp:
            unique_days.add(r.timestamp.date())
            r.late_minutes = compute_late_minutes(r.timestamp, db, faculty.department)
        total_late += r.late_minutes or 0

    approved_leaves = (
        db.query(func.count())
        .select_from(models.LeaveRequest)
        .filter(
            models.LeaveRequest.faculty_id == faculty_id,
            models.LeaveRequest.status == "approved",
        )
        .scalar()
    )

    b.working_days_attended = len(unique_days)
    b.late_margin_used_minutes = total_late
    b.casual_leave_used = approved_leaves or 0
    db.commit()
    db.refresh(b)

    return schemas.LeaveBalanceOut(
        faculty_id=b.faculty_id,
        semester_label=b.semester_label,
        casual_leave_total=b.casual_leave_total,
        casual_leave_used=b.casual_leave_used,
        casual_leave_remaining=b.casual_leave_total - b.casual_leave_used,
        working_days_total=b.working_days_total,
        working_days_attended=b.working_days_attended,
        working_days_remaining=max(0, b.working_days_total - b.working_days_attended),
        late_margin_total=b.late_margin_total_minutes,
        late_margin_used=b.late_margin_used_minutes,
        late_margin_remaining=max(0, b.late_margin_total_minutes - b.late_margin_used_minutes),
    )


@app.get("/api/holidays", response_model=List[schemas.HolidayOut])
def list_holidays(db: Session = Depends(get_db), admin: models.Admin = Depends(get_current_admin)):
    return db.query(models.Holiday).order_by(models.Holiday.date).all()


@app.post("/api/holidays", response_model=schemas.HolidayOut)
def create_holiday(
    payload: schemas.HolidayCreate,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    holiday = models.Holiday(date=payload.date, label=payload.label, department=payload.department)
    db.add(holiday)
    db.commit()
    db.refresh(holiday)
    return holiday


@app.delete("/api/holidays/{holiday_id}", status_code=204)
def delete_holiday(
    holiday_id: int,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    holiday = db.query(models.Holiday).filter(models.Holiday.id == holiday_id).first()
    if not holiday:
        raise HTTPException(status_code=404, detail="Holiday not found")
    db.delete(holiday)
    db.commit()
    return None


# ---------------------------------------------------------------------------
# TIME WINDOWS — working-hour presets that drive late-arrival calculation.
# ---------------------------------------------------------------------------

def _time_window_out(w: models.TimeWindow) -> schemas.TimeWindowOut:
    """Overrides live as a JSON string in the DB but go over the wire as an object."""
    return schemas.TimeWindowOut(
        id=w.id,
        name=w.name,
        start_time=w.start_time,
        end_time=w.end_time,
        grace_minutes=w.grace_minutes,
        overrides=_parse_overrides(w),
        is_active=w.is_active,
    )


@app.get("/api/time-windows/today")
def get_todays_schedule(db: Session = Depends(get_db)):
    """
    Public endpoint — no auth required. Faculty app calls this to show
    the current working hours and grace period so teachers know exactly
    what schedule their lateness is being judged against.
    Returns the resolved schedule for today (PKT), respecting day
    overrides and marking the day as off if applicable.
    """
    local_now = datetime.utcnow() + timedelta(hours=PKT_OFFSET_HOURS)
    window = get_active_time_window(db)

    if not window:
        return {
            "has_preset": False,
            "is_day_off": False,
            "start_time": "08:00",
            "end_time": "16:00",
            "grace_minutes": LATE_GRACE_MINUTES,
            "preset_name": None,
            "weekday": local_now.strftime("%A"),
        }

    resolved = resolve_expected_start(local_now, db)
    if resolved is None:
        return {
            "has_preset": True,
            "is_day_off": True,
            "start_time": None,
            "end_time": None,
            "grace_minutes": window.grace_minutes,
            "preset_name": window.name,
            "weekday": local_now.strftime("%A"),
        }

    hour, minute, grace = resolved
    overrides = _parse_overrides(window)
    weekday = local_now.strftime("%A").lower()
    day_cfg = overrides.get(weekday, {})
    end_time = day_cfg.get("end") or window.end_time

    return {
        "has_preset": True,
        "is_day_off": False,
        "start_time": f"{hour:02d}:{minute:02d}",
        "end_time": end_time,
        "grace_minutes": grace,
        "preset_name": window.name,
        "weekday": local_now.strftime("%A"),
    }


@app.get("/api/time-windows", response_model=List[schemas.TimeWindowOut])
def list_time_windows(
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    windows = db.query(models.TimeWindow).order_by(models.TimeWindow.name).all()
    return [_time_window_out(w) for w in windows]


@app.post("/api/time-windows", response_model=schemas.TimeWindowOut)
def create_time_window(
    payload: schemas.TimeWindowCreate,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    existing = db.query(models.TimeWindow).filter(models.TimeWindow.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f'A preset named "{payload.name}" already exists')

    window = models.TimeWindow(
        name=payload.name,
        start_time=payload.start_time,
        end_time=payload.end_time,
        grace_minutes=payload.grace_minutes,
        overrides=json.dumps(payload.overrides or {}),
        # First preset created becomes active automatically, so late
        # calculation never sits in the fallback state unnoticed.
        is_active=db.query(models.TimeWindow).count() == 0,
    )
    db.add(window)
    db.commit()
    db.refresh(window)
    return _time_window_out(window)


@app.patch("/api/time-windows/{window_id}", response_model=schemas.TimeWindowOut)
def update_time_window(
    window_id: int,
    payload: schemas.TimeWindowCreate,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    window = db.query(models.TimeWindow).filter(models.TimeWindow.id == window_id).first()
    if not window:
        raise HTTPException(status_code=404, detail="Time window not found")

    clash = (
        db.query(models.TimeWindow)
        .filter(models.TimeWindow.name == payload.name, models.TimeWindow.id != window_id)
        .first()
    )
    if clash:
        raise HTTPException(status_code=400, detail=f'A preset named "{payload.name}" already exists')

    window.name = payload.name
    window.start_time = payload.start_time
    window.end_time = payload.end_time
    window.grace_minutes = payload.grace_minutes
    window.overrides = json.dumps(payload.overrides or {})
    db.commit()
    db.refresh(window)
    return _time_window_out(window)


@app.post("/api/time-windows/{window_id}/activate", response_model=schemas.TimeWindowOut)
def activate_time_window(
    window_id: int,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """Exactly one preset is active at a time — clear the rest in the same transaction."""
    window = db.query(models.TimeWindow).filter(models.TimeWindow.id == window_id).first()
    if not window:
        raise HTTPException(status_code=404, detail="Time window not found")

    db.query(models.TimeWindow).update({models.TimeWindow.is_active: False})
    window.is_active = True
    db.commit()
    db.refresh(window)
    return _time_window_out(window)


@app.delete("/api/time-windows/{window_id}", status_code=204)
def delete_time_window(
    window_id: int,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    window = db.query(models.TimeWindow).filter(models.TimeWindow.id == window_id).first()
    if not window:
        raise HTTPException(status_code=404, detail="Time window not found")

    was_active = window.is_active
    db.delete(window)
    db.commit()

    # Never leave the system with zero active presets if others remain —
    # that would silently revert late calculation to the hardcoded fallback.
    if was_active:
        replacement = db.query(models.TimeWindow).order_by(models.TimeWindow.id).first()
        if replacement:
            replacement.is_active = True
            db.commit()
    return None


# ---------------------------------------------------------------------------
# SEMESTER MANAGEMENT
# ---------------------------------------------------------------------------

DEDUCTION_THRESHOLD_MINUTES = 480  # floor(late_minutes / threshold) = deduction days


@app.get("/api/semesters", response_model=List[schemas.SemesterOut])
def list_semesters(
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    return db.query(models.Semester).order_by(models.Semester.created_at.desc()).all()


@app.post("/api/semesters", response_model=schemas.SemesterOut)
def create_semester(
    payload: schemas.SemesterCreate,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    semester = models.Semester(
        label=payload.label,
        start_date=payload.start_date,
        end_date=payload.end_date,
        # First semester created becomes active automatically so the
        # faculty app and HOD dashboard always have a current label.
        is_active=db.query(models.Semester).count() == 0,
        is_closed=False,
    )
    db.add(semester)
    db.commit()
    db.refresh(semester)
    return semester


@app.post("/api/semesters/{semester_id}/activate", response_model=schemas.SemesterOut)
def activate_semester(
    semester_id: int,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    semester = db.query(models.Semester).filter(models.Semester.id == semester_id).first()
    if not semester:
        raise HTTPException(status_code=404, detail="Semester not found")
    if semester.is_closed:
        raise HTTPException(status_code=400, detail="Cannot activate a closed semester")
    db.query(models.Semester).update({models.Semester.is_active: False})
    semester.is_active = True
    db.commit()
    db.refresh(semester)
    return semester


@app.post("/api/semesters/{semester_id}/close", response_model=schemas.SemesterOut)
def close_semester(
    semester_id: int,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """
    Closing a semester:
      1. Locks it — no further changes to its HR numbers.
      2. Snapshots every faculty member's current LeaveBalance into
         SemesterSnapshot (immutable HR record).
      3. Resets all LeaveBalance counters so the next semester starts fresh.

    This is irreversible. The snapshot preserves the numbers permanently.
    """
    semester = db.query(models.Semester).filter(models.Semester.id == semester_id).first()
    if not semester:
        raise HTTPException(status_code=404, detail="Semester not found")
    if semester.is_closed:
        raise HTTPException(status_code=400, detail="Semester is already closed")

    all_faculty = db.query(models.Faculty).filter(
        models.Faculty.approval_status == "approved"
    ).all()

    for faculty in all_faculty:
        balance = get_or_create_leave_balance(db, faculty.id)
        late_min = balance.late_margin_used_minutes
        deduction = late_min // DEDUCTION_THRESHOLD_MINUTES

        snapshot = models.SemesterSnapshot(
            semester_id=semester.id,
            faculty_id=faculty.id,
            faculty_name=faculty.name,
            teacher_id=faculty.teacher_id,
            department=faculty.department,
            late_minutes=late_min,
            deduction_days=deduction,
            days_attended=balance.working_days_attended,
            casual_leave_used=balance.casual_leave_used,
        )
        db.add(snapshot)

        # Reset for the next semester
        balance.late_margin_used_minutes = 0
        balance.working_days_attended = 0
        balance.casual_leave_used = 0
        balance.semester_label = semester.label  # record what period just closed

    semester.is_closed = True
    semester.is_active = False
    semester.closed_at = datetime.utcnow()
    db.commit()
    db.refresh(semester)
    return semester


@app.get("/api/semesters/active", response_model=schemas.SemesterOut)
def get_active_semester(db: Session = Depends(get_db)):
    """Public endpoint — faculty app and HOD dashboard call this to show
    the current semester label next to late-minute stats."""
    semester = db.query(models.Semester).filter(
        models.Semester.is_active == True  # noqa: E712
    ).first()
    if not semester:
        raise HTTPException(status_code=404, detail="No active semester")
    return semester


@app.get("/api/time-windows/today", response_model=schemas.TimeWindowOut)
def get_todays_time_window(db: Session = Depends(get_db)):
    """
    Public endpoint — returns the schedule details for today so the faculty
    app can show teachers what time they're expected to arrive and whether
    today is a working day. Returns 404 if no preset is active (faculty app
    should show a graceful fallback, not an error).
    """
    window = get_active_time_window(db)
    if not window:
        raise HTTPException(status_code=404, detail="No active time window")
    return _time_window_out(window)


@app.get("/api/semesters/{semester_id}/snapshots",
         response_model=List[schemas.SemesterSnapshotOut])
def get_semester_snapshots(
    semester_id: int,
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """Full HR export for a closed semester."""
    semester = db.query(models.Semester).filter(models.Semester.id == semester_id).first()
    if not semester:
        raise HTTPException(status_code=404, detail="Semester not found")
    return db.query(models.SemesterSnapshot).filter(
        models.SemesterSnapshot.semester_id == semester_id
    ).order_by(models.SemesterSnapshot.department, models.SemesterSnapshot.faculty_name).all()


SPOOF_REASONS = {"photo_replay", "flat_image", "screen_display", "spoof_detected"}


def _save_spoof_attempt(db: Session, faculty, face_result, location_check, best_reading, record_type, face_images):
    """Save a spoof attempt record when face anti-spoofing triggers."""
    reason = face_result.get("reason", "")
    if reason not in SPOOF_REASONS:
        return 0

    full_frame_b64 = None
    if face_images:
        raw = face_images[0]
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        if len(raw) < 500_000:
            full_frame_b64 = raw

    attempt = models.SpoofAttempt(
        faculty_id=faculty.id,
        attempt_type=reason,
        face_reason=reason,
        full_frame_b64=full_frame_b64,
        face_crop_b64=face_result.get("face_crop_b64"),
        spoof_scores=json.dumps(face_result.get("liveness_scores", {})),
        gps_lat=best_reading.latitude if best_reading else None,
        gps_lng=best_reading.longitude if best_reading else None,
        gps_distance_m=location_check.get("distance_to_boundary_m"),
        record_type=record_type,
    )
    db.add(attempt)
    db.flush()

    count = db.query(func.count(models.SpoofAttempt.id)).filter(
        models.SpoofAttempt.faculty_id == faculty.id
    ).scalar() or 0

    manager.broadcast_threadsafe({
        "event": "spoof_attempt",
        "data": {
            "faculty_id": faculty.id,
            "faculty_name": faculty.name,
            "department": faculty.department,
            "attempt_type": reason,
            "record_type": record_type,
            "total_attempts": count,
        },
    })

    _alert_body = f"{faculty.name} ({faculty.department}) — {reason.replace('_', ' ')}. Total attempts: {count}."
    push.push_to_all_admins(
        db, title="Spoof attempt detected", body=_alert_body,
        url="./smartattend-admin.html", tag=f"spoof-{faculty.id}",
    )
    if faculty.department:
        push.push_to_department_hods(
            db, faculty.department, title="Spoof attempt detected", body=_alert_body,
            url="./smartattend-hod.html", tag=f"spoof-{faculty.id}",
        )

    return count


def _face_reason_detail(face_result):
    """Map face_verify reason codes to human-readable rejection details."""
    reason = face_result.get("reason")
    if reason == "no_usable_face_detected":
        return "no_face_detected", "No face detected. Please face the camera directly in good lighting."
    if reason == "low_detection_confidence":
        return "low_confidence", "Face not clearly visible. Try better lighting and face the camera."
    if reason in SPOOF_REASONS:
        return reason, None
    if reason is None and face_result.get("verified") == "fail":
        score_pct = round(face_result.get("score", 0) * 100, 1)
        return "face_mismatch", f"Face does not match your enrolled profile (score: {score_pct}%)."
    return reason, None


def _client_ip(request):
    """Best-effort client IP, honouring the proxy's X-Forwarded-For (Railway)."""
    forwarded_for = request.headers.get("x-forwarded-for") if request else None
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request and request.client else None


def _enforce_device_binding(db, faculty, fp, client_ip):
    """
    One-account-one-device, enforced where it matters: check-in / check-out.

    Login already binds the first device, but the attendance actions carried no
    device info, so a second device could still mark attendance. This closes
    that hole using the SAME model as login:
      - no device bound yet     -> bind this one (first device wins),
      - fingerprint matches      -> allow,
      - fingerprint differs      -> block, and file a DeviceSwitchRequest (deduped)
                                    so the admin can approve moving to the new
                                    device from the existing "Device Switches" panel.

    A missing fingerprint (older client, or an offline-queued record) skips the
    check rather than hard-failing, so it can never lock out a legitimate teacher
    on an app version that predates this field.
    Returns None if allowed; raises HTTPException(403) if blocked.
    """
    if not fp:
        return
    if not faculty.last_device_fingerprint:
        faculty.last_device_fingerprint = fp
        db.commit()
        return
    if fp == faculty.last_device_fingerprint:
        return

    existing = db.query(models.DeviceSwitchRequest).filter(
        models.DeviceSwitchRequest.faculty_id == faculty.id,
        models.DeviceSwitchRequest.new_fingerprint == fp,
        models.DeviceSwitchRequest.status == "pending",
    ).first()
    if not existing:
        db.add(models.DeviceSwitchRequest(
            faculty_id=faculty.id, new_ip=client_ip, new_fingerprint=fp
        ))
        db.commit()
        manager.broadcast_threadsafe({
            "event": "device_switch_request",
            "data": {"faculty_id": faculty.id, "faculty_name": faculty.name},
        })
    raise HTTPException(
        status_code=403,
        detail=(
            "This device isn't registered to your account. A switch request has "
            "been sent to the admin — once it's approved you can mark attendance "
            "from here."
        ),
    )


def _pkt_day_bounds_utc():
    """UTC start/end of the current PKT calendar day (same convention used elsewhere)."""
    local_now = datetime.utcnow() + timedelta(hours=PKT_OFFSET_HOURS)
    local_day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start_utc = local_day_start - timedelta(hours=PKT_OFFSET_HOURS)
    return day_start_utc, day_start_utc + timedelta(days=1)


def _enforce_once_per_day(db, faculty_id, record_type):
    """
    One successful check-in and one successful check-out per faculty per PKT day.
    Only 'present' records count, so a rejected attempt (face/location/spoof) can
    still be retried. Raises HTTPException(409) on a duplicate.
    """
    start_utc, end_utc = _pkt_day_bounds_utc()
    exists = db.query(models.AttendanceRecord).filter(
        models.AttendanceRecord.faculty_id == faculty_id,
        models.AttendanceRecord.record_type == record_type,
        models.AttendanceRecord.status == "present",
        models.AttendanceRecord.timestamp >= start_utc,
        models.AttendanceRecord.timestamp < end_utc,
    ).first()
    if exists:
        word = "checked in" if record_type == "check_in" else "checked out"
        raise HTTPException(status_code=409, detail=f"You've already {word} today. Only one {record_type.replace('_',' ')} is allowed per day.")


def _fmt_ampm(h, m):
    return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def _enforce_time_window(db, record_type):
    """
    Restrict attendance to the active working-hours window (PKT). Fail-open: if
    no preset is configured or the config can't be parsed, enforcement is
    skipped so a misconfiguration never locks everyone out.
      - Non-working day (weekday marked "off", or a holiday): block both.
      - check-in: allowed until the day's end time (+ grace); early is fine.
      - check-out: allowed from the day's start time onward; leaving late is fine.
    """
    window = get_active_time_window(db)
    if not window:
        return
    local_now = datetime.utcnow() + timedelta(hours=PKT_OFFSET_HOURS)
    resolved = resolve_expected_start(local_now, db)
    if resolved is None:
        raise HTTPException(status_code=403, detail="Today is a non-working day — attendance is disabled.")
    start_h, start_m, grace = resolved
    overrides = _parse_overrides(window)
    day_cfg = overrides.get(local_now.strftime("%A").lower(), {})
    end_str = day_cfg.get("end") or window.end_time
    try:
        end_h, end_m = (int(p) for p in end_str.split(":"))
    except (ValueError, AttributeError):
        return  # end time unreadable — don't enforce
    now_min = local_now.hour * 60 + local_now.minute
    start_min = start_h * 60 + start_m
    end_min = end_h * 60 + end_m
    win = f"{_fmt_ampm(start_h, start_m)}–{_fmt_ampm(end_h, end_m)}"
    if record_type == "check_in":
        # Check-in only inside the working-hours window (open at start, close at
        # end + grace). Blocks both before-start (e.g. 2 AM) and after-close.
        if now_min < start_min:
            raise HTTPException(status_code=403, detail=f"Check-in opens at {_fmt_ampm(start_h, start_m)}. Today's hours: {win}.")
        if now_min > end_min + (grace or 0):
            raise HTTPException(status_code=403, detail=f"Check-in is closed for today. Today's hours: {win}.")
    else:  # check_out
        # Check-out opens at start; leaving after the end time is allowed.
        if now_min < start_min:
            raise HTTPException(status_code=403, detail=f"Check-out opens at {_fmt_ampm(start_h, start_m)}. Today's hours: {win}.")


@app.post("/api/attendance/check-in", response_model=schemas.CheckInResponse)
def check_in(payload: schemas.CheckInRequest, request: Request, db: Session = Depends(get_db), auth_faculty: models.Faculty = Depends(get_current_faculty)):
    if auth_faculty.id != payload.faculty_id:
        raise HTTPException(status_code=403, detail="Token does not match faculty_id")
    faculty = auth_faculty

    if faculty.approval_status != "approved":
        raise HTTPException(status_code=403, detail=f"Faculty is not approved (status: {faculty.approval_status})")
    if not faculty.is_active:
        raise HTTPException(status_code=403, detail="This faculty account has been deactivated. Contact the admin.")

    now_utc = datetime.utcnow()
    if faculty.face_locked_until and faculty.face_locked_until > now_utc:
        remaining = int((faculty.face_locked_until - now_utc).total_seconds() / 60) + 1
        raise HTTPException(status_code=403, detail=f"Account temporarily locked — too many failed face scans. Try again in {remaining} min.")

    _enforce_device_binding(db, faculty, payload.device_fingerprint, _client_ip(request))
    _enforce_time_window(db, "check_in")
    _enforce_once_per_day(db, faculty.id, "check_in")

    # 1. Pick best GPS reading from the batch sent by frontend
    if not payload.gps_readings:
        raise HTTPException(status_code=400, detail="At least one GPS reading is required")
    best_reading = pick_best_reading(payload.gps_readings)

    # 2. Check geofence
    location_check = check_location(best_reading)

    # 3. Verify face
    face_result = verify_face_from_frames(payload.face_images, faculty.face_embeddings)

    # 4. Decide final status
    spoof_check = check_gps_spoofing(payload.gps_readings)
    if spoof_check["spoofed"] or not location_check["allowed"]:
        status = "rejected_location"
    elif face_result["verified"] != "pass":
        status = "rejected_face"
        faculty.face_fail_count = (faculty.face_fail_count or 0) + 1
        if faculty.face_fail_count >= 5:
            faculty.face_locked_until = now_utc + timedelta(minutes=30)
            faculty.face_fail_count = 0
    else:
        status = "present"
        if faculty.face_fail_count and faculty.face_fail_count > 0:
            faculty.face_fail_count = 0
            faculty.face_locked_until = None

    # 5. Save spoof attempt if anti-spoofing triggered
    spoof_count = 0
    face_reason, face_detail = _face_reason_detail(face_result)
    spoof_warning = None
    if status == "rejected_face" and face_result.get("reason") in SPOOF_REASONS:
        spoof_count = _save_spoof_attempt(db, faculty, face_result, location_check, best_reading, "check_in", payload.face_images)
        spoof_warning = "SPOOFING DETECTED. This attempt has been logged and reported to administration. Further violations may result in account suspension."

    # 6. Flag (don't block) if travel since last known location was implausibly fast
    now = resolve_record_timestamp(payload.captured_at)
    movement_check = check_movement_against_last_record(
        db, faculty.id, best_reading.latitude, best_reading.longitude, now
    )

    record_late_minutes = compute_late_minutes(now, db, faculty.department) if status == "present" else 0

    record = models.AttendanceRecord(
        faculty_id=faculty.id,
        timestamp=now,
        latitude=best_reading.latitude,
        longitude=best_reading.longitude,
        gps_accuracy=best_reading.accuracy,
        readings_used=len(payload.gps_readings),
        wifi_ssid=payload.wifi_ssid,
        face_match_score=face_result["score"],
        face_verified=face_result["verified"],
        status=status,
        notes=location_check["reason"],
        record_type="check_in",
        late_minutes=record_late_minutes,
        flagged_suspicious=movement_check["flagged"],
        flag_reason=movement_check["reason"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    if status == "present":
        local_now = datetime.utcnow() + timedelta(hours=PKT_OFFSET_HOURS)
        local_today_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_start = local_today_start - timedelta(hours=PKT_OFFSET_HOURS)
        already_counted_today = (
            db.query(models.AttendanceRecord)
            .filter(
                models.AttendanceRecord.faculty_id == faculty.id,
                models.AttendanceRecord.record_type == "check_in",
                models.AttendanceRecord.status == "present",
                models.AttendanceRecord.timestamp >= today_start,
                models.AttendanceRecord.id != record.id,
            )
            .first()
        )
        if not already_counted_today:
            balance = get_or_create_leave_balance(db, faculty.id)
            balance.working_days_attended += 1

            if record_late_minutes > 0:
                balance.late_margin_used_minutes += record_late_minutes

            db.commit()

    manager.broadcast_threadsafe({
        "event": "attendance",
        "data": {
            "record_id": record.id,
            "faculty_id": faculty.id,
            "faculty_name": faculty.name,
            "department": faculty.department,
            "record_type": "check_in",
            "status": status,
            "timestamp": to_utc_iso(record.timestamp),
            "face_match_score": face_result["score"],
            "distance_to_boundary_m": location_check["distance_to_boundary_m"],
            "flagged_suspicious": movement_check["flagged"],
            "flag_reason": movement_check["reason"],
        },
    })

    resp_reason = face_detail or location_check["reason"] if status != "present" else None
    return schemas.CheckInResponse(
        status=status,
        reason=resp_reason,
        face_reason=face_reason if status == "rejected_face" else None,
        distance_to_boundary_m=location_check["distance_to_boundary_m"],
        face_match_score=face_result["score"],
        gps_accuracy_used=best_reading.accuracy,
        record_id=record.id,
        spoof_warning=spoof_warning,
        spoof_attempt_count=spoof_count if spoof_count > 0 else None,
    )


@app.get("/api/attendance", response_model=List[schemas.AttendanceOut])
def list_attendance(
    faculty_id: int = None,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
):
    admin = None
    hod = None
    auth_fac = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        admin = validate_admin_token(token, db)
        if not admin:
            hod_session = db.query(models.HODSession).filter(models.HODSession.token == token).first()
            if hod_session and hod_session.expires_at > datetime.utcnow():
                hod = db.query(models.HOD).filter(models.HOD.id == hod_session.hod_id).first()
        if not admin and not hod:
            auth_fac = validate_faculty_token(token, db)

    if not admin and not hod and not auth_fac:
        raise HTTPException(status_code=401, detail="Valid authorization required")

    if auth_fac:
        if faculty_id is not None and faculty_id != auth_fac.id:
            raise HTTPException(status_code=403, detail="You can only view your own attendance")
        faculty_id = auth_fac.id

    q = db.query(models.AttendanceRecord)
    if faculty_id:
        q = q.filter(models.AttendanceRecord.faculty_id == faculty_id)
    elif hod:
        # HOD bulk queries are scoped to their own department - a raw
        # SELECT * across every faculty was never their access level.
        q = q.join(models.Faculty).filter(models.Faculty.department == hod.department)

    records = q.options(joinedload(models.AttendanceRecord.faculty)).order_by(models.AttendanceRecord.timestamp.desc()).all()

    if hod and faculty_id:
        # An HOD who supplied a token AND a specific faculty_id must be
        # that person's own HOD - prevents one HOD reading another
        # department's faculty one id at a time.
        target = records[0].faculty if records else None
        if target and target.department != hod.department:
            raise HTTPException(status_code=403, detail="That faculty member is not in your department")

    return [
        schemas.AttendanceOut(
            id=r.id,
            faculty_id=r.faculty_id,
            faculty_name=r.faculty.name if r.faculty else None,
            department=r.faculty.department if r.faculty else None,
            profile_photo=r.faculty.profile_photo if r.faculty else None,
            timestamp=r.timestamp,
            latitude=r.latitude,
            longitude=r.longitude,
            gps_accuracy=r.gps_accuracy,
            wifi_ssid=r.wifi_ssid,
            face_match_score=r.face_match_score,
            status=r.status,
            record_type=r.record_type,
            late_minutes=r.late_minutes,
            is_late=(r.late_minutes or 0) > 0,
            duration_minutes=r.duration_minutes,
            duration_status=r.duration_status,
            flagged_suspicious=r.flagged_suspicious,
            flag_reason=r.flag_reason,
        )
        for r in records
    ]


def _resolve_admin_or_hod(authorization: Optional[str], db: Session):
    """Returns (admin, hod) — at most one is non-None."""
    admin = None
    hod = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        admin = validate_admin_token(token, db)
        if not admin:
            hs = db.query(models.HODSession).filter(models.HODSession.token == token).first()
            if hs and hs.expires_at > datetime.utcnow():
                hod = db.query(models.HOD).filter(models.HOD.id == hs.hod_id).first()
    return admin, hod


def _working_days_in_range(db: Session, department: Optional[str], start_date, end_date) -> int:
    """Count working days (non-holiday, non-'off') in [start_date, end_date] inclusive."""
    count = 0
    d = start_date
    while d <= end_date:
        local_dt = datetime(d.year, d.month, d.day, 12, 0, 0)
        if resolve_expected_start(local_dt, db, department) is not None:
            count += 1
        d += timedelta(days=1)
    return count


@app.get("/api/analytics/summary")
def analytics_summary(
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
):
    """
    Per-faculty / per-department / overall attendance rollup for a date range
    (defaults to the current PKT month). Admin sees everyone; an HOD is scoped
    to their own department. Reuses the same late/working-day rules as check-in.
    """
    admin, hod = _resolve_admin_or_hod(authorization, db)
    if not admin and not hod:
        raise HTTPException(status_code=401, detail="Admin or HOD authorization required")

    from datetime import date as _date
    today_local = (datetime.utcnow() + timedelta(hours=PKT_OFFSET_HOURS)).date()
    try:
        start_d = _date.fromisoformat(from_date) if from_date else today_local.replace(day=1)
        end_d = _date.fromisoformat(to_date) if to_date else today_local
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD")
    if start_d > end_d:
        raise HTTPException(status_code=400, detail="'from' must be on or before 'to'")

    # UTC window covering the inclusive PKT date range
    range_start_utc = datetime(start_d.year, start_d.month, start_d.day) - timedelta(hours=PKT_OFFSET_HOURS)
    range_end_utc = datetime(end_d.year, end_d.month, end_d.day) + timedelta(days=1) - timedelta(hours=PKT_OFFSET_HOURS)

    fac_q = db.query(models.Faculty).filter(models.Faculty.is_active == True)  # noqa: E712
    if hod:
        fac_q = fac_q.filter(models.Faculty.department == hod.department)
    faculty_list = fac_q.all()
    fac_ids = [f.id for f in faculty_list]

    records = []
    if fac_ids:
        records = db.query(models.AttendanceRecord).filter(
            models.AttendanceRecord.faculty_id.in_(fac_ids),
            models.AttendanceRecord.timestamp >= range_start_utc,
            models.AttendanceRecord.timestamp < range_end_utc,
        ).all()

    # Approved leave requests overlapping the range, per faculty
    leave_rows = []
    if fac_ids:
        leave_rows = db.query(models.LeaveRequest).filter(
            models.LeaveRequest.faculty_id.in_(fac_ids),
            models.LeaveRequest.status == "approved",
            models.LeaveRequest.start_date < range_end_utc,
            models.LeaveRequest.end_date >= range_start_utc,
        ).all()

    def _leave_days_in_range(rows):
        days = set()
        for r in rows:
            d = max(r.start_date.date(), start_d)
            last = min(r.end_date.date(), end_d)
            while d <= last:
                days.add(d)
                d += timedelta(days=1)
        return days

    leave_by_fac = {}
    for lr in leave_rows:
        leave_by_fac.setdefault(lr.faculty_id, []).append(lr)

    # Per-faculty present-day set + late stats
    present_days = {fid: set() for fid in fac_ids}
    late_incidents = {fid: 0 for fid in fac_ids}
    late_minutes_total = {fid: 0 for fid in fac_ids}
    for r in records:
        if r.record_type == "check_in" and r.status == "present":
            pk_day = (r.timestamp + timedelta(hours=PKT_OFFSET_HOURS)).date()
            present_days[r.faculty_id].add(pk_day)
            if (r.late_minutes or 0) > 0:
                late_incidents[r.faculty_id] += 1
                late_minutes_total[r.faculty_id] += r.late_minutes

    # Working days per department (memoized — same for all faculty in a dept)
    wd_cache = {}
    def working_days_for(dept):
        if dept not in wd_cache:
            wd_cache[dept] = _working_days_in_range(db, dept, start_d, end_d)
        return wd_cache[dept]

    per_faculty = []
    dept_agg = {}
    for f in faculty_list:
        wd = working_days_for(f.department)
        days_present = len(present_days.get(f.id, set()))
        leave_days = len(_leave_days_in_range(leave_by_fac.get(f.id, [])))
        days_absent = max(0, wd - days_present - leave_days)
        pct = round((days_present / wd) * 100, 1) if wd else 0.0
        row = {
            "faculty_id": f.id,
            "name": f.name,
            "teacher_id": f.teacher_id,
            "department": f.department,
            "working_days": wd,
            "days_present": days_present,
            "days_late": late_incidents.get(f.id, 0),
            "total_late_minutes": late_minutes_total.get(f.id, 0),
            "leave_used": leave_days,
            "days_absent": days_absent,
            "attendance_pct": pct,
        }
        per_faculty.append(row)

        d = dept_agg.setdefault(f.department or "—", {
            "department": f.department or "—", "headcount": 0,
            "sum_pct": 0.0, "late_incidents": 0, "present": 0, "absent": 0,
        })
        d["headcount"] += 1
        d["sum_pct"] += pct
        d["late_incidents"] += row["days_late"]
        d["present"] += days_present
        d["absent"] += days_absent

    per_faculty.sort(key=lambda r: r["attendance_pct"])
    departments = []
    for d in dept_agg.values():
        departments.append({
            "department": d["department"],
            "headcount": d["headcount"],
            "avg_attendance_pct": round(d["sum_pct"] / d["headcount"], 1) if d["headcount"] else 0.0,
            "late_incidents": d["late_incidents"],
            "present": d["present"],
            "absent": d["absent"],
        })
    departments.sort(key=lambda x: x["avg_attendance_pct"])

    total_present = sum(r["days_present"] for r in per_faculty)
    total_late = sum(r["days_late"] for r in per_faculty)
    total_absent = sum(r["days_absent"] for r in per_faculty)
    overall_pct = round(sum(r["attendance_pct"] for r in per_faculty) / len(per_faculty), 1) if per_faculty else 0.0

    return {
        "range": {"from": start_d.isoformat(), "to": end_d.isoformat()},
        "overall": {
            "faculty_count": len(per_faculty),
            "avg_attendance_pct": overall_pct,
            "total_present_days": total_present,
            "total_absent_days": total_absent,
            "total_late_incidents": total_late,
        },
        "departments": departments,
        "faculty": per_faculty,
    }


@app.get("/api/attendance/late-comers", response_model=List[schemas.AttendanceOut])
def get_late_comers(
    date: Optional[str] = None,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
):
    """
    Returns check-in records where late_minutes > 0 for a given date (defaults
    to today PKT). Accessible by admin or HOD (HOD scoped to their department).
    """
    admin = None
    hod = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        admin = validate_admin_token(token, db)
        if not admin:
            hod_session = db.query(models.HODSession).filter(models.HODSession.token == token).first()
            if hod_session and hod_session.expires_at > datetime.utcnow():
                hod = db.query(models.HOD).filter(models.HOD.id == hod_session.hod_id).first()

    if not admin and not hod:
        raise HTTPException(status_code=401, detail="Admin or HOD authorization required")

    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail='date must be in "YYYY-MM-DD" format')
        local_day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        local_now = datetime.utcnow() + timedelta(hours=PKT_OFFSET_HOURS)
        local_day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)

    day_start_utc = local_day_start - timedelta(hours=PKT_OFFSET_HOURS)
    day_end_utc = day_start_utc + timedelta(days=1)

    q = (
        db.query(models.AttendanceRecord)
        .filter(
            models.AttendanceRecord.record_type == "check_in",
            models.AttendanceRecord.status == "present",
            models.AttendanceRecord.late_minutes > 0,
            models.AttendanceRecord.timestamp >= day_start_utc,
            models.AttendanceRecord.timestamp < day_end_utc,
        )
    )

    if hod:
        q = q.join(models.Faculty).filter(models.Faculty.department == hod.department)

    records = q.options(joinedload(models.AttendanceRecord.faculty)).order_by(
        models.AttendanceRecord.late_minutes.desc()
    ).all()

    return [
        schemas.AttendanceOut(
            id=r.id,
            faculty_id=r.faculty_id,
            faculty_name=r.faculty.name if r.faculty else None,
            department=r.faculty.department if r.faculty else None,
            profile_photo=r.faculty.profile_photo if r.faculty else None,
            timestamp=r.timestamp,
            latitude=r.latitude,
            longitude=r.longitude,
            gps_accuracy=r.gps_accuracy,
            wifi_ssid=r.wifi_ssid,
            face_match_score=r.face_match_score,
            status=r.status,
            record_type=r.record_type,
            late_minutes=r.late_minutes,
            is_late=True,
            duration_minutes=r.duration_minutes,
            duration_status=r.duration_status,
            flagged_suspicious=r.flagged_suspicious,
            flag_reason=r.flag_reason,
        )
        for r in records
    ]


@app.post("/api/attendance/check-out", response_model=schemas.CheckInResponse)
def check_out(payload: schemas.CheckOutRequest, request: Request, db: Session = Depends(get_db), auth_faculty: models.Faculty = Depends(get_current_faculty)):
    """
    Marks the teacher's exit from campus. Same GPS + face verification as
    check-in, but does NOT log the teacher out of the app and does NOT
    affect leave/attendance counters — it's just an exit timestamp record.
    """
    if auth_faculty.id != payload.faculty_id:
        raise HTTPException(status_code=403, detail="Token does not match faculty_id")
    faculty = auth_faculty

    if faculty.approval_status != "approved":
        raise HTTPException(status_code=403, detail=f"Faculty is not approved (status: {faculty.approval_status})")
    if not faculty.is_active:
        raise HTTPException(status_code=403, detail="This faculty account has been deactivated. Contact the admin.")

    now_utc = datetime.utcnow()
    if faculty.face_locked_until and faculty.face_locked_until > now_utc:
        remaining = int((faculty.face_locked_until - now_utc).total_seconds() / 60) + 1
        raise HTTPException(status_code=403, detail=f"Account temporarily locked — too many failed face scans. Try again in {remaining} min.")

    _enforce_device_binding(db, faculty, payload.device_fingerprint, _client_ip(request))
    _enforce_time_window(db, "check_out")
    _enforce_once_per_day(db, faculty.id, "check_out")

    if not payload.gps_readings:
        raise HTTPException(status_code=400, detail="At least one GPS reading is required")
    best_reading = pick_best_reading(payload.gps_readings)
    location_check = check_location(best_reading)
    face_result = verify_face_from_frames(payload.face_images, faculty.face_embeddings)

    spoof_check = check_gps_spoofing(payload.gps_readings)
    if spoof_check["spoofed"] or not location_check["allowed"]:
        status = "rejected_location"
    elif face_result["verified"] != "pass":
        status = "rejected_face"
        faculty.face_fail_count = (faculty.face_fail_count or 0) + 1
        if faculty.face_fail_count >= 5:
            faculty.face_locked_until = now_utc + timedelta(minutes=30)
            faculty.face_fail_count = 0
    else:
        status = "present"
        if faculty.face_fail_count and faculty.face_fail_count > 0:
            faculty.face_fail_count = 0
            faculty.face_locked_until = None

    spoof_count = 0
    face_reason, face_detail = _face_reason_detail(face_result)
    spoof_warning = None
    if status == "rejected_face" and face_result.get("reason") in SPOOF_REASONS:
        spoof_count = _save_spoof_attempt(db, faculty, face_result, location_check, best_reading, "check_out", payload.face_images)
        spoof_warning = "SPOOFING DETECTED. This attempt has been logged and reported to administration. Further violations may result in account suspension."

    now = resolve_record_timestamp(payload.captured_at)
    movement_check = check_movement_against_last_record(
        db, faculty.id, best_reading.latitude, best_reading.longitude, now
    )

    record = models.AttendanceRecord(
        faculty_id=faculty.id,
        timestamp=now,
        latitude=best_reading.latitude,
        longitude=best_reading.longitude,
        gps_accuracy=best_reading.accuracy,
        readings_used=len(payload.gps_readings),
        wifi_ssid=payload.wifi_ssid,
        face_match_score=face_result["score"],
        face_verified=face_result["verified"],
        status=status,
        notes=location_check["reason"],
        record_type="check_out",
        flagged_suspicious=movement_check["flagged"],
        flag_reason=movement_check["reason"],
    )
    db.add(record)

    if status == "present":
        update_checkin_duration(db, faculty.id, now)

    db.commit()
    db.refresh(record)

    manager.broadcast_threadsafe({
        "event": "attendance",
        "data": {
            "record_id": record.id,
            "faculty_id": faculty.id,
            "faculty_name": faculty.name,
            "department": faculty.department,
            "record_type": "check_out",
            "status": status,
            "timestamp": to_utc_iso(record.timestamp),
            "face_match_score": face_result["score"],
            "distance_to_boundary_m": location_check["distance_to_boundary_m"],
            "flagged_suspicious": movement_check["flagged"],
            "flag_reason": movement_check["reason"],
        },
    })

    resp_reason = face_detail or location_check["reason"] if status != "present" else None
    return schemas.CheckInResponse(
        status=status,
        reason=resp_reason,
        face_reason=face_reason if status == "rejected_face" else None,
        distance_to_boundary_m=location_check["distance_to_boundary_m"],
        face_match_score=face_result["score"],
        gps_accuracy_used=best_reading.accuracy,
        record_id=record.id,
        spoof_warning=spoof_warning,
        spoof_attempt_count=spoof_count if spoof_count > 0 else None,
    )


@app.get("/api/spoof-attempts", response_model=List[schemas.SpoofAttemptOut])
def list_spoof_attempts(
    resolved: Optional[bool] = None,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
):
    admin = None
    if authorization and authorization.startswith("Bearer "):
        admin = validate_admin_token(authorization.removeprefix("Bearer ").strip(), db)
    if not admin:
        raise HTTPException(status_code=401, detail="Admin authorization required")

    q = db.query(models.SpoofAttempt).options(joinedload(models.SpoofAttempt.faculty))
    if resolved is not None:
        q = q.filter(models.SpoofAttempt.resolved == resolved)
    attempts = q.order_by(models.SpoofAttempt.created_at.desc()).limit(200).all()

    return [
        schemas.SpoofAttemptOut(
            id=a.id,
            faculty_id=a.faculty_id,
            faculty_name=a.faculty.name if a.faculty else None,
            teacher_id=a.faculty.teacher_id if a.faculty else None,
            department=a.faculty.department if a.faculty else None,
            profile_photo=a.faculty.profile_photo if a.faculty else None,
            attempt_type=a.attempt_type,
            face_reason=a.face_reason,
            full_frame_b64=a.full_frame_b64,
            face_crop_b64=a.face_crop_b64,
            spoof_scores=a.spoof_scores,
            gps_lat=a.gps_lat,
            gps_lng=a.gps_lng,
            gps_distance_m=a.gps_distance_m,
            record_type=a.record_type,
            resolved=a.resolved,
            admin_notes=a.admin_notes,
            created_at=a.created_at,
        )
        for a in attempts
    ]


@app.put("/api/spoof-attempts/{attempt_id}/resolve")
def resolve_spoof_attempt(
    attempt_id: int,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
):
    admin = None
    if authorization and authorization.startswith("Bearer "):
        admin = validate_admin_token(authorization.removeprefix("Bearer ").strip(), db)
    if not admin:
        raise HTTPException(status_code=401, detail="Admin authorization required")

    attempt = db.query(models.SpoofAttempt).filter(models.SpoofAttempt.id == attempt_id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Spoof attempt not found")

    attempt.resolved = True
    db.commit()
    return {"status": "ok"}


@app.get("/api/spoof-attempts/count")
def spoof_attempt_count(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
):
    admin = None
    if authorization and authorization.startswith("Bearer "):
        admin = validate_admin_token(authorization.removeprefix("Bearer ").strip(), db)
    if not admin:
        raise HTTPException(status_code=401, detail="Admin authorization required")

    total = db.query(func.count(models.SpoofAttempt.id)).scalar() or 0
    unresolved = db.query(func.count(models.SpoofAttempt.id)).filter(
        models.SpoofAttempt.resolved == False
    ).scalar() or 0
    return {"total": total, "unresolved": unresolved}


@app.get("/api/leave-balance", response_model=schemas.LeaveBalanceOut)
def get_leave_balance(faculty_id: int, db: Session = Depends(get_db), authorization: Optional[str] = Header(None)):
    admin, hod = _resolve_admin_or_hod(authorization, db)
    auth_fac = _extract_faculty_token(authorization, db) if not admin and not hod else None
    if not admin and not hod and not auth_fac:
        raise HTTPException(status_code=401, detail="Valid authorization required")
    if auth_fac and auth_fac.id != faculty_id:
        raise HTTPException(status_code=403, detail="You can only view your own leave balance")
    faculty = db.query(models.Faculty).filter(models.Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    b = get_or_create_leave_balance(db, faculty_id)

    elapsed = 0
    active_sem = db.query(models.Semester).filter(models.Semester.is_active == True).first()  # noqa: E712
    if active_sem:
        try:
            from datetime import date as _date
            sem_start = _date.fromisoformat(active_sem.start_date)
            today = (datetime.utcnow() + timedelta(hours=PKT_OFFSET_HOURS)).date()
            end = min(today, _date.fromisoformat(active_sem.end_date))
            if sem_start <= end:
                elapsed = _working_days_in_range(db, faculty.department, sem_start, end)
        except Exception:
            pass

    return schemas.LeaveBalanceOut(
        faculty_id=b.faculty_id,
        semester_label=b.semester_label,
        casual_leave_total=b.casual_leave_total,
        casual_leave_used=b.casual_leave_used,
        casual_leave_remaining=b.casual_leave_total - b.casual_leave_used,
        working_days_total=b.working_days_total,
        working_days_attended=b.working_days_attended,
        working_days_remaining=max(0, b.working_days_total - b.working_days_attended),
        working_days_elapsed=elapsed,
        late_margin_total=b.late_margin_total_minutes,
        late_margin_used=b.late_margin_used_minutes,
        late_margin_remaining=max(0, b.late_margin_total_minutes - b.late_margin_used_minutes),
    )


@app.get("/api/salary", response_model=List[schemas.SalaryOut])
def get_salary_records(faculty_id: int, db: Session = Depends(get_db), authorization: Optional[str] = Header(None)):
    admin, hod = _resolve_admin_or_hod(authorization, db)
    auth_fac = _extract_faculty_token(authorization, db) if not admin and not hod else None
    if not admin and not hod and not auth_fac:
        raise HTTPException(status_code=401, detail="Valid authorization required")
    if auth_fac and auth_fac.id != faculty_id:
        raise HTTPException(status_code=403, detail="You can only view your own salary records")
    faculty = db.query(models.Faculty).filter(models.Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    records = (
        db.query(models.SalaryRecord)
        .filter(models.SalaryRecord.faculty_id == faculty_id)
        .order_by(models.SalaryRecord.created_at.desc())
        .all()
    )
    return [
        schemas.SalaryOut(
            id=r.id,
            faculty_id=r.faculty_id,
            month=r.month_label,
            amount=r.amount,
            status=r.status,
            pay_date=r.pay_date,
        )
        for r in records
    ]


@app.post("/api/leave-request", response_model=schemas.LeaveRequestOut)
def create_leave_request(payload: schemas.LeaveRequestCreate, db: Session = Depends(get_db), auth_faculty: models.Faculty = Depends(get_current_faculty)):
    if auth_faculty.id != payload.faculty_id:
        raise HTTPException(status_code=403, detail="Token does not match faculty_id")
    faculty = auth_faculty

    if payload.start_date > payload.end_date:
        raise HTTPException(status_code=400, detail="Start date must be before or equal to end date")

    start_utc = payload.start_date.replace(tzinfo=None) if payload.start_date.tzinfo else payload.start_date
    end_utc = payload.end_date.replace(tzinfo=None) if payload.end_date.tzinfo else payload.end_date

    req = models.LeaveRequest(
        faculty_id=payload.faculty_id,
        start_date=start_utc,
        end_date=end_utc,
        reason=payload.reason,
        status="pending",
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    manager.broadcast_threadsafe({
        "event": "new_leave_request",
        "data": {
            "request_id": req.id,
            "faculty_id": req.faculty_id,
            "department": req.faculty.department if req.faculty else None,
        }
    })
    return _leave_request_out(req)


@app.get("/api/leave-requests", response_model=List[schemas.LeaveRequestOut])
def get_faculty_leave_requests(faculty_id: int, db: Session = Depends(get_db), authorization: Optional[str] = Header(None)):
    admin, hod = _resolve_admin_or_hod(authorization, db)
    auth_fac = _extract_faculty_token(authorization, db) if not admin and not hod else None
    if not admin and not hod and not auth_fac:
        raise HTTPException(status_code=401, detail="Valid authorization required")
    if auth_fac and auth_fac.id != faculty_id:
        raise HTTPException(status_code=403, detail="You can only view your own leave requests")
    faculty = db.query(models.Faculty).filter(models.Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    requests = (
        db.query(models.LeaveRequest)
        .filter(models.LeaveRequest.faculty_id == faculty_id)
        .order_by(models.LeaveRequest.created_at.desc())
        .all()
    )
    return [_leave_request_out(r) for r in requests]


# =========================================================================
# WEB PUSH — subscription management
# =========================================================================
@app.get("/api/push/vapid-public-key")
def get_vapid_public_key():
    """
    The browser needs this application-server public key to build a push
    subscription. 404 when push isn't configured so the frontend can hide
    the "enable notifications" control gracefully.
    """
    key = push.vapid_public_key()
    if not key:
        raise HTTPException(status_code=404, detail="Push notifications are not configured")
    return {"key": key}


@app.post("/api/push/subscribe")
def push_subscribe(
    payload: schemas.PushSubscribeRequest,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
):
    """
    Stores (or upserts on endpoint) a browser push subscription.
      - faculty identify by faculty_id in the body (same trust model as check-in)
      - admin / hod identify by their bearer session token
    """
    stype = payload.subscriber_type
    if stype == "faculty":
        faculty = None
        if payload.faculty_id:
            faculty = db.query(models.Faculty).filter(models.Faculty.id == payload.faculty_id).first()
        elif payload.teacher_id:
            faculty = db.query(models.Faculty).filter(models.Faculty.teacher_id == payload.teacher_id).first()
        if not faculty:
            raise HTTPException(status_code=404, detail="Faculty not found")
        subscriber_id = faculty.id
    elif stype == "admin":
        token = authorization.removeprefix("Bearer ").strip() if authorization and authorization.startswith("Bearer ") else None
        admin = validate_admin_token(token, db) if token else None
        if not admin:
            raise HTTPException(status_code=401, detail="Admin authorization required")
        subscriber_id = admin.id
    elif stype == "hod":
        hod = get_current_hod(authorization, db)
        subscriber_id = hod.id
    else:
        raise HTTPException(status_code=400, detail="Invalid subscriber_type")

    existing = db.query(models.PushSubscription).filter(
        models.PushSubscription.endpoint == payload.endpoint
    ).first()
    if existing:
        existing.subscriber_type = stype
        existing.subscriber_id = subscriber_id
        existing.p256dh = payload.keys.p256dh
        existing.auth = payload.keys.auth
        existing.user_agent = payload.user_agent
    else:
        db.add(models.PushSubscription(
            subscriber_type=stype,
            subscriber_id=subscriber_id,
            endpoint=payload.endpoint,
            p256dh=payload.keys.p256dh,
            auth=payload.keys.auth,
            user_agent=payload.user_agent,
        ))
    db.commit()
    return {"status": "subscribed"}


@app.post("/api/push/unsubscribe")
def push_unsubscribe(payload: schemas.PushUnsubscribeRequest, db: Session = Depends(get_db), authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    token = authorization.removeprefix("Bearer ").strip()
    admin = validate_admin_token(token, db)
    hod = None
    auth_fac = None
    if not admin:
        hod_session = db.query(models.HODSession).filter(models.HODSession.token == token).first()
        if hod_session and hod_session.expires_at > datetime.utcnow():
            hod = db.query(models.HOD).filter(models.HOD.id == hod_session.hod_id).first()
    if not admin and not hod:
        auth_fac = validate_faculty_token(token, db)
    if not admin and not hod and not auth_fac:
        raise HTTPException(status_code=401, detail="Valid authorization required")
    sub = db.query(models.PushSubscription).filter(
        models.PushSubscription.endpoint == payload.endpoint
    ).first()
    if sub:
        if auth_fac and (sub.subscriber_type != "faculty" or sub.subscriber_id != auth_fac.id):
            raise HTTPException(status_code=403, detail="You can only unsubscribe your own push endpoints")
        db.delete(sub)
        db.commit()
    return {"status": "unsubscribed"}


@app.post("/api/push/test")
def push_test(
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """Admin-only. Sends a test notification to the calling admin's devices."""
    if not push.is_enabled():
        raise HTTPException(status_code=503, detail="Push notifications are not configured")
    push.push_to_subscribers(
        db, "admin", admin.id,
        title="SmartAttend test",
        body="Push notifications are working.",
        url="./smartattend-admin.html",
        tag="test",
    )
    return {"status": "sent"}


@app.post("/api/tasks/run-checkout-reminders")
def trigger_checkout_reminders(
    db: Session = Depends(get_db),
    admin: models.Admin = Depends(get_current_admin),
):
    """Admin-only manual trigger for the daily checkout-reminder job (for testing)."""
    run_checkout_reminders()
    return {"status": "ok"}


@app.post("/api/admin/logout")
def admin_logout(db: Session = Depends(get_db), authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        db.query(models.AdminSession).filter(models.AdminSession.token == token).delete(synchronize_session=False)
        db.commit()
    return {"status": "ok"}


@app.post("/api/hod/logout")
def hod_logout(db: Session = Depends(get_db), authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        db.query(models.HODSession).filter(models.HODSession.token == token).delete(synchronize_session=False)
        db.commit()
    return {"status": "ok"}


@app.post("/api/auth/logout")
def faculty_logout(db: Session = Depends(get_db), authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        db.query(models.FacultySession).filter(models.FacultySession.token == token).delete(synchronize_session=False)
        db.commit()
    return {"status": "ok"}
