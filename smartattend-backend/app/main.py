from fastapi import FastAPI, Depends, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import asyncio
import json
import secrets

from . import models, schemas
from .database import engine, get_db, run_simple_migrations, SessionLocal
from .geofence import pick_best_reading, check_location, check_impossible_movement
from .face_verify import verify_face_from_frames, enroll_from_frames, embeddings_to_str
from .auth import hash_pin, verify_pin, hash_password, verify_password
from .ws_manager import manager

models.Base.metadata.create_all(bind=engine)
run_simple_migrations()

app = FastAPI(title="SmartAttend API")


@app.on_event("startup")
async def on_startup():
    manager.set_loop(asyncio.get_running_loop())


# Locked to actual known frontend origins — GitHub Pages (teacher app +
# admin dashboard, same host) and the Netlify fallback used during testing.
# Add any new real domain here before pointing a frontend at this backend.
ALLOWED_ORIGINS = [
    "https://zaryabmalik314-code.github.io",
    "https://preeminent-kulfi-f8576a.netlify.app",
    "https://brilliant-pothos-ee9a7a.netlify.app",
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


ADMIN_SESSION_TTL_HOURS = 24

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
    return int((local_time - grace_deadline).total_seconds() // 60)


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


@app.get("/")
def root():
    return {"status": "ok", "service": "smartattend-backend"}


@app.post("/api/faculty/enroll", response_model=schemas.FacultyOut)
def enroll_faculty(payload: schemas.FacultyEnrollRequest, db: Session = Depends(get_db)):
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
    record = models.AttendanceRecord(
        faculty_id=faculty.id,
        record_type=payload.type,
        status="present",
        timestamp=datetime.utcnow(),
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
    db.commit()
    db.refresh(record)
    return schemas.CheckInResponse(
        status="present",
        record_id=record.id,
        face_match_score=None,
        gps_accuracy_used=None,
    )


@app.post("/api/auth/login", response_model=schemas.LoginResponse)
def login(payload: schemas.LoginRequest, db: Session = Depends(get_db)):
    faculty = db.query(models.Faculty).filter(models.Faculty.teacher_id == payload.teacher_id).first()
    if not faculty or not verify_pin(payload.pin, faculty.pin_hash):
        return schemas.LoginResponse(status="invalid_credentials", faculty=None)

    if faculty.approval_status != "approved":
        return schemas.LoginResponse(status=faculty.approval_status, faculty=faculty)

    # Offboarded accounts keep their approval_status and all their history,
    # so approval_status alone can't gate them out — check the flag too.
    if not faculty.is_active:
        return schemas.LoginResponse(status="deactivated", faculty=faculty)

    return schemas.LoginResponse(status="approved", faculty=faculty)


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


@app.post("/api/admin/bootstrap", response_model=schemas.AdminLoginResponse)
def bootstrap_admin(payload: schemas.AdminBootstrapRequest, db: Session = Depends(get_db)):
    """
    Creates the FIRST admin account. Only works if no admin exists yet.
    The dashboard has no signup UI, so run this once yourself via curl/Postman:
      curl -X POST <backend-url>/api/admin/bootstrap -H "Content-Type: application/json" \
        -d '{"email":"you@example.com","password":"yourpassword","name":"Your Name"}'
    Then log in normally through the dashboard.
    """
    existing_count = db.query(models.Admin).count()
    if existing_count > 0:
        raise HTTPException(status_code=400, detail="An admin already exists — use the dashboard login instead")

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


HOD_SESSION_TTL_HOURS = 24


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

    out = []
    for f in faculty_rows:
        item = schemas.FacultyOut.model_validate(f)
        item.checked_in_today = f.id in present_today_ids
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
        days = (req.end_date.date() - req.start_date.date()).days + 1
        balance.casual_leave_used = min(balance.casual_leave_total, balance.casual_leave_used + max(days, 1))

    db.commit()
    db.refresh(req)
    manager.broadcast_threadsafe({
        "event": "leave_decided",
        "data": {"request_id": req.id, "status": req.status,
                  "faculty_id": req.faculty_id,
                  "department": req.faculty.department if req.faculty else None}
    })
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
        days = (req.end_date.date() - req.start_date.date()).days + 1
        balance.casual_leave_used = min(balance.casual_leave_total, balance.casual_leave_used + max(days, 1))

    db.commit()
    db.refresh(req)
    manager.broadcast_threadsafe({
        "event": "leave_decided",
        "data": {"request_id": req.id, "status": req.status,
                  "faculty_id": req.faculty_id,
                  "department": req.faculty.department if req.faculty else None}
    })
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
        req.faculty.last_device_ip = req.new_ip
    db.commit()
    db.refresh(req)

    item = schemas.DeviceSwitchRequestOut.model_validate(req)
    item.faculty_name = req.faculty.name if req.faculty else None
    item.department = req.faculty.department if req.faculty else None
    return item


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


@app.post("/api/attendance/check-in", response_model=schemas.CheckInResponse)
def check_in(payload: schemas.CheckInRequest, db: Session = Depends(get_db)):
    faculty = db.query(models.Faculty).filter(models.Faculty.id == payload.faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    if faculty.approval_status != "approved":
        raise HTTPException(status_code=403, detail=f"Faculty is not approved (status: {faculty.approval_status})")
    if not faculty.is_active:
        raise HTTPException(status_code=403, detail="This faculty account has been deactivated. Contact the admin.")

    # 1. Pick best GPS reading from the batch sent by frontend
    if not payload.gps_readings:
        raise HTTPException(status_code=400, detail="At least one GPS reading is required")
    best_reading = pick_best_reading(payload.gps_readings)

    # 2. Check geofence
    location_check = check_location(best_reading)

    # 3. Verify face
    face_result = verify_face_from_frames(payload.face_images, faculty.face_embeddings)

    # 4. Decide final status
    if not location_check["allowed"]:
        status = "rejected_location"
    elif face_result["verified"] != "pass":
        status = "rejected_face"
    else:
        status = "present"

    # 5. Flag (don't block) if travel since last known location was implausibly fast
    now = resolve_record_timestamp(payload.captured_at)
    movement_check = check_movement_against_last_record(
        db, faculty.id, best_reading.latitude, best_reading.longitude, now
    )

    # Late minutes belong to THIS check-in event, independent of whether it's
    # the day's first (that distinction only matters for the running
    # LeaveBalance totals below, not for what happened at this specific time).
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

    # Count this as an attended working day (once per calendar day, only if present)
    # and track late-arrival minutes against the fixed 9:00 AM start time.
    if status == "present":
        # Calculate PKT calendar day boundary in UTC
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

    return schemas.CheckInResponse(
        status=status,
        reason=location_check["reason"] if status != "present" else face_result.get("reason"),
        distance_to_boundary_m=location_check["distance_to_boundary_m"],
        face_match_score=face_result["score"],
        gps_accuracy_used=best_reading.accuracy,
        record_id=record.id,
    )


@app.get("/api/attendance", response_model=List[schemas.AttendanceOut])
def list_attendance(
    faculty_id: int = None,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
):
    # The faculty app has no session-token system anywhere (check-in itself
    # only trusts a faculty_id in the body) - so a single-faculty_id query
    # stays open here too, to not break that flow. What changed: an HOD
    # bearer token, if presented, is now actually validated and scoped to
    # that HOD's department, instead of silently ignored. Bulk (no
    # faculty_id) queries still require either admin or HOD.
    admin = None
    hod = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        admin = validate_admin_token(token, db)
        if not admin:
            hod_session = db.query(models.HODSession).filter(models.HODSession.token == token).first()
            if hod_session and hod_session.expires_at > datetime.utcnow():
                hod = db.query(models.HOD).filter(models.HOD.id == hod_session.hod_id).first()

    if faculty_id is None and not admin and not hod:
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header for bulk query")

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
            timestamp=r.timestamp,
            latitude=r.latitude,
            longitude=r.longitude,
            gps_accuracy=r.gps_accuracy,
            wifi_ssid=r.wifi_ssid,
            face_match_score=r.face_match_score,
            status=r.status,
            record_type=r.record_type,
            late_minutes=r.late_minutes,
            flagged_suspicious=r.flagged_suspicious,
            flag_reason=r.flag_reason,
        )
        for r in records
    ]


@app.post("/api/attendance/check-out", response_model=schemas.CheckInResponse)
def check_out(payload: schemas.CheckOutRequest, db: Session = Depends(get_db)):
    """
    Marks the teacher's exit from campus. Same GPS + face verification as
    check-in, but does NOT log the teacher out of the app and does NOT
    affect leave/attendance counters — it's just an exit timestamp record.
    """
    faculty = db.query(models.Faculty).filter(models.Faculty.id == payload.faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    if faculty.approval_status != "approved":
        raise HTTPException(status_code=403, detail=f"Faculty is not approved (status: {faculty.approval_status})")
    if not faculty.is_active:
        raise HTTPException(status_code=403, detail="This faculty account has been deactivated. Contact the admin.")

    if not payload.gps_readings:
        raise HTTPException(status_code=400, detail="At least one GPS reading is required")
    best_reading = pick_best_reading(payload.gps_readings)
    location_check = check_location(best_reading)
    face_result = verify_face_from_frames(payload.face_images, faculty.face_embeddings)

    if not location_check["allowed"]:
        status = "rejected_location"
    elif face_result["verified"] != "pass":
        status = "rejected_face"
    else:
        status = "present"  # "present" here just means "exit successfully logged"

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

    return schemas.CheckInResponse(
        status=status,
        reason=location_check["reason"] if status != "present" else face_result.get("reason"),
        distance_to_boundary_m=location_check["distance_to_boundary_m"],
        face_match_score=face_result["score"],
        gps_accuracy_used=best_reading.accuracy,
        record_id=record.id,
    )


@app.get("/api/leave-balance", response_model=schemas.LeaveBalanceOut)
def get_leave_balance(faculty_id: int, db: Session = Depends(get_db)):
    faculty = db.query(models.Faculty).filter(models.Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    b = get_or_create_leave_balance(db, faculty_id)

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


@app.get("/api/salary", response_model=List[schemas.SalaryOut])
def get_salary_records(faculty_id: int, db: Session = Depends(get_db)):
    """
    Placeholder — no payroll system wired up yet. Returns whatever rows
    exist in salary_records for this faculty (admin dashboard would need
    to create these; nothing auto-generates them yet).
    """
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
def create_leave_request(payload: schemas.LeaveRequestCreate, db: Session = Depends(get_db)):
    faculty = db.query(models.Faculty).filter(models.Faculty.id == payload.faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

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
    return req
    faculty = db.query(models.Faculty).filter(models.Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    requests = (
        db.query(models.LeaveRequest)
        .filter(models.LeaveRequest.faculty_id == faculty_id)
        .order_by(models.LeaveRequest.created_at.desc())
        .all()
    )
    return requests
