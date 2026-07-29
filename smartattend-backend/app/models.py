from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class Faculty(Base):
    __tablename__ = "faculty"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    teacher_id = Column(String, unique=True, index=True, nullable=False)  # roll number / employee ID, used for login
    department = Column(String, nullable=True)
    # Up to 3 ArcFace (InsightFace) embeddings, one per angle bucket
    # (left/center/right), stored as a JSON-encoded list of float lists.
    # Legacy rows may still hold a single comma-separated face-api.js
    # descriptor — face_verify.str_to_embeddings() handles both formats.
    face_embeddings = Column(Text, nullable=False)
    # Small compressed JPEG thumbnails (one per angle bucket), JSON list of
    # data: URLs — for admin approval review only, not used in matching.
    face_photos = Column(Text, nullable=True)
    pin_hash = Column(String, nullable=False)  # bcrypt hash of login PIN
    approval_status = Column(String, default="pending", nullable=False)  # "pending" | "approved" | "rejected"
    review_note = Column(Text, nullable=True)  # admin's feedback when rejecting/flagging an enrollment — shown to the faculty on their pending/rejected screen
    is_active = Column(Boolean, default=True, nullable=False)  # False once admin has deactivated/offboarded this faculty
    last_device_ip = Column(String, nullable=True)  # set once a device switch is approved; enforcement (blocking login from a new IP) is a separate follow-up, not wired in yet
    profile_photo = Column(Text, nullable=True)  # base64-encoded image (data URL), synced across devices
    created_at = Column(DateTime, default=datetime.utcnow)

    attendance_records = relationship("AttendanceRecord", back_populates="faculty")


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    faculty_id = Column(Integer, ForeignKey("faculty.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # GPS data
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    gps_accuracy = Column(Float, nullable=False)  # meters, from navigator.geolocation
    readings_used = Column(Integer, default=1)  # how many GPS samples were averaged

    # Secondary signal
    wifi_ssid = Column(String, nullable=True)

    # Face verification
    face_match_score = Column(Float, nullable=False)  # cosine similarity 0-1
    face_verified = Column(String, nullable=False)  # "pass" | "fail" | "manual_review"

    # Final decision
    status = Column(String, nullable=False)  # "present" | "rejected_location" | "rejected_face"
    notes = Column(Text, nullable=True)
    record_type = Column(String, default="check_in", nullable=False)  # "check_in" | "check_out"

    # Anti-spoofing: flagged (not blocked) if the implied travel speed since
    # this faculty's last recorded location was physically impossible.
    flagged_suspicious = Column(Boolean, default=False, nullable=False)
    flag_reason = Column(Text, nullable=True)

    faculty = relationship("Faculty", back_populates="attendance_records")


class LeaveBalance(Base):
    """
    One row per faculty per semester. Admin dashboard (separate app) would
    normally seed/adjust these; for now a default row is auto-created on
    first read if none exists, using the constants below.
    """
    __tablename__ = "leave_balances"

    DEFAULT_CASUAL_LEAVE_TOTAL = 20
    DEFAULT_WORKING_DAYS_TOTAL = 100
    DEFAULT_LATE_MARGIN_MINUTES = 480  # 8 hours

    id = Column(Integer, primary_key=True, index=True)
    faculty_id = Column(Integer, ForeignKey("faculty.id"), nullable=False, unique=True)
    semester_label = Column(String, default="Current Semester", nullable=False)

    casual_leave_total = Column(Integer, default=DEFAULT_CASUAL_LEAVE_TOTAL, nullable=False)
    casual_leave_used = Column(Integer, default=0, nullable=False)

    working_days_total = Column(Integer, default=DEFAULT_WORKING_DAYS_TOTAL, nullable=False)
    working_days_attended = Column(Integer, default=0, nullable=False)

    late_margin_total_minutes = Column(Integer, default=DEFAULT_LATE_MARGIN_MINUTES, nullable=False)
    late_margin_used_minutes = Column(Integer, default=0, nullable=False)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SalaryRecord(Base):
    """
    Placeholder table — no payroll logic exists yet. Admin dashboard would
    populate/update these manually until a real payroll integration exists.
    """
    __tablename__ = "salary_records"

    id = Column(Integer, primary_key=True, index=True)
    faculty_id = Column(Integer, ForeignKey("faculty.id"), nullable=False, index=True)
    month_label = Column(String, nullable=False)  # e.g. "July 2026"
    amount = Column(Float, nullable=True)  # nullable since not all apps show exact figures to teachers
    status = Column(String, default="pending", nullable=False)  # "paid" | "pending" | "processing"
    pay_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Admin(Base):
    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AdminSession(Base):
    """
    Simple opaque-token session store (not JWT) — plenty for a small admin
    dashboard. Token is a random string; check table + expiry to validate.
    """
    __tablename__ = "admin_sessions"

    token = Column(String, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("admins.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)


class LeaveRequest(Base):
    __tablename__ = "leave_requests"

    id = Column(Integer, primary_key=True, index=True)
    faculty_id = Column(Integer, ForeignKey("faculty.id"), nullable=False, index=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    reason = Column(Text, nullable=False)
    status = Column(String, default="pending", nullable=False)  # "pending" | "approved" | "rejected"
    reviewed_by = Column(String, nullable=True)  # email of the HOD/admin who decided this, for the audit trail
    reviewed_at = Column(DateTime, nullable=True)
    decision_note = Column(Text, nullable=True)  # optional note from the HOD/admin, shown back to the faculty (esp. useful on rejection)
    created_at = Column(DateTime, default=datetime.utcnow)

    faculty = relationship("Faculty")


class HOD(Base):
    """
    One HOD per department (18 departments total). Deliberately a separate
    table/login from Admin, not a shared account with a role flag — the
    two were specified as fully separate systems, and keeping them apart
    means an HOD account can never accidentally end up with admin-wide
    reach through a bug in a shared permissions check.
    """
    __tablename__ = "hods"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    department = Column(String, nullable=False, index=True)  # which department this HOD reviews
    created_at = Column(DateTime, default=datetime.utcnow)


class HODSession(Base):
    """Same opaque-token pattern as AdminSession, kept as its own table
    so an HOD's session can never be mistaken for an admin's."""
    __tablename__ = "hod_sessions"

    token = Column(String, primary_key=True, index=True)
    hod_id = Column(Integer, ForeignKey("hods.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)


class DeviceSwitchRequest(Base):
    """
    Created automatically when a faculty member's login IP doesn't match
    their last recorded device IP — enforces one active device per
    account. Sits pending until an admin approves or rejects it.
    """
    __tablename__ = "device_switch_requests"

    id = Column(Integer, primary_key=True, index=True)
    faculty_id = Column(Integer, ForeignKey("faculty.id"), nullable=False, index=True)
    new_ip = Column(String, nullable=False)
    status = Column(String, default="pending", nullable=False)  # "pending" | "approved" | "rejected"
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)

    faculty = relationship("Faculty")


class Holiday(Base):
    """
    University holiday calendar. A date here is excluded from absence
    counts everywhere attendance is tallied (analytics, monthly HR
    report). department=None means it applies university-wide; a
    specific department name scopes it to just that department.
    """
    __tablename__ = "holidays"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(String, nullable=False, index=True)  # "YYYY-MM-DD" — stored as a plain date string, no time component needed
    label = Column(String, nullable=True)
    department = Column(String, nullable=True)  # None = applies to every department
    created_at = Column(DateTime, default=datetime.utcnow)


class TimeWindow(Base):
    """
    A named working-hours preset (e.g. "Normal", "Ramadan", "Spring").

    Exactly one row should have is_active=True at a time — that's the preset
    late-arrival calculation runs against. Previously these presets lived in
    the admin browser's localStorage, which meant they were per-browser and
    never reached the server, so compute_late_minutes() silently fell back to
    a hardcoded 8:00 start. Storing them here makes them authoritative.

    `overrides` is a JSON object keyed by lowercase weekday name, e.g.
        {"friday": {"start": "08:00", "end": "13:00"},
         "saturday": {"off": true},
         "sunday": {"off": true}}
    A day marked {"off": true} is a non-working day: arrivals on it never
    accrue late minutes. Days absent from the object use start_time/end_time.
    """
    __tablename__ = "time_windows"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    start_time = Column(String, nullable=False)   # "HH:MM" local (PKT)
    end_time = Column(String, nullable=False)     # "HH:MM" local (PKT)
    grace_minutes = Column(Integer, default=10, nullable=False)
    overrides = Column(Text, nullable=True)       # JSON string, see above
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
