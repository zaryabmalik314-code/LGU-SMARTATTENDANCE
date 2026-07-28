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
    created_at = Column(DateTime, default=datetime.utcnow)

    faculty = relationship("Faculty")
