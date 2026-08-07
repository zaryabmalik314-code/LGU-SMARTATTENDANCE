from pydantic import BaseModel, field_serializer, field_validator
from datetime import datetime, timezone
from typing import List, Optional
import json


class GPSReading(BaseModel):
    latitude: float
    longitude: float
    accuracy: float  # meters
    altitude: Optional[float] = None
    altitude_accuracy: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    is_mock: Optional[bool] = None


class CheckInRequest(BaseModel):
    faculty_id: int
    gps_readings: List[GPSReading]  # send 3-5 readings from frontend, backend picks best
    wifi_ssid: Optional[str] = None
    face_images: List[str]  # base64 candidate frames (from ~5s video, client-filtered for blur); backend picks the best one and embeds it
    captured_at: Optional[datetime] = None  # for offline-queued check-ins — see main.py validation

    @field_validator("face_images")
    @classmethod
    def validate_face_images(cls, v):
        for img in v:
            if len(img) > 1_500_000:
                raise ValueError("Image payload too large (max 1.5M base64 characters)")
        return v


class CheckInResponse(BaseModel):
    status: str
    reason: Optional[str] = None
    distance_to_boundary_m: Optional[float] = None
    face_match_score: Optional[float] = None
    gps_accuracy_used: Optional[float] = None
    record_id: Optional[int] = None


class FacultyEnrollRequest(BaseModel):
    name: str
    email: str
    teacher_id: str
    department: Optional[str] = None
    face_images: List[str]  # base64 frames, one per angle bucket (left/center/right), max 3
    pin: str  # 4-6 digit PIN the teacher will use to log in

    @field_validator("face_images")
    @classmethod
    def validate_face_images(cls, v):
        for img in v:
            if len(img) > 1_500_000:
                raise ValueError("Image payload too large (max 1.5M base64 characters)")
        return v

    @field_validator("pin")
    @classmethod
    def validate_pin(cls, v):
        if not v.isdigit() or not (4 <= len(v) <= 6):
            raise ValueError("PIN must be a 4-to-6 digit numeric string")
        return v


class FacultyOut(BaseModel):
    id: int
    name: str
    email: str
    teacher_id: str
    department: Optional[str] = None
    approval_status: str
    review_note: Optional[str] = None
    is_active: bool = True
    checked_in_today: bool = False  # computed by the HOD endpoint — True if a "present" check-in record exists today (PKT)
    on_leave_today: bool = False  # computed by the HOD endpoint — True if an approved leave request covers today (PKT)
    profile_photo: Optional[str] = None
    face_photos: Optional[List[str]] = None  # up to 3 small thumbnails, one per enrolled angle — admin review only

    @field_validator("face_photos", mode="before")
    @classmethod
    def parse_face_photos(cls, v):
        # Stored in the DB as a JSON-encoded string; ORM hands it over raw.
        if v is None or isinstance(v, list):
            return v
        try:
            return json.loads(v)
        except (TypeError, ValueError):
            return None

    class Config:
        from_attributes = True


class UploadPhotoRequest(BaseModel):
    teacher_id: str
    pin: str
    photo_base64: str  # data URL or raw base64 string, e.g. "data:image/jpeg;base64,..."


class UploadPhotoResponse(BaseModel):
    status: str  # "ok" | "invalid_credentials" | "not_approved" | "too_large"
    faculty: Optional[FacultyOut] = None


class LoginRequest(BaseModel):
    teacher_id: str
    pin: str
    device_fingerprint: Optional[str] = None


class LoginResponse(BaseModel):
    status: str  # "approved" | "pending" | "rejected" | "deactivated" | "device_switch_pending" | "invalid_credentials"
    faculty: Optional[FacultyOut] = None


class ReEnrollFaceRequest(BaseModel):
    teacher_id: str
    pin: str
    face_images: List[str]  # base64 frames, one per angle bucket, replaces all stored embeddings

    @field_validator("face_images")
    @classmethod
    def validate_face_images(cls, v):
        for img in v:
            if len(img) > 1_500_000:
                raise ValueError("Image payload too large (max 1.5M base64 characters)")
        return v

    @field_validator("pin")
    @classmethod
    def validate_pin(cls, v):
        if not v.isdigit() or not (4 <= len(v) <= 6):
            raise ValueError("PIN must be a 4-to-6 digit numeric string")
        return v


class ReEnrollFaceResponse(BaseModel):
    status: str  # "ok" | "invalid_credentials" | "not_approved"
    faculty: Optional[FacultyOut] = None


class ApprovalRequest(BaseModel):
    approval_status: str  # "approved" | "rejected"
    note: Optional[str] = None  # shown to the faculty on their pending/rejected screen — required in practice when rejecting so they know what to fix


class ManualAttendanceRequest(BaseModel):
    faculty_id: int
    type: str  # "check_in" | "check_out" — matches AttendanceRecord.record_type values
    note: Optional[str] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v not in ("check_in", "check_out"):
            raise ValueError('type must be "check_in" or "check_out"')
        return v


class CheckOutRequest(BaseModel):
    faculty_id: int
    gps_readings: List[GPSReading]
    wifi_ssid: Optional[str] = None
    face_images: List[str]  # base64 candidate frames, same as check-in
    captured_at: Optional[datetime] = None  # for offline-queued check-outs

    @field_validator("face_images")
    @classmethod
    def validate_face_images(cls, v):
        for img in v:
            if len(img) > 1_500_000:
                raise ValueError("Image payload too large (max 1.5M base64 characters)")
        return v


class AttendanceOut(BaseModel):
    id: int
    faculty_id: int
    faculty_name: Optional[str] = None
    department: Optional[str] = None
    profile_photo: Optional[str] = None
    timestamp: datetime
    latitude: float
    longitude: float
    gps_accuracy: float
    wifi_ssid: Optional[str] = None
    face_match_score: float
    status: str
    record_type: str
    late_minutes: int = 0
    is_late: bool = False
    duration_minutes: Optional[int] = None
    duration_status: Optional[str] = None
    flagged_suspicious: bool = False
    flag_reason: Optional[str] = None

    # DB timestamps are stored naive (no tzinfo) but are always UTC (written
    # via datetime.utcnow()). Without an explicit offset, clients parse the
    # string as their OWN local time instead of UTC, silently shifting every
    # displayed time. Stamp the UTC offset on the way out so it's unambiguous.
    @field_serializer('timestamp')
    def serialize_timestamp(self, dt: datetime, _info):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    class Config:
        from_attributes = True


class LeaveBalanceOut(BaseModel):
    faculty_id: int
    semester_label: str
    casual_leave_total: int
    casual_leave_used: int
    casual_leave_remaining: int
    working_days_total: int
    working_days_attended: int
    working_days_remaining: int
    late_margin_total: int
    late_margin_used: int
    late_margin_remaining: int

    class Config:
        from_attributes = True


class AdjustBalanceRequest(BaseModel):
    faculty_id: int
    field: str
    delta: int


class SalaryOut(BaseModel):
    id: int
    faculty_id: int
    month: str
    amount: Optional[float] = None
    status: str
    pay_date: Optional[datetime] = None

    class Config:
        from_attributes = True


class AdminBootstrapRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class AdminLoginRequest(BaseModel):
    email: str
    password: str


class AdminOut(BaseModel):
    name: str
    email: str


class AdminLoginResponse(BaseModel):
    access_token: str
    admin: AdminOut


class LeaveRequestCreate(BaseModel):
    faculty_id: int
    start_date: datetime
    end_date: datetime
    reason: str


class LeaveRequestOut(BaseModel):
    id: int
    faculty_id: int
    start_date: datetime
    end_date: datetime
    reason: str
    status: str
    created_at: datetime
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    decision_note: Optional[str] = None
    faculty_name: Optional[str] = None
    department: Optional[str] = None

    @field_serializer('start_date')
    def serialize_start_date(self, dt: datetime, _info):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    @field_serializer('end_date')
    def serialize_end_date(self, dt: datetime, _info):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    @field_serializer('created_at')
    def serialize_created_at(self, dt: datetime, _info):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    @field_serializer('reviewed_at')
    def serialize_reviewed_at(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    class Config:
        from_attributes = True


class LeaveDecisionRequest(BaseModel):
    status: str  # "approved" | "rejected"
    note: Optional[str] = None  # shown back to the faculty on their leave-request list — especially useful on rejection so they know why

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        if v not in ("approved", "rejected"):
            raise ValueError('status must be "approved" or "rejected"')
        return v


class HODLoginRequest(BaseModel):
    email: str
    password: str


class HODCreateRequest(BaseModel):
    email: str
    password: str
    department: str
    name: Optional[str] = None


class HODUpdateRequest(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    password: Optional[str] = None  # if provided, resets the HOD's password


class HODOut(BaseModel):
    name: Optional[str] = None
    email: str
    department: str


class HODLoginResponse(BaseModel):
    access_token: str
    hod: HODOut


class DeviceSwitchRequestOut(BaseModel):
    id: int
    faculty_id: int
    new_ip: Optional[str] = None
    new_fingerprint: Optional[str] = None
    status: str
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    faculty_name: Optional[str] = None
    department: Optional[str] = None

    @field_serializer('created_at')
    def serialize_created_at(self, dt: datetime, _info):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    @field_serializer('reviewed_at')
    def serialize_reviewed_at(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    class Config:
        from_attributes = True


class DeviceSwitchDecisionRequest(BaseModel):
    status: str  # "approved" | "rejected"

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        if v not in ("approved", "rejected"):
            raise ValueError('status must be "approved" or "rejected"')
        return v


class HolidayCreate(BaseModel):
    date: str  # "YYYY-MM-DD"
    label: Optional[str] = None
    department: Optional[str] = None

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError('date must be in "YYYY-MM-DD" format')
        return v


class HolidayOut(BaseModel):
    id: int
    date: str
    label: Optional[str] = None
    department: Optional[str] = None

    class Config:
        from_attributes = True


VALID_DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}


def _validate_hhmm(v: str, field: str) -> str:
    try:
        datetime.strptime(v, "%H:%M")
    except ValueError:
        raise ValueError(f'{field} must be in "HH:MM" 24-hour format')
    return v


class TimeWindowCreate(BaseModel):
    name: str
    start_time: str = "08:00"
    end_time: str = "16:00"
    grace_minutes: int = 10
    # {"friday": {"start": "08:00", "end": "13:00"}, "saturday": {"off": true}}
    overrides: dict = {}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 60:
            raise ValueError("name must be 60 characters or fewer")
        return v

    @field_validator("start_time")
    @classmethod
    def validate_start(cls, v):
        return _validate_hhmm(v, "start_time")

    @field_validator("end_time")
    @classmethod
    def validate_end(cls, v):
        return _validate_hhmm(v, "end_time")

    @field_validator("grace_minutes")
    @classmethod
    def validate_grace(cls, v):
        if v < 0 or v > 240:
            raise ValueError("grace_minutes must be between 0 and 240")
        return v

    @field_validator("overrides")
    @classmethod
    def validate_overrides(cls, v):
        if not v:
            return {}
        cleaned = {}
        for day, cfg in v.items():
            day_l = str(day).lower()
            if day_l not in VALID_DAYS:
                raise ValueError(f"unknown day in overrides: {day}")
            if not isinstance(cfg, dict):
                raise ValueError(f"overrides.{day} must be an object")
            if cfg.get("off"):
                cleaned[day_l] = {"off": True}
                continue
            entry = {}
            if cfg.get("start"):
                entry["start"] = _validate_hhmm(cfg["start"], f"overrides.{day}.start")
            if cfg.get("end"):
                entry["end"] = _validate_hhmm(cfg["end"], f"overrides.{day}.end")
            if entry:
                cleaned[day_l] = entry
        return cleaned


class TimeWindowOut(BaseModel):
    id: int
    name: str
    start_time: str
    end_time: str
    grace_minutes: int
    overrides: dict = {}
    is_active: bool

    class Config:
        from_attributes = True


class SemesterCreate(BaseModel):
    label: str
    start_date: str
    end_date: str

    @field_validator("label")
    @classmethod
    def validate_label(cls, v):
        v = (v or "").strip()
        if not v:
            raise ValueError("label is required")
        if len(v) > 80:
            raise ValueError("label must be 80 characters or fewer")
        return v

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError('date must be in "YYYY-MM-DD" format')
        return v


class SemesterOut(BaseModel):
    id: int
    label: str
    start_date: str
    end_date: str
    is_active: bool
    is_closed: bool
    created_at: datetime
    closed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SemesterSnapshotOut(BaseModel):
    id: int
    semester_id: int
    faculty_id: int
    faculty_name: Optional[str] = None
    teacher_id: Optional[str] = None
    department: Optional[str] = None
    late_minutes: int
    deduction_days: int
    days_attended: int
    casual_leave_used: int
    created_at: datetime

    class Config:
        from_attributes = True
