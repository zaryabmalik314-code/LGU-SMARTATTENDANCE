import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Railway injects DATABASE_URL for Postgres. Falls back to local SQLite for dev.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smartattend.db")

# Railway's Postgres URL starts with postgres:// or postgresql:// — force the
# pg8000 driver (pure Python, no native libpq dependency) instead of the
# default psycopg2, which has had native-library issues on Railway's builder.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+pg8000://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+pg8000://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args=connect_args)
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=50,
        pool_pre_ping=True
    )
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_simple_migrations():
    """
    SQLAlchemy's Base.metadata.create_all() only creates tables that don't
    exist yet — it does NOT add new columns to tables that already exist.
    Since this project has no Alembic migration setup, this function checks
    for a few specific columns added after the tables were first created,
    and adds them via raw ALTER TABLE if missing. Safe to call on every
    startup — it's a no-op once the column already exists.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    with engine.connect() as conn:
        if "faculty" in table_names:
            existing_columns = {col["name"] for col in inspector.get_columns("faculty")}
            if "profile_photo" not in existing_columns:
                conn.execute(text("ALTER TABLE faculty ADD COLUMN profile_photo TEXT"))
                conn.commit()
                print("[migration] Added missing column: faculty.profile_photo")

            # face_embedding (single face-api.js descriptor) -> face_embeddings
            # (JSON list of up to 3 InsightFace/ArcFace embeddings). Rename in
            # place and carry over any existing legacy value so old enrolled
            # faculty keep working (as a 1-item list) until they re-enroll.
            if "face_embeddings" not in existing_columns and "face_embedding" in existing_columns:
                conn.execute(text("ALTER TABLE faculty ADD COLUMN face_embeddings TEXT"))
                conn.commit()
                conn.execute(text("UPDATE faculty SET face_embeddings = face_embedding WHERE face_embeddings IS NULL"))
                conn.commit()
                print("[migration] Added faculty.face_embeddings and backfilled from face_embedding")

            if "face_photos" not in existing_columns:
                conn.execute(text("ALTER TABLE faculty ADD COLUMN face_photos TEXT"))
                conn.commit()
                print("[migration] Added missing column: faculty.face_photos")

            if "review_note" not in existing_columns:
                conn.execute(text("ALTER TABLE faculty ADD COLUMN review_note TEXT"))
                conn.commit()
                print("[migration] Added missing column: faculty.review_note")

            if "is_active" not in existing_columns:
                conn.execute(text("ALTER TABLE faculty ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT TRUE"))
                conn.commit()
                print("[migration] Added missing column: faculty.is_active")

            if "last_device_ip" not in existing_columns:
                conn.execute(text("ALTER TABLE faculty ADD COLUMN last_device_ip TEXT"))
                conn.commit()
                print("[migration] Added missing column: faculty.last_device_ip")

            if "last_device_fingerprint" not in existing_columns:
                conn.execute(text("ALTER TABLE faculty ADD COLUMN last_device_fingerprint TEXT"))
                conn.commit()
                print("[migration] Added missing column: faculty.last_device_fingerprint")

            if "face_fail_count" not in existing_columns:
                conn.execute(text("ALTER TABLE faculty ADD COLUMN face_fail_count INTEGER NOT NULL DEFAULT 0"))
                conn.commit()
                print("[migration] Added missing column: faculty.face_fail_count")

            if "face_locked_until" not in existing_columns:
                conn.execute(text("ALTER TABLE faculty ADD COLUMN face_locked_until TIMESTAMP"))
                conn.commit()
                print("[migration] Added missing column: faculty.face_locked_until")

        if "attendance_records" in table_names:
            existing_columns = {col["name"] for col in inspector.get_columns("attendance_records")}
            if "flagged_suspicious" not in existing_columns:
                conn.execute(text(
                    "ALTER TABLE attendance_records ADD COLUMN flagged_suspicious BOOLEAN NOT NULL DEFAULT FALSE"
                ))
                conn.commit()
                print("[migration] Added missing column: attendance_records.flagged_suspicious")
            if "flag_reason" not in existing_columns:
                conn.execute(text("ALTER TABLE attendance_records ADD COLUMN flag_reason TEXT"))
                conn.commit()
                print("[migration] Added missing column: attendance_records.flag_reason")
            if "late_minutes" not in existing_columns:
                conn.execute(text(
                    "ALTER TABLE attendance_records ADD COLUMN late_minutes INTEGER NOT NULL DEFAULT 0"
                ))
                conn.commit()
                # Deliberately NOT backfilled: existing rows predate the
                # TimeWindow system and were never scheduled against it, so
                # recomputing "late" for them now would apply today's rules
                # retroactively to attendance that happened under a different
                # (or no) schedule. They start at 0; only new check-ins from
                # this point forward get a real value.
                print("[migration] Added missing column: attendance_records.late_minutes (existing rows default 0, not backfilled)")
            
            if "duration_minutes" not in existing_columns:
                conn.execute(text("ALTER TABLE attendance_records ADD COLUMN duration_minutes INTEGER"))
                conn.commit()
                print("[migration] Added missing column: attendance_records.duration_minutes")
            if "duration_status" not in existing_columns:
                conn.execute(text("ALTER TABLE attendance_records ADD COLUMN duration_status TEXT"))
                conn.commit()
                print("[migration] Added missing column: attendance_records.duration_status")

            # Create indexes if they don't exist
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_attendance_records_faculty_id ON attendance_records (faculty_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_attendance_records_timestamp ON attendance_records (timestamp)"))
            conn.commit()
            print("[migration] Verified indexes on attendance_records table")

        if "salary_records" in table_names:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_salary_records_faculty_id ON salary_records (faculty_id)"))
            conn.commit()
            print("[migration] Verified indexes on salary_records table")

        # semesters and semester_snapshots are new tables created by
        # create_all() above — no ALTER TABLE needed since they're fresh.
        # The active-semester constraint (at most one is_active row) is
        # enforced in application code, not as a DB unique index, so that
        # closing one semester and activating another can happen atomically
        # in a single transaction without a temporary constraint violation.

        if "device_switch_requests" in table_names:
            existing_columns = {col["name"] for col in inspector.get_columns("device_switch_requests")}
            if "new_fingerprint" not in existing_columns:
                conn.execute(text("ALTER TABLE device_switch_requests ADD COLUMN new_fingerprint TEXT"))
                conn.commit()
                print("[migration] Added missing column: device_switch_requests.new_fingerprint")

        if "leave_requests" in table_names:
            existing_columns = {col["name"] for col in inspector.get_columns("leave_requests")}
            if "reviewed_by" not in existing_columns:
                conn.execute(text("ALTER TABLE leave_requests ADD COLUMN reviewed_by TEXT"))
                conn.commit()
                print("[migration] Added missing column: leave_requests.reviewed_by")
            if "reviewed_at" not in existing_columns:
                conn.execute(text("ALTER TABLE leave_requests ADD COLUMN reviewed_at TIMESTAMP"))
                conn.commit()
                print("[migration] Added missing column: leave_requests.reviewed_at")
            if "decision_note" not in existing_columns:
                conn.execute(text("ALTER TABLE leave_requests ADD COLUMN decision_note TEXT"))
                conn.commit()
                print("[migration] Added missing column: leave_requests.decision_note")
