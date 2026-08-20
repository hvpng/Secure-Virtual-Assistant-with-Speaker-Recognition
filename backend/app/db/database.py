"""SQLAlchemy database model and session lifecycle for M4."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import JSON, Boolean, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import settings


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DATABASE_PATH = DATA_DIR / "employees.db"
DATABASE_URL = settings.database_url.strip() or f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}"


class Base(DeclarativeBase):
    pass


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    leave_days_left: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    meetings_today: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    salary_mock: Mapped[str] = mapped_column(String(120), nullable=False, default="Chưa cập nhật")
    insurance_status: Mapped[str] = mapped_column(
        String(120), nullable=False, default="Chưa cập nhật"
    )
    password_hash_mock: Mapped[str] = mapped_column(
        String(255), nullable=False, default="mock-password-unset"
    )
    voice_enrolled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


def create_database_engine(database_url: str = DATABASE_URL) -> Engine:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


engine = create_database_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(target_engine: Engine = engine) -> None:
    if target_engine is engine:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=target_engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
