from datetime import datetime

from sqlalchemy import String, DateTime, Index
from sqlalchemy.dialects.postgresql import ARRAY, JSON
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    pass

class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index('ix_jobs_status_updated_at', 'status', 'updated_at'),
    )
    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String)
    emails: Mapped[list] = mapped_column(ARRAY(String))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    artifacts: Mapped[dict] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(String)
    warning: Mapped[str | None] = mapped_column(String)

