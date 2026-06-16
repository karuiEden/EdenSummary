from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint,
)
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
    quality_flags: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    quality_eval: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class JobFieldEdit(Base):
    """One user correction of a single summary field (Q3 calibration label).

    A PATCH /result is treated as a review event: every judged field gets a row
    with `edited` = whether the user changed it. Unchanged fields are genuine
    negatives — that is how we manufacture 0-labels without guessing whether the
    user looked at the result. The judge score is intentionally NOT stored here:
    Tier-2 eval is async/post-terminal and may finish after the edit, so a
    snapshot would be null-biased toward fast editors. It is joined from
    jobs.quality_eval.field_scores at calibration (fit) time instead.

    Repeat PATCHes upsert on (job_id, field) — last edit wins, no double-counting.
    """
    __tablename__ = "job_field_edits"
    __table_args__ = (
        UniqueConstraint('job_id', 'field', name='uq_job_field_edits_job_field'),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String, ForeignKey('jobs.id'))
    field: Mapped[str] = mapped_column(String)
    edited: Mapped[bool] = mapped_column(Boolean)
    original: Mapped[object | None] = mapped_column(JSON, nullable=True)
    corrected: Mapped[object | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

