"""Async SQLAlchemy engine and session factory. NullPool because connections must
not be shared across event loops (each Celery task runs its own asyncio.run)."""
from functools import lru_cache

from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from .config import get_db_cfg


@lru_cache(maxsize=None)
def _get_engine():
    return create_async_engine(get_db_cfg().db_url, echo=False, poolclass=NullPool)


@lru_cache(maxsize=None)
def _get_session_factory():
    return async_sessionmaker(_get_engine(), expire_on_commit=False)


def engine():
    """The cached process-wide async engine."""
    return _get_engine()


def AsyncLocalSession():
    """A new AsyncSession; use as `async with AsyncLocalSession() as session:`."""
    return _get_session_factory()()


async def get_session():
    """FastAPI dependency yielding a request-scoped session."""
    async with _get_session_factory()() as session:
        yield session