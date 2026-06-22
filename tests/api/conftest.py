import os

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from eden_summary.api.api import app
from eden_summary.core.db import get_session
from eden_summary.core.models import Base


def _test_db_url() -> str:
    # same variables the app uses (DBConfig)
    return (
        f"postgresql+asyncpg://"
        f"{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ.get('DB_HOST', 'postgres')}:{os.environ.get('DB_PORT', '5432')}"
        f"/{os.environ['DB_NAME']}"
    )


@pytest.fixture(scope="session")
async def test_engine():
    # NullPool — as in db.py: connections don't survive across event loops
    engine = create_async_engine(_test_db_url(), poolclass=NullPool, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # idempotent (checkfirst)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(test_engine):
    return async_sessionmaker(test_engine, expire_on_commit=False)


@pytest.fixture(autouse=True)
async def clean_jobs(test_engine):
    # truncate the table BEFORE each test — isolation.
    # CASCADE: job_field_edits references jobs (FK), TRUNCATE without it would fail.
    async with test_engine.begin() as conn:
        await conn.execute(text("TRUNCATE TABLE jobs CASCADE"))
    yield


@pytest.fixture
async def db_session(session_factory):
    # for preparing data directly in a test (insert a ready-made Job)
    async with session_factory() as session:
        yield session


@pytest.fixture
async def client(session_factory):
    # override the get_session dependency with the test DB
    async def _override():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    return {"x-api-key": os.environ["X_API_KEY"]}