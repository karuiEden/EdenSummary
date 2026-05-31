from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from eden_summary.core import get_db_cfg

cfg = get_db_cfg()

engine = create_async_engine(cfg.db_url, echo=True)

AsyncLocalSession = async_sessionmaker(engine, expire_on_commit=False)

async def get_session():
    async with AsyncLocalSession() as session:
        yield session