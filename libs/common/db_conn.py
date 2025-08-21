# ==============================================================================
# File: db_conn.py (NEW FILE)
# Purpose: Manages the async database connection using SQLAlchemy.
# ==============================================================================
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("No DATABASE_URL found in environment variables")

# Create an async engine. `echo=True` is useful for debugging as it logs all SQL.
engine = create_async_engine(DATABASE_URL, echo=False)

# Create a sessionmaker that will be used to create new sessions.
# expire_on_commit=False prevents attributes from being expired after commit.
AsyncDBSession = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db_session() -> AsyncSession:
    """
    Dependency function that yields a new SQLAlchemy AsyncSession.
    This will be used by our query functions.
    """
    async with AsyncDBSession() as session:
        yield session