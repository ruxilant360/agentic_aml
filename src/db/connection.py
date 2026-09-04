"""
SQLAlchemy engine/session management for the PostgreSQL database.

The database and its two tables (`transactions`, `model_outputs`) already
exist -- this module only opens connections to them. Nothing here issues
DDL (no `create_all()` call), by design: schema ownership stays with
whatever provisioned the DB.
"""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from src.config import get_database_url

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def get_engine() -> Engine:
    """Lazily create (and cache) the SQLAlchemy engine for the AML database."""
    global _engine
    if _engine is None:
        _engine = create_engine(get_database_url(), pool_pre_ping=True, future=True)
    return _engine


def _get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionLocal


@contextmanager
def get_session() -> Iterator[Session]:
    """
    Context-managed session for a single unit of work.

    Usage:
        with get_session() as session:
            session.add(obj)
            session.commit()

    Rolls back automatically on exception so callers (tools) can catch and
    convert failures into error strings without leaving the session dirty.
    """
    session = _get_session_factory()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
