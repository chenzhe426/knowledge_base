from __future__ import annotations

from app.db.connection import get_cursor
from app.db.migrations import run_migrations
from app.db.schema import init_schema


def init_db() -> None:
    with get_cursor(commit=True) as (_, cursor):
        init_schema(cursor)
        run_migrations(cursor)