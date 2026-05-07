from contextlib import asynccontextmanager, contextmanager
import os
import re
from uuid import uuid4

from psycopg import AsyncConnection, Connection

from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore, PostgresStore

DEFAULT_POSTGRES_URI = os.environ.get(
    "POSTGRES_URI", "postgres://postgres:postgres@localhost:5442/"
)


def _validate_database_name(database: str) -> str:
    if not re.match(r'^[a-zA-Z0-9_]+$', database):
        raise ValueError(f"Invalid database name: {database}")
    return database


def _hitl_confirm_drop(database: str) -> bool:
    import sys
    if not sys.stdin.isatty():
        return True
    response = input(
        f"[HITL Approval Required] About to DROP DATABASE '{database}'. "
        "Type 'yes' to confirm: "
    ).strip().lower()
    return response == "yes"


async def _hitl_confirm_drop_async(database: str) -> bool:
    import sys
    if not sys.stdin.isatty():
        return True
    import asyncio
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: input(
            f"[HITL Approval Required] About to DROP DATABASE '{database}'. "
            "Type 'yes' to confirm: "
        ).strip().lower()
    )
    return response == "yes"


@contextmanager
def _store_memory():
    store = InMemoryStore()
    yield store


@contextmanager
def _store_postgres():
    database = _validate_database_name(f"test_{uuid4().hex[:16]}")
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute("CREATE DATABASE " + database)
    try:
        # yield store
        with PostgresStore.from_conn_string(DEFAULT_POSTGRES_URI + database) as store:
            store.setup()
            yield store
    finally:
        # drop unique db
        if _hitl_confirm_drop(database):
            with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
                conn.execute("DROP DATABASE " + database)


@contextmanager
def _store_postgres_pipe():
    database = _validate_database_name(f"test_{uuid4().hex[:16]}")
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute("CREATE DATABASE " + database)
    try:
        # yield store
        with PostgresStore.from_conn_string(DEFAULT_POSTGRES_URI + database) as store:
            store.setup()  # Run in its own transaction
        with PostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database, pipeline=True
        ) as store:
            yield store
    finally:
        # drop unique db
        if _hitl_confirm_drop(database):
            with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
                conn.execute("DROP DATABASE " + database)


@contextmanager
def _store_postgres_pool():
    database = _validate_database_name(f"test_{uuid4().hex[:16]}")
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute("CREATE DATABASE " + database)
    try:
        # yield store
        with PostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database, pool_config={"max_size": 10}
        ) as store:
            store.setup()
            yield store
    finally:
        # drop unique db
        if _hitl_confirm_drop(database):
            with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
                conn.execute("DROP DATABASE " + database)


@asynccontextmanager
async def _store_postgres_aio():
    database = _validate_database_name(f"test_{uuid4().hex[:16]}")
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute("CREATE DATABASE " + database)
    try:
        async with AsyncPostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database
        ) as store:
            await store.setup()
            yield store
    finally:
        if await _hitl_confirm_drop_async(database):
            async with await AsyncConnection.connect(
                DEFAULT_POSTGRES_URI, autocommit=True
            ) as conn:
                await conn.execute("DROP DATABASE " + database)


@asynccontextmanager
async def _store_postgres_aio_pipe():
    database = _validate_database_name(f"test_{uuid4().hex[:16]}")
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute("CREATE DATABASE " + database)
    try:
        async with AsyncPostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database
        ) as store:
            await store.setup()  # Run in its own transaction
        async with AsyncPostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database, pipeline=True
        ) as store:
            yield store
    finally:
        if await _hitl_confirm_drop_async(database):
            async with await AsyncConnection.connect(
                DEFAULT_POSTGRES_URI, autocommit=True
            ) as conn:
                await conn.execute("DROP DATABASE " + database)


@asynccontextmanager
async def _store_postgres_aio_pool():
    database = _validate_database_name(f"test_{uuid4().hex[:16]}")
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute("CREATE DATABASE " + database)
    try:
        async with AsyncPostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database,
            pool_config={"max_size": 10},
        ) as store:
            await store.setup()
            yield store
    finally:
        if await _hitl_confirm_drop_async(database):
            async with await AsyncConnection.connect(
                DEFAULT_POSTGRES_URI, autocommit=True
            ) as conn:
                await conn.execute("DROP DATABASE " + database)


__all__ = [
    "_store_memory",
    "_store_postgres",
    "_store_postgres_pipe",
    "_store_postgres_pool",
    "_store_postgres_aio",
    "_store_postgres_aio_pipe",
    "_store_postgres_aio_pool",
]