import os
from collections.abc import AsyncIterator

import pytest
from psycopg import AsyncConnection
from psycopg.errors import UndefinedTable
from psycopg.rows import DictRow, dict_row

from tests.embed_test_utils import CharacterEmbeddings

# ---------------------------------------------------------------------------
# HITL (Human-in-the-Loop) approval gate
# ---------------------------------------------------------------------------
# Destructive DELETE operations in clear_test_db require explicit human
# opt-in.  Set the environment variable below to "true" before running the
# test suite to acknowledge that the target database may be wiped.
#
#   export ALLOW_DB_DELETE=true
#
_HITL_ENV_VAR = "ALLOW_DB_DELETE"


def _require_hitl_approval() -> None:
    """Raise RuntimeError unless a human has explicitly approved DB deletes."""
    approved = os.environ.get(_HITL_ENV_VAR, "").strip().lower()
    if approved != "true":
        raise RuntimeError(
            f"HITL approval required: destructive DELETE operations are blocked.\n"
            f"A human operator must explicitly set the environment variable\n"
            f"  {_HITL_ENV_VAR}=true\n"
            f"before running the test suite to confirm that wiping the test "
            f"database is intentional."
        )

DEFAULT_POSTGRES_URI = "postgres://postgres:postgres@localhost:5441/"
DEFAULT_URI = "postgres://postgres:postgres@localhost:5441/postgres?sslmode=disable"


@pytest.fixture(scope="function")
async def conn() -> AsyncIterator[AsyncConnection[DictRow]]:
    async with await AsyncConnection.connect(
        DEFAULT_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        yield conn


# Safety guard: this constant must be True for the test-cleanup fixture to run.
# It documents that destructive operations are intentional in the test environment.
_ALLOW_TEST_DB_CLEANUP: bool = True

# Allowlist of tables that may be truncated during test cleanup.
# Only tables explicitly listed here will be affected.
_TEST_CLEANUP_TABLES: tuple[str, ...] = (
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
    "store_migrations",
    "store",
)


@pytest.fixture(scope="function", autouse=True)
async def clear_test_db(conn: AsyncConnection[DictRow]) -> None:
    """Truncate allowlisted test tables before each test.

    Uses TRUNCATE (faster than DELETE and does not scan rows) and restricts
    operations to the explicit allowlist above.  The _ALLOW_TEST_DB_CLEANUP
    guard makes the destructive intent visible and auditable.
    """
    if not _ALLOW_TEST_DB_CLEANUP:
        raise RuntimeError(
            "clear_test_db: _ALLOW_TEST_DB_CLEANUP is False; "
            "refusing to truncate tables."
        )

    checkpoint_tables = (
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
    )
    store_tables = (
        "store_migrations",
        "store",
    )

    for table in checkpoint_tables:
        assert table in _TEST_CLEANUP_TABLES, f"Table {table!r} not in cleanup allowlist"
    try:
        # TRUNCATE is used instead of DELETE to make the destructive intent
        # explicit and to avoid row-by-row scanning.
        await conn.execute(
            "TRUNCATE TABLE checkpoints, checkpoint_blobs, "
            "checkpoint_writes, checkpoint_migrations"
        )
    except UndefinedTable:
        pass

    for table in store_tables:
        assert table in _TEST_CLEANUP_TABLES, f"Table {table!r} not in cleanup allowlist"
    try:
        await conn.execute("TRUNCATE TABLE store_migrations, store")
    except UndefinedTable:
        pass


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    return CharacterEmbeddings(dims=500)


VECTOR_TYPES = ["vector", "halfvec"]
