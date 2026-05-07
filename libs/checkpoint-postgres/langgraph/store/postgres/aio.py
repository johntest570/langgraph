from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, cast

import orjson
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline, Capabilities
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.postgres import _ainternal
from langgraph.store.postgres.base import (
    PLACEHOLDER,
    BasePostgresStore,
    PoolConfig,
    PostgresIndexConfig,
    Row,
    TTLConfig,
    _decode_ns_bytes,
    _ensure_index_config,
    _group_ops,
    _row_to_item,
    _row_to_search_item,
)

logger = logging.getLogger(__name__)

# Approved model registry: only models listed here are permitted for embedding workloads.
_APPROVED_EMBEDDING_MODELS: set[str] = {
    "openai:text-embedding-3-small",
    "openai:text-embedding-3-large",
    "openai:text-embedding-ada-002",
}


def _validate_embedding_model(index_config: PostgresIndexConfig) -> None:
    """Validate that the embedding model specified in index_config is in the approved registry."""
    embed = index_config.get("embed") if isinstance(index_config, dict) else getattr(index_config, "embed", None)
    model_id: str | None = None
    if isinstance(embed, str):
        model_id = embed
    elif embed is not None and hasattr(embed, "model"):
        model_id = getattr(embed, "model", None)
    elif embed is not None and hasattr(embed, "model_name"):
        model_id = getattr(embed, "model_name", None)

    if model_id is not None and model_id not in _APPROVED_EMBEDDING_MODELS:
        raise ValueError(
            f"Embedding model '{model_id}' is not in the approved model registry. "
            f"Approved models: {sorted(_APPROVED_EMBEDDING_MODELS)}"
        )
    if model_id is None and embed is not None:
        # embed is a callable or object without a detectable model id; log a warning
        logger.warning(
            "audit",
            extra={
                "event": "embedding_model_unverified",
                "message": "Cannot verify embedding model identity against approved registry; "
                           "ensure only approved models are used.",
            },
        )


def _make_trace_id() -> str:
    return str(uuid.uuid4())


def _hash_ops(ops: Iterable[Op]) -> str:
    ops_list = list(ops)
    try:
        serialized = orjson.dumps([repr(op) for op in ops_list])
    except Exception:
        serialized = repr(ops_list).encode()
    return hashlib.sha256(serialized).hexdigest()


def _log_audit_event(
    event: str,
    trace_id: str,
    *,
    op_types: list[str] | None = None,
    num_ops: int | None = None,
    input_hash: str | None = None,
    model_id: str | None = None,
    timestamp: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    record: dict[str, Any] = {
        "event": event,
        "trace_id": trace_id,
        "timestamp": timestamp or time.time(),
    }
    if op_types is not None:
        record["op_types"] = op_types
    if num_ops is not None:
        record["num_ops"] = num_ops
    if input_hash is not None:
        record["input_hash"] = input_hash
    if model_id is not None:
        record["model_id"] = model_id
    if extra:
        record.update(extra)
    logger.info("audit", extra=record)


class AsyncPostgresStore(AsyncBatchedBaseStore, BasePostgresStore[_ainternal.Conn]):
    """Asynchronous Postgres-backed store with optional vector search using pgvector.

    !!! example "Examples"
        Basic setup and usage:
        ```python
        from langgraph.store.postgres import AsyncPostgresStore

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        async with AsyncPostgresStore.from_conn_string(conn_string) as store:
            await store.setup()  # Run migrations. Done once

            # Store and retrieve data
            await store.aput(("users", "123"), "prefs", {"theme": "dark"})
            item = await store.aget(("users", "123"), "prefs")
        ```

        Vector search using LangChain embeddings:
        ```python
        from langchain.embeddings import init_embeddings
        from langgraph.store.postgres import AsyncPostgresStore

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        async with AsyncPostgresStore.from_conn_string(
            conn_string,
            index={
                "dims": 1536,
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "fields": ["text"]  # specify which fields to embed. Default is the whole serialized value
            }
        ) as store:
            await store.setup()  # Run migrations. Done once

            # Store documents
            await store.aput(("docs",), "doc1", {"text": "Python tutorial"})
            await store.aput(("docs",), "doc2", {"text": "TypeScript guide"})
            await store.aput(("docs",), "doc3", {"text": "Other guide"}, index=False)  # don't index

            # Search by similarity
            results = await store.asearch(("docs",), query="programming guides", limit=2)
        ```

        Using connection pooling for better performance:
        ```python
        from langgraph.store.postgres import AsyncPostgresStore, PoolConfig

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        async with AsyncPostgresStore.from_conn_string(
            conn_string,
            pool_config=PoolConfig(
                min_size=5,
                max_size=20
            )
        ) as store:
            await store.setup()  # Run migrations. Done once
            # Use store with connection pooling...
        ```

    Warning:
        Make sure to:
        1. Call `setup()` before first use to create necessary tables and indexes
        2. Have the pgvector extension available to use vector search
        3. Use Python 3.10+ for async functionality

    Note:
        Semantic search is disabled by default. You can enable it by providing an `index` configuration
        when creating the store. Without this configuration, all `index` arguments passed to
        `put` or `aput` will have no effect.

    Note:
        If you provide a TTL configuration, you must explicitly call `start_ttl_sweeper()` to begin
        the background task that removes expired items. Call `stop_ttl_sweeper()` to properly
        clean up resources when you're done with the store.
    """

    __slots__ = (
        "_deserializer",
        "pipe",
        "lock",
        "supports_pipeline",
        "index_config",
        "embeddings",
        "ttl_config",
        "_ttl_sweeper_task",
        "_ttl_stop_event",
    )
    supports_ttl: bool = True

    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        pipe: AsyncPipeline | None = None,
        deserializer: Callable[[bytes | orjson.Fragment], dict[str, Any]] | None = None,
        index: PostgresIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        if isinstance(conn, AsyncConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single AsyncConnection, not AsyncConnectionPool."
            )
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.supports_pipeline = Capabilities().has_pipeline()
        self.index_config = index
        if self.index_config:
            _validate_embedding_model(self.index_config)
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
            _model_id = getattr(self.embeddings, "model", None) or getattr(self.embeddings, "model_name", None)
            _log_audit_event(
                "embedding_model_initialized",
                _make_trace_id(),
                model_id=str(_model_id) if _model_id else None,
                extra={"store_class": self.__class__.__name__},
            )
        else:
            self.embeddings = None

        self.ttl_config = ttl
        self._ttl_sweeper_task: asyncio.Task[None] | None = None
        self._ttl_stop_event = asyncio.Event()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        ops_list = list(ops)
        trace_id = _make_trace_id()
        input_hash = _hash_ops(ops_list)
        op_types = list({type(op).__name__ for op in ops_list})
        timestamp = time.time()

        _log_audit_event(
            "abatch_start",
            trace_id,
            op_types=op_types,
            num_ops=len(ops_list),
            input_hash=input_hash,
            timestamp=timestamp,
        )

        grouped_ops, num_ops = _group_ops(ops_list)
        results: list[Result] = [None] * num_ops

        try:
            if self.pipe:
                async with self.pipe:
                    await self._execute_batch(grouped_ops, results)
            else:
                await self._execute_batch(grouped_ops, results)
        except Exception as exc:
            _log_audit_event(
                "abatch_error",
                trace_id,
                op_types=op_types,
                num_ops=num_ops,
                input_hash=input_hash,
                extra={"error": str(exc)},
            )
            raise

        _log_audit_event(
            "abatch_complete",
            trace_id,
            op_types=op_types,
            num_ops=num_ops,
            input_hash=input_hash,
            extra={"result_count": len(results)},
        )

        return results

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        pool_config: PoolConfig | None = None,
        index: PostgresIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> AsyncIterator[AsyncPostgresStore]:
        """Create a new AsyncPostgresStore instance from a connection string.

        Args:
            conn_string: The Postgres connection info string.
            pipeline: Whether to use AsyncPipeline (only for single connections)
            pool_config: Configuration for the connection pool.
                If provided, will create a connection pool and use it instead of a single connection.
                This overrides the `pipeline` argument.
            index: The embedding config.

        Returns:
            AsyncPostgresStore: A new AsyncPostgresStore instance.
        """
        if pool_config is not None:
            pc = pool_config.copy()
            async with cast(
                AsyncConnectionPool[AsyncConnection[DictRow]],
                AsyncConnectionPool(
                    conn_string,
                    min_size=pc.pop("min_size", 1),
                    max_size=pc.pop("max_size", None),
                    kwargs={
                        "autocommit": True,
                        "prepare_threshold": 0,
                        "row_factory": dict_row,
                        **(pc.pop("kwargs", None) or {}),
                    },
                    **cast(dict, pc),
                ),
            ) as pool:
                yield cls(conn=pool, index=index, ttl=ttl)
        else:
            async with await AsyncConnection.connect(
                conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
            ) as conn:
                if pipeline:
                    async with conn.pipeline() as pipe:
                        yield cls(conn=conn, pipe=pipe, index=index, ttl=ttl)
                else:
                    yield cls(conn=conn, index=index, ttl=ttl)

    async def setup(self) -> None:
        """Set up the store database asynchronously.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """

        async def _get_version(cur: AsyncCursor[DictRow], table: str) -> int:
            await cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    v INTEGER PRIMARY KEY
                )
            """
            )
            await cur.execute(f"SELECT v FROM {table} ORDER BY v DESC LIMIT 1")
            row = cast(dict, await cur.fetchone())
            if row is None:
                version = -1
            else:
                version = row["v"]
            return version

        async with self._cursor() as cur:
            version = await _get_version(cur, table="store_migrations")
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                await cur.execute(sql)
                await cur.execute("INSERT INTO store_migrations (v) VALUES (%s)", (v,))

            if self.index_config:
                version = await _get_version(cur, table="vector_migrations")
                for v, migration in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    sql = migration.sql
                    if migration.params:
                        params = {
                            k: v(self) if v is not None and callable(v) else v
                            for k, v in migration.params.items()
                        }
                        if "dims" in params:
                            try:
                                params["dims"] = int(params["dims"])
                            except Exception as e:
                                raise ValueError(
                                    f"Invalid dims for vector index: {params['dims']}"
                                ) from e
                        if "vector_type" in params:
                            vt = str(params["vector_type"])
                            if vt not in ("vector", "halfvec"):
                                raise ValueError(
                                    f"Invalid vector_type for pgvector: {vt}"
                                )
                            params["vector_type"] = vt
                        if "index_type" in params:
                            it = str(params["index_type"])
                            if it not in ("hnsw", "ivfflat"):
                                raise ValueError(
                                    f"Invalid index_type for pgvector: {it}"
                                )
                            params["index_type"] = it
                        sql = sql % params
                    await cur.execute(sql)
                    await cur.execute(
                        "INSERT INTO vector_migrations (v) VALUES (%s)", (v,)
                    )

    async def sweep_ttl(self) -> int:
        """Delete expired store items based on TTL.

        Returns:
            int: The number of deleted items.
        """
        async with self._cursor() as cur:
            await cur.execute(
                """
                DELETE FROM store
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                """
            )
            deleted_count = cur.rowcount
            return deleted_count

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> asyncio.Task[None]:
        """Periodically delete expired store items based on TTL.

        Returns:
            Task that can be awaited or cancelled.
        """
        if not self.ttl_config:
            return asyncio.create_task(asyncio.sleep(0))

        if self._ttl_sweeper_task is not None and not self._ttl_sweeper_task.done():
            return self._ttl_sweeper_task

        self._ttl_stop_event.clear()

        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")

        async def _sweep_loop() -> None:
            while not self._ttl_stop_event.is_set():
                try:
                    try:
                        await asyncio.wait_for(
                            self._ttl_stop_event.wait(),
                            timeout=interval * 60,
                        )
                        break
                    except asyncio.TimeoutError:
                        pass

                    expired_items = await self.sweep_ttl()
                    if expired_items > 0:
                        logger.info(f"Store swept {expired_items} expired items")
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.exception("Store TTL sweep iteration failed", exc_info=exc)

        task = asyncio.create_task(_sweep_loop())
        task.set_name("ttl_sweeper")
        self._ttl_sweeper_task = task
        return task

    async def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper task if it's running.

        Args:
            timeout: Maximum time to wait for the task to stop, in seconds.
                If `None`, wait indefinitely.

        Returns:
            bool: True if the task was successfully stopped or wasn't running,
                False if the timeout was reached before the task stopped.
        """
        if self._ttl_sweeper_task is None or self._ttl_sweeper_task.done():
            return True

        logger.info("Stopping TTL sweeper task")
        self._ttl_stop_event.set()

        if timeout is not None:
            try:
                await asyncio.wait_for(self._ttl_sweeper_task, timeout=timeout)
                success = True
            except asyncio.TimeoutError:
                success = False
        else:
            await self._ttl_sweeper_task
            success = True

        if success:
            self._ttl_sweeper_task = None
            logger.info("TTL sweeper task stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper task to stop")

        return success

    async def __aenter__(self) -> AsyncPostgresStore:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Ensure the TTL sweeper task is stopped when exiting the context
        if hasattr(self, "_ttl_sweeper_task") and self._ttl_sweeper_task is not None:
            # Set the event to signal the task to stop
            self._ttl_stop_event.set()
            # We don't wait for the task to complete here to avoid blocking
            # The task will clean up itself gracefully

    async def _execute_batch(
        self,
        grouped_ops: dict,
        results: list[Result],
        conn: AsyncConnection[DictRow] | None = None,
    ) -> None:
        # Keep `conn` for compatibility with subclasses overriding this private hook.
        # All database I/O goes through `_cursor()`, which owns connection acquisition.
        trace_id = _make_trace_id()
        _log_audit_event(
            "execute_batch_start",
            trace_id,
            op_types=[k.__name__ for k in grouped_ops],
            num_ops=sum(len(v) for v in grouped_ops.values()),
        )
        async with self._cursor(pipeline=True) as cur:
            if GetOp in grouped_ops:
                _log_audit_event(
                    "batch_get_ops",
                    trace_id,
                    num_ops=len(grouped_ops[GetOp]),
                )
                await self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]),
                    results,
                    cur,
                )

            if SearchOp in grouped_ops:
                _model_id = getattr(self.embeddings, "model", None) or getattr(self.embeddings, "model_name", None) if self.embeddings else None
                _log_audit_event(
                    "batch_search_ops",
                    trace_id,
                    num_ops=len(grouped_ops[SearchOp]),
                    model_id=str(_model_id) if _model_id else None,
                )
                await self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                _log_audit_event(
                    "batch_list_namespaces_ops",
                    trace_id,
                    num_ops=len(grouped_ops[ListNamespacesOp]),
                )
                await self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )

            if PutOp in grouped_ops:
                _log_audit_event(
                    "batch_put_ops",
                    trace_id,
                    num_ops=len(grouped_ops[PutOp]),
                )
                await self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]),
                    cur,
                )

        _log_audit_event(
            "execute_batch_complete",
            trace_id,
            op_types=[k.__name__ for k in grouped_ops],
        )

    async def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            await cur.execute(query, params)
            rows = cast(list[Row], await cur.fetchall())
            key_to_row = {row["key"]: row for row in rows}
            for idx, key in items:
                row = key_to_row.get(key)
                if row:
                    results[idx] = _row_to_item(
                        namespace, row, loader=self._deserializer
                    )
                else:
                    results[idx] = None

    async def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        cur: AsyncCursor[DictRow],
    ) -> None:
        queries, embedding_request = self._prepare_batch_PUT_queries(put_ops)
        if embedding_request:
            if self.embeddings is None:
                # Should not get here since the embedding config is required
                # to return an embedding_request above
                raise ValueError(
                    "Embedding configuration is required for vector operations "
                    f"(for semantic search). "
                    f"Please provide an EmbeddingConfig when initializing the {self.__class__.__name__}."
                )
            query, txt_params = embedding_request
            _model_id = getattr(self.embeddings, "model", None) or getattr(self.embeddings, "model_name", None)
            _log_audit_event(
                "embedding_invoked",
                _make_trace_id(),
                num_ops=len(txt_params),
                model_id=str(_model_id) if _model_id else None,
                extra={"operation": "put"},
            )
            vectors = await self.embeddings.aembed_documents(
                [param[-1] for param in txt_params]
            )
            queries.append(
                (
                    query,
                    [
                        p
                        for (ns, k, pathname, _), vector in zip(
                            txt_params, vectors, strict=False
                        )
                        for p in (ns, k, pathname, vector)
                    ],
                )
            )

        for query, params in queries:
            await cur.execute(query, params)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)

        if embedding_requests and self.embeddings:
            _model_id = getattr(self.embeddings, "model", None) or getattr(self.embeddings, "model_name", None)
            _log_audit_event(
                "embedding_invoked",
                _make_trace_id(),
                num_ops=len(embedding_requests),
                model_id=str(_model_id) if _model_id else None,
                extra={"operation": "search"},
            )
            vectors = await self.embeddings.aembed_documents(
                [query for _, query in embedding_requests]
            )
            for (idx, _), vector in zip(embedding_requests, vectors, strict=False):
                _paramslist = queries[idx][1]
                for i in range(len(_paramslist)):
                    if _paramslist[i] is PLACEHOLDER:
                        _paramslist[i] = vector

        for (idx, _), (query, params) in zip(search_ops, queries, strict=False):
            await cur.execute(query, params)
            rows = cast(list[Row], await cur.fetchall())
            items = [
                _row_to_search_item(
                    _decode_ns_bytes(row["prefix"]), row, loader=self._deserializer
                )
                for row in rows
            ]
            results[idx] = items

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        for (query, params), (idx, _) in zip(queries, list_ops, strict=False):
            await cur.execute(query, params)
            rows = cast(list[dict], await cur.fetchall())
            namespaces = [_decode_ns_bytes(row["truncated_prefix"]) for row in rows]
            results[idx] = namespaces

    @asynccontextmanager
    async def _cursor(
        self, *, pipeline: bool = False
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        """Create a database cursor as a context manager.

        Args:
            pipeline: whether to use pipeline for the DB operations inside the context manager.
                Will be applied regardless of whether the PostgresStore instance was initialized with a pipeline.
                If pipeline mode is not supported, will fall back to using transaction context manager.
        """
        is_pooled_conn = isinstance(self.conn, AsyncConnectionPool)
        # With AsyncConnectionPool, each _cursor() call checks out its own connection.
        # The pool does not hand out the same connection concurrently, so a shared lock
        # across calls is unnecessary here.
        lock = asyncio.Lock() if is_pooled_conn else self.lock
        async with _ainternal.get_connection(self.conn) as conn:
            if self.pipe:
                # a connection in pipeline mode can be used concurrently
                # in multiple threads/coroutines, but only one cursor can be
                # used at a time
                try:
                    async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                finally:
                    if pipeline:
                        await self.pipe.sync()
            elif pipeline:
                # a connection not in pipeline mode can only be used by one
                # thread/coroutine at a time, so we acquire a lock
                if self.supports_pipeline:
                    async with (
                        lock,
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    async with (
                        lock,
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                async with (
                    lock,
                    conn.cursor(binary=True, row_factory=dict_row) as cur,
                ):
                    yield cur