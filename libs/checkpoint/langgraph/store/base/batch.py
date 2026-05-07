"""Utilities for batching operations in a background task."""

from __future__ import annotations

import asyncio
import datetime
import functools
import hashlib
import json
import logging
import weakref
from collections.abc import Callable, Iterable
from typing import Any, Literal, TypeVar

from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    NamespacePath,
    NotProvided,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    _ensure_refresh,
    _ensure_ttl,
    _validate_namespace,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

_HITL_DELETE_APPROVER: Callable[[tuple[str, ...], str], bool] | None = None


def set_delete_approver(approver: Callable[[tuple[str, ...], str], bool] | None) -> None:
    """Set a Human-in-the-Loop approver callback for delete operations.

    The approver callable receives (namespace, key) and must return True to allow
    the delete, or False/raise to deny it.
    """
    global _HITL_DELETE_APPROVER
    _HITL_DELETE_APPROVER = approver


def _require_delete_approval(namespace: tuple[str, ...], key: str) -> None:
    """Enforce HITL approval for delete operations."""
    approver = _HITL_DELETE_APPROVER
    if approver is None:
        raise PermissionError(
            f"Delete operation on namespace={namespace!r}, key={key!r} requires "
            "Human-in-the-Loop approval. Register an approver via "
            "`set_delete_approver(callable)` before performing delete operations."
        )
    approved = approver(namespace, key)
    if not approved:
        _audit_log(
            action="delete_denied",
            namespace=namespace,
            key=key,
            details="HITL approver denied the delete operation.",
        )
        raise PermissionError(
            f"Delete operation on namespace={namespace!r}, key={key!r} was denied "
            "by the Human-in-the-Loop approver."
        )
    _audit_log(
        action="delete_approved",
        namespace=namespace,
        key=key,
        details="HITL approver approved the delete operation.",
    )


def _audit_log(
    action: str,
    namespace: tuple[str, ...] | None = None,
    key: str | None = None,
    ops: list[Op] | None = None,
    results: list[Any] | None = None,
    error: Exception | None = None,
    details: str | None = None,
) -> None:
    """Write a structured audit record to the logger."""
    record: dict[str, Any] = {
        "audit": True,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "action": action,
    }
    if namespace is not None:
        record["namespace"] = list(namespace)
    if key is not None:
        record["key"] = key
    if ops is not None:
        try:
            ops_repr = [repr(op) for op in ops]
            ops_hash = hashlib.sha256(
                json.dumps(ops_repr, sort_keys=True, default=str).encode()
            ).hexdigest()
            record["ops_count"] = len(ops)
            record["ops_hash"] = ops_hash
            record["ops_types"] = [type(op).__name__ for op in ops]
        except Exception:
            record["ops_repr_error"] = "failed to serialize ops"
    if results is not None:
        try:
            results_repr = [repr(r) for r in results]
            results_hash = hashlib.sha256(
                json.dumps(results_repr, sort_keys=True, default=str).encode()
            ).hexdigest()
            record["results_count"] = len(results)
            record["results_hash"] = results_hash
        except Exception:
            record["results_repr_error"] = "failed to serialize results"
    if error is not None:
        record["error"] = type(error).__name__
        record["error_message"] = str(error)
    if details is not None:
        record["details"] = details
    try:
        logger.info("AUDIT: %s", json.dumps(record, default=str))
    except Exception as log_exc:
        # Logging failure must not be silent — emit to stderr via fallback
        import sys
        print(f"AUDIT LOG FAILURE: {log_exc!r} | record={record!r}", file=sys.stderr)


def _check_loop(func: F) -> F:
    @functools.wraps(func)
    def wrapper(store: AsyncBatchedBaseStore, *args: Any, **kwargs: Any) -> Any:
        method_name: str = func.__name__
        try:
            current_loop = asyncio.get_running_loop()
            if current_loop is store._loop:
                replacement_str = (
                    f"Specifically, replace `store.{method_name}(...)` with `await store.a{method_name}(...)"
                    if method_name
                    else "For example, replace `store.get(...)` with `await store.aget(...)`"
                )
                raise asyncio.InvalidStateError(
                    f"Synchronous calls to {store.__class__.__name__} detected in the main event loop. "
                    "This can lead to deadlocks or performance issues. "
                    "Please use the asynchronous interface for main thread operations. "
                    f"{replacement_str} "
                )
        except RuntimeError:
            pass
        return func(store, *args, **kwargs)

    return wrapper


class AsyncBatchedBaseStore(BaseStore):
    """Efficiently batch operations in a background task."""

    __slots__ = ("_loop", "_aqueue", "_task")

    def __init__(self) -> None:
        super().__init__()
        self._loop = asyncio.get_running_loop()
        self._aqueue: asyncio.Queue[tuple[asyncio.Future, Op]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._ensure_task()

    def __del__(self) -> None:
        try:
            if self._task:
                self._task.cancel()
        except RuntimeError:
            pass

    def _ensure_task(self) -> None:
        """Ensure the background processing loop is running."""
        if self._task is None or self._task.done():
            self._task = self._loop.create_task(_run(self._aqueue, weakref.ref(self)))

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        self._ensure_task()
        fut = self._loop.create_future()
        self._aqueue.put_nowait(
            (
                fut,
                GetOp(
                    namespace,
                    key,
                    refresh_ttl=_ensure_refresh(self.ttl_config, refresh_ttl),
                ),
            )
        )
        return await fut

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        self._ensure_task()
        fut = self._loop.create_future()
        self._aqueue.put_nowait(
            (
                fut,
                SearchOp(
                    namespace_prefix,
                    filter,
                    limit,
                    offset,
                    query,
                    refresh_ttl=_ensure_refresh(self.ttl_config, refresh_ttl),
                ),
            )
        )
        return await fut

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        self._ensure_task()
        _validate_namespace(namespace)
        fut = self._loop.create_future()
        self._aqueue.put_nowait(
            (
                fut,
                PutOp(
                    namespace, key, value, index, ttl=_ensure_ttl(self.ttl_config, ttl)
                ),
            )
        )
        return await fut

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        _audit_log(
            action="delete_requested",
            namespace=namespace,
            key=key,
            details="HITL approval check initiated for adelete.",
        )
        _require_delete_approval(namespace, key)
        self._ensure_task()
        fut = self._loop.create_future()
        self._aqueue.put_nowait((fut, PutOp(namespace, key, None)))
        _audit_log(
            action="delete_enqueued",
            namespace=namespace,
            key=key,
            details="Delete operation enqueued after HITL approval.",
        )
        return await fut

    async def alist_namespaces(
        self,
        *,
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        self._ensure_task()
        fut = self._loop.create_future()
        match_conditions = []
        if prefix:
            match_conditions.append(MatchCondition(match_type="prefix", path=prefix))
        if suffix:
            match_conditions.append(MatchCondition(match_type="suffix", path=suffix))

        op = ListNamespacesOp(
            match_conditions=tuple(match_conditions),
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )
        self._aqueue.put_nowait((fut, op))
        return await fut

    @_check_loop
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return asyncio.run_coroutine_threadsafe(self.abatch(ops), self._loop).result()

    @_check_loop
    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return asyncio.run_coroutine_threadsafe(
            self.aget(namespace, key=key, refresh_ttl=refresh_ttl), self._loop
        ).result()

    @_check_loop
    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        return asyncio.run_coroutine_threadsafe(
            self.asearch(
                namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            ),
            self._loop,
        ).result()

    @_check_loop
    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        _validate_namespace(namespace)
        asyncio.run_coroutine_threadsafe(
            self.aput(
                namespace,
                key=key,
                value=value,
                index=index,
                ttl=_ensure_ttl(self.ttl_config, ttl),
            ),
            self._loop,
        ).result()

    @_check_loop
    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        _audit_log(
            action="delete_requested",
            namespace=namespace,
            key=key,
            details="HITL approval check initiated for delete.",
        )
        _require_delete_approval(namespace, key)
        asyncio.run_coroutine_threadsafe(
            self.adelete(namespace, key=key), self._loop
        ).result()

    @_check_loop
    def list_namespaces(
        self,
        *,
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return asyncio.run_coroutine_threadsafe(
            self.alist_namespaces(
                prefix=prefix,
                suffix=suffix,
                max_depth=max_depth,
                limit=limit,
                offset=offset,
            ),
            self._loop,
        ).result()


def _dedupe_ops(values: list[Op]) -> tuple[list[int] | None, list[Op]]:
    """Dedupe operations while preserving order for results.

    Args:
        values: List of operations to dedupe

    Returns:
        Tuple of (listen indices, deduped operations)
        where listen indices map deduped operation results back to original positions
    """
    if len(values) <= 1:
        return None, list(values)

    dedupped: list[Op] = []
    listen: list[int] = []
    puts: dict[tuple[tuple[str, ...], str], int] = {}

    for op in values:
        if isinstance(op, (GetOp, SearchOp, ListNamespacesOp)):
            try:
                listen.append(dedupped.index(op))
            except ValueError:
                listen.append(len(dedupped))
                dedupped.append(op)
        elif isinstance(op, PutOp):
            putkey = (op.namespace, op.key)
            if putkey in puts:
                # Overwrite previous put
                ix = puts[putkey]
                dedupped[ix] = op
                listen.append(ix)
            else:
                puts[putkey] = len(dedupped)
                listen.append(len(dedupped))
                dedupped.append(op)

        else:  # Any new ops will be treated regularly
            listen.append(len(dedupped))
            dedupped.append(op)

    return listen, dedupped


async def _run(
    aqueue: asyncio.Queue[tuple[asyncio.Future, Op]],
    store: weakref.ReferenceType[BaseStore],
) -> None:
    while item := await aqueue.get():
        # don't run batch if the future is done (e.g. cancelled)
        if item[0].done():
            continue
        # check if store is still alive
        if s := store():
            try:
                # accumulate operations scheduled in same tick
                items = [item]
                try:
                    while item := aqueue.get_nowait():
                        # don't insert if the future is done (e.g. cancelled)
                        if item[0].done():
                            continue
                        items.append(item)
                except asyncio.QueueEmpty:
                    pass
                # get the operations to run
                futs = [item[0] for item in items]
                values = [item[1] for item in items]
                # action each operation
                try:
                    listen, dedupped = _dedupe_ops(values)
                    _audit_log(
                        action="batch_execute_start",
                        ops=dedupped,
                        details=f"Executing batch of {len(dedupped)} deduped ops from {len(values)} total ops.",
                    )
                    results = await s.abatch(dedupped)
                    if listen is not None:
                        results = [results[ix] for ix in listen]

                    _audit_log(
                        action="batch_execute_success",
                        ops=dedupped,
                        results=results,
                        details=f"Batch execution succeeded with {len(results)} results.",
                    )

                    # set the results of each operation
                    for fut, result in zip(futs, results, strict=False):
                        # guard against future being done (e.g. cancelled)
                        if not fut.done():
                            fut.set_result(result)
                except Exception as e:
                    _audit_log(
                        action="batch_execute_error",
                        ops=dedupped,
                        error=e,
                        details="Batch execution raised an exception; propagating to futures.",
                    )
                    for fut in futs:
                        # guard against future being done (e.g. cancelled)
                        if not fut.done():
                            fut.set_exception(e)
            finally:
                # remove strong ref to store
                del s
        else:
            break