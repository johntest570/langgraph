from __future__ import annotations

import datetime
import threading
from collections.abc import Mapping, Sequence

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol


class InMemoryCache(BaseCache[ValueT]):
    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
        hitl_approval_callback=None,
    ):
        """Initialize the cache.

        Args:
            serde: Optional serializer/deserializer protocol.
            hitl_approval_callback: Optional callable for Human-in-the-Loop approval
                of destructive operations (clear/delete). The callable receives a
                string describing the operation and must return True to approve or
                False/raise to deny. If None, all destructive operations are blocked
                by default unless explicitly approved.
        """
        super().__init__(serde=serde)
        self._cache: dict[Namespace, dict[str, tuple[str, bytes, float | None]]] = {}
        self._lock = threading.RLock()
        self._hitl_approval_callback = hitl_approval_callback

    def _require_hitl_approval(self, operation_description: str) -> None:
        """Enforce HITL approval for risky (destructive) operations.

        Raises:
            PermissionError: If no approval callback is configured or the callback
                returns a falsy value, indicating the operation was not approved.
        """
        if self._hitl_approval_callback is None:
            raise PermissionError(
                f"HITL approval required for destructive operation: "
                f"'{operation_description}'. "
                "Provide a `hitl_approval_callback` to InMemoryCache to allow "
                "delete/clear operations."
            )
        approved = self._hitl_approval_callback(operation_description)
        if not approved:
            raise PermissionError(
                f"HITL approval denied for destructive operation: "
                f"'{operation_description}'."
            )

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""
        with self._lock:
            if not keys:
                return {}
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            values: dict[FullKey, ValueT] = {}
            for ns_tuple, key in keys:
                ns = Namespace(ns_tuple)
                if ns in self._cache and key in self._cache[ns]:
                    enc, val, expiry = self._cache[ns][key]
                    if expiry is None or now < expiry:
                        values[(ns, key)] = self.serde.loads_typed((enc, val))
                    else:
                        del self._cache[ns][key]
            return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""
        return self.get(keys)

    def set(self, keys: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys."""
        with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc)
            for (ns, key), (value, ttl) in keys.items():
                if ttl is not None:
                    delta = datetime.timedelta(seconds=ttl)
                    expiry: float | None = (now + delta).timestamp()
                else:
                    expiry = None
                if ns not in self._cache:
                    self._cache[ns] = {}
                self._cache[ns][key] = (
                    *self.serde.dumps_typed(value),
                    expiry,
                )

    async def aset(self, keys: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys."""
        self.set(keys)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values.

        Requires HITL approval via the `hitl_approval_callback` provided at
        construction time. Raises PermissionError if approval is not granted.
        """
        if namespaces is None:
            self._require_hitl_approval(
                "clear: delete ALL cached values across all namespaces"
            )
        else:
            self._require_hitl_approval(
                f"clear: delete cached values for namespaces {list(namespaces)}"
            )
        with self._lock:
            if namespaces is None:
                self._cache.clear()
            else:
                for ns in namespaces:
                    if ns in self._cache:
                        del self._cache[ns]

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values.

        Requires HITL approval via the `hitl_approval_callback` provided at
        construction time. Raises PermissionError if approval is not granted.
        """
        # Approval is enforced inside self.clear(); delegating preserves HITL.
        self.clear(namespaces)
