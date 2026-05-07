from __future__ import annotations

import datetime
import logging
import threading
from collections.abc import Mapping, Sequence

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol

logger = logging.getLogger(__name__)


class InMemoryCache(BaseCache[ValueT]):
    def __init__(self, *, serde: SerializerProtocol | None = None):
        super().__init__(serde=serde)
        self._cache: dict[Namespace, dict[str, tuple[str, bytes, float | None]]] = {}
        self._lock = threading.RLock()
        self._audit_log: list[dict] = []

    def _record_audit(self, operation: str, detail: object) -> None:
        """Record an audit log entry for forensic readiness."""
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "operation": operation,
            "detail": detail,
        }
        self._audit_log.append(entry)
        logger.info("AUDIT: operation=%s detail=%s timestamp=%s", operation, detail, entry["timestamp"])

    def _request_hitl_approval(self, operation: str, detail: object) -> bool:
        """Request Human-in-the-Loop approval for risky operations.

        This method logs the pending destructive operation and prompts for
        human approval via stdin. Returns True if approved, False otherwise.
        """
        logger.warning(
            "HITL approval required for operation=%s detail=%s", operation, detail
        )
        try:
            response = input(
                f"[HITL] Approve destructive cache operation '{operation}' "
                f"for '{detail}'? (yes/no): "
            ).strip().lower()
        except (EOFError, OSError):
            logger.warning(
                "HITL: Could not obtain interactive approval for operation=%s; denying.",
                operation,
            )
            return False
        approved = response in ("yes", "y")
        self._record_audit(
            f"HITL_{'APPROVED' if approved else 'DENIED'}",
            {"operation": operation, "detail": detail},
        )
        return approved

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""
        with self._lock:
            if not keys:
                return {}
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            values: dict[FullKey, ValueT] = {}
            expired_keys: list[tuple[Namespace, str]] = []
            for ns_tuple, key in keys:
                ns = Namespace(ns_tuple)
                if ns in self._cache and key in self._cache[ns]:
                    enc, val, expiry = self._cache[ns][key]
                    if expiry is None or now < expiry:
                        values[(ns, key)] = self.serde.loads_typed((enc, val))
                        self._record_audit("GET_HIT", {"namespace": ns, "key": key})
                    else:
                        expired_keys.append((ns, key))
            for ns, key in expired_keys:
                self._record_audit("EXPIRE_DELETE", {"namespace": ns, "key": key})
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
                self._record_audit("SET", {"namespace": ns, "key": key, "ttl": ttl})

    async def aset(self, keys: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys."""
        self.set(keys)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values.
        Requires Human-in-the-Loop approval before executing."""
        with self._lock:
            if namespaces is None:
                if not self._request_hitl_approval("CLEAR_ALL", "all namespaces"):
                    logger.warning("HITL: CLEAR_ALL operation denied by human reviewer.")
                    return
                self._record_audit("CLEAR_ALL", "all namespaces")
                self._cache.clear()
            else:
                approved_ns = []
                for ns in namespaces:
                    if not self._request_hitl_approval("CLEAR_NAMESPACE", ns):
                        logger.warning(
                            "HITL: CLEAR_NAMESPACE operation denied for namespace=%s", ns
                        )
                        continue
                    approved_ns.append(ns)
                for ns in approved_ns:
                    if ns in self._cache:
                        self._record_audit("CLEAR_NAMESPACE", ns)
                        del self._cache[ns]

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values.
        Requires Human-in-the-Loop approval before executing."""
        self.clear(namespaces)