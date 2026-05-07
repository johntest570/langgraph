from __future__ import annotations

import logging
import datetime
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol

logger = logging.getLogger(__name__)


def _audit_log(operation: str, details: dict) -> None:
    """Emit a structured audit log entry for cache operations."""
    entry = {
        "audit": True,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "operation": operation,
        **details,
    }
    logger.info("AUDIT: %s", entry)


def _request_hitl_approval(operation: str, keys: list) -> bool:
    """Request Human-in-the-Loop approval for risky operations.

    In a production environment, this should integrate with an external
    approval workflow (e.g., PagerDuty, Slack, ticketing system).
    This implementation logs the request and raises an error to block
    automatic execution, requiring explicit human override.

    Returns True if approved, raises RuntimeError if not approved.
    """
    _audit_log(
        "HITL_APPROVAL_REQUIRED",
        {
            "risky_operation": operation,
            "keys_affected": [str(k) for k in keys],
            "keys_count": len(keys),
            "status": "PENDING_HUMAN_APPROVAL",
        },
    )
    raise RuntimeError(
        f"HITL approval required for risky operation '{operation}' affecting "
        f"{len(keys)} key(s). A human operator must explicitly approve this "
        f"delete/purge operation. Audit record has been logged."
    )


class RedisCache(BaseCache[ValueT]):
    """Redis-based cache implementation with TTL support."""

    def __init__(
        self,
        redis: Any,
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "langgraph:cache:",
        hitl_approver: Any | None = None,
    ) -> None:
        """Initialize the cache with a Redis client.

        Args:
            redis: Redis client instance (sync or async)
            serde: Serializer to use for values
            prefix: Key prefix for all cached values
            hitl_approver: Optional callable(operation, keys) -> bool for HITL approval.
                           If provided, called instead of the default blocking behavior.
                           Must return True to allow the operation to proceed.
        """
        super().__init__(serde=serde)
        self.redis = redis
        self.prefix = prefix
        self.hitl_approver = hitl_approver

    def _make_key(self, ns: Namespace, key: str) -> str:
        """Create a Redis key from namespace and key."""
        ns_str = ":".join(ns) if ns else ""
        return f"{self.prefix}{ns_str}:{key}" if ns_str else f"{self.prefix}{key}"

    def _parse_key(self, redis_key: str) -> tuple[Namespace, str]:
        """Parse a Redis key back to namespace and key."""
        if not redis_key.startswith(self.prefix):
            raise ValueError(
                f"Key {redis_key} does not start with prefix {self.prefix}"
            )

        remaining = redis_key[len(self.prefix) :]
        if ":" in remaining:
            parts = remaining.split(":")
            key = parts[-1]
            ns_parts = parts[:-1]
            return (tuple(ns_parts), key)
        else:
            return (tuple(), remaining)

    def _check_hitl_approval(self, operation: str, keys: list) -> bool:
        """Check HITL approval for a risky operation.

        If a custom hitl_approver is configured, delegates to it.
        Otherwise, uses the default blocking behavior.
        """
        if self.hitl_approver is not None:
            approved = self.hitl_approver(operation, keys)
            _audit_log(
                "HITL_APPROVAL_RESULT",
                {
                    "risky_operation": operation,
                    "keys_count": len(keys),
                    "approved": approved,
                },
            )
            return approved
        return _request_hitl_approval(operation, keys)

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""
        if not keys:
            return {}

        _audit_log(
            "CACHE_GET",
            {
                "keys_count": len(keys),
                "keys": [str(k) for k in keys],
            },
        )

        # Build Redis keys
        redis_keys = [self._make_key(ns, key) for ns, key in keys]

        # Get values from Redis using MGET
        try:
            raw_values = self.redis.mget(redis_keys)
        except Exception as exc:
            logger.error(
                "AUDIT: Redis unavailable during CACHE_GET operation. "
                "keys_count=%d error=%r",
                len(keys),
                exc,
            )
            _audit_log(
                "CACHE_GET_FAILURE",
                {
                    "keys_count": len(keys),
                    "error": repr(exc),
                    "result": "empty_dict_returned",
                },
            )
            return {}

        values: dict[FullKey, ValueT] = {}
        for i, raw_value in enumerate(raw_values):
            if raw_value is not None:
                try:
                    # Deserialize the value
                    encoding, data = raw_value.split(b":", 1)
                    values[keys[i]] = self.serde.loads_typed((encoding.decode(), data))
                except Exception as exc:
                    logger.error(
                        "AUDIT: Failed to deserialize cache entry for key=%r error=%r",
                        keys[i],
                        exc,
                    )
                    _audit_log(
                        "CACHE_GET_DESERIALIZE_ERROR",
                        {
                            "key": str(keys[i]),
                            "error": repr(exc),
                        },
                    )
                    continue

        _audit_log(
            "CACHE_GET_RESULT",
            {
                "keys_requested": len(keys),
                "keys_found": len(values),
            },
        )
        return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""
        return self.get(keys)

    def set(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""
        if not mapping:
            return

        _audit_log(
            "CACHE_SET",
            {
                "keys_count": len(mapping),
                "keys": [str(k) for k in mapping.keys()],
            },
        )

        # Use pipeline for efficient batch operations
        pipe = self.redis.pipeline()

        for (ns, key), (value, ttl) in mapping.items():
            redis_key = self._make_key(ns, key)
            encoding, data = self.serde.dumps_typed(value)

            # Store as "encoding:data" format
            serialized_value = f"{encoding}:".encode() + data

            if ttl is not None:
                pipe.setex(redis_key, ttl, serialized_value)
            else:
                pipe.set(redis_key, serialized_value)

        try:
            pipe.execute()
            _audit_log(
                "CACHE_SET_SUCCESS",
                {
                    "keys_count": len(mapping),
                },
            )
        except Exception as exc:
            logger.error(
                "AUDIT: Redis pipeline execution failed during CACHE_SET. "
                "keys_count=%d error=%r",
                len(mapping),
                exc,
            )
            _audit_log(
                "CACHE_SET_FAILURE",
                {
                    "keys_count": len(mapping),
                    "error": repr(exc),
                    "alert": "PIPELINE_EXECUTION_FAILED",
                },
            )

    async def aset(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""
        self.set(mapping)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values.

        This is a risky delete operation and requires HITL approval.
        """
        try:
            if namespaces is None:
                # Clear all keys with our prefix
                pattern = f"{self.prefix}*"
                keys = self.redis.keys(pattern)
                if keys:
                    _audit_log(
                        "CACHE_CLEAR_ALL_REQUESTED",
                        {
                            "pattern": pattern,
                            "keys_count": len(keys),
                            "keys": [k.decode() if isinstance(k, bytes) else k for k in keys],
                        },
                    )
                    approved = self._check_hitl_approval("clear_all", keys)
                    if approved:
                        self.redis.delete(*keys)
                        _audit_log(
                            "CACHE_CLEAR_ALL_EXECUTED",
                            {
                                "pattern": pattern,
                                "keys_deleted": len(keys),
                                "status": "SUCCESS",
                            },
                        )
            else:
                # Clear specific namespaces
                keys_to_delete = []
                for ns in namespaces:
                    ns_str = ":".join(ns) if ns else ""
                    pattern = (
                        f"{self.prefix}{ns_str}:*" if ns_str else f"{self.prefix}*"
                    )
                    keys = self.redis.keys(pattern)
                    keys_to_delete.extend(keys)

                if keys_to_delete:
                    _audit_log(
                        "CACHE_CLEAR_NAMESPACES_REQUESTED",
                        {
                            "namespaces": [list(ns) for ns in namespaces],
                            "keys_count": len(keys_to_delete),
                            "keys": [
                                k.decode() if isinstance(k, bytes) else k
                                for k in keys_to_delete
                            ],
                        },
                    )
                    approved = self._check_hitl_approval(
                        "clear_namespaces", keys_to_delete
                    )
                    if approved:
                        self.redis.delete(*keys_to_delete)
                        _audit_log(
                            "CACHE_CLEAR_NAMESPACES_EXECUTED",
                            {
                                "namespaces": [list(ns) for ns in namespaces],
                                "keys_deleted": len(keys_to_delete),
                                "status": "SUCCESS",
                            },
                        )
        except RuntimeError:
            # Re-raise HITL approval errors — do not swallow them
            raise
        except Exception as exc:
            logger.error(
                "AUDIT: Redis unavailable or error during CACHE_CLEAR operation. "
                "namespaces=%r error=%r",
                namespaces,
                exc,
            )
            _audit_log(
                "CACHE_CLEAR_FAILURE",
                {
                    "namespaces": [list(ns) for ns in namespaces] if namespaces else None,
                    "error": repr(exc),
                    "alert": "CLEAR_OPERATION_FAILED",
                },
            )

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        self.clear(namespaces)