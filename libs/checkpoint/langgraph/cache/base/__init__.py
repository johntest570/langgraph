from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Generic, TypeVar

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

ValueT = TypeVar("ValueT")
Namespace = tuple[str, ...]
FullKey = tuple[Namespace, str]


class HITLApprovalDeniedError(RuntimeError):
    """Raised when a human operator denies approval for a risky operation."""
    pass


class BaseCache(ABC, Generic[ValueT]):
    """Base class for a cache."""

    serde: SerializerProtocol = JsonPlusSerializer(pickle_fallback=False)

    #: When True, `clear` and `aclear` require explicit human confirmation
    #: before executing the destructive delete operation.
    require_human_approval: bool = True

    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
        require_human_approval: bool = True,
    ) -> None:
        """Initialize the cache with a serializer.

        Args:
            serde: Optional serializer to use instead of the default.
            require_human_approval: If True (default), `clear` and `aclear`
                will prompt for human confirmation before deleting cached data.
        """
        self.serde = serde or self.serde
        self.require_human_approval = require_human_approval

    # ------------------------------------------------------------------
    # HITL helpers
    # ------------------------------------------------------------------

    def _request_human_approval(self, namespaces: Sequence[Namespace] | None) -> None:
        """Synchronous HITL gate for destructive cache operations.

        Prints a summary of the operation about to be performed and reads a
        confirmation from stdin.  Raises :class:`HITLApprovalDeniedError` if
        the operator does not confirm.

        Override this method in a subclass to integrate with an external
        approval workflow (e.g. a ticketing system or chat-ops bot).
        """
        if not self.require_human_approval:
            return
        scope = (
            f"namespaces={list(namespaces)}"
            if namespaces is not None
            else "ALL namespaces (full cache wipe)"
        )
        print(
            f"[HITL] A destructive cache-clear operation has been requested.\n"
            f"       Scope: {scope}\n"
            f"       This will permanently delete cached values."
        )
        answer = input("[HITL] Type 'yes' to approve or anything else to deny: ").strip().lower()
        if answer != "yes":
            raise HITLApprovalDeniedError(
                f"Human operator denied the cache-clear operation (scope={scope})."
            )

    async def _arequest_human_approval(
        self, namespaces: Sequence[Namespace] | None
    ) -> None:
        """Asynchronous HITL gate for destructive cache operations.

        The default implementation delegates to the synchronous helper via
        :func:`asyncio.to_thread` so that blocking stdin I/O does not stall
        the event loop.  Override in a subclass for a fully async approval
        workflow.
        """
        if not self.require_human_approval:
            return
        import asyncio
        await asyncio.to_thread(self._request_human_approval, namespaces)

    @abstractmethod
    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""

    @abstractmethod
    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""

    @abstractmethod
    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""

    @abstractmethod
    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.

        Requires human approval before executing when `require_human_approval`
        is True (the default).  Raises :class:`HITLApprovalDeniedError` if the
        operator denies the request.

        If no namespaces are provided, clear all cached values.
        """
        self._request_human_approval(namespaces)
        self._clear(namespaces)

    @abstractmethod
    def _clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Internal implementation of clear — called only after HITL approval."""

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.

        Requires human approval before executing when `require_human_approval`
        is True (the default).  Raises :class:`HITLApprovalDeniedError` if the
        operator denies the request.

        If no namespaces are provided, clear all cached values.
        """
        await self._arequest_human_approval(namespaces)
        await self._aclear(namespaces)

    @abstractmethod
    async def _aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Internal async implementation of aclear — called only after HITL approval."""
