from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeAlias, runtime_checkable

from langgraph._internal._typing import EMPTY_SEQ
from langgraph.runtime import Runtime
from langgraph.types import CachePolicy, RetryPolicy, StreamWriter, TimeoutPolicy
from langgraph.typing import ContextT, NodeInputT, NodeInputT_contra

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

from typing import TypedDict


class RunnableConfig(TypedDict, total=False):
    tags: list[str]
    metadata: dict[str, Any]
    callbacks: Any
    run_name: str
    max_concurrency: NotRequired[int | None]
    recursion_limit: int
    configurable: dict[str, Any]
    run_id: Any


@runtime_checkable
class BaseStore(Protocol):
    def get(self, namespace: tuple[str, ...], key: str) -> Any: ...
    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None: ...
    def delete(self, namespace: tuple[str, ...], key: str) -> None: ...
    def search(self, namespace_prefix: tuple[str, ...], **kwargs: Any) -> list[Any]: ...


class Runnable(Protocol[NodeInputT_contra, Any]):
    def invoke(self, input: NodeInputT_contra, config: RunnableConfig | None = None) -> Any: ...


class _Node(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra) -> Any: ...


class _NodeWithConfig(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, config: RunnableConfig) -> Any: ...


class _NodeWithWriter(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, writer: StreamWriter) -> Any: ...


class _NodeWithStore(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, store: BaseStore) -> Any: ...


class _NodeWithWriterStore(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, writer: StreamWriter, store: BaseStore
    ) -> Any: ...


class _NodeWithConfigWriter(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, config: RunnableConfig, writer: StreamWriter
    ) -> Any: ...


class _NodeWithConfigStore(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, config: RunnableConfig, store: BaseStore
    ) -> Any: ...


class _NodeWithConfigWriterStore(Protocol[NodeInputT_contra]):
    def __call__(
        self,
        state: NodeInputT_contra,
        *,
        config: RunnableConfig,
        writer: StreamWriter,
        store: BaseStore,
    ) -> Any: ...


class _NodeWithRuntime(Protocol[NodeInputT_contra, ContextT]):
    def __call__(
        self, state: NodeInputT_contra, *, runtime: Runtime[ContextT]
    ) -> Any: ...


# TODO: we probably don't want to explicitly support the config / store signatures once
# we move to adding a context arg. Maybe what we do is we add support for kwargs with param spec
# this is purely for typing purposes though, so can easily change in the coming weeks.
StateNode: TypeAlias = (
    _Node[NodeInputT]
    | _NodeWithConfig[NodeInputT]
    | _NodeWithWriter[NodeInputT]
    | _NodeWithStore[NodeInputT]
    | _NodeWithWriterStore[NodeInputT]
    | _NodeWithConfigWriter[NodeInputT]
    | _NodeWithConfigStore[NodeInputT]
    | _NodeWithConfigWriterStore[NodeInputT]
    | _NodeWithRuntime[NodeInputT, ContextT]
    | Runnable[NodeInputT, Any]
)


@dataclass(slots=True)
class StateNodeSpec(Generic[NodeInputT, ContextT]):
    runnable: StateNode[NodeInputT, ContextT]
    metadata: dict[str, Any] | None
    input_schema: type[NodeInputT]
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    is_error_handler: bool = False
    error_handler_node: str | None = None
    ends: tuple[str, ...] | dict[str, str] | None = EMPTY_SEQ
    defer: bool = False
    timeout: TimeoutPolicy | None = None