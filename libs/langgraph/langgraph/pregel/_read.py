from __future__ import annotations

import base64
import hashlib
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from datetime import timedelta
from functools import cached_property
from typing import (
    Any,
)

from langchain_core.runnables import Runnable, RunnableConfig

from langgraph._internal._config import merge_configs
from langgraph._internal._constants import CONF, CONFIG_KEY_READ
from langgraph._internal._runnable import RunnableCallable, RunnableSeq
from langgraph._internal._timeout import coerce_timeout_policy
from langgraph.pregel._utils import find_subgraph_pregel
from langgraph.pregel._write import ChannelWrite
from langgraph.pregel.protocol import PregelProtocol
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy

READ_TYPE = Callable[[str | Sequence[str], bool], Any | dict[str, Any]]
INPUT_CACHE_KEY_TYPE = tuple[Callable[..., Any], tuple[str, ...]]

_audit_logger = logging.getLogger("langgraph.audit")

_APPROVED_FRAMEWORKS: frozenset[str] = frozenset()

_MALICIOUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(ignore\s+(previous|prior|above)\s+instructions?)"),
    re.compile(r"(?i)(system\s*prompt|you\s+are\s+now|act\s+as\s+if)"),
    re.compile(r"(?i)(exec\s*\(|eval\s*\(|subprocess|os\.system|shell\s*=\s*True)"),
    re.compile(r"(?i)(\bsudo\b|\brm\s+-rf\b|\bchmod\b|\bchown\b)"),
    re.compile(r"(?i)(base64\.b64decode|__import__|importlib\.import_module)"),
    re.compile(r"(?i)(drop\s+table|delete\s+from|insert\s+into|union\s+select)"),
]

_BASE64_PATTERN = re.compile(
    r"(?:[A-Za-z0-9+/]{4}){10,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?"
)

_LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
}


def _decode_leet(text: str) -> str:
    return "".join(_LEET_MAP.get(c, c) for c in text)


def _check_base64_payload(text: str) -> bool:
    for match in _BASE64_PATTERN.finditer(text):
        try:
            decoded = base64.b64decode(match.group()).decode("utf-8", errors="ignore")
            for pattern in _MALICIOUS_PATTERNS:
                if pattern.search(decoded):
                    return True
        except Exception:
            pass
    return False


def _sanitize_and_validate_input(input: Any) -> Any:
    if input is None:
        return input
    text_to_check: str | None = None
    if isinstance(input, str):
        text_to_check = input
    elif isinstance(input, dict):
        text_to_check = str(input)
    elif isinstance(input, (list, tuple)):
        text_to_check = str(input)
    if text_to_check is not None:
        for pattern in _MALICIOUS_PATTERNS:
            if pattern.search(text_to_check):
                raise ValueError(
                    f"Input rejected by security policy: potentially malicious content detected."
                )
        leet_decoded = _decode_leet(text_to_check)
        for pattern in _MALICIOUS_PATTERNS:
            if pattern.search(leet_decoded):
                raise ValueError(
                    f"Input rejected by security policy: potentially malicious content detected (leet variant)."
                )
        if _check_base64_payload(text_to_check):
            raise ValueError(
                f"Input rejected by security policy: potentially malicious base64-encoded content detected."
            )
    return input


def _compute_input_hash(input: Any) -> str:
    try:
        return hashlib.sha256(str(input).encode("utf-8", errors="replace")).hexdigest()
    except Exception:
        return "unhashable"


def _emit_audit_record(
    *,
    operation: str,
    input_hash: str,
    node_bound: Any,
    config: RunnableConfig | None,
    trace_id: str,
    extra: dict[str, Any] | None = None,
) -> None:
    bound_name = getattr(node_bound, "name", None) or type(node_bound).__name__
    run_id = None
    if config:
        run_id = str(config.get("run_id", "")) or None
    record: dict[str, Any] = {
        "audit_event": "pregel_node_invocation",
        "operation": operation,
        "timestamp": time.time(),
        "trace_id": trace_id,
        "run_id": run_id,
        "node_bound": bound_name,
        "input_hash": input_hash,
    }
    if extra:
        record.update(extra)
    _audit_logger.info("AUDIT: %s", record)


class ChannelRead(RunnableCallable):
    """Implements the logic for reading state from CONFIG_KEY_READ.
    Usable both as a runnable as well as a static method to call imperatively."""

    channel: str | list[str]

    fresh: bool = False

    mapper: Callable[[Any], Any] | None = None

    def __init__(
        self,
        channel: str | list[str],
        *,
        fresh: bool = False,
        mapper: Callable[[Any], Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        super().__init__(
            func=self._read,
            afunc=self._aread,
            tags=tags,
            name=None,
            trace=False,
        )
        self.fresh = fresh
        self.mapper = mapper
        self.channel = channel

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        if name:
            pass
        elif isinstance(self.channel, str):
            name = f"ChannelRead<{self.channel}>"
        else:
            name = f"ChannelRead<{','.join(self.channel)}>"
        return super().get_name(suffix, name=name)

    def _read(self, _: Any, config: RunnableConfig) -> Any:
        result = self.do_read(
            config, select=self.channel, fresh=self.fresh, mapper=self.mapper
        )
        _audit_logger.info(
            "AUDIT: %s",
            {
                "audit_event": "channel_read",
                "operation": "_read",
                "timestamp": time.time(),
                "channel": self.channel,
                "fresh": self.fresh,
                "result_hash": _compute_input_hash(result),
            },
        )
        return result

    async def _aread(self, _: Any, config: RunnableConfig) -> Any:
        result = self.do_read(
            config, select=self.channel, fresh=self.fresh, mapper=self.mapper
        )
        _audit_logger.info(
            "AUDIT: %s",
            {
                "audit_event": "channel_read",
                "operation": "_aread",
                "timestamp": time.time(),
                "channel": self.channel,
                "fresh": self.fresh,
                "result_hash": _compute_input_hash(result),
            },
        )
        return result

    @staticmethod
    def do_read(
        config: RunnableConfig,
        *,
        select: str | list[str],
        fresh: bool = False,
        mapper: Callable[[Any], Any] | None = None,
    ) -> Any:
        try:
            read: READ_TYPE = config[CONF][CONFIG_KEY_READ]
        except KeyError:
            raise RuntimeError(
                "Not configured with a read function"
                "Make sure to call in the context of a Pregel process"
            )
        if mapper:
            result = mapper(read(select, fresh))
        else:
            result = read(select, fresh)
        _audit_logger.info(
            "AUDIT: %s",
            {
                "audit_event": "channel_read",
                "operation": "do_read",
                "timestamp": time.time(),
                "select": select,
                "fresh": fresh,
                "result_hash": _compute_input_hash(result),
            },
        )
        return result


DEFAULT_BOUND = RunnableCallable(lambda input: input)


class PregelNode:
    """A node in a Pregel graph. This won't be invoked as a runnable by the graph
    itself, but instead acts as a container for the components necessary to make
    a PregelExecutableTask for a node."""

    channels: str | list[str]
    """The channels that will be passed as input to `bound`.
    If a str, the node will be invoked with its value if it isn't empty.
    If a list, the node will be invoked with a dict of those channels' values."""

    triggers: list[str]
    """If any of these channels is written to, this node will be triggered in
    the next step."""

    mapper: Callable[[Any], Any] | None
    """A function to transform the input before passing it to `bound`."""

    writers: list[Runnable]
    """A list of writers that will be executed after `bound`, responsible for
    taking the output of `bound` and writing it to the appropriate channels."""

    bound: Runnable[Any, Any]
    """The main logic of the node. This will be invoked with the input from 
    `channels`."""

    retry_policy: Sequence[RetryPolicy] | None
    """The retry policies to use when invoking the node."""

    cache_policy: CachePolicy | None
    """The cache policy to use when invoking the node."""

    timeout: TimeoutPolicy | None
    """Timeout policy for a single invocation.

    If exceeded, `NodeTimeoutError` is raised and the retry policy (if any)
    decides whether to retry. Supported only for async nodes.
    """

    tags: Sequence[str] | None
    """Tags to attach to the node for tracing."""

    metadata: Mapping[str, Any] | None
    """Metadata to attach to the node for tracing."""

    is_error_handler: bool
    """Whether this node is registered as an error handler node."""

    error_handler_node: str | None
    """Optional handler node name for failures from this node."""

    subgraphs: Sequence[PregelProtocol]
    """Subgraphs used by the node."""

    def __init__(
        self,
        *,
        channels: str | list[str],
        triggers: Sequence[str],
        mapper: Callable[[Any], Any] | None = None,
        writers: list[Runnable] | None = None,
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        bound: Runnable[Any, Any] | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        cache_policy: CachePolicy | None = None,
        is_error_handler: bool = False,
        error_handler_node: str | None = None,
        subgraphs: Sequence[PregelProtocol] | None = None,
        timeout: float | timedelta | TimeoutPolicy | None = None,
    ) -> None:
        self.channels = channels
        self.triggers = list(triggers)
        self.mapper = mapper
        self.writers = writers or []
        self.bound = bound if bound is not None else DEFAULT_BOUND
        self.cache_policy = cache_policy
        if isinstance(retry_policy, RetryPolicy):
            self.retry_policy = (retry_policy,)
        else:
            self.retry_policy = retry_policy
        self.timeout = coerce_timeout_policy(timeout)
        self.tags = tags
        self.metadata = metadata
        self.is_error_handler = is_error_handler
        self.error_handler_node = error_handler_node
        if subgraphs is not None:
            self.subgraphs = subgraphs
        elif self.bound is not DEFAULT_BOUND:
            try:
                subgraph = find_subgraph_pregel(self.bound)
            except Exception:
                subgraph = None
            if subgraph:
                self.subgraphs = [subgraph]
            else:
                self.subgraphs = []
        else:
            self.subgraphs = []

    def copy(self, update: dict[str, Any]) -> PregelNode:
        attrs = {**self.__dict__, **update}
        # Drop the cached properties
        attrs.pop("flat_writers", None)
        attrs.pop("node", None)
        attrs.pop("input_cache_key", None)
        return PregelNode(**attrs)

    @cached_property
    def flat_writers(self) -> list[Runnable]:
        """Get writers with optimizations applied. Dedupes consecutive ChannelWrites."""
        writers = self.writers.copy()
        while (
            len(writers) > 1
            and isinstance(writers[-1], ChannelWrite)
            and isinstance(writers[-2], ChannelWrite)
        ):
            # we can combine writes if they are consecutive
            # careful to not modify the original writers list or ChannelWrite
            writers[-2] = ChannelWrite(
                writes=writers[-2].writes + writers[-1].writes,
            )
            writers.pop()
        return writers

    @cached_property
    def node(self) -> Runnable[Any, Any] | None:
        """Get a runnable that combines `bound` and `writers`."""
        writers = self.flat_writers
        if self.bound is DEFAULT_BOUND and not writers:
            return None
        elif self.bound is DEFAULT_BOUND and len(writers) == 1:
            return writers[0]
        elif self.bound is DEFAULT_BOUND:
            return RunnableSeq(*writers)
        elif writers:
            return RunnableSeq(self.bound, *writers)
        else:
            return self.bound

    @cached_property
    def input_cache_key(self) -> INPUT_CACHE_KEY_TYPE:
        """Get a cache key for the input to the node.
        This is used to avoid calculating the same input multiple times."""
        return (
            self.mapper,
            tuple(self.channels)
            if isinstance(self.channels, list)
            else (self.channels,),
        )

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Any:
        _sanitize_and_validate_input(input)
        trace_id = str(uuid.uuid4())
        input_hash = _compute_input_hash(input)
        _emit_audit_record(
            operation="invoke",
            input_hash=input_hash,
            node_bound=self.bound,
            config=config,
            trace_id=trace_id,
        )
        self_config: RunnableConfig = {"metadata": self.metadata, "tags": self.tags}
        result = self.bound.invoke(
            input,
            merge_configs(self_config, config),
            **kwargs,
        )
        _audit_logger.info(
            "AUDIT: %s",
            {
                "audit_event": "pregel_node_invocation_result",
                "operation": "invoke",
                "timestamp": time.time(),
                "trace_id": trace_id,
                "output_hash": _compute_input_hash(result),
            },
        )
        return result

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Any:
        _sanitize_and_validate_input(input)
        trace_id = str(uuid.uuid4())
        input_hash = _compute_input_hash(input)
        _emit_audit_record(
            operation="ainvoke",
            input_hash=input_hash,
            node_bound=self.bound,
            config=config,
            trace_id=trace_id,
        )
        self_config: RunnableConfig = {"metadata": self.metadata, "tags": self.tags}
        result = await self.bound.ainvoke(
            input,
            merge_configs(self_config, config),
            **kwargs,
        )
        _audit_logger.info(
            "AUDIT: %s",
            {
                "audit_event": "pregel_node_invocation_result",
                "operation": "ainvoke",
                "timestamp": time.time(),
                "trace_id": trace_id,
                "output_hash": _compute_input_hash(result),
            },
        )
        return result

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Any]:
        _sanitize_and_validate_input(input)
        trace_id = str(uuid.uuid4())
        input_hash = _compute_input_hash(input)
        _emit_audit_record(
            operation="stream",
            input_hash=input_hash,
            node_bound=self.bound,
            config=config,
            trace_id=trace_id,
        )
        self_config: RunnableConfig = {"metadata": self.metadata, "tags": self.tags}
        chunk_index = 0
        for item in self.bound.stream(
            input,
            merge_configs(self_config, config),
            **kwargs,
        ):
            _audit_logger.info(
                "AUDIT: %s",
                {
                    "audit_event": "pregel_node_stream_chunk",
                    "operation": "stream",
                    "timestamp": time.time(),
                    "trace_id": trace_id,
                    "chunk_index": chunk_index,
                    "chunk_hash": _compute_input_hash(item),
                },
            )
            chunk_index += 1
            yield item

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        _sanitize_and_validate_input(input)
        trace_id = str(uuid.uuid4())
        input_hash = _compute_input_hash(input)
        _emit_audit_record(
            operation="astream",
            input_hash=input_hash,
            node_bound=self.bound,
            config=config,
            trace_id=trace_id,
        )
        self_config: RunnableConfig = {"metadata": self.metadata, "tags": self.tags}
        chunk_index = 0
        async for item in self.bound.astream(
            input,
            merge_configs(self_config, config),
            **kwargs,
        ):
            _audit_logger.info(
                "AUDIT: %s",
                {
                    "audit_event": "pregel_node_stream_chunk",
                    "operation": "astream",
                    "timestamp": time.time(),
                    "trace_id": trace_id,
                    "chunk_index": chunk_index,
                    "chunk_hash": _compute_input_hash(item),
                },
            )
            chunk_index += 1
            yield item