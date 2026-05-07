import hashlib
import hmac
import json
import logging
import operator
import os
import re
import secrets
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from random import choice
from typing import Annotated

from langgraph.constants import END, START
from langgraph.graph.state import StateGraph

# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------
_audit_logger = logging.getLogger("wide_state.audit")
if not _audit_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    _audit_logger.addHandler(_handler)
_audit_logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Tool allow list
# ---------------------------------------------------------------------------
_ALLOWED_NODES = frozenset({"one", "two", "three", "four", "five", "six"})

# ---------------------------------------------------------------------------
# Session token helpers
# ---------------------------------------------------------------------------
_SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_hex(32))
_SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))


def _generate_session_token() -> str:
    """Return a signed, expiring session token."""
    thread_id = str(uuid.uuid4())
    issued_at = int(time.time())
    payload = f"{thread_id}:{issued_at}"
    sig = hmac.new(
        _SESSION_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload}:{sig}"


def _verify_session_token(token: str) -> str:
    """Verify signature and expiry; return thread_id on success."""
    parts = token.split(":")
    if len(parts) != 3:
        raise ValueError("Invalid session token format.")
    thread_id, issued_at_str, sig = parts
    issued_at = int(issued_at_str)
    payload = f"{thread_id}:{issued_at_str}"
    expected_sig = hmac.new(
        _SESSION_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected_sig, sig):
        raise ValueError("Session token signature verification failed.")
    if time.time() - issued_at > _SESSION_TTL:
        raise ValueError("Session token has expired.")
    return thread_id


# ---------------------------------------------------------------------------
# Authentication stub
# ---------------------------------------------------------------------------
def _authenticate() -> str:
    """
    Authenticate the caller before granting access to the AI agent.
    Returns a signed session token on success.
    Raises PermissionError if authentication fails.
    """
    api_key = os.environ.get("AGENT_API_KEY", "")
    if not api_key:
        raise PermissionError(
            "Authentication required: AGENT_API_KEY environment variable is not set."
        )
    expected = os.environ.get("EXPECTED_API_KEY", "")
    if not expected or not hmac.compare_digest(api_key, expected):
        raise PermissionError("Authentication failed: invalid API key.")
    token = _generate_session_token()
    _audit_logger.info("AUTH_SUCCESS token_issued")
    return token


# ---------------------------------------------------------------------------
# Input sanitisation / validation
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS = re.compile(
    r"(base64|eval\s*\(|exec\s*\(|__import__|subprocess|os\.system"
    r"|cmd\.exe|/bin/sh|/bin/bash|\bpowershell\b|\bwget\b|\bcurl\b"
    r"|ignore previous instructions|disregard.*instructions"
    r"|you are now|act as|pretend you)",
    re.IGNORECASE,
)

_MAX_STRING_LENGTH = 4096
_MAX_LIST_LENGTH = 100
_MAX_DICT_KEYS = 100
_MAX_NESTING_DEPTH = 5


def _sanitize_value(value, depth: int = 0):
    """Recursively validate and sanitise an input value."""
    if depth > _MAX_NESTING_DEPTH:
        raise ValueError(
            f"Input exceeds maximum nesting depth of {_MAX_NESTING_DEPTH}."
        )
    if isinstance(value, str):
        if len(value) > _MAX_STRING_LENGTH:
            raise ValueError(
                f"Input string exceeds maximum length of {_MAX_STRING_LENGTH}."
            )
        if _DANGEROUS_PATTERNS.search(value):
            raise ValueError(
                "Input contains potentially malicious content and was rejected."
            )
        return value
    if isinstance(value, list):
        if len(value) > _MAX_LIST_LENGTH:
            raise ValueError(
                f"Input list exceeds maximum length of {_MAX_LIST_LENGTH}."
            )
        return [_sanitize_value(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > _MAX_DICT_KEYS:
            raise ValueError(
                f"Input dict exceeds maximum key count of {_MAX_DICT_KEYS}."
            )
        return {
            _sanitize_value(k, depth + 1): _sanitize_value(v, depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (bool, int, float, type(None))):
        return value
    raise ValueError(f"Input contains unsupported type: {type(value).__name__}.")


def _validate_and_sanitize_input(input_data: dict) -> dict:
    """Validate and sanitise the full input dictionary."""
    if not isinstance(input_data, dict):
        raise ValueError("Input must be a dictionary.")
    return _sanitize_value(input_data)


# ---------------------------------------------------------------------------
# Node allow-list enforcement
# ---------------------------------------------------------------------------
def _check_node_allowed(node_name: str) -> None:
    if node_name not in _ALLOWED_NODES:
        _audit_logger.warning(
            "TOOL_DENIED node=%s reason=not_in_allowlist", node_name
        )
        raise PermissionError(
            f"Node '{node_name}' is not on the approved tool allow list."
        )
    _audit_logger.info("TOOL_ALLOWED node=%s", node_name)


# ---------------------------------------------------------------------------
# Output provenance watermarking
# ---------------------------------------------------------------------------
_GRAPH_VERSION = "wide_state/v1"


def _attach_provenance(output: dict, node_name: str, trace_id: str) -> dict:
    """Attach provenance metadata to a node's output dict."""
    output["__provenance__"] = {
        "generator": _GRAPH_VERSION,
        "node": node_name,
        "trace_id": trace_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "synthetic": True,
    }
    return output


# ---------------------------------------------------------------------------
# Output data minimisation helper
# ---------------------------------------------------------------------------
_MAX_OUTPUT_LIST_LENGTH = 10
_MAX_OUTPUT_DICT_KEYS = 20


def _minimise_output(value):
    """Truncate lists and dicts in output to enforce data minimisation."""
    if isinstance(value, list):
        return value[:_MAX_OUTPUT_LIST_LENGTH]
    if isinstance(value, dict):
        keys = list(value.keys())[:_MAX_OUTPUT_DICT_KEYS]
        return {k: value[k] for k in keys}
    return value


# ---------------------------------------------------------------------------
# Main graph factory
# ---------------------------------------------------------------------------

def wide_state(n: int) -> StateGraph:
    @dataclass(kw_only=True)
    class State:
        messages: Annotated[list, operator.add] = field(default_factory=list)
        trigger_events: Annotated[list, operator.add] = field(default_factory=list)
        """The external events that are converted by the graph."""
        primary_issue_medium: Annotated[str, lambda x, y: y or x] = field(
            default="email"
        )
        autoresponse: Annotated[dict | None, lambda _, y: y] = field(
            default=None
        )  # Always overwrite
        issue: Annotated[dict | None, lambda x, y: y if y else x] = field(default=None)
        relevant_rules: list[dict] | None = field(default=None)
        """SOPs fetched from the rulebook that are relevant to the current conversation."""
        memory_docs: list[dict] | None = field(default=None)
        """Memory docs fetched from the memory service that are relevant to the current conversation."""
        categorizations: Annotated[list[dict], operator.add] = field(
            default_factory=list
        )
        """The issue categorizations auto-generated by the AI."""
        responses: Annotated[list[dict], operator.add] = field(default_factory=list)
        """The draft responses recommended by the AI."""

        user_info: Annotated[dict | None, lambda x, y: y if y is not None else x] = (
            field(default=None)
        )
        """The current user state (by email)."""
        crm_info: Annotated[dict | None, lambda x, y: y if y is not None else x] = (
            field(default=None)
        )
        """The CRM information for organization the current user is from."""
        email_thread_id: Annotated[
            str | None, lambda x, y: y if y is not None else x
        ] = field(default=None)
        """The current email thread ID."""
        slack_participants: Annotated[dict, operator.or_] = field(default_factory=dict)
        """The growing list of current slack participants."""
        bot_id: str | None = field(default=None)
        """The ID of the bot user in the slack channel."""
        notified_assignees: Annotated[dict, operator.or_] = field(default_factory=dict)

    list_fields = {
        "messages",
        "trigger_events",
        "categorizations",
        "responses",
        "memory_docs",
        "relevant_rules",
    }
    dict_fields = {
        "user_info",
        "crm_info",
        "slack_participants",
        "notified_assignees",
        "autoresponse",
        "issue",
    }

    # Shared trace identifier for this graph invocation lifecycle
    _trace_id = str(uuid.uuid4())

    def read_write(read: str, write: Sequence[str], input: State) -> dict:
        node_name = f"read_write:{read}"
        _audit_logger.info(
            "NODE_EXEC trace_id=%s node=%s read_field=%s write_fields=%s",
            _trace_id,
            node_name,
            read,
            list(write),
        )
        val = getattr(input, read)
        val = {val: val} if isinstance(val, str) else val
        val_single = val[-1] if isinstance(val, list) else val
        val_list = val if isinstance(val, list) else [val]

        result = {
            k: _minimise_output(val_list)
            if k in list_fields
            else _minimise_output(val_single)
            if k in dict_fields
            else "".join(choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
            for k in write
        }

        _attach_provenance(result, node_name, _trace_id)
        _audit_logger.info(
            "NODE_OUTPUT trace_id=%s node=%s output_keys=%s",
            _trace_id,
            node_name,
            list(result.keys()),
        )
        return result

    def _make_allowed_node(node_name: str, fn):
        """Wrap a node function with allow-list enforcement."""
        def _wrapped(*args, **kwargs):
            _check_node_allowed(node_name)
            return fn(*args, **kwargs)
        _wrapped.__name__ = node_name
        return _wrapped

    builder = StateGraph(State)
    builder.add_edge(START, "one")
    builder.add_node(
        "one",
        _make_allowed_node(
            "one",
            partial(read_write, "messages", ["trigger_events", "primary_issue_medium"]),
        ),
    )
    builder.add_edge("one", "two")
    builder.add_node(
        "two",
        _make_allowed_node(
            "two",
            partial(read_write, "trigger_events", ["autoresponse", "issue"]),
        ),
    )
    builder.add_edge("two", "three")
    builder.add_edge("two", "four")
    builder.add_node(
        "three",
        _make_allowed_node(
            "three",
            partial(read_write, "autoresponse", ["relevant_rules"]),
        ),
    )
    builder.add_node(
        "four",
        _make_allowed_node(
            "four",
            partial(
                read_write,
                "trigger_events",
                ["categorizations", "responses", "memory_docs"],
            ),
        ),
    )
    builder.add_node(
        "five",
        _make_allowed_node(
            "five",
            partial(
                read_write,
                "categorizations",
                [
                    "user_info",
                    "crm_info",
                    "email_thread_id",
                    "slack_participants",
                    "bot_id",
                    "notified_assignees",
                ],
            ),
        ),
    )
    builder.add_edge(["three", "four"], "five")
    builder.add_edge("five", "six")
    builder.add_node(
        "six",
        _make_allowed_node(
            "six",
            partial(read_write, "responses", ["messages"]),
        ),
    )
    builder.add_conditional_edges(
        "six", lambda state: END if len(state.messages) > n else "one"
    )

    return builder


if __name__ == "__main__":
    import asyncio

    import uvloop
    from langgraph.checkpoint.memory import InMemorySaver

    # --- Authentication ---
    try:
        session_token = _authenticate()
    except PermissionError as _auth_err:
        _audit_logger.error("AUTH_FAILURE reason=%s", _auth_err)
        raise SystemExit(1) from _auth_err

    # --- Session token verification ---
    try:
        _verified_thread_id = _verify_session_token(session_token)
    except ValueError as _tok_err:
        _audit_logger.error("SESSION_INVALID reason=%s", _tok_err)
        raise SystemExit(1) from _tok_err

    graph = wide_state(1000).compile(checkpointer=InMemorySaver())

    # --- Input sanitisation ---
    _raw_input = {
        "messages": [
            {
                str(i) * 10: {
                    str(j) * 10: ["hi?" * 10, True, 1, 6327816386138, None] * 5
                    for j in range(50)
                }
                for i in range(50)
            }
        ]
    }

    try:
        input = _validate_and_sanitize_input(_raw_input)
    except ValueError as _san_err:
        _audit_logger.error("INPUT_REJECTED reason=%s", _san_err)
        raise SystemExit(1) from _san_err

    config = {
        "configurable": {"thread_id": _verified_thread_id},
        "recursion_limit": 20000000000,
    }

    _audit_logger.info(
        "GRAPH_START thread_id=%s graph_version=%s", _verified_thread_id, _GRAPH_VERSION
    )

    async def run():
        async for c in graph.astream(input, config=config):
            # Output data minimisation: only log keys, not full content
            _audit_logger.info(
                "GRAPH_CHUNK thread_id=%s keys=%s",
                _verified_thread_id,
                list(c.keys()),
            )
            print(c.keys())

    uvloop.install()
    asyncio.run(run())