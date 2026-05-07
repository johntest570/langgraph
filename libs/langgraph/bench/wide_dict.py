import operator
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from functools import partial
from random import choice
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph.state import StateGraph

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Allowed state field keys for output minimisation
ALLOWED_OUTPUT_KEYS = {
    "messages",
    "trigger_events",
    "primary_issue_medium",
    "autoresponse",
    "issue",
    "relevant_rules",
    "memory_docs",
    "categorizations",
    "responses",
    "user_info",
    "crm_info",
    "email_thread_id",
    "slack_participants",
    "bot_id",
    "notified_assignees",
}

# Maximum allowed size (in characters) for any string value in input
MAX_STRING_VALUE_LENGTH = 10000

# Maximum allowed depth for nested input structures
MAX_INPUT_DEPTH = 5

# Maximum allowed number of items in a list or dict in input
MAX_COLLECTION_SIZE = 100

# Patterns indicating potentially malicious prompt content
MALICIOUS_PATTERNS = [
    re.compile(r"(?i)(ignore\s+(previous|above|all)\s+instructions?)"),
    re.compile(r"(?i)(system\s*prompt|you\s+are\s+now|act\s+as\s+)"),
    re.compile(r"(?i)(base64\s*decode|eval\s*\(|exec\s*\()"),
    re.compile(r"(?i)(\bos\.system\b|\bsubprocess\b|\bshutil\b)"),
    re.compile(r"(?i)(rm\s+-rf|del\s+/|format\s+c:)"),
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),  # base64-like blobs
]

_SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_hex(32))
_SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))

_audit_log: list[dict] = []


def _audit(event: str, details: dict) -> None:
    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": details.get("trace_id", ""),
        "thread_id": details.get("thread_id", ""),
        "details": {k: v for k, v in details.items() if k not in ("trace_id", "thread_id")},
    }
    _audit_log.append(record)
    logger.info("AUDIT: %s", json.dumps(record, default=str))


def _generate_session_token(user_id: str) -> str:
    issued_at = int(time.time())
    session_id = str(uuid.uuid4())
    payload = f"{user_id}:{session_id}:{issued_at}"
    sig = hmac.new(
        _SESSION_SECRET.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}:{sig}"


def _verify_session_token(token: str) -> str:
    parts = token.split(":")
    if len(parts) != 4:
        raise ValueError("Invalid session token format.")
    user_id, session_id, issued_at_str, provided_sig = parts
    issued_at = int(issued_at_str)
    now = int(time.time())
    if now - issued_at > _SESSION_TTL:
        raise ValueError("Session token has expired.")
    payload = f"{user_id}:{session_id}:{issued_at_str}"
    expected_sig = hmac.new(
        _SESSION_SECRET.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_sig, provided_sig):
        raise ValueError("Session token signature is invalid.")
    return user_id


def _authenticate_user(user_id: str, credential: str) -> str:
    expected = os.environ.get("BENCH_USER_CREDENTIAL", "bench-secret")
    if not hmac.compare_digest(credential, expected):
        raise PermissionError("Authentication failed: invalid credentials.")
    token = _generate_session_token(user_id)
    _audit("user_authenticated", {"user_id": user_id, "thread_id": "", "trace_id": ""})
    return token


def _check_string_for_malicious_content(value: str) -> None:
    for pattern in MALICIOUS_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"Potentially malicious content detected in input string.")


def _validate_input_value(value, depth: int = 0) -> None:
    if depth > MAX_INPUT_DEPTH:
        raise ValueError("Input exceeds maximum allowed nesting depth.")
    if isinstance(value, str):
        if len(value) > MAX_STRING_VALUE_LENGTH:
            raise ValueError(f"Input string value exceeds maximum allowed length of {MAX_STRING_VALUE_LENGTH}.")
        _check_string_for_malicious_content(value)
    elif isinstance(value, dict):
        if len(value) > MAX_COLLECTION_SIZE:
            raise ValueError(f"Input dict exceeds maximum allowed size of {MAX_COLLECTION_SIZE}.")
        for k, v in value.items():
            _validate_input_value(k, depth + 1)
            _validate_input_value(v, depth + 1)
    elif isinstance(value, (list, tuple)):
        if len(value) > MAX_COLLECTION_SIZE:
            raise ValueError(f"Input collection exceeds maximum allowed size of {MAX_COLLECTION_SIZE}.")
        for item in value:
            _validate_input_value(item, depth + 1)


def _sanitize_and_validate_input(input_data: dict) -> dict:
    if not isinstance(input_data, dict):
        raise ValueError("Input must be a dictionary.")
    _validate_input_value(input_data)
    sanitized = {k: v for k, v in input_data.items() if k in ALLOWED_OUTPUT_KEYS}
    return sanitized


def _minimise_output(chunk: dict) -> dict:
    return {k: v for k, v in chunk.items() if k in ALLOWED_OUTPUT_KEYS}


def wide_dict(n: int) -> StateGraph:
    class State(TypedDict):
        messages: Annotated[list, operator.add]
        trigger_events: Annotated[list, operator.add]
        """The external events that are converted by the graph."""
        primary_issue_medium: Annotated[str, lambda x, y: y or x]
        autoresponse: Annotated[dict | None, lambda _, y: y]  # Always overwrite
        issue: Annotated[dict | None, lambda x, y: y if y else x]
        relevant_rules: list[dict] | None
        """SOPs fetched from the rulebook that are relevant to the current conversation."""
        memory_docs: list[dict] | None
        """Memory docs fetched from the memory service that are relevant to the current conversation."""
        categorizations: Annotated[list[dict], operator.add]
        """The issue categorizations auto-generated by the AI."""
        responses: Annotated[list[dict], operator.add]
        """The draft responses recommended by the AI."""

        user_info: Annotated[dict | None, lambda x, y: y if y is not None else x]
        """The current user state (by email)."""
        crm_info: Annotated[dict | None, lambda x, y: y if y is not None else x]
        """The CRM information for organization the current user is from."""
        email_thread_id: Annotated[str | None, lambda x, y: y if y is not None else x]
        """The current email thread ID."""
        slack_participants: Annotated[dict, operator.or_]
        """The growing list of current slack participants."""
        bot_id: str | None
        """The ID of the bot user in the slack channel."""
        notified_assignees: Annotated[dict, operator.or_]

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

    def read_write(read: str, write: Sequence[str], input: State) -> dict:
        val = input.get(read)
        val = {val: val} if isinstance(val, str) else val
        val_single = val[-1] if isinstance(val, list) else val
        val_list = val if isinstance(val, list) else [val]
        # Restrict output to only allowed keys and known field types
        result = {}
        for k in write:
            if k not in ALLOWED_OUTPUT_KEYS:
                continue
            if k in list_fields:
                result[k] = val_list
            elif k in dict_fields:
                result[k] = val_single
            else:
                result[k] = "".join(choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
        _audit(
            "node_executed",
            {
                "read_field": read,
                "write_fields": list(result.keys()),
                "trace_id": "",
                "thread_id": "",
            },
        )
        return result

    builder = StateGraph(State)
    builder.add_edge(START, "one")
    builder.add_node(
        "one",
        partial(read_write, "messages", ["trigger_events", "primary_issue_medium"]),
    )
    builder.add_edge("one", "two")
    builder.add_node(
        "two",
        partial(read_write, "trigger_events", ["autoresponse", "issue"]),
    )
    builder.add_edge("two", "three")
    builder.add_edge("two", "four")
    builder.add_node(
        "three",
        partial(read_write, "autoresponse", ["relevant_rules"]),
    )
    builder.add_node(
        "four",
        partial(
            read_write,
            "trigger_events",
            ["categorizations", "responses", "memory_docs"],
        ),
    )
    builder.add_node(
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
    )
    builder.add_edge(["three", "four"], "five")
    builder.add_edge("five", "six")
    builder.add_node(
        "six",
        partial(read_write, "responses", ["messages"]),
    )
    builder.add_conditional_edges(
        "six", lambda state: END if len(state["messages"]) > n else "one"
    )

    return builder


if __name__ == "__main__":
    import asyncio

    import uvloop
    from langgraph.checkpoint.memory import InMemorySaver

    # Authentication: verify user identity before accessing the agent
    user_id = os.environ.get("BENCH_USER_ID", "bench-user")
    credential = os.environ.get("BENCH_USER_CREDENTIAL", "bench-secret")
    session_token = _authenticate_user(user_id, credential)
    verified_user = _verify_session_token(session_token)

    trace_id = str(uuid.uuid4())
    thread_id = secrets.token_hex(16)
    session_sig = hmac.new(
        _SESSION_SECRET.encode(),
        f"{thread_id}:{verified_user}".encode(),
        hashlib.sha256,
    ).hexdigest()
    bound_thread_id = f"{thread_id}:{session_sig}"

    graph = wide_dict(10).compile(checkpointer=InMemorySaver())

    # Input is bounded and validated before being passed to the graph
    raw_input = {
        "messages": [
            {
                str(i): {
                    str(j): ["hi", True, 1]
                    for j in range(5)
                }
                for i in range(5)
            }
        ]
    }

    # Sanitize and validate input before passing to the AI graph
    validated_input = _sanitize_and_validate_input(raw_input)

    input_hash = hashlib.sha256(
        json.dumps(validated_input, sort_keys=True, default=str).encode()
    ).hexdigest()

    _audit(
        "graph_invocation_start",
        {
            "trace_id": trace_id,
            "thread_id": bound_thread_id,
            "user_id": verified_user,
            "input_hash": input_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    config = {
        "configurable": {"thread_id": bound_thread_id},
        "recursion_limit": 100,
    }

    async def run():
        async for c in graph.astream(validated_input, config=config):
            minimised = _minimise_output(c)
            _audit(
                "graph_stream_chunk",
                {
                    "trace_id": trace_id,
                    "thread_id": bound_thread_id,
                    "output_keys": list(minimised.keys()),
                },
            )
            print(list(minimised.keys()))

    uvloop.install()
    asyncio.run(run())