import asyncio
import logging
import os
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph, add_messages

logger = logging.getLogger(__name__)

# check that env var is present
os.environ["SOME_ENV_VAR"]

# Approved model registry
APPROVED_MODEL_REGISTRY = {
    "gpt-4": {"provider": "openai", "version": "gpt-4"},
    "gpt-4o": {"provider": "openai", "version": "gpt-4o"},
    "gpt-3.5-turbo": {"provider": "openai", "version": "gpt-3.5-turbo"},
}

APPROVED_MODEL_ID = "gpt-4o"
APPROVED_MODEL_VERSION = APPROVED_MODEL_REGISTRY[APPROVED_MODEL_ID]["version"]

# Approved tool allow list
APPROVED_TOOLS = {"tool"}

# Dangerous patterns for input/output sanitization
DANGEROUS_PATTERNS = [
    re.compile(r"\beval\s*\(", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\bsubprocess\s*\.", re.IGNORECASE),
    re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
    re.compile(r"\bos\.popen\s*\(", re.IGNORECASE),
    re.compile(r"\b__import__\s*\(", re.IGNORECASE),
    re.compile(r"\bcompile\s*\(", re.IGNORECASE),
    re.compile(r"\bexecfile\s*\(", re.IGNORECASE),
    re.compile(r"\bimportlib\b", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),  # base64-encoded content
]

MALICIOUS_COMMAND_PATTERNS = [
    re.compile(r";\s*(rm|wget|curl|bash|sh|python|perl|ruby)\b", re.IGNORECASE),
    re.compile(r"\|\s*(bash|sh|python|perl|ruby)\b", re.IGNORECASE),
    re.compile(r"`[^`]+`"),
    re.compile(r"\$\([^)]+\)"),
]

AI_CONTENT_WATERMARK = "[AI-GENERATED]"


def _require_authenticated_user(config):
    """Enforce that a user is authenticated before accessing the agent."""
    configurable = config.get("configurable", {}) if config else {}
    user_id = configurable.get("user_id") or configurable.get("authenticated_user")
    if not user_id:
        raise PermissionError(
            "Authentication required: no authenticated user found in config. "
            "Provide 'user_id' in config['configurable'] before invoking the agent."
        )
    return user_id


def _validate_and_sanitize_messages(messages):
    """Validate and sanitize input messages before sending to the model."""
    if not messages:
        raise ValueError("Messages list must not be empty.")
    sanitized = []
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(content, str):
            for pattern in DANGEROUS_PATTERNS:
                if pattern.search(content):
                    raise ValueError(
                        f"Input message contains potentially dangerous content matching pattern: {pattern.pattern}"
                    )
            for pattern in MALICIOUS_COMMAND_PATTERNS:
                if pattern.search(content):
                    raise ValueError(
                        f"Input message contains potentially malicious command pattern: {pattern.pattern}"
                    )
        sanitized.append(msg)
    return sanitized


def _validate_and_sanitize_llm_output(response):
    """Validate and sanitize LLM output, checking for dynamic code execution primitives."""
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, str):
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(content):
                raise ValueError(
                    f"LLM output contains potentially dangerous content matching pattern: {pattern.pattern}"
                )
        for pattern in MALICIOUS_COMMAND_PATTERNS:
            if pattern.search(content):
                raise ValueError(
                    f"LLM output contains potentially malicious command pattern: {pattern.pattern}"
                )
    return response


def _check_tool_allow_list(tool_name):
    """Enforce that only approved tools are invoked."""
    if tool_name not in APPROVED_TOOLS:
        logger.warning(
            "AUDIT: Tool invocation DENIED | tool=%s | allowed=%s | timestamp=%s",
            tool_name,
            APPROVED_TOOLS,
            datetime.now(timezone.utc).isoformat(),
        )
        raise PermissionError(
            f"Tool '{tool_name}' is not in the approved tool allow list: {APPROVED_TOOLS}"
        )


def _verify_model_registry(model_id):
    """Verify the model is in the approved registry."""
    if model_id not in APPROVED_MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_id}' is not in the approved model registry. "
            f"Approved models: {list(APPROVED_MODEL_REGISTRY.keys())}"
        )
    return APPROVED_MODEL_REGISTRY[model_id]


class AgentState(TypedDict):
    some_bytes: bytes
    some_byte_array: bytearray
    dict_with_bytes: dict[str, bytes]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sleep: int


async def call_model(state, config):
    # Enforce authentication
    user_id = _require_authenticated_user(config)

    trace_id = str(uuid.uuid4())

    if sleep := state.get("sleep"):
        await asyncio.sleep(sleep)

    messages = state["messages"]

    if len(messages) > 1:
        assert state["some_bytes"] == b"some_bytes"
        assert state["some_byte_array"] == bytearray(b"some_byte_array")
        assert state["dict_with_bytes"] == {"more_bytes": b"more_bytes"}

    # hacky way to reset model to the "first" response
    if isinstance(messages[-1], HumanMessage):
        model.i = 0

    # Validate and sanitize input messages
    sanitized_messages = _validate_and_sanitize_messages(messages)

    # Verify model is in approved registry
    _verify_model_registry(APPROVED_MODEL_ID)

    # Log the interaction with the LLM (input)
    logger.info(
        "AUDIT: LLM invocation START | trace_id=%s | user_id=%s | model=%s | model_version=%s | "
        "message_count=%d | timestamp=%s",
        trace_id,
        user_id,
        APPROVED_MODEL_ID,
        APPROVED_MODEL_VERSION,
        len(sanitized_messages),
        datetime.now(timezone.utc).isoformat(),
    )

    response = await model.ainvoke(sanitized_messages)

    # Validate and sanitize LLM output
    response = _validate_and_sanitize_llm_output(response)

    # Log the interaction with the LLM (output)
    response_content = response.content if hasattr(response, "content") else str(response)
    logger.info(
        "AUDIT: LLM invocation END | trace_id=%s | user_id=%s | model=%s | model_version=%s | "
        "response_preview=%s | timestamp=%s | provenance=%s",
        trace_id,
        user_id,
        APPROVED_MODEL_ID,
        APPROVED_MODEL_VERSION,
        str(response_content)[:200],
        datetime.now(timezone.utc).isoformat(),
        AI_CONTENT_WATERMARK,
    )

    # Label AI-generated content with provenance metadata
    if hasattr(response, "additional_kwargs"):
        response.additional_kwargs["ai_provenance"] = {
            "model_id": APPROVED_MODEL_ID,
            "model_version": APPROVED_MODEL_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "label": AI_CONTENT_WATERMARK,
        }

    return {
        "messages": [response],
        "some_bytes": b"some_bytes",
        "some_byte_array": bytearray(b"some_byte_array"),
        "dict_with_bytes": {"more_bytes": b"more_bytes"},
    }


def call_tool(state):
    trace_id = str(uuid.uuid4())
    tool_name = "tool"

    # Enforce tool allow list
    _check_tool_allow_list(tool_name)

    last_message_content = state["messages"][-1].content

    # Validate tool input content
    if isinstance(last_message_content, str):
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(last_message_content):
                logger.warning(
                    "AUDIT: Tool input REJECTED | trace_id=%s | tool=%s | reason=dangerous_pattern | timestamp=%s",
                    trace_id,
                    tool_name,
                    datetime.now(timezone.utc).isoformat(),
                )
                raise ValueError(
                    f"Tool input contains potentially dangerous content: {pattern.pattern}"
                )
        for pattern in MALICIOUS_COMMAND_PATTERNS:
            if pattern.search(last_message_content):
                logger.warning(
                    "AUDIT: Tool input REJECTED | trace_id=%s | tool=%s | reason=malicious_command | timestamp=%s",
                    trace_id,
                    tool_name,
                    datetime.now(timezone.utc).isoformat(),
                )
                raise ValueError(
                    f"Tool input contains potentially malicious command: {pattern.pattern}"
                )

    logger.info(
        "AUDIT: Tool invocation | trace_id=%s | tool=%s | input_preview=%s | timestamp=%s",
        trace_id,
        tool_name,
        str(last_message_content)[:200],
        datetime.now(timezone.utc).isoformat(),
    )

    result = ToolMessage(
        f"tool_call__{last_message_content}", tool_call_id="tool_call_id"
    )

    logger.info(
        "AUDIT: Tool result | trace_id=%s | tool=%s | result_preview=%s | timestamp=%s",
        trace_id,
        tool_name,
        str(result.content)[:200],
        datetime.now(timezone.utc).isoformat(),
    )

    return {
        "messages": [result]
    }


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content == "end":
        return END
    else:
        next_node = "tool"
        # Enforce tool allow list before routing
        if next_node not in APPROVED_TOOLS:
            logger.warning(
                "AUDIT: Routing DENIED | target=%s | allowed=%s | timestamp=%s",
                next_node,
                APPROVED_TOOLS,
                datetime.now(timezone.utc).isoformat(),
            )
            raise PermissionError(
                f"Routing to '{next_node}' is not permitted. Approved tools: {APPROVED_TOOLS}"
            )
        return next_node


# Verify model is in approved registry at module load time
_verify_model_registry(APPROVED_MODEL_ID)

# NOTE: uses approved model from registry with version pinning
model = ChatOpenAI(model=APPROVED_MODEL_VERSION)
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tool", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tool", "agent")

graph = workflow.compile()