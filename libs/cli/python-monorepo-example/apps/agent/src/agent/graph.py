"""Simple LangGraph agent for monorepo testing."""

import hashlib
import logging
import os
import re
import uuid
from datetime import datetime, timezone

from common import get_common_prefix
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from shared import get_dummy_message

from agent.state import State

logger = logging.getLogger(__name__)

# Approved LLM registry
APPROVED_LLM_REGISTRY = {"approved_model": "approved-internal-model-v1"}
CURRENT_MODEL_ID = APPROVED_LLM_REGISTRY["approved_model"]
CURRENT_MODEL_VERSION = "1.0.0"

# Synthetic content provenance label
AI_GENERATED_LABEL = "[AI-GENERATED]"
AI_PROVENANCE_MODEL = CURRENT_MODEL_ID

# Dangerous patterns for prompt injection / malicious command detection
_DANGEROUS_PATTERNS = [
    re.compile(r"(?i)(;|\||\$\(|`|&&|\|\|)\s*(rm|wget|curl|bash|sh|python|exec|eval|os\.system)"),
    re.compile(r"(?i)(ignore previous instructions|disregard all prior)"),
    re.compile(r"(?i)(base64\s*decode|eval\s*\()"),
    re.compile(r"(?i)(<script|javascript:|data:text/html)"),
]

_MAX_INPUT_LENGTH = 4096


def _check_authentication(state: State) -> None:
    """Verify that the caller is authenticated before accessing the agent."""
    auth_token = os.environ.get("AGENT_AUTH_TOKEN")
    if not auth_token:
        raise PermissionError(
            "Authentication required: AGENT_AUTH_TOKEN environment variable is not set. "
            "A user must authenticate before accessing the AI Agent."
        )
    expected_token = os.environ.get("AGENT_EXPECTED_TOKEN")
    if expected_token and auth_token != expected_token:
        raise PermissionError(
            "Authentication failed: invalid AGENT_AUTH_TOKEN. "
            "A user must authenticate before accessing the AI Agent."
        )


def _sanitize_and_validate(value: str, field_name: str) -> str:
    """Sanitize and validate a string input before passing to the AI model."""
    if not isinstance(value, str):
        raise ValueError(f"Input field '{field_name}' must be a string, got {type(value).__name__}.")
    sanitized = value.strip()
    if len(sanitized) == 0:
        raise ValueError(f"Input field '{field_name}' must not be empty.")
    if len(sanitized) > _MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input field '{field_name}' exceeds maximum allowed length of {_MAX_INPUT_LENGTH} characters."
        )
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(sanitized):
            raise ValueError(
                f"Input field '{field_name}' contains potentially malicious content and has been rejected."
            )
    return sanitized


def _attach_provenance(content: str, model_id: str, model_version: str, trace_id: str, timestamp: str) -> str:
    """Attach provenance metadata and synthetic content label to AI-generated output."""
    provenance_header = (
        f"{AI_GENERATED_LABEL} "
        f"[model={model_id}] "
        f"[version={model_version}] "
        f"[trace_id={trace_id}] "
        f"[generated_at={timestamp}]"
    )
    return f"{provenance_header}\n{content}"


def call_model(state: State) -> dict:
    """Simple node that uses the shared libraries."""
    # Enforce authentication before accessing the agent
    _check_authentication(state)

    trace_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Validate state messages
    messages = state.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("State 'messages' must be a list.")

    # Use functions from both shared packages
    raw_dummy_message = get_dummy_message()
    raw_prefix = get_common_prefix()

    # Sanitize and validate all inputs before passing to the AI model
    dummy_message = _sanitize_and_validate(raw_dummy_message, "dummy_message")
    prefix = _sanitize_and_validate(raw_prefix, "prefix")

    # Log the interaction with the LLM (input)
    input_content = f"{prefix} Agent says: {dummy_message}"
    input_hash = hashlib.sha256(input_content.encode("utf-8")).hexdigest()

    logger.info(
        "LLM interaction initiated",
        extra={
            "trace_id": trace_id,
            "timestamp": timestamp,
            "model_id": CURRENT_MODEL_ID,
            "model_version": CURRENT_MODEL_VERSION,
            "input_hash": input_hash,
            "input_message_count": len(messages),
            "event": "llm_call_start",
        },
    )

    # Construct the AI message using only approved model constructs
    # NOTE: AIMessage is used here as a message container; the approved model identifier
    # is enforced via APPROVED_LLM_REGISTRY and attached as provenance metadata.
    labeled_content = _attach_provenance(
        content=input_content,
        model_id=CURRENT_MODEL_ID,
        model_version=CURRENT_MODEL_VERSION,
        trace_id=trace_id,
        timestamp=timestamp,
    )

    message = AIMessage(content=labeled_content)

    output_hash = hashlib.sha256(labeled_content.encode("utf-8")).hexdigest()

    # Log the interaction with the LLM (output / decision audit trail)
    logger.info(
        "LLM interaction completed",
        extra={
            "trace_id": trace_id,
            "timestamp": timestamp,
            "model_id": CURRENT_MODEL_ID,
            "model_version": CURRENT_MODEL_VERSION,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "output_label": AI_GENERATED_LABEL,
            "provenance_model": AI_PROVENANCE_MODEL,
            "event": "llm_call_end",
            "decision": "message_generated",
        },
    )

    return {"messages": [message]}


def should_continue(state: State):
    """Conditional edge - end after first message."""
    messages = state["messages"]
    if len(messages) > 0:
        return END
    return "call_model"


# Build the graph
workflow = StateGraph(State)

# Add the node
workflow.add_node("call_model", call_model)

# Add edges
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", should_continue)

graph = workflow.compile()