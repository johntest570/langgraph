import hashlib
import logging
import os
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict
from urllib.parse import urlparse

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_aws import ChatBedrock
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

APPROVED_MODEL_REGISTRY = {
    "amazon.titan-text-express-v1": "amazon.titan-text-express-v1",
    "amazon.titan-text-lite-v1": "amazon.titan-text-lite-v1",
}

APPROVED_MODEL_ID = "amazon.titan-text-express-v1"

ALLOWED_TOOL_NAMES = {"tavily_search_results_json"}

ALLOWED_URL_PREFIXES = (
    "https://api.tavily.com/",
)

DANGEROUS_PATTERNS = re.compile(
    r"\b(eval|exec|subprocess|__import__|os\.system|shell=True|base64\.b64decode)\b",
    re.IGNORECASE,
)

PRIVILEGE_ESCALATION_PATTERNS = re.compile(
    r"\b(sudo|chmod|chown|setuid|setgid|escalate|admin|root|privilege)\b",
    re.IGNORECASE,
)

CORRELATION_ID = str(uuid.uuid4())


def _require_authenticated_user(config: dict) -> str:
    user_id = (config or {}).get("configurable", {}).get("user_id")
    if not user_id:
        raise PermissionError(
            "Authentication required: no authenticated user found in config. "
            "Provide a valid 'user_id' in config['configurable']."
        )
    return user_id


def _validate_api_key() -> None:
    api_key = os.environ.get("BEDROCK_API_KEY") or os.environ.get("AWS_ACCESS_KEY_ID")
    if not api_key:
        raise PermissionError(
            "LLM endpoint authentication required: AWS credentials not set."
        )


def _sanitize_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    sanitized = []
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else ""
        if isinstance(content, str):
            if DANGEROUS_PATTERNS.search(content):
                raise ValueError(
                    f"Input message contains forbidden dynamic code execution primitive: {content[:200]}"
                )
            if PRIVILEGE_ESCALATION_PATTERNS.search(content):
                raise ValueError(
                    f"Input message contains potential privilege escalation attempt: {content[:200]}"
                )
            try:
                import base64
                decoded = base64.b64decode(content + "==", validate=False)
                decoded_str = decoded.decode("utf-8", errors="ignore")
                if DANGEROUS_PATTERNS.search(decoded_str):
                    raise ValueError(
                        "Input message contains base64-encoded forbidden content."
                    )
            except Exception as exc:
                if "forbidden" in str(exc):
                    raise
        sanitized.append(msg)
    return sanitized


def _validate_output(response: BaseMessage) -> BaseMessage:
    content = response.content if hasattr(response, "content") else ""
    if isinstance(content, str):
        if DANGEROUS_PATTERNS.search(content):
            raise ValueError(
                f"LLM output contains forbidden dynamic code execution primitive: {content[:200]}"
            )
        if PRIVILEGE_ESCALATION_PATTERNS.search(content):
            raise ValueError(
                f"LLM output contains potential privilege escalation attempt: {content[:200]}"
            )
    return response


def _validate_tool_calls(last_message: BaseMessage) -> None:
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
        if tool_name not in ALLOWED_TOOL_NAMES:
            logger.warning(
                "AUDIT: Denied tool call to '%s' not in allow list. correlation_id=%s timestamp=%s",
                tool_name,
                CORRELATION_ID,
                datetime.now(timezone.utc).isoformat(),
            )
            raise PermissionError(
                f"Tool '{tool_name}' is not in the approved tool allow list: {ALLOWED_TOOL_NAMES}"
            )


def _validate_url_allowlist(url: str) -> None:
    parsed = urlparse(url)
    if not any(url.startswith(prefix) for prefix in ALLOWED_URL_PREFIXES):
        raise PermissionError(
            f"Outbound URL '{url}' is not in the approved allowlist: {ALLOWED_URL_PREFIXES}"
        )
    if parsed.scheme not in ("https",):
        raise PermissionError(
            f"Outbound URL scheme '{parsed.scheme}' is not allowed. Only 'https' is permitted."
        )


def _attach_provenance(response: BaseMessage, model_id: str) -> BaseMessage:
    timestamp = datetime.now(timezone.utc).isoformat()
    content = response.content if hasattr(response, "content") else ""
    content_hash = hashlib.sha256(
        content.encode("utf-8") if isinstance(content, str) else b""
    ).hexdigest()
    provenance = {
        "ai_generated": True,
        "model_id": model_id,
        "model_version": APPROVED_MODEL_REGISTRY.get(model_id, model_id),
        "timestamp": timestamp,
        "content_hash": content_hash,
        "correlation_id": CORRELATION_ID,
        "label": "[AI-GENERATED CONTENT]",
    }
    if hasattr(response, "additional_kwargs"):
        response.additional_kwargs["provenance"] = provenance
    return response


def _minimise_response(response: BaseMessage) -> BaseMessage:
    allowed_fields = {"content", "type", "tool_calls", "additional_kwargs"}
    for attr in list(vars(response).keys()):
        if attr.startswith("_"):
            continue
        if attr not in allowed_fields:
            try:
                setattr(response, attr, None)
            except Exception:
                pass
    return response


_validate_api_key()

model_approved = ChatBedrock(
    model_id=APPROVED_MODEL_ID,
    model_kwargs={"temperature": 0},
)

tools = [TavilySearchResults(max_results=1)]

model_approved = model_approved.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class ContextSchema(TypedDict):
    model: Literal["amazon.titan-text-express-v1", "amazon.titan-text-lite-v1"]


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    _validate_tool_calls(last_message)
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


def call_model(state, config):
    user_id = _require_authenticated_user(config)
    model = model_approved
    messages = state["messages"]

    sanitized_messages = _sanitize_messages(messages)

    input_contents = [
        (m.content if hasattr(m, "content") else "") for m in sanitized_messages
    ]
    input_hash = hashlib.sha256(
        str(input_contents).encode("utf-8")
    ).hexdigest()

    logger.info(
        "AUDIT: LLM invocation started. model_id=%s correlation_id=%s user_id=%s "
        "input_hash=%s timestamp=%s",
        APPROVED_MODEL_ID,
        CORRELATION_ID,
        user_id,
        input_hash,
        datetime.now(timezone.utc).isoformat(),
    )

    response = model.invoke(sanitized_messages)

    _validate_output(response)
    response = _attach_provenance(response, APPROVED_MODEL_ID)
    response = _minimise_response(response)

    output_content = response.content if hasattr(response, "content") else ""
    output_hash = hashlib.sha256(
        output_content.encode("utf-8") if isinstance(output_content, str) else b""
    ).hexdigest()

    logger.info(
        "AUDIT: LLM invocation completed. model_id=%s correlation_id=%s user_id=%s "
        "output_hash=%s timestamp=%s label=[AI-GENERATED CONTENT]",
        APPROVED_MODEL_ID,
        CORRELATION_ID,
        user_id,
        output_hash,
        datetime.now(timezone.utc).isoformat(),
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)

workflow = StateGraph(AgentState, context_schema=ContextSchema)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")

graph = workflow.compile()