import hashlib
import logging
import os
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from urllib.parse import urlparse

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Approved model registry — only models listed here may be used
APPROVED_MODEL_REGISTRY = {
    "approved-model-v1": "approved-model-v1",
}

# Explicit tool allow list
TOOL_ALLOW_LIST = {"tavily_search_results_json"}

# URL allowlist for outbound HTTP
ALLOWED_URL_HOSTS = {"api.tavily.com"}

# Dynamic code execution primitives to block in LLM output
DANGEROUS_PATTERNS = re.compile(
    r"\b(eval|exec|subprocess|os\.system|os\.popen|__import__|compile|execfile"
    r"|open\s*\(|importlib|ctypes|pickle\.loads|marshal\.loads)\b",
    re.IGNORECASE,
)

# Prompt injection / malicious content patterns
MALICIOUS_PROMPT_PATTERNS = re.compile(
    r"(ignore previous instructions|disregard.*instructions"
    r"|you are now|pretend you|act as if|system prompt"
    r"|<script|javascript:|data:text/html"
    r"|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}"
    r"|base64[,:]|eval\(|exec\(|subprocess)",
    re.IGNORECASE,
)

# Correlation ID for this session
_SESSION_TRACE_ID = str(uuid.uuid4())


def _require_env_key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            "Authentication is mandatory before accessing the AI agent."
        )
    return value


def _validate_api_keys() -> None:
    _require_env_key("AGENT_API_KEY")


_validate_api_keys()

# Raise immediately if approved model is not available
_APPROVED_MODEL_ID = os.environ.get("APPROVED_MODEL_ID", "approved-model-v1")
if _APPROVED_MODEL_ID not in APPROVED_MODEL_REGISTRY:
    raise ValueError(
        f"Model '{_APPROVED_MODEL_ID}' is not in the approved model registry. "
        "Only approved models may be used."
    )

try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI
    _approved_model = _ChatOpenAI(
        model=_APPROVED_MODEL_ID,
        temperature=0,
    )
except Exception:
    raise RuntimeError(
        "Could not instantiate approved model. Ensure the approved model endpoint "
        "is configured and the model is in the registry."
    )


def _validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.hostname in ALLOWED_URL_HOSTS
    except Exception:
        return False


def _sanitize_text(text: str, context: str = "input") -> str:
    if MALICIOUS_PROMPT_PATTERNS.search(text):
        logger.warning(
            "Malicious content detected in %s. Blocking.",
            context,
            extra={"trace_id": _SESSION_TRACE_ID},
        )
        raise ValueError(f"Malicious content detected in {context}. Request blocked.")
    return text


def _validate_prompt_file(content: str, filename: str) -> str:
    _sanitize_text(content, context=f"prompt file '{filename}'")
    if len(content) > 65536:
        raise ValueError(f"Prompt file '{filename}' exceeds maximum allowed size.")
    return content


def _validate_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    MAX_MESSAGES = 50
    truncated = list(messages[-MAX_MESSAGES:])
    for msg in truncated:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        _sanitize_text(content, context="input message")
    return truncated


def _validate_llm_output(response: BaseMessage) -> BaseMessage:
    content = response.content if isinstance(response.content, str) else str(response.content)
    if DANGEROUS_PATTERNS.search(content):
        logger.warning(
            "Dangerous code execution primitive detected in LLM output. Blocking.",
            extra={"trace_id": _SESSION_TRACE_ID},
        )
        raise ValueError(
            "LLM output contains dangerous code execution primitives. Response blocked."
        )
    _sanitize_text(content, context="LLM output")
    return response


def _check_tool_allow_list(tool_calls) -> None:
    for tc in tool_calls:
        tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if tool_name not in TOOL_ALLOW_LIST:
            logger.error(
                "Tool '%s' is not in the allow list. Blocking invocation. trace_id=%s",
                tool_name,
                _SESSION_TRACE_ID,
            )
            raise ValueError(
                f"Tool '{tool_name}' is not permitted. Only tools in the allow list may be invoked."
            )
        logger.info(
            "Tool invocation permitted: tool=%s trace_id=%s",
            tool_name,
            _SESSION_TRACE_ID,
        )


def _watermark_response(response: BaseMessage, model_id: str, timestamp: str) -> BaseMessage:
    if hasattr(response, "additional_kwargs"):
        response.additional_kwargs["x_ai_generated"] = True
        response.additional_kwargs["x_model_id"] = model_id
        response.additional_kwargs["x_generated_at"] = timestamp
        response.additional_kwargs["x_trace_id"] = _SESSION_TRACE_ID
        response.additional_kwargs["x_content_origin"] = "ai-synthetic"
    return response


def _log_audit_record(
    event: str,
    model_id: str,
    input_hash: str,
    output_summary: str,
    timestamp: str,
) -> None:
    logger.info(
        "AUDIT event=%s model_id=%s model_version=%s input_hash=%s "
        "output_summary=%.120s timestamp=%s trace_id=%s",
        event,
        model_id,
        APPROVED_MODEL_REGISTRY.get(model_id, "unknown"),
        input_hash,
        output_summary,
        timestamp,
        _SESSION_TRACE_ID,
    )


_raw_prompt = open(Path(__file__).parent.parent / "prompt.txt").read()
_raw_subprompt = open(Path(__file__).parent / "subprompt.txt").read()

prompt = _validate_prompt_file(_raw_prompt, "prompt.txt")
subprompt = _validate_prompt_file(_raw_subprompt, "subprompt.txt")

tools = [TavilySearchResults(max_results=1)]

model_anth = _approved_model
model_oai = _approved_model

model_anth = model_anth.bind_tools(tools)
model_oai = model_oai.bind_tools(tools)


class AgentContext(TypedDict):
    model: Literal["anthropic", "openai"]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    _check_tool_allow_list(last_message.tool_calls)
    return "continue"


def call_model(state, runtime: Runtime[AgentContext]):
    model_id = _APPROVED_MODEL_ID
    if model_id not in APPROVED_MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_id}' is not in the approved model registry."
        )

    model = model_anth

    raw_messages = state["messages"]
    messages = _validate_messages(raw_messages)

    input_text = " ".join(
        (m.content if isinstance(m.content, str) else str(m.content))
        for m in messages
    )
    input_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(
        "LLM invocation start: model=%s input_hash=%s message_count=%d trace_id=%s timestamp=%s",
        model_id,
        input_hash,
        len(messages),
        _SESSION_TRACE_ID,
        timestamp,
    )

    response = model.invoke(messages)

    response = _validate_llm_output(response)

    output_text = response.content if isinstance(response.content, str) else str(response.content)
    output_summary = output_text[:200]

    response = _watermark_response(response, model_id, timestamp)

    _log_audit_record(
        event="llm_inference",
        model_id=model_id,
        input_hash=input_hash,
        output_summary=output_summary,
        timestamp=timestamp,
    )

    logger.info(
        "LLM invocation complete: model=%s output_length=%d trace_id=%s timestamp=%s",
        model_id,
        len(output_text),
        _SESSION_TRACE_ID,
        timestamp,
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)

workflow = StateGraph(AgentState, context_schema=AgentContext)

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