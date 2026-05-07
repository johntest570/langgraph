import hashlib
import hmac
import logging
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from langgraph.pregel import Pregel

# ---------------------------------------------------------------------------
# Security configuration
# ---------------------------------------------------------------------------

# Approved model registry (policy: approved model list + version pinning)
APPROVED_MODEL_REGISTRY = {
    "FakeFunctionChatModel": {
        "version": "1.0.0",
        "provider": "internal-test",
        "approved": True,
    }
}

# Approved tool allow list (policy: explicit tool allow list)
APPROVED_TOOL_NAMES: set[str] = set()  # populated at runtime after tool creation

# Dangerous patterns for input/output sanitization
_DANGEROUS_PATTERNS = re.compile(
    r"\b(eval|exec|subprocess|__import__|compile|open|os\.system"
    r"|shell=True|base64|leetspeak)\b",
    re.IGNORECASE,
)

# Session token signing secret (policy: session token integrity)
_SESSION_SECRET = secrets.token_bytes(32)
_SESSION_EXPIRY_SECONDS = 3600

# Logging setup (policy: decision logging + LLM interaction logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
_audit_logger = logging.getLogger("ai_agent.audit")
_llm_logger = logging.getLogger("ai_agent.llm")

# Output size limits (policy: output data minimisation)
_MAX_TOOL_RESULT_LEN = 500
_MAX_LLM_CONTENT_LEN = 500

# Model identity constants (policy: model identity + version pinning)
_MODEL_ID = "FakeFunctionChatModel"
_MODEL_VERSION = APPROVED_MODEL_REGISTRY[_MODEL_ID]["version"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sanitize_text(text: str, context: str = "input") -> str:
    """Sanitize text by rejecting dangerous patterns."""
    if _DANGEROUS_PATTERNS.search(text):
        _audit_logger.warning(
            "Dangerous pattern detected in %s; content blocked.", context
        )
        raise ValueError(f"Blocked: dangerous pattern found in {context}.")
    return text


def _validate_input_messages(messages: list) -> list:
    """Validate and sanitize all input messages before sending to the model."""
    sanitized = []
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            _sanitize_text(msg.content, context="input message")
        sanitized.append(msg)
    return sanitized


def _validate_llm_output(message: BaseMessage) -> BaseMessage:
    """Validate LLM output for dangerous code execution primitives."""
    if isinstance(message.content, str) and message.content:
        _sanitize_text(message.content, context="LLM output")
    if hasattr(message, "tool_calls"):
        for tc in message.tool_calls or []:
            tool_name = tc.get("name", "")
            if tool_name not in APPROVED_TOOL_NAMES:
                _audit_logger.warning(
                    "Tool invocation denied: tool '%s' not in allow list. "
                    "Approved tools: %s",
                    tool_name,
                    APPROVED_TOOL_NAMES,
                )
                raise ValueError(
                    f"Tool '{tool_name}' is not in the approved tool allow list."
                )
            for arg_val in (tc.get("args") or {}).values():
                if isinstance(arg_val, str):
                    _sanitize_text(arg_val, context=f"tool_call arg for {tool_name}")
    return message


def _truncate(value: str, max_len: int) -> str:
    """Truncate a string to max_len characters."""
    if len(value) > max_len:
        return value[:max_len] + "[TRUNCATED]"
    return value


def _sanitize_tool_query(query: str) -> str:
    """Sanitize tool input query."""
    _sanitize_text(query, context="tool query input")
    return query


def _tool_result(query: str) -> str:
    """Tool implementation with input sanitization and output minimisation."""
    _sanitize_tool_query(query)
    raw = f"result for query: {query}"
    return _truncate(raw, _MAX_TOOL_RESULT_LEN)


def _label_ai_output(message: AIMessage, model_id: str, model_version: str) -> AIMessage:
    """Attach provenance metadata to AI-generated output (watermarking/labeling)."""
    provenance = {
        "ai_generated": True,
        "model_id": model_id,
        "model_version": model_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "content_origin": "synthetic",
    }
    existing = message.additional_kwargs or {}
    labeled = message.copy(
        update={"additional_kwargs": {**existing, "provenance": provenance}}
    )
    return labeled


def _create_session_token(thread_id: str) -> str:
    """Create a signed, expiry-bound session token."""
    expiry = int(time.time()) + _SESSION_EXPIRY_SECONDS
    payload = f"{thread_id}:{expiry}"
    sig = hmac.new(_SESSION_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def _verify_session_token(token: str) -> str:
    """Verify session token integrity and expiry; return thread_id."""
    try:
        parts = token.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError("Malformed session token.")
        thread_id, expiry_str, sig = parts
        expiry = int(expiry_str)
        payload = f"{thread_id}:{expiry_str}"
        expected_sig = hmac.new(
            _SESSION_SECRET, payload.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            raise ValueError("Session token signature invalid.")
        if int(time.time()) > expiry:
            raise ValueError("Session token has expired.")
        return thread_id
    except (ValueError, AttributeError) as exc:
        _audit_logger.error("Session token verification failed: %s", exc)
        raise


def _authenticate_user(user_id: str | None) -> str:
    """Stub authentication gate. Raises if user is not authenticated."""
    if not user_id:
        raise PermissionError(
            "Authentication required: no user_id provided. "
            "Access to the AI agent is denied."
        )
    _audit_logger.info("User authenticated: %s", user_id)
    return user_id


def _check_model_registry(model_id: str) -> None:
    """Verify model is in the approved registry."""
    entry = APPROVED_MODEL_REGISTRY.get(model_id)
    if not entry or not entry.get("approved"):
        raise ValueError(
            f"Model '{model_id}' is not in the approved model registry."
        )
    _audit_logger.info(
        "Model registry check passed: model_id=%s version=%s",
        model_id,
        entry.get("version"),
    )


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------

def react_agent(n_tools: int, checkpointer: BaseCheckpointSaver | None) -> Pregel:

    # Policy: verify model is in approved registry before instantiation
    _check_model_registry(_MODEL_ID)

    class FakeFunctionChatModel(FakeMessagesListChatModel):
        def bind_tools(self, functions: list):
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            # Policy: validate and sanitize input messages before LLM call
            validated_messages = _validate_input_messages(list(messages))

            # Policy: log LLM interaction (input)
            _llm_logger.info(
                "LLM input | model_id=%s model_version=%s message_count=%d",
                _MODEL_ID,
                _MODEL_VERSION,
                len(validated_messages),
            )

            response = self.responses[self.i].copy()
            if self.i < len(self.responses) - 1:
                self.i += 1
            else:
                self.i = 0

            # Policy: validate LLM output for dangerous primitives
            _validate_llm_output(response)

            # Policy: truncate oversized content (output data minimisation)
            if isinstance(response.content, str):
                response = response.copy(
                    update={"content": _truncate(response.content, _MAX_LLM_CONTENT_LEN)}
                )

            # Policy: label/watermark AI-generated output
            response = _label_ai_output(response, _MODEL_ID, _MODEL_VERSION)

            # Policy: log LLM interaction (output)
            _llm_logger.info(
                "LLM output | model_id=%s model_version=%s content_len=%d",
                _MODEL_ID,
                _MODEL_VERSION,
                len(response.content) if isinstance(response.content, str) else 0,
            )

            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation])

    tool_name = str(uuid4())

    tool = StructuredTool.from_function(
        _tool_result,
        name=tool_name,
        description="approved-benchmark-tool",
    )

    # Policy: register tool in allow list before agent creation
    APPROVED_TOOL_NAMES.add(tool.name)
    _audit_logger.info(
        "Tool registered in allow list: tool_name=%s", tool.name
    )

    # Policy: build model responses with provenance labels and size limits
    tool_call_responses = []
    for _ in range(n_tools):
        raw_query = str(uuid4())
        sanitized_query = _truncate(raw_query, 64)
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": str(uuid4()),
                    "name": tool.name,
                    "args": {"query": sanitized_query},
                }
            ],
            id=str(uuid4()),
        )
        msg = _label_ai_output(msg, _MODEL_ID, _MODEL_VERSION)
        tool_call_responses.append(msg)

    final_msg = AIMessage(
        content=_truncate("answer" * 100, _MAX_LLM_CONTENT_LEN),
        id=str(uuid4()),
    )
    final_msg = _label_ai_output(final_msg, _MODEL_ID, _MODEL_VERSION)

    model = FakeFunctionChatModel(
        responses=tool_call_responses + [final_msg]
    )

    agent = create_react_agent(model, [tool], checkpointer=checkpointer)

    # Policy: log agent creation decision
    _audit_logger.info(
        "Agent created | model_id=%s model_version=%s tool_count=%d "
        "checkpointer=%s",
        _MODEL_ID,
        _MODEL_VERSION,
        1,
        type(checkpointer).__name__ if checkpointer else "None",
    )

    return agent


if __name__ == "__main__":
    import asyncio

    import uvloop
    from langgraph.checkpoint.memory import InMemorySaver

    # Policy: authenticate user before accessing the agent
    _authenticated_user = _authenticate_user(user_id="bench-user-001")

    # Policy: create a signed, expiry-bound session token
    _raw_thread_id = secrets.token_hex(16)
    _session_token = _create_session_token(_raw_thread_id)
    _verified_thread_id = _verify_session_token(_session_token)

    graph = react_agent(100, checkpointer=InMemorySaver())

    # Policy: validate and sanitize input before passing to agent
    _raw_input_text = "hi?"
    _sanitize_text(_raw_input_text, context="user input")
    agent_input = {"messages": [HumanMessage(_raw_input_text)]}

    config = {
        "configurable": {"thread_id": _verified_thread_id},
        "recursion_limit": 10000,
    }

    # Policy: log decision / audit trail before agent invocation
    _trace_id = str(uuid4())
    _audit_logger.info(
        "Agent invocation | trace_id=%s user=%s thread_id=%s "
        "model_id=%s model_version=%s input_hash=%s timestamp_utc=%s",
        _trace_id,
        _authenticated_user,
        _verified_thread_id,
        _MODEL_ID,
        _MODEL_VERSION,
        hashlib.sha256(str(agent_input).encode()).hexdigest(),
        datetime.now(timezone.utc).isoformat(),
    )

    async def run():
        chunks = [c async for c in graph.astream(agent_input, config=config)]
        _audit_logger.info(
            "Agent completed | trace_id=%s chunk_count=%d timestamp_utc=%s",
            _trace_id,
            len(chunks),
            datetime.now(timezone.utc).isoformat(),
        )
        return len(chunks)

    uvloop.install()
    asyncio.run(run())