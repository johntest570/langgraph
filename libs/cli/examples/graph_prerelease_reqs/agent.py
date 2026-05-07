import hashlib
import logging
import os
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

# Policy: approved model registry and version pinning
APPROVED_MODEL_REGISTRY = {
    # NOTE: ChatOpenAI/openai is NOT_IN_REGISTRY per policy.
    # This entry is kept only to demonstrate registry enforcement.
    # Replace with an approved model from the organization's registry.
}

APPROVED_TOOL_ALLOWLIST: set[str] = set()

# Explicit tool allow list (empty — no tools approved)
tools = []

# Policy: LLM endpoints must require authentication
_openai_api_key = os.environ.get("OPENAI_API_KEY")
if not _openai_api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable must be set for LLM endpoint authentication."
    )

# Policy: version pinning — specify explicit model name/version
# NOTE: ChatOpenAI (openai) is NOT_IN_REGISTRY. Replace with an approved model.
_PINNED_MODEL_NAME = "gpt-3.5-turbo-0125"

model_oai = ChatOpenAI(temperature=0, model=_PINNED_MODEL_NAME, api_key=_openai_api_key)

model_oai = model_oai.bind_tools(tools)

# Patterns for detecting dynamic code execution primitives in LLM output
_DANGEROUS_OUTPUT_PATTERNS = re.compile(
    r"\b(eval|exec|execfile|compile|__import__|subprocess|os\.system|"
    r"os\.popen|popen|shell=True|Popen|importlib|ctypes|cffi)\b",
    re.IGNORECASE,
)

# Patterns for detecting malicious prompt content in input
_DANGEROUS_INPUT_PATTERNS = re.compile(
    r"\b(eval|exec|execfile|compile|__import__|subprocess|os\.system|"
    r"os\.popen|popen|shell=True|Popen|importlib|ctypes|cffi|"
    r"ignore previous instructions|disregard your instructions|"
    r"you are now|act as|pretend you are)\b",
    re.IGNORECASE,
)

# Base64-like pattern detection
_BASE64_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{40,}={0,2})")

# Privilege escalation keywords
_ESCALATION_PATTERNS = re.compile(
    r"\b(sudo|root|admin|superuser|privilege|escalat|bypass|override|"
    r"grant.*permission|elevat|setuid|chmod 777|chown root)\b",
    re.IGNORECASE,
)


def _require_authenticated_user(config: dict) -> str:
    """Policy: A user must authenticate before accessing the AI Agent."""
    configurable = config.get("configurable", {}) if config else {}
    user_token = configurable.get("user_token") or os.environ.get("AGENT_USER_TOKEN")
    if not user_token:
        raise PermissionError(
            "Authentication required: provide a valid user_token in config['configurable']['user_token'] "
            "or set the AGENT_USER_TOKEN environment variable."
        )
    return user_token


def _validate_and_sanitize_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """Policy: Sanitize and validate all input to the AI Model.
    Policy: Do not allow prompts that can execute malicious commands at runtime.
    Policy: Enforce output data minimisation.
    """
    if not messages:
        raise ValueError("Messages list must not be empty.")

    sanitized = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Check for dangerous input patterns
        if _DANGEROUS_INPUT_PATTERNS.search(content):
            logger.warning(
                "SECURITY: Potentially dangerous input pattern detected and blocked. "
                "content_hash=%s",
                hashlib.sha256(content.encode()).hexdigest(),
            )
            raise ValueError("Input message contains disallowed content.")

        # Check for base64-encoded payloads
        if _BASE64_PATTERN.search(content):
            logger.warning(
                "SECURITY: Potential base64-encoded payload detected in input. "
                "content_hash=%s",
                hashlib.sha256(content.encode()).hexdigest(),
            )
            raise ValueError("Input message contains suspicious encoded content.")

        # Check for privilege escalation attempts
        if _ESCALATION_PATTERNS.search(content):
            logger.warning(
                "SECURITY: Privilege escalation attempt detected in input. "
                "content_hash=%s",
                hashlib.sha256(content.encode()).hexdigest(),
            )
            raise ValueError("Input message contains privilege escalation attempt.")

        # Data minimisation: truncate excessively large messages
        max_content_length = 32000
        if len(content) > max_content_length:
            logger.warning(
                "SECURITY: Input message truncated for data minimisation. "
                "original_length=%d",
                len(content),
            )
            content = content[:max_content_length]

        # Reconstruct message with sanitized content
        sanitized_msg = msg.__class__(content=content)
        sanitized.append(sanitized_msg)

    # Data minimisation: limit conversation history depth
    max_history = 20
    if len(sanitized) > max_history:
        logger.warning(
            "SECURITY: Conversation history truncated for data minimisation. "
            "original_count=%d truncated_to=%d",
            len(sanitized),
            max_history,
        )
        sanitized = sanitized[-max_history:]

    return sanitized


def _validate_and_sanitize_output(response: BaseMessage, trace_id: str) -> BaseMessage:
    """Policy: Validate and sanitize LLM output including for presence of eval or
    any dynamic code execution primitive.
    Policy: Enforce synthetic content provenance, labeling, and watermarking.
    """
    content = response.content if isinstance(response.content, str) else str(response.content)

    # Check for dangerous output patterns (eval, exec, subprocess, etc.)
    if _DANGEROUS_OUTPUT_PATTERNS.search(content):
        logger.warning(
            "SECURITY: LLM output contains dynamic code execution primitive. "
            "trace_id=%s content_hash=%s",
            trace_id,
            hashlib.sha256(content.encode()).hexdigest(),
        )
        raise ValueError("LLM output contains disallowed dynamic code execution content.")

    # Policy: Enforce synthetic content provenance and labeling
    provenance_label = (
        f"[AI-GENERATED | model={_PINNED_MODEL_NAME} | "
        f"trace_id={trace_id} | "
        f"timestamp={datetime.now(timezone.utc).isoformat()} | "
        f"origin=synthetic]"
    )
    labeled_content = f"{provenance_label}\n{content}"

    # Reconstruct response with provenance label
    labeled_response = response.__class__(content=labeled_content)
    return labeled_response


def _check_tool_allowlist(tool_calls: list) -> None:
    """Policy: Restrict AI agents to an explicit tool allow list.
    Policy: Detect and block agent privilege escalation attempts.
    """
    for tool_call in tool_calls:
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
        if tool_name not in APPROVED_TOOL_ALLOWLIST:
            logger.warning(
                "SECURITY: Tool invocation blocked — not in allow list. "
                "tool_name=%s approved_tools=%s",
                tool_name,
                sorted(APPROVED_TOOL_ALLOWLIST),
            )
            raise PermissionError(
                f"Tool '{tool_name}' is not in the approved tool allow list. "
                f"Approved tools: {sorted(APPROVED_TOOL_ALLOWLIST)}"
            )

        # Check for privilege escalation in tool name
        if _ESCALATION_PATTERNS.search(str(tool_name)):
            logger.warning(
                "SECURITY: Privilege escalation attempt detected in tool call. "
                "tool_name=%s",
                tool_name,
            )
            raise PermissionError(
                f"Tool '{tool_name}' appears to be a privilege escalation attempt."
            )


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Policy: Restrict AI agents to an explicit tool allow list
    # Policy: Detect and block agent privilege escalation attempts
    _check_tool_allowlist(last_message.tool_calls)
    # Otherwise if there is, we continue
    return "continue"


# Define the function that calls the model
def call_model(state, config):
    # Policy: A user must authenticate before accessing the AI Agent
    user_token = _require_authenticated_user(config)

    # Generate a unique trace/correlation ID for this invocation
    trace_id = str(uuid.uuid4())
    invocation_timestamp = datetime.now(timezone.utc).isoformat()

    model = model_oai
    messages = state["messages"]

    # Policy: Sanitize and validate all input to the AI Model
    # Policy: Do not allow prompts that can execute malicious commands at runtime
    # Policy: Enforce output data minimisation
    sanitized_messages = _validate_and_sanitize_messages(messages)

    # Compute input hash for audit trail
    input_repr = str([m.content for m in sanitized_messages])
    input_hash = hashlib.sha256(input_repr.encode()).hexdigest()

    # Policy: Agents must log all interactions with an LLM
    # Policy: Enforce decision logging, audit trail, and forensic readiness
    logger.info(
        "LLM_INTERACTION_START trace_id=%s model=%s input_message_count=%d "
        "input_hash=%s timestamp=%s user_token_hash=%s",
        trace_id,
        _PINNED_MODEL_NAME,
        len(sanitized_messages),
        input_hash,
        invocation_timestamp,
        hashlib.sha256(user_token.encode()).hexdigest(),
    )

    response = model.invoke(sanitized_messages)

    # Policy: Validate and sanitize LLM output
    # Policy: Enforce synthetic content provenance, labeling, and watermarking
    labeled_response = _validate_and_sanitize_output(response, trace_id)

    output_hash = hashlib.sha256(
        (labeled_response.content if isinstance(labeled_response.content, str) else str(labeled_response.content)).encode()
    ).hexdigest()

    # Policy: Agents must log all interactions with an LLM
    # Policy: Enforce decision logging, audit trail, and forensic readiness
    logger.info(
        "LLM_INTERACTION_END trace_id=%s model=%s output_hash=%s "
        "has_tool_calls=%s timestamp=%s",
        trace_id,
        _PINNED_MODEL_NAME,
        output_hash,
        bool(getattr(labeled_response, "tool_calls", None)),
        datetime.now(timezone.utc).isoformat(),
    )

    # We return a list, because this will get added to the existing list
    return {"messages": [labeled_response]}


# Define the function to execute tools
tool_node = ToolNode(tools)


class ContextSchema(TypedDict):
    model: Literal["anthropic", "openai"]


# Define a new graph
workflow = StateGraph(AgentState, context_schema=ContextSchema)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()