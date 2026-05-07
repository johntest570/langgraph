import hashlib
import hmac
import logging
import operator
import os
import secrets
import time as time_module
import uuid
from datetime import datetime, timezone
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.types import Send

logger = logging.getLogger(__name__)

_AGENT_SECRET = os.environ.get("AGENT_SHARED_SECRET", secrets.token_hex(32))
_SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_hex(32))
_SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))

AI_CONTENT_ORIGIN_TAG = "ai-generated"
AI_MODEL_IDENTIFIER = "fanout-subgraph-bench-v1"


def _sign_payload(payload: str) -> str:
    return hmac.new(
        _AGENT_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()


def _make_auth_token(subject: str) -> dict:
    nonce = secrets.token_hex(16)
    issued_at = time_module.time()
    payload = f"{subject}:{nonce}:{issued_at}"
    signature = _sign_payload(payload)
    return {
        "_auth_subject": subject,
        "_auth_nonce": nonce,
        "_auth_issued_at": issued_at,
        "_auth_signature": signature,
    }


def _verify_auth_token(data: dict) -> bool:
    subject = data.get("_auth_subject", "")
    nonce = data.get("_auth_nonce", "")
    issued_at = data.get("_auth_issued_at", 0)
    signature = data.get("_auth_signature", "")
    payload = f"{subject}:{nonce}:{issued_at}"
    expected = _sign_payload(payload)
    if not hmac.compare_digest(expected, signature):
        logger.warning("Auth token signature verification failed for subject=%s", subject)
        return False
    age = time_module.time() - issued_at
    if age > _SESSION_TTL_SECONDS:
        logger.warning("Auth token expired for subject=%s age=%.1fs", subject, age)
        return False
    return True


def _make_session_token() -> str:
    raw_id = secrets.token_hex(32)
    issued_at = time_module.time()
    payload = f"{raw_id}:{issued_at}"
    sig = hmac.new(_SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{raw_id}.{issued_at}.{sig}"


def _verify_session_token(token: str) -> str:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid session token format")
    raw_id, issued_at_str, sig = parts
    issued_at = float(issued_at_str)
    payload = f"{raw_id}:{issued_at}"
    expected_sig = hmac.new(
        _SESSION_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected_sig, sig):
        raise ValueError("Session token signature invalid")
    age = time_module.time() - issued_at
    if age > _SESSION_TTL_SECONDS:
        raise ValueError(f"Session token expired (age={age:.1f}s)")
    return raw_id


def _audit_log(event: str, node: str, input_data: object, output_data: object, correlation_id: str) -> None:
    input_hash = hashlib.sha256(str(input_data).encode()).hexdigest()
    output_hash = hashlib.sha256(str(output_data).encode()).hexdigest()
    record = {
        "event": event,
        "node": node,
        "model_identifier": AI_MODEL_IDENTIFIER,
        "correlation_id": correlation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_hash": input_hash,
        "output_hash": output_hash,
        "principal": "bench-runner",
        "retention_policy": "90d",
    }
    logger.info("AUDIT: %s", record)


_CORRELATION_ID = str(uuid.uuid4())


def fanout_to_subgraph() -> StateGraph:
    class OverallState(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], operator.add]

    async def continue_to_jokes(state: OverallState):
        sends = []
        for s in state["subjects"]:
            auth = _make_auth_token(s)
            payload = {"subject": s}
            payload.update(auth)
            sends.append(Send("generate_joke", payload))
        return sends

    class JokeInput(TypedDict):
        subject: str

    class JokeOutput(TypedDict):
        jokes: list[str]

    class JokeState(JokeInput, JokeOutput): ...

    async def bump(state: JokeOutput):
        input_data = dict(state)
        result = {"jokes": [state["jokes"][0] + " a"]}
        _audit_log("node_execution", "bump", input_data, result, _CORRELATION_ID)
        return result

    async def generate(state: JokeInput):
        if not _verify_auth_token(state):
            logger.warning("Unauthenticated call to generate node; rejecting.")
            raise PermissionError("Inter-agent authentication failed for generate node")
        input_data = dict(state)
        subject = state["subject"]
        timestamp = datetime.now(timezone.utc).isoformat()
        joke_text = f"Joke about {subject}"
        result = {
            "jokes": [
                f"{joke_text} [origin:{AI_CONTENT_ORIGIN_TAG}|model:{AI_MODEL_IDENTIFIER}|ts:{timestamp}]"
            ]
        }
        _audit_log("node_execution", "generate", input_data, result, _CORRELATION_ID)
        return result

    async def edit(state: JokeInput):
        if not _verify_auth_token(state):
            logger.warning("Unauthenticated call to edit node; rejecting.")
            raise PermissionError("Inter-agent authentication failed for edit node")
        input_data = dict(state)
        subject = state["subject"]
        result = {"subject": f"{subject} - hohoho"}
        _audit_log("node_execution", "edit", input_data, result, _CORRELATION_ID)
        return result

    async def bump_loop(state: JokeOutput):
        decision = END if state["jokes"][0].endswith(" a" * 10) else "bump"
        _audit_log("decision", "bump_loop", dict(state), {"next": decision}, _CORRELATION_ID)
        return decision

    # subgraph
    subgraph = StateGraph(JokeState, input_schema=JokeInput, output_schema=JokeOutput)
    subgraph.add_node("edit", edit)
    subgraph.add_node("generate", generate)
    subgraph.add_node("bump", bump)
    subgraph.set_entry_point("edit")
    subgraph.add_edge("edit", "generate")
    subgraph.add_edge("generate", "bump")
    subgraph.add_conditional_edges("bump", bump_loop)
    subgraph.set_finish_point("generate")
    subgraphc = subgraph.compile()

    # parent graph
    builder = StateGraph(OverallState)
    builder.add_node("generate_joke", subgraphc)
    builder.add_conditional_edges(START, continue_to_jokes)
    builder.add_edge("generate_joke", END)

    return builder


def fanout_to_subgraph_sync() -> StateGraph:
    class OverallState(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], operator.add]

    def continue_to_jokes(state: OverallState):
        sends = []
        for s in state["subjects"]:
            auth = _make_auth_token(s)
            payload = {"subject": s}
            payload.update(auth)
            sends.append(Send("generate_joke", payload))
        return sends

    class JokeInput(TypedDict):
        subject: str

    class JokeOutput(TypedDict):
        jokes: list[str]

    class JokeState(JokeInput, JokeOutput): ...

    def bump(state: JokeOutput):
        input_data = dict(state)
        result = {"jokes": [state["jokes"][0] + " a"]}
        _audit_log("node_execution", "bump", input_data, result, _CORRELATION_ID)
        return result

    def generate(state: JokeInput):
        if not _verify_auth_token(state):
            logger.warning("Unauthenticated call to generate node; rejecting.")
            raise PermissionError("Inter-agent authentication failed for generate node")
        input_data = dict(state)
        subject = state["subject"]
        timestamp = datetime.now(timezone.utc).isoformat()
        joke_text = f"Joke about {subject}"
        result = {
            "jokes": [
                f"{joke_text} [origin:{AI_CONTENT_ORIGIN_TAG}|model:{AI_MODEL_IDENTIFIER}|ts:{timestamp}]"
            ]
        }
        _audit_log("node_execution", "generate", input_data, result, _CORRELATION_ID)
        return result

    def edit(state: JokeInput):
        if not _verify_auth_token(state):
            logger.warning("Unauthenticated call to edit node; rejecting.")
            raise PermissionError("Inter-agent authentication failed for edit node")
        input_data = dict(state)
        subject = state["subject"]
        result = {"subject": f"{subject} - hohoho"}
        _audit_log("node_execution", "edit", input_data, result, _CORRELATION_ID)
        return result

    def bump_loop(state: JokeOutput):
        decision = END if state["jokes"][0].endswith(" a" * 10) else "bump"
        _audit_log("decision", "bump_loop", dict(state), {"next": decision}, _CORRELATION_ID)
        return decision

    # subgraph
    subgraph = StateGraph(JokeState, input_schema=JokeInput, output_schema=JokeOutput)
    subgraph.add_node("edit", edit)
    subgraph.add_node("generate", generate)
    subgraph.add_node("bump", bump)
    subgraph.set_entry_point("edit")
    subgraph.add_edge("edit", "generate")
    subgraph.add_edge("generate", "bump")
    subgraph.add_conditional_edges("bump", bump_loop)
    subgraph.set_finish_point("generate")
    subgraphc = subgraph.compile()

    # parent graph
    builder = StateGraph(OverallState)
    builder.add_node("generate_joke", subgraphc)
    builder.add_conditional_edges(START, continue_to_jokes)
    builder.add_edge("generate_joke", END)

    return builder


if __name__ == "__main__":
    import asyncio
    import random
    import time

    import uvloop
    from langgraph.checkpoint.memory import InMemorySaver

    logging.basicConfig(level=logging.INFO)

    session_token = _make_session_token()
    try:
        thread_raw_id = _verify_session_token(session_token)
    except ValueError as e:
        raise RuntimeError(f"Session token validation failed: {e}") from e

    run_correlation_id = str(uuid.uuid4())
    _CORRELATION_ID = run_correlation_id

    graph = fanout_to_subgraph().compile(checkpointer=InMemorySaver())
    input = {
        "subjects": [
            random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(1000)
        ]
    }
    config = {"configurable": {"thread_id": thread_raw_id}}

    logger.info(
        "AUDIT: workflow_start correlation_id=%s thread_id=%s principal=bench-runner retention_policy=90d",
        run_correlation_id,
        thread_raw_id,
    )

    async def run():
        len([c async for c in graph.astream(input, config=config)])

    uvloop.install()
    start = time.time()
    asyncio.run(run())
    end = time.time()

    logger.info(
        "AUDIT: workflow_end correlation_id=%s thread_id=%s elapsed=%.4fs principal=bench-runner",
        run_correlation_id,
        thread_raw_id,
        end - start,
    )
    print(f"Time taken: {end - start:.4f} seconds")