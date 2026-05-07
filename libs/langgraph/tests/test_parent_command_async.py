from __future__ import annotations

import secrets
import pytest
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

pytestmark = pytest.mark.anyio

ALLOWED_NODES = {"node", "parent_first", "parent_second"}

def _make_auth_config(config: RunnableConfig, token: str) -> RunnableConfig:
    metadata = dict(config.get("metadata") or {})
    metadata["auth_token"] = token
    return {**config, "metadata": metadata}

def _verify_auth_token(config: RunnableConfig, expected_token: str) -> None:
    metadata = config.get("metadata") or {}
    token = metadata.get("auth_token")
    if token != expected_token:
        raise PermissionError("Inter-agent authentication failed: invalid or missing auth token.")

def _check_node_allowed(node_name: str) -> None:
    if node_name not in ALLOWED_NODES:
        raise PermissionError(f"Tool/node '{node_name}' is not in the allowed list. Denied.")


async def test_parent_command_from_nested_subgraph() -> None:
    class ParentState(TypedDict):
        jump_from_idx: int

    class ChildState(TypedDict):
        jump: bool

    child_builder: StateGraph[ChildState] = StateGraph(ChildState)

    _check_node_allowed("node")

    async def child_node(state: ChildState) -> Command | ChildState:
        if state["jump"]:
            return Command(graph=Command.PARENT, goto="parent_second")
        return state

    child_builder.add_node("node", child_node)
    child_builder.add_edge(START, "node")

    child_0 = child_builder.compile()
    child_1 = child_builder.compile()

    parent_builder: StateGraph[ParentState] = StateGraph(ParentState)

    _check_node_allowed("parent_first")
    _check_node_allowed("parent_second")

    _session_token = secrets.token_hex(32)

    async def parent_first(state: ParentState, config: RunnableConfig) -> ParentState:
        _verify_auth_token(config, _session_token)

        auth_config = _make_auth_config(config, _session_token)

        await child_0.ainvoke({"jump": state["jump_from_idx"] == 1}, auth_config)
        if state["jump_from_idx"] == 1:
            raise AssertionError("Shouldn't be here")

        await child_1.ainvoke({"jump": state["jump_from_idx"] == 2}, auth_config)
        if state["jump_from_idx"] == 2:
            raise AssertionError("Shouldn't be here")

        return state

    async def parent_second(state: ParentState) -> ParentState:
        return state

    parent_builder.add_node("parent_first", parent_first)
    parent_builder.add_node("parent_second", parent_second)
    parent_builder.add_edge(START, "parent_first")
    parent_builder.add_edge("parent_second", END)

    graph = parent_builder.compile().with_config(
        recursion_limit=10,
        metadata={"auth_token": _session_token},
    )

    assert await graph.ainvoke({"jump_from_idx": 1}) == {"jump_from_idx": 1}
    assert await graph.ainvoke({"jump_from_idx": 2}) == {"jump_from_idx": 2}