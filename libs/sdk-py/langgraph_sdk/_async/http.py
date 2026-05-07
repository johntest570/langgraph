"""HTTP client for async operations."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import sys
import time
import uuid
import warnings
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, cast

import httpx
import orjson

from langgraph_sdk._shared.utilities import (
    _orjson_default,
    _validate_reconnect_location,
)
from langgraph_sdk.errors import _araise_for_status_typed
from langgraph_sdk.schema import QueryParamTypes, StreamPart
from langgraph_sdk.sse import SSEDecoder, aiter_lines_raw

logger = logging.getLogger(__name__)

# Maximum response body size to log (to avoid logging huge payloads)
_MAX_LOG_BODY_SIZE = 512

# Suspicious prompt injection patterns to detect in inputs
_SUSPICIOUS_PATTERNS = [
    re.compile(r"ignore\s+(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"<\s*script\s*>", re.IGNORECASE),
    re.compile(r"\beval\s*\(", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
    re.compile(r"\bsubprocess\s*\.", re.IGNORECASE),
    re.compile(r"__import__\s*\(", re.IGNORECASE),
]

# Allowed URL schemes for outbound requests
_ALLOWED_SCHEMES = {"http", "https"}


def _generate_trace_id() -> str:
    """Generate a unique trace/correlation ID for a request."""
    return str(uuid.uuid4())


def _truncate_for_log(value: Any, max_size: int = _MAX_LOG_BODY_SIZE) -> str:
    """Truncate a value for safe logging."""
    text = repr(value)
    if len(text) > max_size:
        return text[:max_size] + "...[truncated]"
    return text


def _sanitize_json_input(json: Any) -> None:
    """Validate and sanitize JSON input for suspicious content."""
    if json is None:
        return
    text = repr(json)
    for pattern in _SUSPICIOUS_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"Input contains potentially unsafe content matching pattern: {pattern.pattern}"
            )
    # Check for base64-encoded suspicious content
    # Extract potential base64 strings and check them
    b64_candidates = re.findall(r"[A-Za-z0-9+/]{20,}={0,2}", text)
    for candidate in b64_candidates:
        try:
            decoded = base64.b64decode(candidate).decode("utf-8", errors="ignore")
            for pattern in _SUSPICIOUS_PATTERNS:
                if pattern.search(decoded):
                    raise ValueError(
                        "Input contains potentially unsafe base64-encoded content"
                    )
        except Exception as exc:
            if "unsafe" in str(exc):
                raise


def _sanitize_path(path: str) -> str:
    """Validate and sanitize a URL path."""
    if not isinstance(path, str):
        raise ValueError(f"Path must be a string, got {type(path)}")
    # Check for null bytes or other dangerous characters
    if "\x00" in path:
        raise ValueError("Path contains null bytes")
    return path


def _validate_url_allowlist(url: str | httpx.URL, base_url: httpx.URL) -> None:
    """Enforce URL allowlist: only allow same scheme and host as base_url."""
    parsed = httpx.URL(str(url)) if not isinstance(url, httpx.URL) else url
    if parsed.scheme and parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. Allowed schemes: {_ALLOWED_SCHEMES}"
        )
    if parsed.host and parsed.host != base_url.host:
        raise ValueError(
            f"URL host '{parsed.host}' is not in the allowlist. Expected host: '{base_url.host}'"
        )


def _sanitize_response(data: Any) -> Any:
    """Sanitize and validate response data from the server."""
    if data is None:
        return None
    # Validate that the response is a basic JSON-compatible type
    if not isinstance(data, (dict, list, str, int, float, bool, type(None))):
        raise ValueError(f"Unexpected response type from server: {type(data)}")
    return data


def _verify_server_authentication(client: httpx.AsyncClient) -> None:
    """Verify that the httpx client is configured with server authentication."""
    base_url = str(client.base_url)
    # Enforce HTTPS for non-localhost connections
    if base_url.startswith("http://"):
        host = client.base_url.host
        if host not in ("localhost", "127.0.0.1", "::1", ""):
            logger.warning(
                "HttpClient is connecting to a non-local server over HTTP (not HTTPS). "
                "Server identity cannot be verified. Consider using HTTPS with proper "
                "certificate verification for production use."
            )
    # Log the server being connected to for audit purposes
    logger.debug(
        "HttpClient initialized",
        extra={
            "audit": True,
            "server_base_url": base_url,
            "ssl_verify": getattr(client, "_verify", "unknown"),
        },
    )


def _require_hitl_approval(path: str, json: Any) -> None:
    """Human-in-the-Loop approval gate for DELETE (risky) operations."""
    import os

    # Check for an environment variable that pre-approves deletions (for automated pipelines)
    if os.environ.get("LANGGRAPH_HITL_AUTO_APPROVE_DELETE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.warning(
            "HITL approval for DELETE bypassed via LANGGRAPH_HITL_AUTO_APPROVE_DELETE env var",
            extra={"audit": True, "path": path},
        )
        return

    # In interactive environments, prompt the user
    try:
        answer = input(
            f"[HITL] Confirm DELETE request to '{path}' (this is a destructive operation). "
            f"Type 'yes' to proceed: "
        )
    except (EOFError, OSError):
        # Non-interactive environment and auto-approve not set — deny by default
        raise PermissionError(
            f"DELETE request to '{path}' requires Human-in-the-Loop approval. "
            "Set LANGGRAPH_HITL_AUTO_APPROVE_DELETE=true to allow automated deletions."
        )
    if answer.strip().lower() != "yes":
        raise PermissionError(
            f"DELETE request to '{path}' was not approved by the operator."
        )
    logger.info(
        "HITL approval granted for DELETE",
        extra={"audit": True, "path": path},
    )


class HttpClient:
    """Handle async requests to the LangGraph API.

    Adds additional error messaging & content handling above the
    provided httpx client.

    Attributes:
        client (httpx.AsyncClient): Underlying HTTPX async client.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        _verify_server_authentication(client)
        self.client = client

    async def get(
        self,
        path: str,
        *,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `GET` request."""
        trace_id = _generate_trace_id()
        path = _sanitize_path(path)
        logger.info(
            "MCP request: GET %s",
            path,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": "GET",
                "path": path,
                "params": _truncate_for_log(params),
                "timestamp": time.time(),
            },
        )
        r = await self.client.get(path, params=params, headers=headers)
        logger.info(
            "MCP response: GET %s -> %s",
            path,
            r.status_code,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": "GET",
                "path": path,
                "status_code": r.status_code,
                "timestamp": time.time(),
            },
        )
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        result = _sanitize_response(await _adecode_json(r))
        return result

    async def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | list | None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `POST` request."""
        trace_id = _generate_trace_id()
        path = _sanitize_path(path)
        _sanitize_json_input(json)
        logger.info(
            "MCP request: POST %s",
            path,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": "POST",
                "path": path,
                "params": _truncate_for_log(params),
                "input_hash": hashlib.sha256(repr(json).encode()).hexdigest() if json is not None else None,
                "timestamp": time.time(),
            },
        )
        if json is not None:
            request_headers, content = await _aencode_json(json)
        else:
            request_headers, content = {}, b""
        # Merge headers, with runtime headers taking precedence
        if headers:
            request_headers.update(headers)
        r = await self.client.post(
            path, headers=request_headers, content=content, params=params
        )
        logger.info(
            "MCP response: POST %s -> %s",
            path,
            r.status_code,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": "POST",
                "path": path,
                "status_code": r.status_code,
                "timestamp": time.time(),
            },
        )
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        result = _sanitize_response(await _adecode_json(r))
        return result

    async def put(
        self,
        path: str,
        *,
        json: dict,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `PUT` request."""
        trace_id = _generate_trace_id()
        path = _sanitize_path(path)
        _sanitize_json_input(json)
        logger.info(
            "MCP request: PUT %s",
            path,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": "PUT",
                "path": path,
                "params": _truncate_for_log(params),
                "input_hash": hashlib.sha256(repr(json).encode()).hexdigest(),
                "timestamp": time.time(),
            },
        )
        request_headers, content = await _aencode_json(json)
        if headers:
            request_headers.update(headers)
        r = await self.client.put(
            path, headers=request_headers, content=content, params=params
        )
        logger.info(
            "MCP response: PUT %s -> %s",
            path,
            r.status_code,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": "PUT",
                "path": path,
                "status_code": r.status_code,
                "timestamp": time.time(),
            },
        )
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        result = _sanitize_response(await _adecode_json(r))
        return result

    async def patch(
        self,
        path: str,
        *,
        json: dict,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `PATCH` request."""
        trace_id = _generate_trace_id()
        path = _sanitize_path(path)
        _sanitize_json_input(json)
        logger.info(
            "MCP request: PATCH %s",
            path,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": "PATCH",
                "path": path,
                "params": _truncate_for_log(params),
                "input_hash": hashlib.sha256(repr(json).encode()).hexdigest(),
                "timestamp": time.time(),
            },
        )
        request_headers, content = await _aencode_json(json)
        if headers:
            request_headers.update(headers)
        r = await self.client.patch(
            path, headers=request_headers, content=content, params=params
        )
        logger.info(
            "MCP response: PATCH %s -> %s",
            path,
            r.status_code,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": "PATCH",
                "path": path,
                "status_code": r.status_code,
                "timestamp": time.time(),
            },
        )
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)
        result = _sanitize_response(await _adecode_json(r))
        return result

    async def delete(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> None:
        """Send a `DELETE` request."""
        trace_id = _generate_trace_id()
        path = _sanitize_path(path)
        # HITL approval gate for destructive DELETE operations
        _require_hitl_approval(path, json)
        logger.info(
            "MCP request: DELETE %s",
            path,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": "DELETE",
                "path": path,
                "params": _truncate_for_log(params),
                "timestamp": time.time(),
            },
        )
        r = await self.client.request(
            "DELETE", path, json=json, params=params, headers=headers
        )
        logger.info(
            "MCP response: DELETE %s -> %s",
            path,
            r.status_code,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": "DELETE",
                "path": path,
                "status_code": r.status_code,
                "timestamp": time.time(),
            },
        )
        if on_response:
            on_response(r)
        await _araise_for_status_typed(r)

    async def request_reconnect(
        self,
        path: str,
        method: str,
        *,
        json: dict[str, Any] | None = None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
        reconnect_limit: int = 5,
    ) -> Any:
        """Send a request that automatically reconnects to Location header."""
        trace_id = _generate_trace_id()
        path = _sanitize_path(path)
        _sanitize_json_input(json)
        logger.info(
            "MCP request: %s %s (reconnect)",
            method,
            path,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": method,
                "path": path,
                "reconnect_limit": reconnect_limit,
                "timestamp": time.time(),
            },
        )
        request_headers, content = await _aencode_json(json)
        if headers:
            request_headers.update(headers)
        async with self.client.stream(
            method, path, headers=request_headers, content=content, params=params
        ) as r:
            logger.info(
                "MCP response: %s %s -> %s (reconnect)",
                method,
                path,
                r.status_code,
                extra={
                    "audit": True,
                    "trace_id": trace_id,
                    "method": method,
                    "path": path,
                    "status_code": r.status_code,
                    "timestamp": time.time(),
                },
            )
            if on_response:
                on_response(r)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = (await r.aread()).decode()
                # Truncate error body to avoid leaking sensitive operational details
                truncated_body = body[:_MAX_LOG_BODY_SIZE] + ("...[truncated]" if len(body) > _MAX_LOG_BODY_SIZE else "")
                if sys.version_info >= (3, 11):
                    e.add_note(f"Server error (truncated): {truncated_body}")
                else:
                    logger.error(
                        "Error from langgraph-api (truncated): %s",
                        truncated_body,
                        extra={"audit": True, "trace_id": trace_id},
                    )
                raise e
            loc = r.headers.get("location")
            if reconnect_limit <= 0 or not loc:
                result = _sanitize_response(await _adecode_json(r))
                return result
            _validate_reconnect_location(self.client.base_url, loc)
            _validate_url_allowlist(loc, self.client.base_url)
            try:
                result = _sanitize_response(await _adecode_json(r))
                return result
            except httpx.HTTPError as exc:
                logger.warning(
                    "Request failed during reconnect attempt, retrying to Location: %s",
                    loc,
                    extra={
                        "audit": True,
                        "trace_id": trace_id,
                        "reconnect_location": loc,
                        "error": str(exc),
                        "timestamp": time.time(),
                    },
                )
                warnings.warn(
                    f"Request failed, attempting reconnect to Location: {loc}",
                    stacklevel=2,
                )
                await r.aclose()
                return await self.request_reconnect(
                    loc,
                    "GET",
                    headers=request_headers,
                    # don't pass on_response so it's only called once
                    reconnect_limit=reconnect_limit - 1,
                )

    async def stream(
        self,
        path: str,
        method: str,
        *,
        json: dict[str, Any] | None = None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> AsyncIterator[StreamPart]:
        """Stream results using SSE."""
        trace_id = _generate_trace_id()
        path = _sanitize_path(path)
        _sanitize_json_input(json)
        logger.info(
            "MCP stream request: %s %s",
            method,
            path,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": method,
                "path": path,
                "params": _truncate_for_log(params),
                "input_hash": hashlib.sha256(repr(json).encode()).hexdigest() if json is not None else None,
                "timestamp": time.time(),
            },
        )
        request_headers, content = await _aencode_json(json)
        request_headers["Accept"] = "text/event-stream"
        request_headers["Cache-Control"] = "no-store"
        # Add runtime headers with precedence
        if headers:
            request_headers.update(headers)

        reconnect_headers = {
            key: value
            for key, value in request_headers.items()
            if key.lower() not in {"content-length", "content-type"}
        }

        last_event_id: str | None = None
        reconnect_path: str | None = None
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        event_count = 0

        while True:
            current_headers = dict(
                request_headers if reconnect_path is None else reconnect_headers
            )
            if last_event_id is not None:
                current_headers["Last-Event-ID"] = last_event_id

            current_method = method if reconnect_path is None else "GET"
            current_content = content if reconnect_path is None else None
            current_params = params if reconnect_path is None else None

            retry = False
            async with self.client.stream(
                current_method,
                reconnect_path or path,
                headers=current_headers,
                content=current_content,
                params=current_params,
            ) as res:
                logger.info(
                    "MCP stream response: %s %s -> %s",
                    current_method,
                    reconnect_path or path,
                    res.status_code,
                    extra={
                        "audit": True,
                        "trace_id": trace_id,
                        "method": current_method,
                        "path": reconnect_path or path,
                        "status_code": res.status_code,
                        "timestamp": time.time(),
                    },
                )
                if reconnect_path is None and on_response:
                    on_response(res)
                # check status
                await _araise_for_status_typed(res)
                # check content type
                content_type = res.headers.get("content-type", "").partition(";")[0]
                if "text/event-stream" not in content_type:
                    raise httpx.TransportError(
                        "Expected response header Content-Type to contain 'text/event-stream', "
                        f"got {content_type!r}"
                    )

                reconnect_location = res.headers.get("location")
                if reconnect_location:
                    _validate_reconnect_location(
                        self.client.base_url, reconnect_location
                    )
                    _validate_url_allowlist(reconnect_location, self.client.base_url)
                    reconnect_path = reconnect_location

                # parse SSE
                decoder = SSEDecoder()
                try:
                    async for line in aiter_lines_raw(res):
                        sse = decoder.decode(line=cast("bytes", line).rstrip(b"\n"))
                        if sse is not None:
                            if decoder.last_event_id is not None:
                                last_event_id = decoder.last_event_id
                            if sse.event or sse.data is not None:
                                event_count += 1
                                logger.debug(
                                    "MCP stream event received",
                                    extra={
                                        "audit": True,
                                        "trace_id": trace_id,
                                        "event_count": event_count,
                                        "event_type": sse.event,
                                        "timestamp": time.time(),
                                    },
                                )
                                yield sse
                except httpx.HTTPError as exc:
                    # httpx.TransportError inherits from HTTPError, so transient
                    # disconnects during streaming land here.
                    if reconnect_path is None:
                        logger.error(
                            "MCP stream error (no reconnect path): %s",
                            str(exc),
                            extra={
                                "audit": True,
                                "trace_id": trace_id,
                                "timestamp": time.time(),
                            },
                        )
                        raise
                    logger.warning(
                        "MCP stream transport error, will retry: %s",
                        str(exc),
                        extra={
                            "audit": True,
                            "trace_id": trace_id,
                            "reconnect_path": reconnect_path,
                            "timestamp": time.time(),
                        },
                    )
                    retry = True
                else:
                    if sse := decoder.decode(b""):
                        if decoder.last_event_id is not None:
                            last_event_id = decoder.last_event_id
                        if sse.event or sse.data is not None:
                            # decoder.decode(b"") flushes the in-flight event and may
                            # return an empty placeholder when there is no pending
                            # message. Skip these no-op events so the stream doesn't
                            # emit a trailing blank item after reconnects.
                            event_count += 1
                            logger.debug(
                                "MCP stream final event received",
                                extra={
                                    "audit": True,
                                    "trace_id": trace_id,
                                    "event_count": event_count,
                                    "event_type": sse.event,
                                    "timestamp": time.time(),
                                },
                            )
                            yield sse
            if retry:
                reconnect_attempts += 1
                if reconnect_attempts > max_reconnect_attempts:
                    logger.error(
                        "MCP stream exceeded maximum reconnection attempts",
                        extra={
                            "audit": True,
                            "trace_id": trace_id,
                            "max_reconnect_attempts": max_reconnect_attempts,
                            "timestamp": time.time(),
                        },
                    )
                    raise httpx.TransportError(
                        "Exceeded maximum SSE reconnection attempts"
                    )
                continue
            break
        logger.info(
            "MCP stream completed: %s %s, total events: %d",
            method,
            path,
            event_count,
            extra={
                "audit": True,
                "trace_id": trace_id,
                "method": method,
                "path": path,
                "total_events": event_count,
                "timestamp": time.time(),
            },
        )


async def _aencode_json(json: Any) -> tuple[dict[str, str], bytes | None]:
    if json is None:
        return {}, None
    body = await asyncio.get_running_loop().run_in_executor(
        None,
        orjson.dumps,
        json,
        _orjson_default,
        orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
    )
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, body


async def _adecode_json(r: httpx.Response) -> Any:
    body = await r.aread()
    if not body:
        return None
    data = await asyncio.get_running_loop().run_in_executor(None, orjson.loads, body)
    return data