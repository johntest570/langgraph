"""Synchronous HTTP client for LangGraph API."""

from __future__ import annotations

import base64
import logging
import os
import re
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import Any, cast
from urllib.parse import urlparse

import httpx
import orjson

from langgraph_sdk._shared.utilities import (
    _orjson_default,
    _validate_reconnect_location,
)
from langgraph_sdk.errors import _raise_for_status_typed
from langgraph_sdk.schema import QueryParamTypes, StreamPart
from langgraph_sdk.sse import SSEDecoder, iter_lines_raw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_ALLOWED_SCHEMES = {"http", "https"}

# Patterns that may indicate prompt injection / malicious command content
_SUSPICIOUS_PATTERNS = [
    re.compile(r"(?i)(ignore\s+(previous|above|prior)\s+instructions?)"),
    re.compile(r"(?i)(system\s*prompt|you\s+are\s+now)"),
    re.compile(r"(?i)(<\s*script[\s>])"),
    re.compile(r"(?i)(eval\s*\(|exec\s*\()"),
    re.compile(r"(?i)(\bos\.system\b|\bsubprocess\b|\bshutil\b)"),
    re.compile(r"(?i)(rm\s+-rf|del\s+/[sqf]|format\s+c:)"),
]

_B64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{20,}={0,2}$")


def _validate_url_path(path: str) -> None:
    """Validate that a URL path/URL does not point to a disallowed scheme."""
    if "://" in path:
        parsed = urlparse(path)
        if parsed.scheme and parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            raise ValueError(
                f"URL scheme '{parsed.scheme}' is not allowed. "
                f"Allowed schemes: {_ALLOWED_SCHEMES}"
            )


def _check_suspicious_content(value: str, context: str = "") -> None:
    """Warn if a string value looks like a prompt injection or shell command."""
    for pattern in _SUSPICIOUS_PATTERNS:
        if pattern.search(value):
            logger.warning(
                "Suspicious content detected in %s: matched pattern '%s'",
                context or "input",
                pattern.pattern,
            )
    # Heuristic: long base64-looking strings may encode hidden instructions
    if _B64_PATTERN.match(value.strip()):
        try:
            decoded = base64.b64decode(value.strip()).decode("utf-8", errors="ignore")
            for pattern in _SUSPICIOUS_PATTERNS:
                if pattern.search(decoded):
                    logger.warning(
                        "Suspicious base64-encoded content detected in %s",
                        context or "input",
                    )
                    break
        except Exception:
            pass


def _sanitize_json_payload(payload: Any, context: str = "json payload") -> None:
    """Recursively inspect a JSON-serialisable payload for suspicious content."""
    if isinstance(payload, str):
        _check_suspicious_content(payload, context)
    elif isinstance(payload, dict):
        for k, v in payload.items():
            _sanitize_json_payload(v, context=f"{context}.{k}")
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            _sanitize_json_payload(item, context=f"{context}[{i}]")


def _sanitize_response(data: Any, context: str = "response") -> Any:
    """Validate and sanitise data received from the server."""
    if isinstance(data, str):
        _check_suspicious_content(data, context)
    elif isinstance(data, dict):
        for k, v in data.items():
            _sanitize_response(v, context=f"{context}.{k}")
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            _sanitize_response(item, context=f"{context}[{i}]")
    return data


def _require_hitl_approval(path: str, json: Any) -> None:
    """Human-in-the-Loop approval gate for destructive operations.

    Reads the environment variable ``LANGGRAPH_HITL_APPROVED`` to determine
    whether a human has pre-approved the operation.  In interactive
    environments (TTY) the user is prompted directly.  If neither mechanism
    grants approval the operation is aborted.
    """
    env_approved = os.environ.get("LANGGRAPH_HITL_APPROVED", "").lower()
    if env_approved in {"1", "true", "yes"}:
        logger.info("HITL: DELETE operation approved via environment variable for path: %s", path)
        return

    if sys.stdin.isatty():
        print(
            f"\n[HITL] A DELETE request is about to be sent to: {path}\n"
            f"Payload: {json}\n"
            "Do you approve this operation? [yes/no]: ",
            end="",
            flush=True,
        )
        answer = sys.stdin.readline().strip().lower()
        if answer in {"yes", "y"}:
            logger.info("HITL: DELETE operation approved by user for path: %s", path)
            return
        raise PermissionError(
            f"DELETE operation to '{path}' was rejected by the human operator."
        )

    raise PermissionError(
        f"DELETE operation to '{path}' requires Human-in-the-Loop approval. "
        "Set the environment variable LANGGRAPH_HITL_APPROVED=true to pre-approve, "
        "or run in an interactive terminal."
    )


class SyncHttpClient:
    """Handle synchronous requests to the LangGraph API.

    Provides error messaging and content handling enhancements above the
    underlying httpx client, mirroring the interface of [HttpClient](#HttpClient)
    but for sync usage.

    Attributes:
        client (httpx.Client): Underlying HTTPX sync client.
    """

    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    def get(
        self,
        path: str,
        *,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `GET` request."""
        _validate_url_path(path)
        logger.debug("GET request: path=%s params=%s", path, params)
        r = self.client.get(path, params=params, headers=headers)
        logger.debug("GET response: path=%s status=%s", path, r.status_code)
        if on_response:
            on_response(r)
        _raise_for_status_typed(r)
        result = _decode_json(r)
        return _sanitize_response(result, context=f"GET {path}")

    def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | list | None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `POST` request."""
        _validate_url_path(path)
        if json is not None:
            _sanitize_json_payload(json, context=f"POST {path}")
            request_headers, content = _encode_json(json)
        else:
            request_headers, content = {}, b""
        if headers:
            request_headers.update(headers)
        logger.debug("POST request: path=%s params=%s", path, params)
        r = self.client.post(
            path, headers=request_headers, content=content, params=params
        )
        logger.debug("POST response: path=%s status=%s", path, r.status_code)
        if on_response:
            on_response(r)
        _raise_for_status_typed(r)
        result = _decode_json(r)
        return _sanitize_response(result, context=f"POST {path}")

    def put(
        self,
        path: str,
        *,
        json: dict,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `PUT` request."""
        _validate_url_path(path)
        _sanitize_json_payload(json, context=f"PUT {path}")
        request_headers, content = _encode_json(json)
        if headers:
            request_headers.update(headers)

        logger.debug("PUT request: path=%s params=%s", path, params)
        r = self.client.put(
            path, headers=request_headers, content=content, params=params
        )
        logger.debug("PUT response: path=%s status=%s", path, r.status_code)
        if on_response:
            on_response(r)
        _raise_for_status_typed(r)
        result = _decode_json(r)
        return _sanitize_response(result, context=f"PUT {path}")

    def patch(
        self,
        path: str,
        *,
        json: dict,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a `PATCH` request."""
        _validate_url_path(path)
        _sanitize_json_payload(json, context=f"PATCH {path}")
        request_headers, content = _encode_json(json)
        if headers:
            request_headers.update(headers)
        logger.debug("PATCH request: path=%s params=%s", path, params)
        r = self.client.patch(
            path, headers=request_headers, content=content, params=params
        )
        logger.debug("PATCH response: path=%s status=%s", path, r.status_code)
        if on_response:
            on_response(r)
        _raise_for_status_typed(r)
        result = _decode_json(r)
        return _sanitize_response(result, context=f"PATCH {path}")

    def delete(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> None:
        """Send a `DELETE` request."""
        _validate_url_path(path)
        if json is not None:
            _sanitize_json_payload(json, context=f"DELETE {path}")
        _require_hitl_approval(path, json)
        logger.debug("DELETE request: path=%s params=%s", path, params)
        r = self.client.request(
            "DELETE", path, json=json, params=params, headers=headers
        )
        logger.debug("DELETE response: path=%s status=%s", path, r.status_code)
        if on_response:
            on_response(r)
        _raise_for_status_typed(r)

    def request_reconnect(
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
        _validate_url_path(path)
        if json is not None:
            _sanitize_json_payload(json, context=f"{method} {path}")
        request_headers, content = _encode_json(json)
        if headers:
            request_headers.update(headers)
        logger.debug("request_reconnect: method=%s path=%s params=%s", method, path, params)
        with self.client.stream(
            method, path, headers=request_headers, content=content, params=params
        ) as r:
            logger.debug("request_reconnect response: method=%s path=%s status=%s", method, path, r.status_code)
            if on_response:
                on_response(r)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = r.read().decode()
                if sys.version_info >= (3, 11):
                    e.add_note(body)
                else:
                    logger.error(f"Error from langgraph-api: {body}", exc_info=e)
                raise e
            loc = r.headers.get("location")
            if reconnect_limit <= 0 or not loc:
                result = _decode_json(r)
                return _sanitize_response(result, context=f"{method} {path}")
            _validate_reconnect_location(self.client.base_url, loc)
            try:
                result = _decode_json(r)
                return _sanitize_response(result, context=f"{method} {path}")
            except httpx.HTTPError:
                warnings.warn(
                    f"Request failed, attempting reconnect to Location: {loc}",
                    stacklevel=2,
                )
                r.close()
                return self.request_reconnect(
                    loc,
                    "GET",
                    headers=request_headers,
                    # don't pass on_response so it's only called once
                    reconnect_limit=reconnect_limit - 1,
                )

    def stream(
        self,
        path: str,
        method: str,
        *,
        json: dict[str, Any] | None = None,
        params: QueryParamTypes | None = None,
        headers: Mapping[str, str] | None = None,
        on_response: Callable[[httpx.Response], None] | None = None,
    ) -> Iterator[StreamPart]:
        """Stream the results of a request using SSE."""
        _validate_url_path(path)
        if json is not None:
            _sanitize_json_payload(json, context=f"{method} {path} (stream)")
            request_headers, content = _encode_json(json)
        else:
            request_headers, content = {}, None
        request_headers["Accept"] = "text/event-stream"
        request_headers["Cache-Control"] = "no-store"
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

        while True:
            current_headers = dict(
                request_headers if reconnect_path is None else reconnect_headers
            )
            if last_event_id is not None:
                current_headers["Last-Event-ID"] = last_event_id

            current_method = method if reconnect_path is None else "GET"
            current_content = content if reconnect_path is None else None
            current_params = params if reconnect_path is None else None

            current_path = reconnect_path or path
            _validate_url_path(current_path)

            logger.debug(
                "stream request: method=%s path=%s params=%s",
                current_method,
                current_path,
                current_params,
            )

            retry = False
            with self.client.stream(
                current_method,
                current_path,
                headers=current_headers,
                content=current_content,
                params=current_params,
            ) as res:
                logger.debug(
                    "stream response: method=%s path=%s status=%s",
                    current_method,
                    current_path,
                    res.status_code,
                )
                if reconnect_path is None and on_response:
                    on_response(res)
                # check status
                _raise_for_status_typed(res)
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
                    reconnect_path = reconnect_location

                decoder = SSEDecoder()
                try:
                    for line in iter_lines_raw(res):
                        sse = decoder.decode(cast(bytes, line).rstrip(b"\n"))
                        if sse is not None:
                            if decoder.last_event_id is not None:
                                last_event_id = decoder.last_event_id
                            if sse.event or sse.data is not None:
                                # Sanitise SSE event data before yielding
                                if isinstance(sse.data, str):
                                    _check_suspicious_content(
                                        sse.data,
                                        context=f"SSE event data from {current_path}",
                                    )
                                logger.debug(
                                    "stream SSE event: path=%s event=%s",
                                    current_path,
                                    sse.event,
                                )
                                yield sse
                except httpx.HTTPError:
                    # httpx.TransportError inherits from HTTPError, so transient
                    # disconnects during streaming land here.
                    if reconnect_path is None:
                        raise
                    retry = True
                else:
                    if sse := decoder.decode(b""):
                        if decoder.last_event_id is not None:
                            last_event_id = decoder.last_event_id
                        if sse.event or sse.data is not None:
                            # See async stream implementation for rationale on
                            # skipping empty flush events.
                            if isinstance(sse.data, str):
                                _check_suspicious_content(
                                    sse.data,
                                    context=f"SSE flush event data from {current_path}",
                                )
                            logger.debug(
                                "stream SSE flush event: path=%s event=%s",
                                current_path,
                                sse.event,
                            )
                            yield sse
            if retry:
                reconnect_attempts += 1
                if reconnect_attempts > max_reconnect_attempts:
                    raise httpx.TransportError(
                        "Exceeded maximum SSE reconnection attempts"
                    )
                continue
            break


def _encode_json(json: Any) -> tuple[dict[str, str], bytes]:
    body = orjson.dumps(
        json,
        _orjson_default,
        orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
    )
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, body


def _decode_json(r: httpx.Response) -> Any:
    body = r.read()
    return orjson.loads(body) if body else None