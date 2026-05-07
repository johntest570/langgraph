"""Utilities for working with embedding functions and LangChain's Embeddings interface.

This module provides tools to wrap arbitrary embedding functions (both sync and async)
into LangChain's Embeddings interface. This enables using custom embedding functions
with LangChain-compatible tools while maintaining support for both synchronous and
asynchronous operations.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import re
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Approved model registry
# ---------------------------------------------------------------------------

#: Allowlist of approved embedding model identifiers (provider:model-version).
#: Only models listed here may be loaded via ensure_embeddings().
APPROVED_EMBEDDING_MODELS: frozenset[str] = frozenset()
"""Set of approved embedding model identifiers.

This set is intentionally empty by default because neither LangChain nor the
RAG embedding model are on the organisation's approved list.  Operators must
populate this set with approved, version-pinned identifiers before passing a
string to ensure_embeddings().
"""

# ---------------------------------------------------------------------------
# Minimal Embeddings protocol (replaces langchain_core dependency)
# ---------------------------------------------------------------------------


@runtime_checkable
class Embeddings(Protocol):
    """Minimal protocol mirroring LangChain's Embeddings interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    async def aembed_query(self, text: str) -> list[float]:
        ...


EmbeddingsFunc = Callable[[Sequence[str]], list[list[float]]]
"""Type for synchronous embedding functions.

The function should take a sequence of strings and return a list of embeddings,
where each embedding is a list of floats. The dimensionality of the embeddings
should be consistent for all inputs.
"""

AEmbeddingsFunc = Callable[[Sequence[str]], Awaitable[list[list[float]]]]
"""Type for asynchronous embedding functions.

Similar to EmbeddingsFunc, but returns an awaitable that resolves to the embeddings.
"""

# ---------------------------------------------------------------------------
# Input sanitization / validation helpers
# ---------------------------------------------------------------------------

# Maximum allowed length for a single text input (characters).
_MAX_TEXT_LENGTH: int = 32_768

# Patterns that indicate potentially malicious prompt content.
_SHELL_COMMAND_PATTERN = re.compile(
    r"(?:^|\s)(?:sudo|rm\s+-rf|chmod|chown|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat|eval|exec)\b",
    re.IGNORECASE | re.MULTILINE,
)

_INJECTION_PATTERN = re.compile(
    r"(?:ignore\s+(?:previous|above|prior)\s+instructions?|"
    r"disregard\s+(?:previous|above|prior)\s+instructions?|"
    r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbreak)|"
    r"act\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:an?\s+)?(?:evil|unrestricted|unfiltered)|"
    r"system\s*:\s*you\s+are)",
    re.IGNORECASE | re.DOTALL,
)

# Detect strings that are suspiciously large base64 blobs (potential binary payloads).
_BASE64_BLOB_PATTERN = re.compile(r"(?:[A-Za-z0-9+/]{76,}\n?){4,}={0,2}")


def _sanitize_texts(texts: Sequence[str]) -> list[str]:
    """Validate and sanitize a sequence of text inputs before embedding.

    Raises:
        ValueError: If any text fails validation.

    Returns:
        The sanitized list of texts (strings are stripped of leading/trailing
        whitespace; no other mutation is performed so embeddings remain
        semantically equivalent).
    """
    sanitized: list[str] = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(
                f"Input at index {i} is not a string (got {type(text).__name__!r}). "
                "All inputs must be plain text strings."
            )

        # Normalise unicode to NFC to prevent homoglyph / encoding tricks.
        text = unicodedata.normalize("NFC", text)

        # Strip leading/trailing whitespace.
        text = text.strip()

        if len(text) == 0:
            raise ValueError(
                f"Input at index {i} is empty after stripping whitespace. "
                "Empty strings are not valid embedding inputs."
            )

        if len(text) > _MAX_TEXT_LENGTH:
            raise ValueError(
                f"Input at index {i} exceeds the maximum allowed length of "
                f"{_MAX_TEXT_LENGTH} characters (got {len(text)})."
            )

        # Check for null bytes (common in binary / exploit payloads).
        if "\x00" in text:
            raise ValueError(
                f"Input at index {i} contains null bytes, which are not permitted."
            )

        # Check for shell-command-like content.
        if _SHELL_COMMAND_PATTERN.search(text):
            raise ValueError(
                f"Input at index {i} appears to contain shell commands or "
                "executable directives, which are not permitted."
            )

        # Check for prompt-injection patterns.
        if _INJECTION_PATTERN.search(text):
            raise ValueError(
                f"Input at index {i} appears to contain prompt-injection content, "
                "which is not permitted."
            )

        # Check for large base64 blobs (potential binary executable payloads).
        if _BASE64_BLOB_PATTERN.search(text):
            # Attempt to decode and check for ELF/PE magic bytes.
            for match in _BASE64_BLOB_PATTERN.finditer(text):
                try:
                    decoded = base64.b64decode(match.group(0).replace("\n", ""))
                    if decoded[:4] in (b"\x7fELF", b"MZ\x90\x00", b"MZ"):
                        raise ValueError(
                            f"Input at index {i} contains a base64-encoded binary "
                            "executable, which is not permitted."
                        )
                except Exception as exc:
                    if "not permitted" in str(exc):
                        raise
                    # Decoding failed — not a valid base64 blob; ignore.

        sanitized.append(text)

    return sanitized


# ---------------------------------------------------------------------------
# Registry validation helper
# ---------------------------------------------------------------------------


def _validate_model_identifier(identifier: str) -> None:
    """Raise ValueError if *identifier* is not in the approved model registry.

    Args:
        identifier: A provider:model-version string, e.g. 'openai:text-embedding-3-small'.

    Raises:
        ValueError: Always, because the approved registry is empty by default
            (neither LangChain nor the RAG embedding model are approved).
    """
    if identifier not in APPROVED_EMBEDDING_MODELS:
        raise ValueError(
            f"Embedding model identifier '{identifier}' is not in the organisation's "
            "approved model registry. "
            "Only version-pinned, approved models may be used. "
            "Update APPROVED_EMBEDDING_MODELS with an approved, version-pinned "
            "identifier before calling ensure_embeddings()."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_embeddings(
    embed: Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str | None,
) -> "EmbeddingsLambda | Embeddings":
    """Ensure that an embedding function conforms to the Embeddings interface.

    This function wraps arbitrary embedding functions to make them compatible with
    the Embeddings interface. It handles both synchronous and asynchronous
    functions.

    Args:
        embed: Either an existing Embeddings instance, or a function that converts
            text to embeddings. If the function is async, it will be used for both
            sync and async operations.
            String identifiers are validated against the approved model registry
            before use.

    Returns:
        An Embeddings instance that wraps the provided function(s).

    Raises:
        ValueError: If *embed* is None, if a string identifier is not in the
            approved registry, or if the required dependencies are unavailable.

    ??? example "Examples"

        Wrap a synchronous embedding function:

        ```python
        def my_embed_fn(texts):
            return [[0.1, 0.2] for _ in texts]

        embeddings = ensure_embeddings(my_embed_fn)
        result = embeddings.embed_query("hello")  # Returns [0.1, 0.2]
        ```

        Wrap an asynchronous embedding function:

        ```python
        async def my_async_fn(texts):
            return [[0.1, 0.2] for _ in texts]

        embeddings = ensure_embeddings(my_async_fn)
        result = await embeddings.aembed_query("hello")  # Returns [0.1, 0.2]
        ```
    """
    if embed is None:
        raise ValueError("embed must be provided")

    if isinstance(embed, str):
        # Validate against the approved registry before attempting to load.
        _validate_model_identifier(embed)

        # The block below is intentionally unreachable when the registry is
        # empty (the default).  It is retained so that operators who populate
        # APPROVED_EMBEDDING_MODELS can still use string identifiers, but the
        # unapproved langchain.embeddings.init_embeddings path is blocked.
        raise ValueError(
            f"Loading embedding models via string identifiers is disabled. "
            f"Identifier '{embed}' was validated against the approved registry "
            "but the string-based loader has been disabled because LangChain "
            "is not on the organisation's approved model list. "
            "Provide an approved Embeddings instance or callable directly."
        )

    if isinstance(embed, Embeddings):
        return embed  # type: ignore[return-value]

    return EmbeddingsLambda(embed)


class EmbeddingsLambda:
    """Wrapper to convert embedding functions into the Embeddings interface.

    This class allows arbitrary embedding functions to be used with
    Embeddings-compatible tools. It supports both synchronous and asynchronous
    operations, and can handle:
    1. A synchronous function for sync operations (async operations will use sync function)
    2. An async function for both sync/async operations (sync operations will raise an error)

    All text inputs are sanitized and validated before being passed to the
    underlying embedding function.

    The embedding functions should convert text into fixed-dimensional vectors that
    capture the semantic meaning of the text.

    Args:
        func: Function that converts text to embeddings. Can be sync or async.
            If async, it will be used for async operations, but sync operations
            will raise an error. If sync, it will be used for both sync and async operations.

    ??? example "Examples"

        With a sync function:

        ```python
        def my_embed_fn(texts):
            # Return 2D embeddings for each text
            return [[0.1, 0.2] for _ in texts]

        embeddings = EmbeddingsLambda(my_embed_fn)
        result = embeddings.embed_query("hello")  # Returns [0.1, 0.2]
        await embeddings.aembed_query("hello")  # Also returns [0.1, 0.2]
        ```

        With an async function:

        ```python
        async def my_async_fn(texts):
            return [[0.1, 0.2] for _ in texts]

        embeddings = EmbeddingsLambda(my_async_fn)
        await embeddings.aembed_query("hello")  # Returns [0.1, 0.2]
        # Note: embed_query() would raise an error
        ```
    """

    def __init__(
        self,
        func: EmbeddingsFunc | AEmbeddingsFunc,
    ) -> None:
        if func is None:
            raise ValueError("func must be provided")
        if _is_async_callable(func):
            self.afunc = func
        else:
            self.func = func

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors.

        Args:
            texts: list of texts to convert to embeddings.

        Returns:
            list of embeddings, one per input text. Each embedding is a list of floats.

        Raises:
            ValueError: If the instance was initialized with only an async function,
                or if any input text fails validation.
        """
        func = getattr(self, "func", None)
        if func is None:
            raise ValueError(
                "EmbeddingsLambda was initialized with an async function but no sync function. "
                "Use aembed_documents for async operation or provide a sync function."
            )
        sanitized = _sanitize_texts(texts)
        return func(sanitized)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single piece of text.

        Args:
            text: Text to convert to an embedding.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            ValueError: If the input text fails validation.

        Note:
            This is equivalent to calling embed_documents with a single text
            and taking the first result.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed a list of texts into vectors.

        Args:
            texts: list of texts to convert to embeddings.

        Returns:
            list of embeddings, one per input text. Each embedding is a list of floats.

        Raises:
            ValueError: If any input text fails validation.

        Note:
            If no async function was provided, this falls back to the sync implementation.
        """
        sanitized = _sanitize_texts(texts)
        afunc = getattr(self, "afunc", None)
        if afunc is None:
            # Fall back to sync implementation via a thread executor.
            func = getattr(self, "func", None)
            if func is None:
                raise ValueError(
                    "EmbeddingsLambda has neither a sync nor an async function."
                )
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, sanitized)
        return await afunc(sanitized)

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously embed a single piece of text.

        Args:
            text: Text to convert to an embedding.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            ValueError: If the input text fails validation.

        Note:
            This is equivalent to calling aembed_documents with a single text
            and taking the first result.
        """
        afunc = getattr(self, "afunc", None)
        if afunc is None:
            return (await self.aembed_documents([text]))[0]
        sanitized = _sanitize_texts([text])
        return (await afunc(sanitized))[0]


def get_text_at_path(obj: Any, path: str | list[str]) -> list[str]:
    """Extract text from an object using a path expression or pre-tokenized path.

    Args:
        obj: The object to extract text from
        path: Either a path string or pre-tokenized path list.

    !!! info "Path types handled"
        - Simple paths: "field1.field2"
        - Array indexing: "[0]", "[*]", "[-1]"
        - Wildcards: "*"
        - Multi-field selection: "{field1,field2}"
        - Nested paths in multi-field: "{field1,nested.field2}"
    """
    if not path or path == "$":
        return [json.dumps(obj, sort_keys=True, ensure_ascii=False)]

    tokens = tokenize_path(path) if isinstance(path, str) else path

    def _extract_from_obj(obj: Any, tokens: list[str], pos: int) -> list[str]:
        if pos >= len(tokens):
            if isinstance(obj, (str, int, float, bool)):
                return [str(obj)]
            elif obj is None:
                return []
            elif isinstance(obj, (list, dict)):
                return [json.dumps(obj, sort_keys=True, ensure_ascii=False)]
            return []

        token = tokens[pos]
        results = []

        if token.startswith("[") and token.endswith("]"):
            if not isinstance(obj, list):
                return []

            index = token[1:-1]
            if index == "*":
                for item in obj:
                    results.extend(_extract_from_obj(item, tokens, pos + 1))
            else:
                try:
                    idx = int(index)
                    if idx < 0:
                        idx = len(obj) + idx
                    if 0 <= idx < len(obj):
                        results.extend(_extract_from_obj(obj[idx], tokens, pos + 1))
                except (ValueError, IndexError):
                    return []

        elif token.startswith("{") and token.endswith("}"):
            if not isinstance(obj, dict):
                return []

            fields = [f.strip() for f in token[1:-1].split(",")]
            for field in fields:
                nested_tokens = tokenize_path(field)
                if nested_tokens:
                    current_obj: dict | None = obj
                    for nested_token in nested_tokens:
                        if (
                            isinstance(current_obj, dict)
                            and nested_token in current_obj
                        ):
                            current_obj = current_obj[nested_token]
                        else:
                            current_obj = None
                            break
                    if current_obj is not None:
                        if isinstance(current_obj, (str, int, float, bool)):
                            results.append(str(current_obj))
                        elif isinstance(current_obj, (list, dict)):
                            results.append(
                                json.dumps(
                                    current_obj, sort_keys=True, ensure_ascii=False
                                )
                            )

        # Handle wildcard
        elif token == "*":
            if isinstance(obj, dict):
                for value in obj.values():
                    results.extend(_extract_from_obj(value, tokens, pos + 1))
            elif isinstance(obj, list):
                for item in obj:
                    results.extend(_extract_from_obj(item, tokens, pos + 1))

        # Handle regular field
        else:
            if isinstance(obj, dict) and token in obj:
                results.extend(_extract_from_obj(obj[token], tokens, pos + 1))

        return results

    return _extract_from_obj(obj, tokens, 0)


# Private utility functions


def tokenize_path(path: str) -> list[str]:
    """Tokenize a path into components.

    !!! info "Types handled"
        - Simple paths: "field1.field2"
        - Array indexing: "[0]", "[*]", "[-1]"
        - Wildcards: "*"
        - Multi-field selection: "{field1,field2}"
    """
    if not path:
        return []

    tokens = []
    current: list[str] = []
    i = 0
    while i < len(path):
        char = path[i]

        if char == "[":  # Handle array index
            if current:
                tokens.append("".join(current))
                current = []
            bracket_count = 1
            index_chars = ["["]
            i += 1
            while i < len(path) and bracket_count > 0:
                if path[i] == "[":
                    bracket_count += 1
                elif path[i] == "]":
                    bracket_count -= 1
                index_chars.append(path[i])
                i += 1
            tokens.append("".join(index_chars))
            continue

        elif char == "{":  # Handle multi-field selection
            if current:
                tokens.append("".join(current))
                current = []
            brace_count = 1
            field_chars = ["{"]
            i += 1
            while i < len(path) and brace_count > 0:
                if path[i] == "{":
                    brace_count += 1
                elif path[i] == "}":
                    brace_count -= 1
                field_chars.append(path[i])
                i += 1
            tokens.append("".join(field_chars))
            continue

        elif char == ".":  # Handle regular field
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
        i += 1

    if current:
        tokens.append("".join(current))

    return tokens


def _is_async_callable(
    func: Any,
) -> bool:
    """Check if a function is async.

    This includes both async def functions and classes with async __call__ methods.

    Args:
        func: Function or callable object to check.

    Returns:
        True if the function is async, False otherwise.
    """
    return (
        asyncio.iscoroutinefunction(func)
        or hasattr(func, "__call__")  # noqa: B004
        and asyncio.iscoroutinefunction(func.__call__)
    )


@functools.lru_cache
def _get_init_embeddings() -> None:
    """Disabled: loading embeddings via LangChain is not permitted.

    LangChain and the RAG embedding model are not on the organisation's
    approved model registry.  This function always returns None to prevent
    accidental use of the string-based loader.
    """
    return None


__all__ = [
    "ensure_embeddings",
    "EmbeddingsFunc",
    "AEmbeddingsFunc",
    "APPROVED_EMBEDDING_MODELS",
]