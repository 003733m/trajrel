"""Structured identifier extraction for trajectory relevance."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Identifier:
    token: str
    kind: str


# Conservative categories that are useful in software-engineering trajectories.
_PATH_RE = re.compile(
    r"(?<![\w.-])"
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
    r"(?![\w.-])"
)

_TEST_RE = re.compile(
    r"\btest_[A-Za-z0-9_]+\b"
)

_EXCEPTION_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b"
)

_UPPER_CONFIG_RE = re.compile(
    r"\b[A-Z][A-Z0-9_]{2,}\b"
)

_SNAKE_RE = re.compile(
    r"\b_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"
)

_CAMEL_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*[a-z][A-Z][A-Za-z0-9]*\b"
)

_REQUEST_ID_RE = re.compile(
    r"\b(?:req|request|trace|call|job|task)[-_][A-Za-z0-9_-]+\b",
    re.IGNORECASE,
)


def _add(
    out: list[Identifier],
    seen: set[tuple[str, str]],
    token: str,
    kind: str,
) -> None:
    token = token.strip()

    if not token:
        return

    key = (token, kind)
    if key in seen:
        return

    seen.add(key)
    out.append(Identifier(token=token, kind=kind))


def extract_identifiers(text: str) -> tuple[Identifier, ...]:
    """Extract conservative structured identifiers from arbitrary text.

    This deliberately favors precision over broad natural-language recall.
    """
    out: list[Identifier] = []
    seen: set[tuple[str, str]] = set()

    for match in _PATH_RE.finditer(text):
        _add(out, seen, match.group(0), "path")

    for match in _TEST_RE.finditer(text):
        _add(out, seen, match.group(0), "test_name")

    for match in _EXCEPTION_RE.finditer(text):
        _add(out, seen, match.group(0), "exception")

    for match in _UPPER_CONFIG_RE.finditer(text):
        _add(out, seen, match.group(0), "config")

    for match in _SNAKE_RE.finditer(text):
        _add(out, seen, match.group(0), "symbol")

    for match in _CAMEL_RE.finditer(text):
        _add(out, seen, match.group(0), "symbol")

    for match in _REQUEST_ID_RE.finditer(text):
        _add(out, seen, match.group(0), "request_id")

    return tuple(out)


def extract_identifier_tokens(text: str) -> tuple[str, ...]:
    """Return unique extracted tokens in stable order."""
    return tuple(dict.fromkeys(identifier.token for identifier in extract_identifiers(text)))
