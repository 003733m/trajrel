"""Faithful generic port of the frozen KTH recruitment bridge scorer.

Reference implementation:
7c4fe80c19b3a9537f395f515a64432e41ee1d43

This module preserves the reference scorer's behavior so later ablations can
change one component at a time rather than mixing refactoring with algorithmic
changes.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from trajrel.events import ToolEvent

DEFAULT_TOP_K = 6

TARGET_CORROBORATION_KINDS = frozenset(
    {
        "test_name",
        "request_id",
        "exception",
    }
)


@dataclass(frozen=True, slots=True)
class ReferenceScoringConfig:
    causal_weight: float = 4.0
    cross_tool_weight: float = 2.0
    occurrence_weight: float = 2.0
    recency_weight: float = 1.0
    speculation_weight: float = 2.0

    max_cross_tool_steps: int = 2
    max_occurrences: int = 3

    min_score: float | None = 2.0


@dataclass(frozen=True, slots=True)
class ReferenceBridgeCandidate:
    token: str
    kind: str
    score: float
    distinct_tools: int
    occurrences: int
    positive_evidence: int
    negative_evidence: int
    recency: float


@dataclass(frozen=True, slots=True)
class _Hit:
    token: str
    kind: str
    line_no: int


_PATTERN_SPECS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    (
        "file_path",
        re.compile(
            r"(?<![\w.-])"
            r"((?:[A-Za-z0-9_.-]+/)+"
            r"[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)"
        ),
        1,
    ),
    (
        "request_id",
        re.compile(
            r"\b("
            r"(?:req|request|trace|session|job|task)"
            r"[-_][A-Za-z0-9][A-Za-z0-9_-]*"
            r")\b",
            re.IGNORECASE,
        ),
        1,
    ),
    (
        "test_name",
        re.compile(r"\b(test_[A-Za-z0-9_]+)\b"),
        1,
    ),
    (
        "exception",
        re.compile(
            r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception))\b"
        ),
        1,
    ),
    (
        "config_key",
        re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b"),
        1,
    ),
    (
        "function_name",
        re.compile(r"\b([a-z_][a-z0-9_]{2,})\s*\("),
        1,
    ),
    (
        "class_name",
        re.compile(
            r"\b([A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+)\b"
        ),
        1,
    ),
)


_POSITIVE_CUES: dict[str, int] = {
    "root cause": 10,
    "affected": 9,
    "confirmed": 8,
    "identified": 8,
    "failed": 7,
    "failure": 7,
    "exception raised": 7,
    "traceback": 6,
    "assertion": 6,
    "expected": 4,
    "actual": 4,
}


_NEGATIVE_CUES: dict[str, int] = {
    "suspicion": 5,
    "suspect": 5,
    "hypothesis": 5,
    "possible": 4,
    "maybe": 4,
    "candidate": 3,
}


_MAX_POSITIVE_CUE = max(_POSITIVE_CUES.values())
_MAX_NEGATIVE_CUE = max(_NEGATIVE_CUES.values())


def _find_candidates(text: str) -> list[_Hit]:
    hits: list[_Hit] = []

    for line_no, line in enumerate(text.splitlines()):
        occupied: list[tuple[int, int]] = []

        for kind, pattern, group in _PATTERN_SPECS:
            for match in pattern.finditer(line):
                start, end = match.span(group)

                if any(
                    not (end <= old_start or start >= old_end)
                    for old_start, old_end in occupied
                ):
                    continue

                token = match.group(group)
                if not token:
                    continue

                occupied.append((start, end))
                hits.append(
                    _Hit(
                        token=token,
                        kind=kind,
                        line_no=line_no,
                    )
                )

    return hits


def _context_evidence(
    line: str,
    token: str,
) -> tuple[int, int]:
    masked = line.replace(
        token,
        " [BRIDGE] ",
        1,
    ).lower()

    positive = max(
        (
            score
            for cue, score in _POSITIVE_CUES.items()
            if cue in masked
        ),
        default=0,
    )

    negative = max(
        (
            score
            for cue, score in _NEGATIVE_CUES.items()
            if cue in masked
        ),
        default=0,
    )

    return positive, negative


def rank_headroom_reference_outputs(
    tool_outputs: Sequence[str],
    *,
    top_k: int | None = DEFAULT_TOP_K,
    scoring: ReferenceScoringConfig | None = None,
    provisional_kinds: frozenset[str] | None = None,
) -> tuple[ReferenceBridgeCandidate, ...]:
    """Rank identifiers exactly as the frozen reference scorer did."""

    scoring = scoring or ReferenceScoringConfig()

    if not tool_outputs:
        return ()

    if top_k is not None and top_k <= 0:
        return ()

    occurrence_count: Counter[str] = Counter()
    tool_presence: dict[str, set[int]] = defaultdict(set)
    best_positive: dict[str, int] = defaultdict(int)
    strongest_negative: dict[str, int] = defaultdict(int)
    last_seen_tool: dict[str, int] = {}
    token_kind: dict[str, str] = {}

    n_tools = len(tool_outputs)

    for tool_index, text in enumerate(tool_outputs):
        lines = text.splitlines()

        for hit in _find_candidates(text):
            token = hit.token

            occurrence_count[token] += 1

            # Historical naming retained for parity: "distinct_tools"
            # means distinct prior output positions in the reference code.
            tool_presence[token].add(tool_index)

            last_seen_tool[token] = tool_index
            token_kind[token] = hit.kind

            line = lines[hit.line_no]

            positive, negative = _context_evidence(
                line,
                token,
            )

            best_positive[token] = max(
                best_positive[token],
                positive,
            )

            strongest_negative[token] = max(
                strongest_negative[token],
                negative,
            )

    ranked: list[ReferenceBridgeCandidate] = []

    for token, occurrences in occurrence_count.items():
        distinct_tools = len(tool_presence[token])
        positive = best_positive[token]
        negative = strongest_negative[token]

        provisional = (
            provisional_kinds is not None
            and token_kind[token] in provisional_kinds
            and distinct_tools == 1
            and positive <= 0
        )

        if distinct_tools < 2 and positive <= 0 and not provisional:
            continue

        cross_tool_steps = min(
            max(distinct_tools - 1, 0),
            scoring.max_cross_tool_steps,
        )

        bounded_occurrences = min(
            occurrences,
            scoring.max_occurrences,
        )

        if n_tools > 1:
            recency = last_seen_tool[token] / (n_tools - 1)
        else:
            recency = 0.0

        causal_signal = (
            float(positive)
            / float(_MAX_POSITIVE_CUE)
        )

        if scoring.max_cross_tool_steps > 0:
            cross_tool_signal = (
                float(cross_tool_steps)
                / float(scoring.max_cross_tool_steps)
            )
        else:
            cross_tool_signal = 0.0

        if scoring.max_occurrences > 0:
            occurrence_signal = (
                float(bounded_occurrences)
                / float(scoring.max_occurrences)
            )
        else:
            occurrence_signal = 0.0

        speculation_signal = (
            float(negative)
            / float(_MAX_NEGATIVE_CUE)
        )

        score = (
            scoring.causal_weight * causal_signal
            + scoring.cross_tool_weight * cross_tool_signal
            + scoring.occurrence_weight * occurrence_signal
            + scoring.recency_weight * recency
            - scoring.speculation_weight * speculation_signal
        )

        if (
            scoring.min_score is not None
            and score < scoring.min_score
            and not provisional
        ):
            continue

        ranked.append(
            ReferenceBridgeCandidate(
                token=token,
                kind=token_kind[token],
                score=score,
                distinct_tools=distinct_tools,
                occurrences=occurrences,
                positive_evidence=positive,
                negative_evidence=negative,
                recency=recency,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.score,
            -item.distinct_tools,
            -item.recency,
            item.token,
        )
    )

    if top_k is None:
        return tuple(ranked)

    return tuple(ranked[:top_k])


def rank_headroom_reference(
    history: Sequence[ToolEvent],
    *,
    top_k: int | None = DEFAULT_TOP_K,
    scoring: ReferenceScoringConfig | None = None,
    provisional_kinds: frozenset[str] | None = None,
) -> tuple[ReferenceBridgeCandidate, ...]:
    """Apply the frozen scorer to generic TrajRel history."""
    return rank_headroom_reference_outputs(
        [event.content for event in history],
        top_k=top_k,
        scoring=scoring,
        provisional_kinds=provisional_kinds,
    )
