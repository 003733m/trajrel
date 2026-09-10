"""Historical bridge discovery from prior trajectory evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .events import ToolEvent
from .extractors import extract_identifier_tokens


@dataclass(frozen=True, slots=True)
class BridgeCandidate:
    token: str
    occurrences: int
    distinct_tools: int
    last_seen_index: int
    score: float


def rank_historical_bridges(
    history: tuple[ToolEvent, ...],
) -> tuple[BridgeCandidate, ...]:
    """Rank identifiers grounded in prior tool observations.

    The first generic implementation intentionally uses only observable,
    deterministic evidence:

    - occurrence count,
    - distinct-tool corroboration,
    - recency.

    More detailed KTH scoring will be ported after the generic contract is
    validated.
    """
    occurrences: dict[str, int] = defaultdict(int)
    tools: dict[str, set[str]] = defaultdict(set)
    last_seen: dict[str, int] = {}

    for index, event in enumerate(history):
        for token in extract_identifier_tokens(event.content):
            occurrences[token] += 1
            tools[token].add(event.tool_name)
            last_seen[token] = index

    candidates: list[BridgeCandidate] = []

    n_events = max(len(history), 1)

    for token, count in occurrences.items():
        distinct_tools = len(tools[token])
        recency = (last_seen[token] + 1) / n_events

        # Conservative deterministic score.
        score = (
            float(count)
            + 1.5 * max(0, distinct_tools - 1)
            + 0.5 * recency
        )

        candidates.append(
            BridgeCandidate(
                token=token,
                occurrences=count,
                distinct_tools=distinct_tools,
                last_seen_index=last_seen[token],
                score=score,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            -candidate.distinct_tools,
            -candidate.occurrences,
            candidate.token,
        )
    )

    return tuple(candidates)


def select_historical_bridges(
    history: tuple[ToolEvent, ...],
    *,
    min_occurrences: int = 2,
    min_distinct_tools: int = 1,
    max_bridges: int = 32,
) -> tuple[str, ...]:
    """Select B: historically supported structured identifiers."""
    ranked = rank_historical_bridges(history)

    selected = [
        candidate.token
        for candidate in ranked
        if candidate.occurrences >= min_occurrences
        and candidate.distinct_tools >= min_distinct_tools
    ]

    return tuple(selected[:max_bridges])
