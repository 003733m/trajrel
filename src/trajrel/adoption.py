"""Current reuse and adopted-bridge construction."""

from __future__ import annotations

from .bridges import select_historical_bridges
from .events import ToolEvent
from .extractors import extract_identifier_tokens
from .policies import PolicySignals


def current_reused_identifiers(current_action: str) -> tuple[str, ...]:
    """Extract Q: structured identifiers explicitly reused in the current action."""
    return extract_identifier_tokens(current_action)


def adopted_bridge_identifiers(
    historical_bridges: tuple[str, ...],
    current_identifiers: tuple[str, ...],
) -> tuple[str, ...]:
    """Return B ∩ Q while preserving B's ranking/order."""
    current = set(current_identifiers)

    return tuple(
        identifier
        for identifier in dict.fromkeys(historical_bridges)
        if identifier in current
    )


def build_policy_signals(
    history: tuple[ToolEvent, ...],
    current_action: str,
    *,
    min_occurrences: int = 2,
    min_distinct_tools: int = 1,
    max_bridges: int = 32,
) -> PolicySignals:
    """Build B and Q directly from raw trajectory evidence."""
    historical = select_historical_bridges(
        history,
        min_occurrences=min_occurrences,
        min_distinct_tools=min_distinct_tools,
        max_bridges=max_bridges,
    )

    current = current_reused_identifiers(current_action)

    return PolicySignals(
        historical_bridges=historical,
        current_identifiers=current,
    )
