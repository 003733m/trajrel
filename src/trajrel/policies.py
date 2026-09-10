"""Generic monotonic preservation policies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .events import CompressionOpportunity


@dataclass(frozen=True, slots=True)
class PolicySignals:
    """Signals computed from raw trajectory state at evaluation time."""

    historical_bridges: tuple[str, ...] = ()
    current_identifiers: tuple[str, ...] = ()

    @property
    def adopted_bridges(self) -> tuple[str, ...]:
        """B ∩ Q, preserving historical bridge order."""
        current = set(self.current_identifiers)
        return tuple(
            identifier
            for identifier in dict.fromkeys(self.historical_bridges)
            if identifier in current
        )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Decision returned by one preservation policy."""

    policy: str
    keep: bool
    forced: bool
    matched_identifiers: tuple[str, ...] = ()


class PreservationPolicy(Protocol):
    name: str

    def decide(
        self,
        opportunity: CompressionOpportunity,
        signals: PolicySignals,
    ) -> PolicyDecision: ...


def _contains_identifier(text: str, identifier: str) -> bool:
    """Conservative exact-boundary lexical match.

    Mirrors the preservation semantics used by the original proof of concept:
    identifier characters cannot be immediately preceded/followed by another
    ASCII alphanumeric or underscore character.
    """
    if not identifier:
        return False

    pattern = (
        rf"(?<![A-Za-z0-9_])"
        rf"{re.escape(identifier)}"
        rf"(?![A-Za-z0-9_])"
    )
    return re.search(pattern, text) is not None


def _matching_identifiers(
    segment: str,
    identifiers: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        identifier
        for identifier in dict.fromkeys(identifiers)
        if _contains_identifier(segment, identifier)
    )


def _apply_monotonic_floor(
    *,
    policy_name: str,
    opportunity: CompressionOpportunity,
    identifiers: tuple[str, ...],
) -> PolicyDecision:
    # Monotonicity: a base KEEP can never become DROP.
    if opportunity.base_keep:
        return PolicyDecision(
            policy=policy_name,
            keep=True,
            forced=False,
        )

    matched = _matching_identifiers(opportunity.segment, identifiers)

    if matched:
        return PolicyDecision(
            policy=policy_name,
            keep=True,
            forced=True,
            matched_identifiers=matched,
        )

    return PolicyDecision(
        policy=policy_name,
        keep=False,
        forced=False,
    )


class BasePolicy:
    """Unmodified compressor decision."""

    name = "base"

    def decide(
        self,
        opportunity: CompressionOpportunity,
        signals: PolicySignals,
    ) -> PolicyDecision:
        del signals
        return PolicyDecision(
            policy=self.name,
            keep=opportunity.base_keep,
            forced=False,
        )


class HistoryOnlyPolicy:
    """Preserve a dropped segment when it contains any bridge in B."""

    name = "history_only"

    def decide(
        self,
        opportunity: CompressionOpportunity,
        signals: PolicySignals,
    ) -> PolicyDecision:
        return _apply_monotonic_floor(
            policy_name=self.name,
            opportunity=opportunity,
            identifiers=signals.historical_bridges,
        )


class QueryOnlyPolicy:
    """Preserve a dropped segment when it contains any identifier in Q."""

    name = "query_only"

    def decide(
        self,
        opportunity: CompressionOpportunity,
        signals: PolicySignals,
    ) -> PolicyDecision:
        return _apply_monotonic_floor(
            policy_name=self.name,
            opportunity=opportunity,
            identifiers=signals.current_identifiers,
        )


class AdoptedBridgePolicy:
    """Preserve a dropped segment only for historically supported reuse, B ∩ Q."""

    name = "adopted_bridge"

    def decide(
        self,
        opportunity: CompressionOpportunity,
        signals: PolicySignals,
    ) -> PolicyDecision:
        return _apply_monotonic_floor(
            policy_name=self.name,
            opportunity=opportunity,
            identifiers=signals.adopted_bridges,
        )
