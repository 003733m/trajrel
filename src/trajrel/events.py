"""Generic trajectory and benchmark data structures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """One grounded event observed during an agent trajectory."""

    tool_name: str
    content: str
    call_id: str | None = None
    arguments: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class CompressionOpportunity:
    """One candidate segment presented to a context-management policy.

    The raw benchmark sample intentionally stores trajectory/action evidence,
    rather than precomputing TrajRel's B, Q, or B-intersection-Q decisions.
    """

    sample_id: str
    history: tuple[ToolEvent, ...]
    current_action: str
    segment: str
    base_keep: bool
    critical: bool
    token_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
