"""Target-conditioned bridge selectors."""

from .headroom_reference import (
    ReferenceTargetBridgeCandidate,
    select_headroom_reference_candidates,
    target_specificity,
)

__all__ = [
    "ReferenceTargetBridgeCandidate",
    "select_headroom_reference_candidates",
    "target_specificity",
]
