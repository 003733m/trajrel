"""Controlled component ablations for TrajRel."""

from .registry import (
    SCORER_ABLATIONS,
    get_scorer_ablation,
)
from .scorer import (
    AblationSpec,
    rank_ablation_outputs,
)

__all__ = [
    "SCORER_ABLATIONS",
    "AblationSpec",
    "get_scorer_ablation",
    "rank_ablation_outputs",
]
