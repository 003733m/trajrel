"""Controlled component ablations for TrajRel."""

from .registry import (
    SCORER_ABLATIONS,
    SELECTOR_ABLATIONS,
    get_scorer_ablation,
    get_selector_ablation,
)
from .scorer import (
    AblationSpec,
    rank_ablation_outputs,
)
from .selector import (
    SelectorAblationSpec,
    select_ablation_candidates,
)

__all__ = [
    "SCORER_ABLATIONS",
    "SELECTOR_ABLATIONS",
    "AblationSpec",
    "SelectorAblationSpec",
    "get_scorer_ablation",
    "get_selector_ablation",
    "rank_ablation_outputs",
    "select_ablation_candidates",
]
