"""Predefined historical-scorer ablations."""

from __future__ import annotations

from .scorer import AblationSpec

FULL = AblationSpec()

NO_CAUSAL_SCORE = AblationSpec(
    causal_score=False,
)

NO_CAUSAL_HISTORICAL_SIGNAL = AblationSpec(
    causal_score=False,
    causal_eligibility=False,
)

NO_CROSS_OUTPUT_SCORE = AblationSpec(
    cross_output_score=False,
)

NO_CROSS_OUTPUT_HISTORICAL_SIGNAL = AblationSpec(
    cross_output_score=False,
    cross_output_eligibility=False,
)

NO_OCCURRENCE_SCORE = AblationSpec(
    occurrence_score=False,
)

NO_RECENCY_SCORE = AblationSpec(
    recency_score=False,
)

NO_SPECULATION_PENALTY = AblationSpec(
    speculation_penalty=False,
)


SCORER_ABLATIONS: dict[str, AblationSpec] = {
    "full": FULL,
    "no_causal_score": NO_CAUSAL_SCORE,
    "no_causal_historical_signal": NO_CAUSAL_HISTORICAL_SIGNAL,
    "no_cross_output_score": NO_CROSS_OUTPUT_SCORE,
    "no_cross_output_historical_signal": (
        NO_CROSS_OUTPUT_HISTORICAL_SIGNAL
    ),
    "no_occurrence_score": NO_OCCURRENCE_SCORE,
    "no_recency_score": NO_RECENCY_SCORE,
    "no_speculation_penalty": NO_SPECULATION_PENALTY,
}


def get_scorer_ablation(name: str) -> AblationSpec:
    """Return a named scorer ablation."""
    try:
        return SCORER_ABLATIONS[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCORER_ABLATIONS))

        raise ValueError(
            f"Unknown scorer ablation {name!r}. "
            f"Available: {available}"
        ) from exc


from .selector import SelectorAblationSpec

FULL_SELECTOR = SelectorAblationSpec()

NO_GENERIC_IDENTIFIER_FILTER = SelectorAblationSpec(
    generic_identifier_filter=False,
)

NO_WEAK_IDENTIFIER_FILTER = SelectorAblationSpec(
    weak_identifier_filter=False,
)

NO_USER_CONTEXT_EXCLUSION = SelectorAblationSpec(
    user_context_exclusion=False,
)

NO_TARGET_SPECIFICITY = SelectorAblationSpec(
    target_specificity=False,
)

NO_SINGLETON_CORROBORATION = SelectorAblationSpec(
    singleton_corroboration=False,
)


SELECTOR_ABLATIONS: dict[str, SelectorAblationSpec] = {
    "full": FULL_SELECTOR,
    "no_generic_identifier_filter": NO_GENERIC_IDENTIFIER_FILTER,
    "no_weak_identifier_filter": NO_WEAK_IDENTIFIER_FILTER,
    "no_user_context_exclusion": NO_USER_CONTEXT_EXCLUSION,
    "no_target_specificity": NO_TARGET_SPECIFICITY,
    "no_singleton_corroboration": NO_SINGLETON_CORROBORATION,
}


def get_selector_ablation(name: str) -> SelectorAblationSpec:
    """Return a named selector ablation."""
    try:
        return SELECTOR_ABLATIONS[name]
    except KeyError as exc:
        available = ", ".join(sorted(SELECTOR_ABLATIONS))

        raise ValueError(
            f"Unknown selector ablation {name!r}. "
            f"Available: {available}"
        ) from exc
