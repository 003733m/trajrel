"""Controlled ablations of the frozen target-conditioned selector."""

from __future__ import annotations

from dataclasses import dataclass

from trajrel.scorers.kth_reference import (
    _MAX_NEGATIVE_CUE,
    _MAX_POSITIVE_CUE,
    TARGET_CORROBORATION_KINDS,
    KTHBridgeCandidate,
    KTHScoringConfig,
)
from trajrel.selectors.kth_reference import (
    KTHTargetBridgeCandidate,
    _contains_candidate,
    _is_generic_bridge_identifier,
    _is_weak_unstructured_bridge_identifier,
    _target_document_frequencies,
    _target_specificity_from_frequency,
)


@dataclass(frozen=True, slots=True)
class SelectorAblationSpec:
    """Switches controlling target-conditioned selection mechanisms."""

    generic_identifier_filter: bool = True
    weak_identifier_filter: bool = True
    user_context_exclusion: bool = True
    target_specificity: bool = True
    singleton_corroboration: bool = True


def select_ablation_candidates(
    candidates: list[KTHBridgeCandidate],
    *,
    target_content: str,
    user_context: str,
    spec: SelectorAblationSpec,
    top_k: int = 6,
    scoring: KTHScoringConfig | None = None,
) -> tuple[KTHTargetBridgeCandidate, ...]:
    """Condition historical candidates under controlled selector ablations."""

    if not candidates or not target_content or top_k <= 0:
        return ()

    scoring = scoring or KTHScoringConfig()

    admissible: list[KTHBridgeCandidate] = []

    for candidate in candidates:
        if (
            spec.generic_identifier_filter
            and _is_generic_bridge_identifier(candidate)
        ):
            continue

        if (
            spec.weak_identifier_filter
            and _is_weak_unstructured_bridge_identifier(candidate)
        ):
            continue

        if (
            spec.user_context_exclusion
            and user_context
            and _contains_candidate(
                user_context,
                candidate.token,
                kind=candidate.kind,
            )
        ):
            continue

        admissible.append(candidate)

    if not admissible:
        return ()

    n_lines, document_frequencies = _target_document_frequencies(
        admissible,
        target_content,
    )

    if n_lines <= 0:
        return ()

    selected: list[KTHTargetBridgeCandidate] = []

    for candidate in admissible:
        document_frequency = document_frequencies[candidate]

        # Target occurrence remains a fixed applicability requirement.
        if document_frequency <= 0:
            continue

        specificity = _target_specificity_from_frequency(
            n_lines=n_lines,
            document_frequency=document_frequency,
        )

        evidence_score = candidate.score

        provisional_singleton = (
            candidate.kind in TARGET_CORROBORATION_KINDS
            and candidate.distinct_tools == 1
            and candidate.positive_evidence <= 0
        )

        if provisional_singleton:
            if not spec.singleton_corroboration:
                continue

            effective_distinct_outputs = 2
            effective_occurrences = candidate.occurrences + 1

            cross_output_steps = min(
                max(effective_distinct_outputs - 1, 0),
                scoring.max_cross_tool_steps,
            )

            bounded_occurrences = min(
                effective_occurrences,
                scoring.max_occurrences,
            )

            causal_signal = (
                float(candidate.positive_evidence)
                / float(_MAX_POSITIVE_CUE)
            )

            if scoring.max_cross_tool_steps > 0:
                cross_output_signal = (
                    float(cross_output_steps)
                    / float(scoring.max_cross_tool_steps)
                )
            else:
                cross_output_signal = 0.0

            if scoring.max_occurrences > 0:
                occurrence_signal = (
                    float(bounded_occurrences)
                    / float(scoring.max_occurrences)
                )
            else:
                occurrence_signal = 0.0

            speculation_signal = (
                float(candidate.negative_evidence)
                / float(_MAX_NEGATIVE_CUE)
            )

            evidence_score = (
                scoring.causal_weight * causal_signal
                + scoring.cross_tool_weight * cross_output_signal
                + scoring.occurrence_weight * occurrence_signal
                + scoring.recency_weight
                - scoring.speculation_weight * speculation_signal
            )

            if (
                scoring.min_score is not None
                and evidence_score < scoring.min_score
            ):
                continue

        adjusted_score = evidence_score

        if spec.target_specificity:
            adjusted_score *= specificity

        if adjusted_score <= 0.0:
            continue

        selected.append(
            KTHTargetBridgeCandidate(
                candidate=candidate,
                score=adjusted_score,
                target_specificity=specificity,
            )
        )

    selected.sort(
        key=lambda item: (
            -item.score,
            -item.candidate.score,
            -item.candidate.distinct_tools,
            -item.candidate.recency,
            item.token,
        )
    )

    return tuple(selected[:top_k])
