"""Controlled ablations of the frozen historical bridge scorer.

This module separates score contributions from historical eligibility gates.

Important:
- ``causal_score=False`` removes only the causal score contribution.
- ``causal_eligibility=False`` removes causal evidence as an admission route.
- The equivalent distinction applies to cross-output evidence.

These are historical-scorer ablations. Target-selector mechanisms are ablated
separately so their effects are not conflated.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from trajrel.scorers.kth_reference import (
    _MAX_NEGATIVE_CUE,
    _MAX_POSITIVE_CUE,
    KTHBridgeCandidate,
    KTHScoringConfig,
    _context_evidence,
    _find_candidates,
)


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """Switches controlling individual historical-scoring mechanisms."""

    causal_score: bool = True
    causal_eligibility: bool = True

    cross_output_score: bool = True
    cross_output_eligibility: bool = True

    occurrence_score: bool = True
    recency_score: bool = True
    speculation_penalty: bool = True


def rank_ablation_outputs(
    tool_outputs: Sequence[str],
    *,
    spec: AblationSpec,
    top_k: int | None = 6,
    scoring: KTHScoringConfig | None = None,
    provisional_kinds: frozenset[str] | None = None,
) -> tuple[KTHBridgeCandidate, ...]:
    """Rank historical identifiers under one controlled ablation.

    With ``AblationSpec()`` this must be behaviorally identical to the frozen
    KTH historical scorer.
    """
    scoring = scoring or KTHScoringConfig()

    if not tool_outputs:
        return ()

    if top_k is not None and top_k <= 0:
        return ()

    occurrence_count: Counter[str] = Counter()
    output_presence: dict[str, set[int]] = defaultdict(set)
    best_positive: dict[str, int] = defaultdict(int)
    strongest_negative: dict[str, int] = defaultdict(int)
    last_seen_output: dict[str, int] = {}
    token_kind: dict[str, str] = {}

    n_outputs = len(tool_outputs)

    for output_index, text in enumerate(tool_outputs):
        lines = text.splitlines()

        for hit in _find_candidates(text):
            token = hit.token

            occurrence_count[token] += 1
            output_presence[token].add(output_index)
            last_seen_output[token] = output_index
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

    ranked: list[KTHBridgeCandidate] = []

    for token, occurrences in occurrence_count.items():
        distinct_outputs = len(output_presence[token])
        raw_positive = best_positive[token]
        negative = strongest_negative[token]

        # If causal evidence is disabled everywhere in this historical scorer,
        # downstream historical state should not retain it as effective
        # evidence merely because it was detected diagnostically.
        effective_positive = (
            raw_positive
            if spec.causal_score or spec.causal_eligibility
            else 0
        )

        provisional = (
            provisional_kinds is not None
            and token_kind[token] in provisional_kinds
            and distinct_outputs == 1
            and effective_positive <= 0
        )

        eligible_by_causal = (
            spec.causal_eligibility
            and raw_positive > 0
        )

        eligible_by_cross_output = (
            spec.cross_output_eligibility
            and distinct_outputs >= 2
        )

        if (
            not eligible_by_causal
            and not eligible_by_cross_output
            and not provisional
        ):
            continue

        cross_output_steps = min(
            max(distinct_outputs - 1, 0),
            scoring.max_cross_tool_steps,
        )

        bounded_occurrences = min(
            occurrences,
            scoring.max_occurrences,
        )

        if n_outputs > 1:
            recency = (
                last_seen_output[token]
                / (n_outputs - 1)
            )
        else:
            recency = 0.0

        causal_signal = (
            float(raw_positive)
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
            float(negative)
            / float(_MAX_NEGATIVE_CUE)
        )

        score = 0.0

        if spec.causal_score:
            score += (
                scoring.causal_weight
                * causal_signal
            )

        if spec.cross_output_score:
            score += (
                scoring.cross_tool_weight
                * cross_output_signal
            )

        if spec.occurrence_score:
            score += (
                scoring.occurrence_weight
                * occurrence_signal
            )

        if spec.recency_score:
            score += (
                scoring.recency_weight
                * recency
            )

        if spec.speculation_penalty:
            score -= (
                scoring.speculation_weight
                * speculation_signal
            )

        if (
            scoring.min_score is not None
            and score < scoring.min_score
            and not provisional
        ):
            continue

        ranked.append(
            KTHBridgeCandidate(
                token=token,
                kind=token_kind[token],
                score=score,
                distinct_tools=distinct_outputs,
                occurrences=occurrences,
                positive_evidence=effective_positive,
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
