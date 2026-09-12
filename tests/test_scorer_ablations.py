import json
from pathlib import Path

import pytest

from trajrel.ablations import (
    SCORER_ABLATIONS,
    AblationSpec,
    get_scorer_ablation,
    rank_ablation_outputs,
)
from trajrel.scorers.kth_reference import KTHScoringConfig

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/parity/kth_reference_scorer_golden.json"
)


@pytest.mark.parametrize("case_index", range(6))
def test_full_ablation_matches_frozen_reference(case_index):
    data = json.loads(FIXTURE.read_text())
    case = data["cases"][case_index]

    provisional = case["provisional_kinds"]

    actual = rank_ablation_outputs(
        case["tool_outputs"],
        spec=AblationSpec(),
        top_k=None,
        provisional_kinds=(
            frozenset(provisional)
            if provisional is not None
            else None
        ),
    )

    expected = case["expected"]

    assert len(actual) == len(expected)

    for got, want in zip(actual, expected, strict=True):
        assert got.token == want["token"]
        assert got.kind == want["kind"]
        assert got.distinct_tools == want["distinct_tools"]
        assert got.occurrences == want["occurrences"]
        assert got.positive_evidence == want["positive_evidence"]
        assert got.negative_evidence == want["negative_evidence"]
        assert got.recency == pytest.approx(
            want["recency"],
            abs=1e-12,
        )
        assert got.score == pytest.approx(
            want["score"],
            abs=1e-12,
        )


def test_registry_contains_predeclared_ablations():
    assert set(SCORER_ABLATIONS) == {
        "full",
        "no_causal_score",
        "no_causal_historical_signal",
        "no_cross_output_score",
        "no_cross_output_historical_signal",
        "no_occurrence_score",
        "no_recency_score",
        "no_speculation_penalty",
    }

    assert get_scorer_ablation("full") == AblationSpec()

    with pytest.raises(ValueError):
        get_scorer_ablation("does_not_exist")


def test_causal_score_and_causal_eligibility_are_distinct():
    outputs = [
        "Confirmed root cause: resolve_backend_target(config) failed."
    ]

    no_threshold = KTHScoringConfig(
        min_score=None,
    )

    score_removed = rank_ablation_outputs(
        outputs,
        spec=get_scorer_ablation("no_causal_score"),
        top_k=None,
        scoring=no_threshold,
    )

    signal_removed = rank_ablation_outputs(
        outputs,
        spec=get_scorer_ablation(
            "no_causal_historical_signal"
        ),
        top_k=None,
        scoring=no_threshold,
    )

    # Causal evidence can still admit the candidate when only its score
    # contribution is removed.
    assert [x.token for x in score_removed] == [
        "resolve_backend_target"
    ]

    # Removing causal eligibility as well eliminates this singleton.
    assert signal_removed == ()


def test_cross_output_score_and_eligibility_are_distinct():
    outputs = [
        "Observed resolve_backend_target(config) during inspection.",
        "Calling resolve_backend_target(config) again.",
    ]

    no_threshold = KTHScoringConfig(
        min_score=None,
    )

    score_removed = rank_ablation_outputs(
        outputs,
        spec=get_scorer_ablation(
            "no_cross_output_score"
        ),
        top_k=None,
        scoring=no_threshold,
    )

    signal_removed = rank_ablation_outputs(
        outputs,
        spec=get_scorer_ablation(
            "no_cross_output_historical_signal"
        ),
        top_k=None,
        scoring=no_threshold,
    )

    assert [x.token for x in score_removed] == [
        "resolve_backend_target"
    ]

    assert signal_removed == ()


def test_removing_speculation_penalty_only_increases_score():
    outputs = [
        "Possible hypothesis: resolve_backend_target(config) may be wrong.",
        "Maybe resolve_backend_target(config) is suspect.",
    ]

    no_threshold = KTHScoringConfig(
        min_score=None,
    )

    full = rank_ablation_outputs(
        outputs,
        spec=get_scorer_ablation("full"),
        top_k=None,
        scoring=no_threshold,
    )

    no_penalty = rank_ablation_outputs(
        outputs,
        spec=get_scorer_ablation(
            "no_speculation_penalty"
        ),
        top_k=None,
        scoring=no_threshold,
    )

    assert len(full) == 1
    assert len(no_penalty) == 1

    assert no_penalty[0].token == full[0].token
    assert no_penalty[0].score > full[0].score
