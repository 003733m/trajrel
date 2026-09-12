import json
from pathlib import Path

import pytest

from trajrel.ablations import (
    SELECTOR_ABLATIONS,
    SelectorAblationSpec,
    get_selector_ablation,
    select_ablation_candidates,
)
from trajrel.scorers.kth_reference import rank_kth_reference_outputs

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/parity/kth_reference_selector_golden.json"
)


@pytest.mark.parametrize("case_index", range(6))
def test_full_selector_ablation_matches_frozen_reference(case_index):
    data = json.loads(FIXTURE.read_text())
    case = data["cases"][case_index]

    provisional = case["provisional_kinds"]

    ranked = rank_kth_reference_outputs(
        case["tool_outputs"],
        top_k=None,
        provisional_kinds=(
            frozenset(provisional)
            if provisional is not None
            else None
        ),
    )

    actual = select_ablation_candidates(
        list(ranked),
        target_content=case["target_content"],
        user_context=case["user_context"],
        spec=SelectorAblationSpec(),
        top_k=6,
    )

    expected = case["expected_selected"]

    assert len(actual) == len(expected)

    for got, want in zip(actual, expected, strict=True):
        expected_candidate = want["candidate"]

        assert got.token == expected_candidate["token"]
        assert got.kind == expected_candidate["kind"]

        assert got.candidate.score == pytest.approx(
            expected_candidate["score"],
            abs=1e-12,
        )

        assert got.target_specificity == pytest.approx(
            want["target_specificity"],
            abs=1e-12,
        )

        assert got.score == pytest.approx(
            want["score"],
            abs=1e-12,
        )


def _fixture_case(case_id: str) -> dict:
    data = json.loads(FIXTURE.read_text())

    return next(
        case
        for case in data["cases"]
        if case["id"] == case_id
    )


def _rank_case(case: dict):
    provisional = case["provisional_kinds"]

    return rank_kth_reference_outputs(
        case["tool_outputs"],
        top_k=None,
        provisional_kinds=(
            frozenset(provisional)
            if provisional is not None
            else None
        ),
    )


def test_registry_contains_predeclared_selector_ablations():
    assert set(SELECTOR_ABLATIONS) == {
        "full",
        "no_generic_identifier_filter",
        "no_weak_identifier_filter",
        "no_user_context_exclusion",
        "no_target_specificity",
        "no_singleton_corroboration",
    }

    assert get_selector_ablation("full") == SelectorAblationSpec()

    with pytest.raises(ValueError):
        get_selector_ablation("does_not_exist")


def test_disabling_generic_filter_allows_builtin_candidate():
    case = _fixture_case("generic_builtin_rejected")
    ranked = _rank_case(case)

    selected = select_ablation_candidates(
        list(ranked),
        target_content=case["target_content"],
        user_context=case["user_context"],
        spec=get_selector_ablation(
            "no_generic_identifier_filter"
        ),
    )

    assert "dict" in {x.token for x in selected}


def test_disabling_weak_filter_allows_plain_function():
    case = _fixture_case("weak_plain_function_rejected")
    ranked = _rank_case(case)

    selected = select_ablation_candidates(
        list(ranked),
        target_content=case["target_content"],
        user_context=case["user_context"],
        spec=get_selector_ablation(
            "no_weak_identifier_filter"
        ),
    )

    assert "get" in {x.token for x in selected}


def test_disabling_user_context_exclusion_allows_duplicate():
    case = _fixture_case("user_context_duplicate_rejected")
    ranked = _rank_case(case)

    selected = select_ablation_candidates(
        list(ranked),
        target_content=case["target_content"],
        user_context=case["user_context"],
        spec=get_selector_ablation(
            "no_user_context_exclusion"
        ),
    )

    assert "resolve_backend_target" in {
        x.token
        for x in selected
    }


def test_disabling_target_specificity_removes_idf_reranking():
    case = _fixture_case(
        "specificity_ranks_rare_identifier_higher"
    )
    ranked = _rank_case(case)

    selected = select_ablation_candidates(
        list(ranked),
        target_content=case["target_content"],
        user_context=case["user_context"],
        spec=get_selector_ablation(
            "no_target_specificity"
        ),
    )

    scores = {
        x.token: x.score
        for x in selected
    }

    # Historical evidence for these two candidates is deliberately equal.
    assert scores["resolve_backend_target"] == pytest.approx(
        scores["parse_provider_config"],
        abs=1e-12,
    )


def test_disabling_singleton_corroboration_rejects_provisional():
    case = _fixture_case(
        "provisional_test_singleton_corroborated"
    )
    ranked = _rank_case(case)

    selected = select_ablation_candidates(
        list(ranked),
        target_content=case["target_content"],
        user_context=case["user_context"],
        spec=get_selector_ablation(
            "no_singleton_corroboration"
        ),
    )

    assert selected == ()
