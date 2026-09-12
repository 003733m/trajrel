import json
from pathlib import Path

import pytest

from trajrel.scorers.kth_reference import rank_kth_reference_outputs
from trajrel.selectors.kth_reference import select_kth_reference_candidates

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/parity/kth_reference_selector_golden.json"
)


@pytest.mark.parametrize("case_index", range(6))
def test_kth_target_selector_matches_frozen_golden(case_index):
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

    selected = select_kth_reference_candidates(
        list(ranked),
        target_content=case["target_content"],
        user_context=case["user_context"],
        top_k=6,
    )

    expected = case["expected_selected"]

    assert len(selected) == len(expected)

    for got, want in zip(selected, expected, strict=True):
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
