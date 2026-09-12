import json
from pathlib import Path

import pytest

from trajrel.scorers.headroom_reference import rank_headroom_reference_outputs

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/parity/headroom_reference_scorer_golden.json"
)


def test_reference_default_config_matches_frozen_fixture():
    data = json.loads(FIXTURE.read_text())

    assert data["scoring_config"] == {
        "causal_weight": 4.0,
        "cross_tool_weight": 2.0,
        "occurrence_weight": 2.0,
        "recency_weight": 1.0,
        "speculation_weight": 2.0,
        "max_cross_tool_steps": 2,
        "max_occurrences": 3,
        "min_score": 2.0,
    }



@pytest.mark.parametrize(
    "case_index",
    range(6),
)
def test_headroom_reference_scorer_matches_frozen_golden(case_index):
    data = json.loads(FIXTURE.read_text())
    case = data["cases"][case_index]

    provisional = case["provisional_kinds"]

    actual = rank_headroom_reference_outputs(
        case["tool_outputs"],
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
        assert got.recency == pytest.approx(want["recency"], abs=1e-12)
        assert got.score == pytest.approx(want["score"], abs=1e-12)
