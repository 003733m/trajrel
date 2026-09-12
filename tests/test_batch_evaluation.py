import json
from pathlib import Path

from trajrel.ablations import (
    SCORER_ABLATIONS,
    SELECTOR_ABLATIONS,
)
from trajrel.evaluation import (
    CAUSAL_ACTION_VIEW,
    LEGACY_MEASUREMENT_VIEW,
    evaluate_variant,
)

CORPUS = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "development"
    / "headroom_hard_matched_v2.json"
)


def _corpus():
    return json.loads(CORPUS.read_text())


def test_legacy_view_uses_frozen_measurement_stratum():
    result = evaluate_variant(
        _corpus(),
        view=LEGACY_MEASUREMENT_VIEW,
        scorer_spec=SCORER_ABLATIONS["full"],
        selector_spec=SELECTOR_ABLATIONS["full"],
        variant="full",
        family="combined",
    )

    assert result["units"] == 10
    assert result["scorable_records"] == 605
    assert result["critical_records"] == 9
    assert result["critical_regressed"] == 0

    assert (
        result["forced_records"]
        == result["forced_critical_records"]
        + result["forced_noncritical_records"]
    )


def test_causal_view_is_monotonic():
    result = evaluate_variant(
        _corpus(),
        view=CAUSAL_ACTION_VIEW,
        scorer_spec=SCORER_ABLATIONS["full"],
        selector_spec=SELECTOR_ABLATIONS["full"],
        variant="full",
        family="combined",
    )

    assert result["units"] == 13
    assert result["critical_regressed"] == 0

    assert (
        result["final_critical_kept"]
        >= result["baseline_critical_kept"]
    )


def test_frozen_legacy_adopted_floor_reproduction():
    from trajrel.evaluation import (
        reproduce_legacy_adopted_floor,
    )

    result = reproduce_legacy_adopted_floor(
        _corpus()
    )

    assert result["units"] == 10
    assert result["scorable_records"] == 605
    assert result["critical_records"] == 9

    assert (
        result["units_with_adopted_bridge"]
        == 3
    )

    assert (
        result["baseline_critical_kept"]
        == 5
    )

    assert (
        result["final_critical_kept"]
        == 6
    )

    assert result["critical_rescued"] == 1
    assert result["critical_regressed"] == 0

    assert result["forced_records"] == 21

    assert (
        result["forced_critical_records"]
        == 1
    )

    assert (
        result["forced_noncritical_records"]
        == 20
    )

    assert (
        result["added_critical_tokens"]
        == 36
    )

    assert (
        result["added_noncritical_tokens"]
        == 504
    )


def test_recomputed_full_matches_historical_mechanism():
    from trajrel.ablations import (
        SCORER_ABLATIONS,
        SELECTOR_ABLATIONS,
    )
    from trajrel.evaluation import (
        LEGACY_MEASUREMENT_VIEW,
        evaluate_variant,
        reproduce_legacy_adopted_floor,
    )

    corpus = _corpus()

    frozen = reproduce_legacy_adopted_floor(
        corpus
    )

    recomputed = evaluate_variant(
        corpus,
        view=LEGACY_MEASUREMENT_VIEW,
        scorer_spec=SCORER_ABLATIONS["full"],
        selector_spec=SELECTOR_ABLATIONS["full"],
        variant="full",
        family="combined",
    )

    keys = [
        "units_with_adopted_bridge",
        "baseline_critical_kept",
        "final_critical_kept",
        "critical_rescued",
        "critical_regressed",
        "forced_records",
        "forced_critical_records",
        "forced_noncritical_records",
        "added_critical_tokens",
        "added_noncritical_tokens",
    ]

    for key in keys:
        assert recomputed[key] == frozen[key]
