from trajrel.events import CompressionOpportunity
from trajrel.policies import (
    AdoptedBridgePolicy,
    BasePolicy,
    HistoryOnlyPolicy,
    PolicySignals,
    QueryOnlyPolicy,
)


def opportunity(
    *,
    segment: str,
    base_keep: bool = False,
    critical: bool = True,
) -> CompressionOpportunity:
    return CompressionOpportunity(
        sample_id="test",
        history=(),
        current_action="",
        segment=segment,
        base_keep=base_keep,
        critical=critical,
    )


def test_adopted_bridge_is_intersection_of_history_and_current_reuse():
    signals = PolicySignals(
        historical_bridges=("OLD_ONLY", "SHARED"),
        current_identifiers=("QUERY_ONLY", "SHARED"),
    )

    assert signals.adopted_bridges == ("SHARED",)


def test_four_policies_distinguish_history_query_and_adoption():
    signals = PolicySignals(
        historical_bridges=("HISTORY_ONLY_ID", "ADOPTED_ID"),
        current_identifiers=("QUERY_ONLY_ID", "ADOPTED_ID"),
    )

    history_segment = opportunity(segment="value = HISTORY_ONLY_ID")
    query_segment = opportunity(segment="value = QUERY_ONLY_ID")
    adopted_segment = opportunity(segment="value = ADOPTED_ID")

    assert not BasePolicy().decide(adopted_segment, signals).keep

    assert HistoryOnlyPolicy().decide(history_segment, signals).keep
    assert not QueryOnlyPolicy().decide(history_segment, signals).keep
    assert not AdoptedBridgePolicy().decide(history_segment, signals).keep

    assert not HistoryOnlyPolicy().decide(query_segment, signals).keep
    assert QueryOnlyPolicy().decide(query_segment, signals).keep
    assert not AdoptedBridgePolicy().decide(query_segment, signals).keep

    assert HistoryOnlyPolicy().decide(adopted_segment, signals).keep
    assert QueryOnlyPolicy().decide(adopted_segment, signals).keep
    assert AdoptedBridgePolicy().decide(adopted_segment, signals).keep


def test_adopted_bridge_requires_same_identifier_not_merely_nonempty_sets():
    signals = PolicySignals(
        historical_bridges=("FOO",),
        current_identifiers=("BAR",),
    )

    sample = opportunity(segment="FOO BAR")

    result = AdoptedBridgePolicy().decide(sample, signals)

    assert signals.adopted_bridges == ()
    assert not result.keep
    assert not result.forced


def test_identifier_matching_uses_exact_boundaries():
    signals = PolicySignals(
        historical_bridges=("DEDUP_AUTO_THRESHOLD",),
        current_identifiers=("DEDUP_AUTO_THRESHOLD",),
    )

    exact = opportunity(segment="DEDUP_AUTO_THRESHOLD = 0.92")
    substring = opportunity(segment="MY_DEDUP_AUTO_THRESHOLD_BACKUP = 0.92")

    assert AdoptedBridgePolicy().decide(exact, signals).keep
    assert not AdoptedBridgePolicy().decide(substring, signals).keep


def test_all_preservation_floors_are_monotonic():
    sample = opportunity(
        segment="completely unrelated evidence",
        base_keep=True,
    )
    signals = PolicySignals(
        historical_bridges=("OTHER",),
        current_identifiers=("OTHER",),
    )

    policies = [
        HistoryOnlyPolicy(),
        QueryOnlyPolicy(),
        AdoptedBridgePolicy(),
    ]

    for policy in policies:
        result = policy.decide(sample, signals)
        assert result.keep
        assert not result.forced


def test_forced_decision_reports_matching_identifier():
    sample = opportunity(segment="DEDUP_AUTO_THRESHOLD = 0.92")
    signals = PolicySignals(
        historical_bridges=("DEDUP_AUTO_THRESHOLD",),
        current_identifiers=("DEDUP_AUTO_THRESHOLD",),
    )

    result = AdoptedBridgePolicy().decide(sample, signals)

    assert result.keep
    assert result.forced
    assert result.matched_identifiers == ("DEDUP_AUTO_THRESHOLD",)
