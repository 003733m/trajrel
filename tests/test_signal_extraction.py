from trajrel.adoption import (
    adopted_bridge_identifiers,
    build_policy_signals,
    current_reused_identifiers,
)
from trajrel.bridges import rank_historical_bridges, select_historical_bridges
from trajrel.events import CompressionOpportunity, ToolEvent
from trajrel.extractors import extract_identifier_tokens
from trajrel.policies import AdoptedBridgePolicy


def test_extracts_common_software_identifiers():
    text = """
    FAILED tests/test_memory.py::test_auto_supersede
    ConfigError: DEDUP_AUTO_THRESHOLD is missing
    call resolve_memory_state from headroom/proxy/memory_handler.py
    """

    tokens = set(extract_identifier_tokens(text))

    assert "tests/test_memory.py" in tokens
    assert "test_auto_supersede" in tokens
    assert "ConfigError" in tokens
    assert "DEDUP_AUTO_THRESHOLD" in tokens
    assert "resolve_memory_state" in tokens
    assert "headroom/proxy/memory_handler.py" in tokens


def test_historical_bridge_requires_repeated_support_by_default():
    history = (
        ToolEvent(
            tool_name="pytest",
            content="ConfigError involving DEDUP_AUTO_THRESHOLD",
        ),
        ToolEvent(
            tool_name="read",
            content="class MemoryHandler uses DEDUP_AUTO_THRESHOLD",
        ),
    )

    bridges = select_historical_bridges(history)

    assert "DEDUP_AUTO_THRESHOLD" in bridges
    assert "ConfigError" not in bridges


def test_cross_tool_corroboration_increases_bridge_score():
    history = (
        ToolEvent(
            tool_name="pytest",
            content="failure in DEDUP_AUTO_THRESHOLD and OTHER_SETTING",
        ),
        ToolEvent(
            tool_name="read",
            content="DEDUP_AUTO_THRESHOLD = 0.92",
        ),
        ToolEvent(
            tool_name="pytest",
            content="OTHER_SETTING remains suspicious",
        ),
    )

    ranked = {
        candidate.token: candidate
        for candidate in rank_historical_bridges(history)
    }

    assert ranked["DEDUP_AUTO_THRESHOLD"].distinct_tools == 2
    assert ranked["OTHER_SETTING"].distinct_tools == 1
    assert ranked["DEDUP_AUTO_THRESHOLD"].score > ranked["OTHER_SETTING"].score


def test_current_action_defines_q():
    q = current_reused_identifiers(
        'rg -n "DEDUP_AUTO_THRESHOLD|resolve_memory_state" headroom/'
    )

    assert "DEDUP_AUTO_THRESHOLD" in q
    assert "resolve_memory_state" in q


def test_adoption_is_exact_intersection():
    adopted = adopted_bridge_identifiers(
        ("DEDUP_AUTO_THRESHOLD", "OLD_ONLY"),
        ("DEDUP_AUTO_THRESHOLD", "QUERY_ONLY"),
    )

    assert adopted == ("DEDUP_AUTO_THRESHOLD",)


def test_raw_trajectory_can_drive_adopted_bridge_policy():
    history = (
        ToolEvent(
            tool_name="pytest",
            content="Failure points to DEDUP_AUTO_THRESHOLD",
        ),
        ToolEvent(
            tool_name="read",
            content="DEDUP_AUTO_THRESHOLD controls auto supersession",
        ),
    )

    current_action = 'rg -n "DEDUP_AUTO_THRESHOLD" headroom/'

    signals = build_policy_signals(
        history,
        current_action,
    )

    sample = CompressionOpportunity(
        sample_id="natural-example",
        history=history,
        current_action=current_action,
        segment="DEDUP_AUTO_THRESHOLD = 0.92",
        base_keep=False,
        critical=True,
    )

    result = AdoptedBridgePolicy().decide(sample, signals)

    assert signals.adopted_bridges == ("DEDUP_AUTO_THRESHOLD",)
    assert result.keep
    assert result.forced
