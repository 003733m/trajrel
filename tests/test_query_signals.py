from trajrel.query_signals import (
    extract_search_pattern,
    structured_search_identifiers_from_command,
)


def test_extracts_rg_pattern_not_search_path():
    command = (
        'rg -nH '
        '"discover_projects|flush_to_file" '
        'tests/test_memory/test_traffic_learner.py'
    )

    assert extract_search_pattern(command) == (
        "discover_projects|flush_to_file"
    )

    assert (
        structured_search_identifiers_from_command(
            command
        )
        == (
            "discover_projects",
            "flush_to_file",
        )
    )


def test_g21_historical_q():
    command = (
        'rg -nH '
        '"_background_dedup|DEDUP_AUTO_THRESHOLD" '
        'headroom'
    )

    assert (
        structured_search_identifiers_from_command(
            command
        )
        == (
            "_background_dedup",
            "DEDUP_AUTO_THRESHOLD",
        )
    )


def test_g07_historical_q():
    command = (
        'rg -nH '
        '"def count_messages|image_url|'
        'type.*image|_count_serialized" '
        'headroom'
    )

    assert (
        structured_search_identifiers_from_command(
            command
        )
        == (
            "_count_serialized",
            "count_messages",
            "image_url",
        )
    )
