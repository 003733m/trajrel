import json
from pathlib import Path

CORPUS = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "development"
    / "headroom_hard_matched_v2.json"
)


def test_headroom_development_corpus_contract():
    data = json.loads(CORPUS.read_text())
    summary = data["summary"]

    assert data["schema_version"] == 2

    assert summary["tasks"] == 6
    assert summary["units"] == 14
    assert summary["records"] == 1444

    assert (
        summary[
            "direct_replay_hash_exact_units"
        ]
        == 10
    )

    assert (
        summary["legacy_strict_units"]
        == 10
    )

    assert (
        summary["legacy_scorable_records"]
        == 702
    )

    assert (
        summary["legacy_strict_records"]
        == 1332
    )

    assert (
        summary[
            "legacy_strict_scorable_records"
        ]
        == 605
    )

    assert (
        summary[
            "legacy_strict_critical_scorable_records"
        ]
        == 9
    )
