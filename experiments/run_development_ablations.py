"""Run TrajRel scorer/selector development ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajrel.evaluation import (
    CAUSAL_ACTION_VIEW,
    LEGACY_MEASUREMENT_VIEW,
    reproduce_legacy_adopted_floor,
    run_ablation_suite,
)
from trajrel.evaluation.batch import (
    load_corpus,
)

DEFAULT_CORPUS = Path(
    "benchmarks/development/"
    "headroom_hard_matched_v2.json"
)

DEFAULT_OUTPUT = Path(
    "experiments/results/"
    "development_ablations_v1.json"
)


def _compact(result):
    return {
        key: result[key]
        for key in [
            "units",
            "scorable_records",
            "critical_records",
            "units_with_selected_bridges",
            "units_with_Q",
            "units_with_adopted_bridge",
            "baseline_critical_kept",
            "final_critical_kept",
            "critical_rescued",
            "forced_records",
            "forced_critical_records",
            "forced_noncritical_records",
            "added_critical_tokens",
            "added_noncritical_tokens",
            "critical_recall_before",
            "critical_recall_after",
            "incremental_precision",
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    corpus = load_corpus(args.corpus)

    result = {
        "evaluation_type": (
            "development_component_ablation"
        ),
        "corpus": corpus["name"],
        "held_out": False,
        "historical_reproduction": (
            reproduce_legacy_adopted_floor(
                corpus
            )
        ),
        "views": {
            LEGACY_MEASUREMENT_VIEW.name:
                run_ablation_suite(
                    corpus,
                    view=LEGACY_MEASUREMENT_VIEW,
                ),
            CAUSAL_ACTION_VIEW.name:
                run_ablation_suite(
                    corpus,
                    view=CAUSAL_ACTION_VIEW,
                ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    for view_name, view in result[
        "views"
    ].items():
        print()
        print("=" * 90)
        print(view_name)

        print()
        print("SCORER ABLATIONS")

        for name, row in view[
            "scorer_ablations"
        ].items():
            print()
            print(name)
            print(
                json.dumps(
                    _compact(row),
                    indent=2,
                )
            )

        print()
        print("SELECTOR ABLATIONS")

        for name, row in view[
            "selector_ablations"
        ].items():
            print()
            print(name)
            print(
                json.dumps(
                    _compact(row),
                    indent=2,
                )
            )

    print()
    print("WROTE:", args.output)


if __name__ == "__main__":
    main()
