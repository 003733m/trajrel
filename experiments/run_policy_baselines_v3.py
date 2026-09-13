"""Run TrajRel policy-level development controls."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from trajrel.ablations import (
    SCORER_ABLATIONS,
    SELECTOR_ABLATIONS,
    rank_ablation_outputs,
    select_ablation_candidates,
)
from trajrel.adoption import (
    adopted_bridge_identifiers,
)
from trajrel.evaluation.policy_baselines import (
    evaluate_budget_matched_b_only,
    evaluate_budget_matched_b_only_token,
    evaluate_identifier_policy,
    evaluate_random_b_cardinality_sham,
    evaluate_random_same_record_budget,
    evaluate_random_same_token_budget,
    evaluate_recent_same_record_budget,
    forced_record_indices,
    record_tokens,
    result_to_dict,
)
from trajrel.query_signals import (
    structured_search_identifiers_from_command,
)

CORPUS = Path(
    "benchmarks/development/"
    "headroom_hard_matched_v2.json"
)

OUTPUT = Path(
    "experiments/results/"
    "policy_baselines_v3.json"
)

RANDOM_RUNS = 10_000


def action_command(
    unit: dict,
) -> str:
    action = unit.get(
        "current_action",
        {},
    )

    if not isinstance(action, dict):
        return ""

    command = action.get(
        "command",
        "",
    )

    return (
        command
        if isinstance(command, str)
        else ""
    )


def history_outputs(
    unit: dict,
) -> tuple[str, ...]:
    history = unit[
        "history_before_action"
    ]

    outputs = tuple(
        event["content"]
        for event in history
        if isinstance(event, dict)
        and isinstance(
            event.get("content"),
            str,
        )
        and event["content"]
    )

    return outputs[-8:]


def ensemble_summary(
    results: list,
    *,
    observed_rescues: int,
) -> dict:
    rescues = [
        row.critical_rescued
        for row in results
    ]

    forced_records = [
        row.forced_records
        for row in results
    ]

    noncritical_tokens = [
        row.added_noncritical_tokens
        for row in results
    ]

    total_tokens = [
        row.added_total_tokens
        for row in results
    ]

    ge_observed = sum(
        value >= observed_rescues
        for value in rescues
    )

    histogram = Counter(
        rescues
    )

    return {
        "runs": len(results),
        "mean_rescues": mean(
            rescues
        ),
        "rescue_histogram": {
            str(key): histogram[key]
            for key in sorted(
                histogram
            )
        },
        "runs_ge_observed_rescues": (
            ge_observed
        ),
        "fraction_ge_observed_rescues": (
            ge_observed
            / len(results)
        ),
        "empirical_p_ge_observed": (
            (ge_observed + 1)
            / (len(results) + 1)
        ),
        "mean_forced_records": mean(
            forced_records
        ),
        "mean_added_noncritical_tokens": mean(
            noncritical_tokens
        ),
        "mean_added_total_tokens": mean(
            total_tokens
        ),
        "min_rescues": min(
            rescues
        ),
        "max_rescues": max(
            rescues
        ),
    }


def shuffle_q_same_cardinality(
    q_by_unit: dict[
        str,
        tuple[str, ...],
    ],
    *,
    seed: int,
) -> dict[
    str,
    tuple[str, ...],
]:
    """Shuffle Q tokens while preserving every unit's |Q| exactly."""
    rng = random.Random(seed)

    unit_ids = list(
        q_by_unit
    )

    pool = [
        identifier
        for unit_id in unit_ids
        for identifier in q_by_unit[
            unit_id
        ]
    ]

    rng.shuffle(pool)

    shuffled: dict[
        str,
        tuple[str, ...],
    ] = {}

    offset = 0

    for unit_id in unit_ids:
        length = len(
            q_by_unit[unit_id]
        )

        shuffled[
            unit_id
        ] = tuple(
            pool[
                offset:
                offset + length
            ]
        )

        offset += length

    assert offset == len(pool)

    for unit_id in unit_ids:
        assert len(
            shuffled[unit_id]
        ) == len(
            q_by_unit[unit_id]
        )

    return shuffled


def main() -> None:
    corpus = json.loads(
        CORPUS.read_text()
    )

    units = [
        unit
        for unit in corpus["units"]
        if unit["replay_valid"]
    ]

    scorer_spec = (
        SCORER_ABLATIONS["full"]
    )

    selector_spec = (
        SELECTOR_ABLATIONS["full"]
    )

    b_by_unit: dict[
        str,
        tuple[str, ...],
    ] = {}

    q_by_unit: dict[
        str,
        tuple[str, ...],
    ] = {}

    a_by_unit: dict[
        str,
        tuple[str, ...],
    ] = {}

    forced_budget_by_unit: dict[
        str,
        int,
    ] = {}

    token_budget_by_unit: dict[
        str,
        int,
    ] = {}

    adopted_cardinality_by_unit: dict[
        str,
        int,
    ] = {}

    for unit in units:
        ranked = (
            rank_ablation_outputs(
                history_outputs(unit),
                spec=scorer_spec,
                top_k=None,
            )
        )

        selected = (
            select_ablation_candidates(
                list(ranked),
                target_content=unit[
                    "target_content"
                ],
                user_context=unit[
                    "user_context"
                ],
                spec=selector_spec,
                top_k=6,
            )
        )

        b = tuple(
            candidate.candidate.token
            for candidate in selected
        )

        q = (
            structured_search_identifiers_from_command(
                action_command(unit)
            )
        )

        a = adopted_bridge_identifiers(
            b,
            q,
        )

        unit_id = unit["unit_id"]

        b_by_unit[unit_id] = b
        q_by_unit[unit_id] = q
        a_by_unit[unit_id] = a

        adopted_cardinality_by_unit[
            unit_id
        ] = len(a)

        forced_indices = (
            forced_record_indices(
                unit,
                a,
            )
        )

        forced_budget_by_unit[
            unit_id
        ] = len(
            forced_indices
        )

        token_budget = sum(
            record_tokens(
                unit["records"][index]
            )
            for index in forced_indices
        )

        token_budget_by_unit[
            unit_id
        ] = token_budget

    b_only = evaluate_identifier_policy(
        units=units,
        policy_name="B_only",
        identifiers_by_unit=b_by_unit,
    )

    q_only = evaluate_identifier_policy(
        units=units,
        policy_name="Q_only",
        identifiers_by_unit=q_by_unit,
    )

    full = evaluate_identifier_policy(
        units=units,
        policy_name="B_intersect_Q",
        identifiers_by_unit=a_by_unit,
    )

    b_record_budget = (
        evaluate_budget_matched_b_only(
            units=units,
            b_by_unit=b_by_unit,
            forced_budget_by_unit=(
                forced_budget_by_unit
            ),
        )
    )

    b_token_budget = (
        evaluate_budget_matched_b_only_token(
            units=units,
            b_by_unit=b_by_unit,
            token_budget_by_unit=(
                token_budget_by_unit
            ),
        )
    )

    recent = (
        evaluate_recent_same_record_budget(
            units=units,
            forced_budget_by_unit=(
                forced_budget_by_unit
            ),
        )
    )

    random_record_results = [
        evaluate_random_same_record_budget(
            units=units,
            forced_budget_by_unit=(
                forced_budget_by_unit
            ),
            seed=seed,
        )
        for seed in range(
            RANDOM_RUNS
        )
    ]

    random_token_results = [
        evaluate_random_same_token_budget(
            units=units,
            token_budget_by_unit=(
                token_budget_by_unit
            ),
            seed=seed,
        )
        for seed in range(
            RANDOM_RUNS
        )
    ]

    random_b_results = [
        evaluate_random_b_cardinality_sham(
            units=units,
            b_by_unit=b_by_unit,
            adopted_cardinality_by_unit=(
                adopted_cardinality_by_unit
            ),
            seed=seed,
        )
        for seed in range(
            RANDOM_RUNS
        )
    ]

    shuffled_q_results = []

    for seed in range(
        RANDOM_RUNS
    ):
        shuffled_q = (
            shuffle_q_same_cardinality(
                q_by_unit,
                seed=seed,
            )
        )

        shuffled_a = {
            unit_id: (
                adopted_bridge_identifiers(
                    b_by_unit[
                        unit_id
                    ],
                    shuffled_q[
                        unit_id
                    ],
                )
            )
            for unit_id in b_by_unit
        }

        shuffled_q_results.append(
            evaluate_identifier_policy(
                units=units,
                policy_name=(
                    "shuffled_Q_"
                    f"seed_{seed}"
                ),
                identifiers_by_unit=(
                    shuffled_a
                ),
            )
        )

    observed_rescues = (
        full.critical_rescued
    )

    result = {
        "evaluation_type": (
            "development_policy_controls_v3"
        ),
        "held_out": False,
        "history_policy": (
            "history_before_action"
        ),
        "random_runs": RANDOM_RUNS,
        "observed_rescues": (
            observed_rescues
        ),
        "trajrel_budget": {
            "forced_records": sum(
                forced_budget_by_unit.values()
            ),
            "added_tokens": sum(
                token_budget_by_unit.values()
            ),
            "adopted_identifiers": sum(
                adopted_cardinality_by_unit.values()
            ),
        },
        "policies": {
            "B_only": result_to_dict(
                b_only
            ),
            "Q_only": result_to_dict(
                q_only
            ),
            "B_intersect_Q": (
                result_to_dict(
                    full
                )
            ),
            "B_only_same_record_budget": (
                result_to_dict(
                    b_record_budget
                )
            ),
            "B_only_same_token_budget": (
                result_to_dict(
                    b_token_budget
                )
            ),
            "recent_same_record_budget": (
                result_to_dict(
                    recent
                )
            ),
        },
        "null_controls": {
            "random_same_record_budget": (
                ensemble_summary(
                    random_record_results,
                    observed_rescues=(
                        observed_rescues
                    ),
                )
            ),
            "random_same_token_budget": (
                ensemble_summary(
                    random_token_results,
                    observed_rescues=(
                        observed_rescues
                    ),
                )
            ),
            "random_B_cardinality_sham": (
                ensemble_summary(
                    random_b_results,
                    observed_rescues=(
                        observed_rescues
                    ),
                )
            ),
            "shuffled_Q_same_cardinality": (
                ensemble_summary(
                    shuffled_q_results,
                    observed_rescues=(
                        observed_rescues
                    ),
                )
            ),
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    print(
        "===== POLICY CONTROLS V3 ====="
    )

    for name, row in result[
        "policies"
    ].items():
        print()
        print(name)

        for key in [
            "critical_rescued",
            "forced_records",
            "forced_critical_records",
            "forced_noncritical_records",
            "added_total_tokens",
            "added_noncritical_tokens",
            "critical_recall_after",
            "incremental_precision",
        ]:
            print(
                f"  {key}: "
                f"{row[key]}"
            )

    print()
    print("===== NULL CONTROLS =====")

    for name, row in result[
        "null_controls"
    ].items():
        print()
        print(name)

        for key in [
            "runs",
            "mean_rescues",
            "rescue_histogram",
            "fraction_ge_observed_rescues",
            "empirical_p_ge_observed",
            "mean_forced_records",
            "mean_added_total_tokens",
            "mean_added_noncritical_tokens",
        ]:
            print(
                f"  {key}: "
                f"{row[key]}"
            )

    print()
    print("===== TRAJREL =====")
    print(
        "observed rescues:",
        observed_rescues,
    )
    print(
        "forced records:",
        result["trajrel_budget"][
            "forced_records"
        ],
    )
    print(
        "added tokens:",
        result["trajrel_budget"][
            "added_tokens"
        ],
    )
    print(
        "adopted identifiers:",
        result["trajrel_budget"][
            "adopted_identifiers"
        ],
    )

    print()
    print("WROTE:", OUTPUT)


if __name__ == "__main__":
    main()
