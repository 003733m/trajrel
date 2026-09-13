"""Run TrajRel policy-level development baselines."""

from __future__ import annotations

import json
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
    evaluate_identifier_policy,
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
    "policy_baselines_v2.json"
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


def random_summary(
    results: list,
) -> dict:
    rescues = [
        row.critical_rescued
        for row in results
    ]

    forced_records = [
        row.forced_records
        for row in results
    ]

    forced_critical = [
        row.forced_critical_records
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

    return {
        "runs": len(results),
        "mean_rescues": mean(
            rescues
        ),
        "runs_with_at_least_one_rescue": sum(
            value >= 1
            for value in rescues
        ),
        "fraction_with_at_least_one_rescue": (
            sum(
                value >= 1
                for value in rescues
            )
            / len(results)
        ),
        "mean_forced_records": mean(
            forced_records
        ),
        "mean_forced_critical_records": mean(
            forced_critical
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

    for unit in units:
        history = history_outputs(
            unit
        )

        ranked = (
            rank_ablation_outputs(
                history,
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

        token_budget = 0

        records = unit["records"]

        for index in forced_indices:
            token_budget += record_tokens(
                records[index]
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

    b_budget = (
        evaluate_budget_matched_b_only(
            units=units,
            b_by_unit=b_by_unit,
            forced_budget_by_unit=(
                forced_budget_by_unit
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

    result = {
        "evaluation_type": (
            "development_policy_baselines_v2"
        ),
        "held_out": False,
        "history_policy": (
            "history_before_action"
        ),
        "base_keep": (
            "legacy_normalized_a_keep"
        ),
        "query_signal": (
            "structured_search_pattern"
        ),
        "units": len(units),
        "trajrel_budget": {
            "forced_records": sum(
                forced_budget_by_unit.values()
            ),
            "added_tokens": sum(
                token_budget_by_unit.values()
            ),
            "forced_records_by_unit": (
                forced_budget_by_unit
            ),
            "token_budget_by_unit": (
                token_budget_by_unit
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
                    b_budget
                )
            ),
            "recent_same_record_budget": (
                result_to_dict(
                    recent
                )
            ),
        },
        "random_same_record_budget": {
            **random_summary(
                random_record_results
            ),
            "results": [
                result_to_dict(
                    row
                )
                for row
                in random_record_results
            ],
        },
        "random_same_token_budget": {
            **random_summary(
                random_token_results
            ),
            "results": [
                result_to_dict(
                    row
                )
                for row
                in random_token_results
            ],
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
        "===== POLICY BASELINES V2 ====="
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
            "added_critical_tokens",
            "added_noncritical_tokens",
            "added_total_tokens",
            "critical_recall_before",
            "critical_recall_after",
            "incremental_precision",
        ]:
            print(
                f"  {key}: "
                f"{row[key]}"
            )

    print()
    print(
        "===== RANDOM SAME-RECORD-BUDGET ====="
    )

    for key, value in result[
        "random_same_record_budget"
    ].items():
        if key == "results":
            continue

        print(
            f"{key}: {value}"
        )

    print()
    print(
        "===== RANDOM SAME-TOKEN-BUDGET ====="
    )

    for key, value in result[
        "random_same_token_budget"
    ].items():
        if key == "results":
            continue

        print(
            f"{key}: {value}"
        )

    print()
    print(
        "===== TRAJREL BUDGET ====="
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

    print()
    print("WROTE:", OUTPUT)


if __name__ == "__main__":
    main()
