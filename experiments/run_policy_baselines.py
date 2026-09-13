"""Run TrajRel policy-level development baselines."""

from __future__ import annotations

import json
from pathlib import Path

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
    evaluate_identifier_policy,
    evaluate_random_same_record_budget,
    identifier_present,
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
    "policy_baselines_v1.json"
)


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

        forced = 0

        for record in unit["records"]:
            if not record.get(
                "legacy_scorable"
            ):
                continue

            if record[
                "legacy_normalized_a_keep"
            ]:
                continue

            raw = record.get(
                "raw",
                record.get(
                    "text",
                    "",
                ),
            )

            if any(
                identifier_present(
                    raw,
                    identifier,
                )
                for identifier in a
            ):
                forced += 1

        forced_budget_by_unit[
            unit_id
        ] = forced

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

    random_results = [
        evaluate_random_same_record_budget(
            units=units,
            forced_budget_by_unit=(
                forced_budget_by_unit
            ),
            seed=seed,
        )
        for seed in range(100)
    ]

    random_rescues = [
        result.critical_rescued
        for result in random_results
    ]

    random_critical = [
        result.forced_critical_records
        for result in random_results
    ]

    random_noncritical_tokens = [
        result.added_noncritical_tokens
        for result in random_results
    ]

    result = {
        "evaluation_type": (
            "development_policy_baselines"
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
        },
        "random_same_record_budget": {
            "runs": len(
                random_results
            ),
            "mean_rescues": (
                sum(random_rescues)
                / len(random_rescues)
            ),
            "runs_with_at_least_one_rescue": (
                sum(
                    value >= 1
                    for value
                    in random_rescues
                )
            ),
            "mean_forced_critical_records": (
                sum(random_critical)
                / len(random_critical)
            ),
            "mean_added_noncritical_tokens": (
                sum(
                    random_noncritical_tokens
                )
                / len(
                    random_noncritical_tokens
                )
            ),
            "results": [
                result_to_dict(
                    row
                )
                for row
                in random_results
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
        "===== POLICY BASELINES ====="
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

    random_summary = result[
        "random_same_record_budget"
    ]

    for key in [
        "runs",
        "mean_rescues",
        "runs_with_at_least_one_rescue",
        "mean_forced_critical_records",
        "mean_added_noncritical_tokens",
    ]:
        print(
            f"{key}: "
            f"{random_summary[key]}"
        )

    print()
    print("WROTE:", OUTPUT)


if __name__ == "__main__":
    main()
