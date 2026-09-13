"""Task- and unit-level robustness analysis for exact TrajRel B∩Q.

This analysis deliberately excludes the exploratory action-context envelope.
The core policy remains exact adopted-bridge preservation:

    A = B ∩ Q
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    base_keep,
    evaluate_identifier_policy,
    forced_record_indices,
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
    "core_robustness_v1.json"
)


def action_command(
    unit: dict[str, Any],
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
    unit: dict[str, Any],
) -> tuple[str, ...]:
    history = unit.get(
        "history_before_action",
        [],
    )

    outputs: list[str] = []

    for event in history:
        if not isinstance(event, dict):
            continue

        content = event.get(
            "content"
        )

        if (
            isinstance(content, str)
            and content
        ):
            outputs.append(content)

    return tuple(
        outputs[-8:]
    )


def build_signals(
    units: Sequence[
        dict[str, Any]
    ],
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
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

    for unit in units:
        ranked = rank_ablation_outputs(
            history_outputs(unit),
            spec=scorer_spec,
            top_k=None,
        )

        selected = (
            select_ablation_candidates(
                list(ranked),
                target_content=str(
                    unit.get(
                        "target_content",
                        "",
                    )
                ),
                user_context=str(
                    unit.get(
                        "user_context",
                        "",
                    )
                ),
                spec=selector_spec,
                top_k=6,
            )
        )

        b = tuple(
            item.candidate.token
            for item in selected
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

        unit_id = str(
            unit["unit_id"]
        )

        b_by_unit[unit_id] = b
        q_by_unit[unit_id] = q
        a_by_unit[unit_id] = a

    return (
        b_by_unit,
        q_by_unit,
        a_by_unit,
    )


def critical_counts(
    units: Sequence[
        dict[str, Any]
    ],
) -> dict[str, int]:
    total = 0
    kept = 0
    dropped = 0

    for unit in units:
        records = unit.get(
            "records",
            [],
        )

        for record in records:
            if not isinstance(
                record,
                dict,
            ):
                continue

            if not record.get(
                "legacy_scorable"
            ):
                continue

            if not record.get(
                "critical"
            ):
                continue

            total += 1

            if base_keep(record):
                kept += 1
            else:
                dropped += 1

    return {
        "critical_records": total,
        "base_critical_kept": kept,
        "base_critical_dropped": (
            dropped
        ),
    }


def evaluate_subset(
    units: Sequence[
        dict[str, Any]
    ],
    *,
    b_by_unit: dict[
        str,
        tuple[str, ...],
    ],
    q_by_unit: dict[
        str,
        tuple[str, ...],
    ],
    a_by_unit: dict[
        str,
        tuple[str, ...],
    ],
) -> dict[str, Any]:
    return {
        "counts": critical_counts(
            units
        ),
        "B_only": result_to_dict(
            evaluate_identifier_policy(
                units=units,
                policy_name="B_only",
                identifiers_by_unit=(
                    b_by_unit
                ),
            )
        ),
        "Q_only": result_to_dict(
            evaluate_identifier_policy(
                units=units,
                policy_name="Q_only",
                identifiers_by_unit=(
                    q_by_unit
                ),
            )
        ),
        "B_intersect_Q": (
            result_to_dict(
                evaluate_identifier_policy(
                    units=units,
                    policy_name=(
                        "B_intersect_Q"
                    ),
                    identifiers_by_unit=(
                        a_by_unit
                    ),
                )
            )
        ),
    }


def unit_details(
    units: Sequence[
        dict[str, Any]
    ],
    *,
    b_by_unit: dict[
        str,
        tuple[str, ...],
    ],
    q_by_unit: dict[
        str,
        tuple[str, ...],
    ],
    a_by_unit: dict[
        str,
        tuple[str, ...],
    ],
) -> list[dict[str, Any]]:
    details: list[
        dict[str, Any]
    ] = []

    for unit in units:
        unit_id = str(
            unit["unit_id"]
        )

        critical_total = 0
        critical_dropped = 0

        for record in unit[
            "records"
        ]:
            if not isinstance(
                record,
                dict,
            ):
                continue

            if not record.get(
                "legacy_scorable"
            ):
                continue

            if not record.get(
                "critical"
            ):
                continue

            critical_total += 1

            if not base_keep(record):
                critical_dropped += 1

        b_forced = (
            forced_record_indices(
                unit,
                b_by_unit[
                    unit_id
                ],
            )
        )

        q_forced = (
            forced_record_indices(
                unit,
                q_by_unit[
                    unit_id
                ],
            )
        )

        a_forced = (
            forced_record_indices(
                unit,
                a_by_unit[
                    unit_id
                ],
            )
        )

        if not (
            critical_total
            or b_forced
            or q_forced
            or a_forced
        ):
            continue

        def forced_critical(
            indices: tuple[
                int,
                ...,
            ],
            records: list[
                dict[str, Any]
            ] = unit["records"],
        ) -> int:
            return sum(
                bool(
                    records[index].get(
                        "critical"
                    )
                )
                for index in indices
            )

        details.append(
            {
                "task_id": (
                    unit[
                        "task_id"
                    ]
                ),
                "unit_id": unit_id,
                "critical_records": (
                    critical_total
                ),
                "base_dropped_critical": (
                    critical_dropped
                ),
                "B_size": len(
                    b_by_unit[
                        unit_id
                    ]
                ),
                "Q_size": len(
                    q_by_unit[
                        unit_id
                    ]
                ),
                "A_size": len(
                    a_by_unit[
                        unit_id
                    ]
                ),
                "B_only_forced": len(
                    b_forced
                ),
                "B_only_forced_critical": (
                    forced_critical(
                        b_forced
                    )
                ),
                "Q_only_forced": len(
                    q_forced
                ),
                "Q_only_forced_critical": (
                    forced_critical(
                        q_forced
                    )
                ),
                "B_intersect_Q_forced": (
                    len(
                        a_forced
                    )
                ),
                (
                    "B_intersect_Q_"
                    "forced_critical"
                ): forced_critical(
                    a_forced
                ),
                "B": list(
                    b_by_unit[
                        unit_id
                    ]
                ),
                "Q": list(
                    q_by_unit[
                        unit_id
                    ]
                ),
                "A": list(
                    a_by_unit[
                        unit_id
                    ]
                ),
            }
        )

    return details


def main() -> None:
    corpus = json.loads(
        CORPUS.read_text()
    )

    units = [
        unit
        for unit in corpus[
            "units"
        ]
        if unit.get(
            "replay_valid"
        )
    ]

    (
        b_by_unit,
        q_by_unit,
        a_by_unit,
    ) = build_signals(
        units
    )

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for unit in units:
        grouped[
            str(
                unit[
                    "task_id"
                ]
            )
        ].append(unit)

    per_task = {
        task_id: evaluate_subset(
            task_units,
            b_by_unit=b_by_unit,
            q_by_unit=q_by_unit,
            a_by_unit=a_by_unit,
        )
        for task_id, task_units
        in sorted(
            grouped.items()
        )
    }

    leave_one_task_out = {}

    for held_out_task in sorted(
        grouped
    ):
        remaining = [
            unit
            for unit in units
            if str(
                unit[
                    "task_id"
                ]
            )
            != held_out_task
        ]

        leave_one_task_out[
            held_out_task
        ] = evaluate_subset(
            remaining,
            b_by_unit=b_by_unit,
            q_by_unit=q_by_unit,
            a_by_unit=a_by_unit,
        )

    overall = evaluate_subset(
        units,
        b_by_unit=b_by_unit,
        q_by_unit=q_by_unit,
        a_by_unit=a_by_unit,
    )

    details = unit_details(
        units,
        b_by_unit=b_by_unit,
        q_by_unit=q_by_unit,
        a_by_unit=a_by_unit,
    )

    supporting_tasks = {
        policy: sorted(
            task_id
            for task_id, result
            in per_task.items()
            if result[
                policy
            ][
                "critical_rescued"
            ]
            > 0
        )
        for policy in [
            "B_only",
            "Q_only",
            "B_intersect_Q",
        ]
    }

    result = {
        "evaluation_type": (
            "core_robustness_v1"
        ),
        "held_out": False,
        "core_policy": (
            "exact_B_intersect_Q"
        ),
        "explicit_action_context": (
            False
        ),
        "units": len(units),
        "tasks": len(grouped),
        "overall": overall,
        "supporting_tasks": (
            supporting_tasks
        ),
        "per_task": per_task,
        "leave_one_task_out": (
            leave_one_task_out
        ),
        "unit_details": details,
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
        "===== CORE ROBUSTNESS ====="
    )

    counts = overall[
        "counts"
    ]

    print(
        "tasks:",
        len(grouped),
    )
    print(
        "units:",
        len(units),
    )
    print(
        "critical:",
        counts[
            "critical_records"
        ],
    )
    print(
        "base dropped critical:",
        counts[
            "base_critical_dropped"
        ],
    )

    print()
    print(
        "===== SUPPORTING TASKS ====="
    )

    for policy, tasks in (
        supporting_tasks.items()
    ):
        print(
            policy,
            "=>",
            tasks,
        )

    print()
    print(
        "===== PER TASK ====="
    )

    for task_id, task in (
        per_task.items()
    ):
        print()
        print(task_id)

        print(
            "  critical:",
            task[
                "counts"
            ][
                "critical_records"
            ],
        )

        print(
            "  base dropped:",
            task[
                "counts"
            ][
                "base_critical_dropped"
            ],
        )

        for policy in [
            "B_only",
            "Q_only",
            "B_intersect_Q",
        ]:
            row = task[policy]

            print(
                f"  {policy}:",
                "rescue=",
                row[
                    "critical_rescued"
                ],
                "forced=",
                row[
                    "forced_records"
                ],
                "junk_tokens=",
                row[
                    "added_noncritical_tokens"
                ],
            )

    print()
    print(
        "===== LEAVE-ONE-TASK-OUT ====="
    )

    for excluded, subset in (
        leave_one_task_out.items()
    ):
        row = subset[
            "B_intersect_Q"
        ]

        print(
            f"without {excluded}:",
            "critical=",
            row[
                "critical_records"
            ],
            "rescues=",
            row[
                "critical_rescued"
            ],
            "forced=",
            row[
                "forced_records"
            ],
        )

    print()
    print(
        "===== UNIT-LEVEL "
        "B∩Q ACTIVATIONS ====="
    )

    for detail in details:
        forced = detail[
            "B_intersect_Q_forced"
        ]

        if not forced:
            continue

        print()
        print(
            detail["task_id"],
            detail["unit_id"],
        )
        print(
            "  A:",
            detail["A"],
        )
        print(
            "  forced:",
            forced,
        )
        print(
            "  forced critical:",
            detail[
                "B_intersect_Q_"
                "forced_critical"
            ],
        )

    print()
    print(
        "WROTE:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
