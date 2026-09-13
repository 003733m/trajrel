"""Probe a causal search-context envelope around adopted bridge anchors."""

from __future__ import annotations

import json
import shlex
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
    base_keep,
    identifier_present,
    record_tokens,
)
from trajrel.query_signals import (
    structured_search_identifiers_from_command,
)

CORPUS = Path(
    "benchmarks/development/"
    "headroom_hard_matched_v2.json"
)


def action_command(unit: dict) -> str:
    action = unit.get("current_action", {})

    if not isinstance(action, dict):
        return ""

    value = action.get("command", "")

    return value if isinstance(value, str) else ""


def history_outputs(unit: dict) -> tuple[str, ...]:
    outputs = []

    for event in unit["history_before_action"]:
        if not isinstance(event, dict):
            continue

        content = event.get("content")

        if isinstance(content, str) and content:
            outputs.append(content)

    return tuple(outputs[-8:])


def search_context_radius(command: str) -> int:
    """Return explicit rg/grep symmetric context radius.

    Supports:
      -C 8
      -C8
      --context 8
      --context=8

    Deliberately does not invent context when none was requested.
    """
    try:
        args = shlex.split(command)
    except ValueError:
        return 0

    if not any(
        arg in {"rg", "grep"}
        for arg in args
    ):
        return 0

    index = 0

    while index < len(args):
        arg = args[index]

        if arg in {"-C", "--context"}:
            if index + 1 >= len(args):
                return 0

            try:
                return max(
                    0,
                    int(args[index + 1]),
                )
            except ValueError:
                return 0

        if (
            arg.startswith("-C")
            and arg != "-C"
        ):
            try:
                return max(
                    0,
                    int(arg[2:]),
                )
            except ValueError:
                return 0

        if arg.startswith("--context="):
            try:
                return max(
                    0,
                    int(
                        arg.split(
                            "=",
                            1,
                        )[1]
                    ),
                )
            except ValueError:
                return 0

        index += 1

    return 0


def record_text(record: dict) -> str:
    raw = record.get("raw")

    if isinstance(raw, str):
        return raw

    return str(
        record.get("text", "")
    )


def line_number(record: dict) -> int | None:
    value = record.get("line")

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None

    return None


def path_value(record: dict) -> str | None:
    value = record.get("path")

    return value if isinstance(value, str) else None


def main() -> None:
    data = json.loads(
        CORPUS.read_text()
    )

    scorer = SCORER_ABLATIONS["full"]
    selector = SELECTOR_ABLATIONS["full"]

    totals = {
        "base_critical": 0,
        "base_critical_kept": 0,
        "exact_forced": 0,
        "exact_forced_critical": 0,
        "exact_added_tokens": 0,
        "exact_added_noncritical_tokens": 0,
        "envelope_forced": 0,
        "envelope_forced_critical": 0,
        "envelope_added_tokens": 0,
        "envelope_added_noncritical_tokens": 0,
    }

    details = []

    for unit in data["units"]:
        if not unit.get("replay_valid"):
            continue

        ranked = rank_ablation_outputs(
            history_outputs(unit),
            spec=scorer,
            top_k=None,
        )

        selected = select_ablation_candidates(
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
            spec=selector,
            top_k=6,
        )

        b = tuple(
            item.candidate.token
            for item in selected
        )

        command = action_command(unit)

        q = (
            structured_search_identifiers_from_command(
                command
            )
        )

        adopted = adopted_bridge_identifiers(
            b,
            q,
        )

        radius = search_context_radius(
            command
        )

        records = unit["records"]

        scorable = [
            (index, record)
            for index, record
            in enumerate(records)
            if isinstance(record, dict)
            and record.get("legacy_scorable")
        ]

        anchors: list[
            tuple[str, int]
        ] = []

        exact_indices: set[int] = set()

        for index, record in scorable:
            if base_keep(record):
                continue

            text = record_text(record)

            if not any(
                identifier_present(
                    text,
                    identifier,
                )
                for identifier in adopted
            ):
                continue

            exact_indices.add(index)

            path = path_value(record)
            line = line_number(record)

            if (
                path is not None
                and line is not None
            ):
                anchors.append(
                    (path, line)
                )

        envelope_indices = set(
            exact_indices
        )

        if radius > 0:
            for index, record in scorable:
                if base_keep(record):
                    continue

                path = path_value(record)
                line = line_number(record)

                if (
                    path is None
                    or line is None
                ):
                    continue

                if any(
                    path == anchor_path
                    and abs(
                        line - anchor_line
                    ) <= radius
                    for (
                        anchor_path,
                        anchor_line,
                    ) in anchors
                ):
                    envelope_indices.add(
                        index
                    )

        unit_detail = {
            "unit_id": unit["unit_id"],
            "task_id": unit["task_id"],
            "command": command,
            "radius": radius,
            "B": list(b),
            "Q": list(q),
            "A": list(adopted),
            "anchors": [
                {
                    "path": path,
                    "line": line,
                }
                for path, line
                in anchors
            ],
            "new_envelope_records": [],
        }

        for index, record in scorable:
            critical = bool(
                record.get("critical")
            )

            before = base_keep(record)

            if critical:
                totals[
                    "base_critical"
                ] += 1

                if before:
                    totals[
                        "base_critical_kept"
                    ] += 1

            tokens = record_tokens(
                record
            )

            if (
                not before
                and index in exact_indices
            ):
                totals[
                    "exact_forced"
                ] += 1

                totals[
                    "exact_added_tokens"
                ] += tokens

                if critical:
                    totals[
                        "exact_forced_critical"
                    ] += 1
                else:
                    totals[
                        "exact_added_noncritical_tokens"
                    ] += tokens

            if (
                not before
                and index in envelope_indices
            ):
                totals[
                    "envelope_forced"
                ] += 1

                totals[
                    "envelope_added_tokens"
                ] += tokens

                if critical:
                    totals[
                        "envelope_forced_critical"
                    ] += 1
                else:
                    totals[
                        "envelope_added_noncritical_tokens"
                    ] += tokens

                if index not in exact_indices:
                    unit_detail[
                        "new_envelope_records"
                    ].append(
                        {
                            "path": (
                                record.get(
                                    "path"
                                )
                            ),
                            "line": (
                                record.get(
                                    "line"
                                )
                            ),
                            "critical": (
                                critical
                            ),
                            "tokens": tokens,
                            "text": (
                                record_text(
                                    record
                                )[:240]
                            ),
                        }
                    )

        if (
            adopted
            or unit_detail[
                "new_envelope_records"
            ]
        ):
            details.append(
                unit_detail
            )

    base_total = totals[
        "base_critical"
    ]

    exact_after = (
        totals[
            "base_critical_kept"
        ]
        + totals[
            "exact_forced_critical"
        ]
    )

    envelope_after = (
        totals[
            "base_critical_kept"
        ]
        + totals[
            "envelope_forced_critical"
        ]
    )

    result = {
        **totals,
        "exact_critical_after": (
            exact_after
        ),
        "envelope_critical_after": (
            envelope_after
        ),
        "exact_recall_after": (
            exact_after / base_total
        ),
        "envelope_recall_after": (
            envelope_after
            / base_total
        ),
        "exact_incremental_precision": (
            totals[
                "exact_forced_critical"
            ]
            / totals["exact_forced"]
            if totals["exact_forced"]
            else None
        ),
        "envelope_incremental_precision": (
            totals[
                "envelope_forced_critical"
            ]
            / totals[
                "envelope_forced"
            ]
            if totals[
                "envelope_forced"
            ]
            else None
        ),
        "details": details,
    }

    output = Path(
        "experiments/results/"
        "action_context_envelope_probe.json"
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    print(
        "===== EXACT B∩Q ====="
    )
    print(
        "critical kept:",
        f"{totals['base_critical_kept']}"
        f" -> {exact_after}"
        f"/{base_total}",
    )
    print(
        "forced:",
        totals["exact_forced"],
    )
    print(
        "forced critical:",
        totals[
            "exact_forced_critical"
        ],
    )
    print(
        "added tokens:",
        totals[
            "exact_added_tokens"
        ],
    )
    print(
        "added noncritical tokens:",
        totals[
            "exact_added_noncritical_tokens"
        ],
    )
    print(
        "incremental precision:",
        result[
            "exact_incremental_precision"
        ],
    )

    print()
    print(
        "===== B∩Q + ACTION CONTEXT ====="
    )
    print(
        "critical kept:",
        f"{totals['base_critical_kept']}"
        f" -> {envelope_after}"
        f"/{base_total}",
    )
    print(
        "forced:",
        totals[
            "envelope_forced"
        ],
    )
    print(
        "forced critical:",
        totals[
            "envelope_forced_critical"
        ],
    )
    print(
        "added tokens:",
        totals[
            "envelope_added_tokens"
        ],
    )
    print(
        "added noncritical tokens:",
        totals[
            "envelope_added_noncritical_tokens"
        ],
    )
    print(
        "incremental precision:",
        result[
            "envelope_incremental_precision"
        ],
    )

    print()
    print(
        "===== NEW ENVELOPE RECORDS ====="
    )

    for detail in details:
        new_records = detail[
            "new_envelope_records"
        ]

        if not new_records:
            continue

        print()
        print(
            detail["task_id"],
            detail["unit_id"],
        )
        print(
            "radius:",
            detail["radius"],
        )
        print(
            "A:",
            detail["A"],
        )

        for record in new_records:
            print(
                " ",
                "CRITICAL"
                if record["critical"]
                else "noncritical",
                record["path"],
                record["line"],
                "tokens=",
                record["tokens"],
                repr(
                    record["text"]
                ),
            )

    print()
    print("WROTE:", output)


if __name__ == "__main__":
    main()
