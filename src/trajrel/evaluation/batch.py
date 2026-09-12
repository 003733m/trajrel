"""Batch evaluation for TrajRel development experiments.

Historical reproduction and new component ablations are intentionally
separated.

Historical reproduction:
    frozen historical B
    + historical conservative search Q
    + B intersection Q

Component ablations:
    recomputed scorer/selector B
    + the same conservative search Q

The primary causal view uses history strictly before the current action.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajrel.ablations import (
    SCORER_ABLATIONS,
    SELECTOR_ABLATIONS,
    AblationSpec,
    SelectorAblationSpec,
    rank_ablation_outputs,
    select_ablation_candidates,
)
from trajrel.adoption import (
    adopted_bridge_identifiers,
)
from trajrel.query_signals import (
    structured_search_identifiers_from_command,
)


@dataclass(frozen=True, slots=True)
class EvaluationView:
    name: str
    history_field: str
    unit_flag: str
    record_flag: str = "legacy_scorable"


LEGACY_MEASUREMENT_VIEW = EvaluationView(
    name="legacy_measurement_stratum",
    history_field="legacy_history_before_target",
    unit_flag="legacy_strict_hash_valid",
)

CAUSAL_ACTION_VIEW = EvaluationView(
    name="causal_action_history",
    history_field="history_before_action",
    unit_flag="replay_valid",
)


def load_corpus(
    path: Path,
) -> dict[str, Any]:
    corpus = json.loads(
        path.read_text()
    )

    if corpus.get("schema_version") != 2:
        raise ValueError(
            "Batch evaluator expects "
            "schema_version=2"
        )

    if not isinstance(
        corpus.get("units"),
        list,
    ):
        raise TypeError(
            "Corpus units is not a list"
        )

    return corpus


def _action_text(
    unit: dict[str, Any],
) -> str:
    action = unit.get("current_action")

    if not isinstance(action, dict):
        return ""

    command = action.get("command")

    if isinstance(command, str):
        return command

    return ""


def _current_q(
    unit: dict[str, Any],
) -> tuple[str, ...]:
    """Q = structured identifiers in the current search pattern."""
    return (
        structured_search_identifiers_from_command(
            _action_text(unit)
        )
    )


def _history_outputs(
    unit: dict[str, Any],
    *,
    view: EvaluationView,
    max_outputs: int,
) -> tuple[str, ...]:
    history = unit.get(
        view.history_field
    )

    if not isinstance(history, list):
        raise TypeError(
            f"{view.history_field} is not a list"
        )

    outputs: list[str] = []

    for event in history:
        if not isinstance(event, dict):
            continue

        content = event.get("content")

        if isinstance(content, str) and content:
            outputs.append(content)

    if max_outputs <= 0:
        return ()

    return tuple(
        outputs[-max_outputs:]
    )


def _identifier_present(
    text: str,
    identifier: str,
) -> bool:
    if not identifier:
        return False

    pattern = re.compile(
        rf"(?<![A-Za-z0-9_])"
        rf"{re.escape(identifier)}"
        rf"(?![A-Za-z0-9_])"
    )

    return bool(
        pattern.search(text)
    )


def _record_tokens(
    record: dict[str, Any],
) -> int:
    value = record.get(
        "legacy_record_tokens"
    )

    if (
        isinstance(value, int)
        and value >= 0
    ):
        return value

    return 0


def _legacy_base_keep(
    record: dict[str, Any],
    *,
    mode: str = "B",
) -> bool:
    """Historical floor baseline.

    The historical adopted-floor probe was layered over the B replay
    condition. A is retained separately as the direct OFF replay measurement.
    """
    kept = record.get(
        "legacy_normalized_kept"
    )

    if not isinstance(kept, dict):
        return False

    return bool(
        kept.get(mode)
    )


def _ablation_base_keep(
    record: dict[str, Any],
) -> bool:
    """Stable base compressor decision used for new component evaluation."""
    value = record.get(
        "legacy_normalized_a_keep"
    )

    if not isinstance(value, bool):
        raise TypeError(
            "Scorable record has no boolean "
            "legacy_normalized_a_keep"
        )

    return value


def reproduce_legacy_adopted_floor(
    corpus: dict[str, Any],
) -> dict[str, Any]:
    """Reproduce the historical frozen-B adopted-floor measurement."""
    result: dict[str, Any] = {
        "units": 0,
        "scorable_records": 0,
        "critical_records": 0,
        "units_with_Q": 0,
        "units_with_frozen_B": 0,
        "units_with_adopted_bridge": 0,
        "baseline_critical_kept": 0,
        "final_critical_kept": 0,
        "critical_rescued": 0,
        "critical_regressed": 0,
        "forced_records": 0,
        "forced_critical_records": 0,
        "forced_noncritical_records": 0,
        "added_critical_tokens": 0,
        "added_noncritical_tokens": 0,
        "details": [],
    }

    for unit in corpus["units"]:
        if not isinstance(unit, dict):
            continue

        if not bool(
            unit.get(
                "legacy_strict_hash_valid"
            )
        ):
            continue

        result["units"] += 1

        q = _current_q(unit)

        if q:
            result["units_with_Q"] += 1

        frozen = unit.get(
            "legacy_frozen_bridges"
        )

        if not isinstance(frozen, list):
            frozen = []

        frozen_b = tuple(
            token
            for token in frozen
            if isinstance(token, str)
        )

        if frozen_b:
            result[
                "units_with_frozen_B"
            ] += 1

        adopted = adopted_bridge_identifiers(
            frozen_b,
            q,
        )

        if adopted:
            result[
                "units_with_adopted_bridge"
            ] += 1

        detail = {
            "unit_id": unit.get("unit_id"),
            "task_id": unit.get("task_id"),
            "Q": list(q),
            "B": list(frozen_b),
            "A": list(adopted),
            "forced_records": [],
        }

        records = unit.get("records")

        if not isinstance(records, list):
            raise TypeError(
                "Unit records is not a list"
            )

        for record in records:
            if not isinstance(record, dict):
                continue

            if not bool(
                record.get(
                    "legacy_scorable"
                )
            ):
                continue

            result[
                "scorable_records"
            ] += 1

            critical = bool(
                record.get("critical")
            )

            if critical:
                result[
                    "critical_records"
                ] += 1

            base_keep = _legacy_base_keep(
                record,
                mode="B",
            )

            if (
                critical
                and base_keep
            ):
                result[
                    "baseline_critical_kept"
                ] += 1

            raw = record.get("raw")

            if not isinstance(raw, str):
                raw = str(
                    record.get("text", "")
                )

            matched = tuple(
                identifier
                for identifier in adopted
                if _identifier_present(
                    raw,
                    identifier,
                )
            )

            forced = (
                not base_keep
                and bool(matched)
            )

            final_keep = (
                base_keep
                or forced
            )

            if (
                critical
                and final_keep
            ):
                result[
                    "final_critical_kept"
                ] += 1

            if (
                critical
                and not base_keep
                and final_keep
            ):
                result[
                    "critical_rescued"
                ] += 1

            if (
                critical
                and base_keep
                and not final_keep
            ):
                result[
                    "critical_regressed"
                ] += 1

            if not forced:
                continue

            tokens = _record_tokens(
                record
            )

            result[
                "forced_records"
            ] += 1

            if critical:
                result[
                    "forced_critical_records"
                ] += 1
                result[
                    "added_critical_tokens"
                ] += tokens
            else:
                result[
                    "forced_noncritical_records"
                ] += 1
                result[
                    "added_noncritical_tokens"
                ] += tokens

            detail[
                "forced_records"
            ].append(
                {
                    "path": record.get(
                        "path"
                    ),
                    "line": record.get(
                        "line"
                    ),
                    "critical": critical,
                    "tokens": tokens,
                    "matched_identifiers": list(
                        matched
                    ),
                }
            )

        result["details"].append(
            detail
        )

    denominator = result[
        "critical_records"
    ]

    result[
        "critical_recall_before"
    ] = (
        result[
            "baseline_critical_kept"
        ]
        / denominator
        if denominator
        else None
    )

    result[
        "critical_recall_after"
    ] = (
        result[
            "final_critical_kept"
        ]
        / denominator
        if denominator
        else None
    )

    result["added_total_tokens"] = (
        result["added_critical_tokens"]
        + result[
            "added_noncritical_tokens"
        ]
    )

    return result


def evaluate_variant(
    corpus: dict[str, Any],
    *,
    view: EvaluationView,
    scorer_spec: AblationSpec,
    selector_spec: SelectorAblationSpec,
    variant: str,
    family: str,
) -> dict[str, Any]:
    """Evaluate one recomputed scorer/selector configuration."""
    max_outputs = corpus.get(
        "reference_history_max_tool_outputs",
        8,
    )

    if not isinstance(max_outputs, int):
        raise TypeError(
            "reference_history_max_tool_outputs "
            "is not an integer"
        )

    result: dict[str, Any] = {
        "family": family,
        "variant": variant,
        "view": view.name,
        "units": 0,
        "scorable_records": 0,
        "critical_records": 0,
        "units_with_ranked_candidates": 0,
        "units_with_selected_bridges": 0,
        "units_with_Q": 0,
        "units_with_adopted_bridge": 0,
        "baseline_critical_kept": 0,
        "final_critical_kept": 0,
        "critical_rescued": 0,
        "critical_regressed": 0,
        "forced_records": 0,
        "forced_critical_records": 0,
        "forced_noncritical_records": 0,
        "added_critical_tokens": 0,
        "added_noncritical_tokens": 0,
        "details": [],
    }

    for unit in corpus["units"]:
        if not isinstance(unit, dict):
            continue

        if not bool(
            unit.get(view.unit_flag)
        ):
            continue

        result["units"] += 1

        history = _history_outputs(
            unit,
            view=view,
            max_outputs=max_outputs,
        )

        # Do not truncate before target conditioning.
        #
        # Historical semantics:
        #   historical scorer -> full candidate pool
        #   target-conditioned selector -> final top-k
        #
        # Premature top-k here can remove a lower-ranked
        # historical identifier that becomes highly relevant
        # once the current target is considered.
        ranked = rank_ablation_outputs(
            history,
            spec=scorer_spec,
            top_k=None,
        )

        if ranked:
            result[
                "units_with_ranked_candidates"
            ] += 1

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
            spec=selector_spec,
            top_k=6,
        )

        selected_tokens = tuple(
            item.candidate.token
            for item in selected
        )

        if selected_tokens:
            result[
                "units_with_selected_bridges"
            ] += 1

        q = _current_q(unit)

        if q:
            result[
                "units_with_Q"
            ] += 1

        adopted = adopted_bridge_identifiers(
            selected_tokens,
            q,
        )

        if adopted:
            result[
                "units_with_adopted_bridge"
            ] += 1

        frozen = unit.get(
            "legacy_frozen_bridges"
        )

        if not isinstance(frozen, list):
            frozen = []

        detail = {
            "unit_id": unit.get("unit_id"),
            "task_id": unit.get("task_id"),
            "ranked_candidates": [
                candidate.token
                for candidate in ranked
            ],
            "selected_bridges": list(
                selected_tokens
            ),
            "frozen_bridges": frozen,
            "current_identifiers": list(q),
            "adopted_bridges": list(
                adopted
            ),
            "forced_records": [],
        }

        records = unit.get("records")

        if not isinstance(records, list):
            raise TypeError(
                "Unit records is not a list"
            )

        for record in records:
            if not isinstance(record, dict):
                continue

            if not bool(
                record.get(
                    view.record_flag
                )
            ):
                continue

            result[
                "scorable_records"
            ] += 1

            critical = bool(
                record.get("critical")
            )

            if critical:
                result[
                    "critical_records"
                ] += 1

            base_keep = (
                _ablation_base_keep(
                    record
                )
            )

            if (
                critical
                and base_keep
            ):
                result[
                    "baseline_critical_kept"
                ] += 1

            raw = record.get("raw")

            if not isinstance(raw, str):
                raw = str(
                    record.get("text", "")
                )

            matched = tuple(
                identifier
                for identifier in adopted
                if _identifier_present(
                    raw,
                    identifier,
                )
            )

            forced = (
                not base_keep
                and bool(matched)
            )

            final_keep = (
                base_keep
                or forced
            )

            if (
                critical
                and final_keep
            ):
                result[
                    "final_critical_kept"
                ] += 1

            if (
                critical
                and not base_keep
                and final_keep
            ):
                result[
                    "critical_rescued"
                ] += 1

            if (
                critical
                and base_keep
                and not final_keep
            ):
                result[
                    "critical_regressed"
                ] += 1

            if not forced:
                continue

            tokens = _record_tokens(
                record
            )

            result[
                "forced_records"
            ] += 1

            if critical:
                result[
                    "forced_critical_records"
                ] += 1
                result[
                    "added_critical_tokens"
                ] += tokens
            else:
                result[
                    "forced_noncritical_records"
                ] += 1
                result[
                    "added_noncritical_tokens"
                ] += tokens

            detail[
                "forced_records"
            ].append(
                {
                    "path": record.get(
                        "path"
                    ),
                    "line": record.get(
                        "line"
                    ),
                    "critical": critical,
                    "tokens": tokens,
                    "matched_identifiers": list(
                        matched
                    ),
                }
            )

        result["details"].append(
            detail
        )

    critical_total = result[
        "critical_records"
    ]

    result[
        "critical_recall_before"
    ] = (
        result[
            "baseline_critical_kept"
        ]
        / critical_total
        if critical_total
        else None
    )

    result[
        "critical_recall_after"
    ] = (
        result[
            "final_critical_kept"
        ]
        / critical_total
        if critical_total
        else None
    )

    result[
        "critical_recall_delta"
    ] = (
        result[
            "critical_recall_after"
        ]
        - result[
            "critical_recall_before"
        ]
        if critical_total
        else None
    )

    forced_total = result[
        "forced_records"
    ]

    result[
        "incremental_precision"
    ] = (
        result[
            "forced_critical_records"
        ]
        / forced_total
        if forced_total
        else None
    )

    result[
        "added_total_tokens"
    ] = (
        result[
            "added_critical_tokens"
        ]
        + result[
            "added_noncritical_tokens"
        ]
    )

    if result[
        "critical_regressed"
    ] != 0:
        raise AssertionError(
            "Monotonic floor produced "
            "KEEP-to-DROP regression"
        )

    return result


def run_ablation_suite(
    corpus: dict[str, Any],
    *,
    view: EvaluationView,
) -> dict[str, Any]:
    full_scorer = (
        SCORER_ABLATIONS["full"]
    )

    full_selector = (
        SELECTOR_ABLATIONS["full"]
    )

    scorer_results = {
        name: evaluate_variant(
            corpus,
            view=view,
            scorer_spec=spec,
            selector_spec=full_selector,
            variant=name,
            family="scorer",
        )
        for name, spec
        in SCORER_ABLATIONS.items()
    }

    selector_results = {
        name: evaluate_variant(
            corpus,
            view=view,
            scorer_spec=full_scorer,
            selector_spec=spec,
            variant=name,
            family="selector",
        )
        for name, spec
        in SELECTOR_ABLATIONS.items()
    }

    return {
        "view": view.name,
        "history_field": (
            view.history_field
        ),
        "unit_flag": view.unit_flag,
        "record_flag": view.record_flag,
        "query_signal": (
            "structured_search_pattern"
        ),
        "scorer_ablations": (
            scorer_results
        ),
        "selector_ablations": (
            selector_results
        ),
    }
