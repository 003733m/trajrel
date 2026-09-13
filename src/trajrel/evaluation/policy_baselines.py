"""Policy-level baselines for TrajRel development evaluation."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PolicyBaselineResult:
    policy: str
    critical_records: int
    baseline_critical_kept: int
    final_critical_kept: int
    critical_rescued: int
    forced_records: int
    forced_critical_records: int
    forced_noncritical_records: int
    added_critical_tokens: int
    added_noncritical_tokens: int

    @property
    def critical_recall_before(self) -> float | None:
        if not self.critical_records:
            return None

        return (
            self.baseline_critical_kept
            / self.critical_records
        )

    @property
    def critical_recall_after(self) -> float | None:
        if not self.critical_records:
            return None

        return (
            self.final_critical_kept
            / self.critical_records
        )

    @property
    def incremental_precision(self) -> float | None:
        if not self.forced_records:
            return None

        return (
            self.forced_critical_records
            / self.forced_records
        )

    @property
    def added_total_tokens(self) -> int:
        return (
            self.added_critical_tokens
            + self.added_noncritical_tokens
        )


def identifier_present(
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


def matching_identifiers(
    text: str,
    identifiers: Iterable[str],
) -> tuple[str, ...]:
    return tuple(
        identifier
        for identifier in identifiers
        if identifier_present(
            text,
            identifier,
        )
    )


def record_tokens(
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


def base_keep(
    record: dict[str, Any],
) -> bool:
    value = record.get(
        "legacy_normalized_a_keep"
    )

    if not isinstance(value, bool):
        raise TypeError(
            "Scorable record has no boolean "
            "legacy_normalized_a_keep"
        )

    return value


def _record_text(
    record: dict[str, Any],
) -> str:
    raw = record.get("raw")

    if isinstance(raw, str):
        return raw

    return str(
        record.get("text", "")
    )


def _scorable_records(
    unit: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    records = unit.get("records")

    if not isinstance(records, list):
        raise TypeError(
            "Unit records is not a list"
        )

    return [
        (index, record)
        for index, record in enumerate(records)
        if isinstance(record, dict)
        and bool(
            record.get(
                "legacy_scorable"
            )
        )
    ]


def forced_record_indices(
    unit: dict[str, Any],
    identifiers: Iterable[str],
) -> tuple[int, ...]:
    identifiers = tuple(identifiers)

    if not identifiers:
        return ()

    result: list[int] = []

    for index, record in _scorable_records(
        unit
    ):
        if base_keep(record):
            continue

        text = _record_text(record)

        if any(
            identifier_present(
                text,
                identifier,
            )
            for identifier in identifiers
        ):
            result.append(index)

    return tuple(result)


def _evaluate_selected_indices(
    *,
    units: Sequence[dict[str, Any]],
    policy_name: str,
    selected_indices_by_unit: dict[
        str,
        set[int],
    ],
) -> PolicyBaselineResult:
    critical_records = 0
    baseline_critical_kept = 0
    final_critical_kept = 0
    critical_rescued = 0
    forced_records = 0
    forced_critical_records = 0
    forced_noncritical_records = 0
    added_critical_tokens = 0
    added_noncritical_tokens = 0

    for unit in units:
        selected = selected_indices_by_unit.get(
            unit["unit_id"],
            set(),
        )

        for index, record in _scorable_records(
            unit
        ):
            critical = bool(
                record.get("critical")
            )

            before = base_keep(record)

            if critical:
                critical_records += 1

                if before:
                    baseline_critical_kept += 1

            forced = (
                not before
                and index in selected
            )

            after = (
                before
                or forced
            )

            if critical and after:
                final_critical_kept += 1

            if (
                critical
                and not before
                and after
            ):
                critical_rescued += 1

            if not forced:
                continue

            forced_records += 1

            tokens = record_tokens(record)

            if critical:
                forced_critical_records += 1
                added_critical_tokens += tokens
            else:
                forced_noncritical_records += 1
                added_noncritical_tokens += (
                    tokens
                )

    return PolicyBaselineResult(
        policy=policy_name,
        critical_records=critical_records,
        baseline_critical_kept=(
            baseline_critical_kept
        ),
        final_critical_kept=(
            final_critical_kept
        ),
        critical_rescued=critical_rescued,
        forced_records=forced_records,
        forced_critical_records=(
            forced_critical_records
        ),
        forced_noncritical_records=(
            forced_noncritical_records
        ),
        added_critical_tokens=(
            added_critical_tokens
        ),
        added_noncritical_tokens=(
            added_noncritical_tokens
        ),
    )


def evaluate_identifier_policy(
    *,
    units: Sequence[dict[str, Any]],
    policy_name: str,
    identifiers_by_unit: dict[
        str,
        tuple[str, ...],
    ],
) -> PolicyBaselineResult:
    selected: dict[str, set[int]] = {}

    for unit in units:
        unit_id = unit["unit_id"]

        selected[unit_id] = set(
            forced_record_indices(
                unit,
                identifiers_by_unit.get(
                    unit_id,
                    (),
                ),
            )
        )

    return _evaluate_selected_indices(
        units=units,
        policy_name=policy_name,
        selected_indices_by_unit=selected,
    )


def evaluate_random_same_record_budget(
    *,
    units: Sequence[dict[str, Any]],
    forced_budget_by_unit: dict[
        str,
        int,
    ],
    seed: int,
) -> PolicyBaselineResult:
    rng = random.Random(seed)

    selected: dict[str, set[int]] = {}

    for unit in units:
        eligible = [
            index
            for index, record
            in _scorable_records(unit)
            if not base_keep(record)
        ]

        budget = min(
            forced_budget_by_unit.get(
                unit["unit_id"],
                0,
            ),
            len(eligible),
        )

        selected[
            unit["unit_id"]
        ] = set(
            rng.sample(
                eligible,
                budget,
            )
        )

    return _evaluate_selected_indices(
        units=units,
        policy_name=(
            "random_same_record_budget_"
            f"seed_{seed}"
        ),
        selected_indices_by_unit=selected,
    )


def evaluate_random_same_token_budget(
    *,
    units: Sequence[dict[str, Any]],
    token_budget_by_unit: dict[
        str,
        int,
    ],
    seed: int,
) -> PolicyBaselineResult:
    """Randomly retain dropped records without exceeding TrajRel's token budget.

    Budget matching is performed independently per unit. Record granularity
    means the random baseline may use fewer tokens than the available budget.
    """
    rng = random.Random(seed)

    selected: dict[str, set[int]] = {}

    for unit in units:
        unit_id = unit["unit_id"]

        remaining = max(
            0,
            token_budget_by_unit.get(
                unit_id,
                0,
            ),
        )

        chosen: set[int] = set()

        if remaining > 0:
            candidates = [
                (index, record)
                for index, record
                in _scorable_records(unit)
                if not base_keep(record)
            ]

            rng.shuffle(candidates)

            for index, record in candidates:
                tokens = record_tokens(
                    record
                )

                if tokens > remaining:
                    continue

                chosen.add(index)
                remaining -= tokens

                if remaining == 0:
                    break

        selected[unit_id] = chosen

    return _evaluate_selected_indices(
        units=units,
        policy_name=(
            "random_same_token_budget_"
            f"seed_{seed}"
        ),
        selected_indices_by_unit=selected,
    )


def evaluate_recent_same_record_budget(
    *,
    units: Sequence[dict[str, Any]],
    forced_budget_by_unit: dict[
        str,
        int,
    ],
) -> PolicyBaselineResult:
    """Keep the most recent eligible drops with TrajRel's per-unit budget.

    Corpus record order is treated as chronological context order.
    """
    selected: dict[str, set[int]] = {}

    for unit in units:
        eligible = [
            index
            for index, record
            in _scorable_records(unit)
            if not base_keep(record)
        ]

        budget = min(
            forced_budget_by_unit.get(
                unit["unit_id"],
                0,
            ),
            len(eligible),
        )

        if budget:
            chosen = set(
                eligible[-budget:]
            )
        else:
            chosen = set()

        selected[
            unit["unit_id"]
        ] = chosen

    return _evaluate_selected_indices(
        units=units,
        policy_name=(
            "recent_same_record_budget"
        ),
        selected_indices_by_unit=selected,
    )


def evaluate_budget_matched_b_only(
    *,
    units: Sequence[dict[str, Any]],
    b_by_unit: dict[
        str,
        tuple[str, ...],
    ],
    forced_budget_by_unit: dict[
        str,
        int,
    ],
) -> PolicyBaselineResult:
    """B-only with the same per-unit record budget as B∩Q.

    B identifiers retain their selector ranking. For each B identifier,
    matching dropped records are considered from most recent to oldest.
    """
    selected: dict[str, set[int]] = {}

    for unit in units:
        unit_id = unit["unit_id"]

        budget = max(
            0,
            forced_budget_by_unit.get(
                unit_id,
                0,
            ),
        )

        chosen: set[int] = set()

        if budget:
            records = _scorable_records(
                unit
            )

            for identifier in b_by_unit.get(
                unit_id,
                (),
            ):
                matching = [
                    index
                    for index, record
                    in reversed(records)
                    if index not in chosen
                    and not base_keep(record)
                    and identifier_present(
                        _record_text(record),
                        identifier,
                    )
                ]

                for index in matching:
                    chosen.add(index)

                    if len(chosen) >= budget:
                        break

                if len(chosen) >= budget:
                    break

        selected[unit_id] = chosen

    return _evaluate_selected_indices(
        units=units,
        policy_name=(
            "B_only_same_record_budget"
        ),
        selected_indices_by_unit=selected,
    )


def result_to_dict(
    result: PolicyBaselineResult,
) -> dict[str, Any]:
    return {
        "policy": result.policy,
        "critical_records": (
            result.critical_records
        ),
        "baseline_critical_kept": (
            result.baseline_critical_kept
        ),
        "final_critical_kept": (
            result.final_critical_kept
        ),
        "critical_rescued": (
            result.critical_rescued
        ),
        "forced_records": (
            result.forced_records
        ),
        "forced_critical_records": (
            result.forced_critical_records
        ),
        "forced_noncritical_records": (
            result.forced_noncritical_records
        ),
        "added_critical_tokens": (
            result.added_critical_tokens
        ),
        "added_noncritical_tokens": (
            result.added_noncritical_tokens
        ),
        "added_total_tokens": (
            result.added_total_tokens
        ),
        "critical_recall_before": (
            result.critical_recall_before
        ),
        "critical_recall_after": (
            result.critical_recall_after
        ),
        "incremental_precision": (
            result.incremental_precision
        ),
    }


def evaluate_budget_matched_b_only_token(
    *,
    units: Sequence[dict[str, Any]],
    b_by_unit: dict[
        str,
        tuple[str, ...],
    ],
    token_budget_by_unit: dict[
        str,
        int,
    ],
) -> PolicyBaselineResult:
    """B-only under the same per-unit token budget as B∩Q.

    B identifiers retain selector ranking. Matching dropped records are
    considered from most recent to oldest. A record is retained only when
    it fits inside the remaining token budget.
    """
    selected: dict[str, set[int]] = {}

    for unit in units:
        unit_id = unit["unit_id"]

        remaining = max(
            0,
            token_budget_by_unit.get(
                unit_id,
                0,
            ),
        )

        chosen: set[int] = set()

        if remaining:
            records = _scorable_records(
                unit
            )

            for identifier in b_by_unit.get(
                unit_id,
                (),
            ):
                matching = [
                    (index, record)
                    for index, record
                    in reversed(records)
                    if index not in chosen
                    and not base_keep(record)
                    and identifier_present(
                        _record_text(record),
                        identifier,
                    )
                ]

                for index, record in matching:
                    tokens = record_tokens(
                        record
                    )

                    if tokens > remaining:
                        continue

                    chosen.add(index)
                    remaining -= tokens

                    if remaining == 0:
                        break

                if remaining == 0:
                    break

        selected[unit_id] = chosen

    return _evaluate_selected_indices(
        units=units,
        policy_name=(
            "B_only_same_token_budget"
        ),
        selected_indices_by_unit=selected,
    )


def evaluate_random_b_cardinality_sham(
    *,
    units: Sequence[dict[str, Any]],
    b_by_unit: dict[
        str,
        tuple[str, ...],
    ],
    adopted_cardinality_by_unit: dict[
        str,
        int,
    ],
    seed: int,
) -> PolicyBaselineResult:
    """Choose random B identifiers using the same |B∩Q| per unit.

    This preserves the number of adopted bridge identifiers while removing
    the current-action Q conditioning.
    """
    rng = random.Random(seed)

    identifiers_by_unit: dict[
        str,
        tuple[str, ...],
    ] = {}

    for unit in units:
        unit_id = unit["unit_id"]

        candidates = tuple(
            b_by_unit.get(
                unit_id,
                (),
            )
        )

        count = min(
            max(
                0,
                adopted_cardinality_by_unit.get(
                    unit_id,
                    0,
                ),
            ),
            len(candidates),
        )

        if count:
            identifiers_by_unit[
                unit_id
            ] = tuple(
                rng.sample(
                    candidates,
                    count,
                )
            )
        else:
            identifiers_by_unit[
                unit_id
            ] = ()

    return evaluate_identifier_policy(
        units=units,
        policy_name=(
            "random_B_cardinality_sham_"
            f"seed_{seed}"
        ),
        identifiers_by_unit=(
            identifiers_by_unit
        ),
    )
