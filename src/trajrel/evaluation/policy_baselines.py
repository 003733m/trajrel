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

    return bool(pattern.search(text))


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


def evaluate_identifier_policy(
    *,
    units: Sequence[dict[str, Any]],
    policy_name: str,
    identifiers_by_unit: dict[
        str,
        tuple[str, ...],
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
        unit_id = unit["unit_id"]

        identifiers = identifiers_by_unit.get(
            unit_id,
            (),
        )

        records = unit["records"]

        for record in records:
            if not record.get(
                "legacy_scorable"
            ):
                continue

            critical = bool(
                record.get("critical")
            )

            before = base_keep(record)

            if critical:
                critical_records += 1

                if before:
                    baseline_critical_kept += 1

            raw = record.get("raw")

            if not isinstance(raw, str):
                raw = str(
                    record.get("text", "")
                )

            matched = matching_identifiers(
                raw,
                identifiers,
            )

            forced = (
                not before
                and bool(matched)
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

            tokens = record_tokens(
                record
            )

            if critical:
                forced_critical_records += 1
                added_critical_tokens += (
                    tokens
                )
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
        records = [
            record
            for record in unit["records"]
            if record.get(
                "legacy_scorable"
            )
        ]

        eligible_drops = [
            record
            for record in records
            if not base_keep(record)
        ]

        budget = min(
            forced_budget_by_unit.get(
                unit["unit_id"],
                0,
            ),
            len(eligible_drops),
        )

        chosen_ids = {
            id(record)
            for record in rng.sample(
                eligible_drops,
                budget,
            )
        }

        for record in records:
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
                and id(record) in chosen_ids
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

            tokens = record_tokens(
                record
            )

            if critical:
                forced_critical_records += 1
                added_critical_tokens += (
                    tokens
                )
            else:
                forced_noncritical_records += 1
                added_noncritical_tokens += (
                    tokens
                )

    return PolicyBaselineResult(
        policy=f"random_same_record_budget_seed_{seed}",
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
