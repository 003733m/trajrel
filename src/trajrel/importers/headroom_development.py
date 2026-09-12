"""Normalize historical Headroom replay artifacts for TrajRel development.

This importer intentionally preserves two temporal histories:

``history_before_action``
    Tool outputs observed strictly before the producing/current action was
    issued. This is the primary history for causal TrajRel evaluation.

``legacy_history_before_target``
    Tool outputs observed before the target tool output. This reproduces the
    historical Headroom/KTH cutoff and may include outputs from concurrently
    issued calls that arrived after the current action.

The distinction prevents later-arriving evidence from being treated as support
for an action that had already been issued.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CALL_TYPES = frozenset(
    {
        "function_call",
        "custom_tool_call",
    }
)

OUTPUT_TYPES = frozenset(
    {
        "function_call_output",
        "custom_tool_call_output",
    }
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content_text(value: Any) -> str:
    """Flatten common Responses content representations to text."""
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts: list[str] = []

        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue

            if not isinstance(item, dict):
                continue

            text = item.get("text")

            if isinstance(text, str):
                parts.append(text)

        return "\n".join(parts)

    if isinstance(value, dict):
        text = value.get("text")

        if isinstance(text, str):
            return text

    return ""


def _output_text(value: Any) -> str:
    """Normalize a tool output to textual benchmark content."""
    if isinstance(value, str):
        return value

    text = _content_text(value)

    if text:
        return text

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


def _calls_by_id(
    items: list[Any],
) -> dict[str, tuple[int, dict[str, Any]]]:
    calls: dict[str, tuple[int, dict[str, Any]]] = {}

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        if item.get("type") not in CALL_TYPES:
            continue

        call_id = item.get("call_id")

        if isinstance(call_id, str):
            calls[call_id] = (index, item)

    return calls


def _history_events(
    items: list[Any],
    *,
    cutoff_index: int,
    calls: dict[str, tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Collect grounded tool outputs strictly before ``cutoff_index``."""
    events: list[dict[str, Any]] = []

    for output_index, item in enumerate(items[:cutoff_index]):
        if not isinstance(item, dict):
            continue

        if item.get("type") not in OUTPUT_TYPES:
            continue

        call_id = item.get("call_id")

        if not isinstance(call_id, str):
            continue

        call_info = calls.get(call_id)

        tool_name = ""
        arguments = ""
        call_index = None

        if call_info is not None:
            call_index, call = call_info

            raw_name = call.get("name")
            raw_arguments = call.get("arguments")

            if isinstance(raw_name, str):
                tool_name = raw_name

            if isinstance(raw_arguments, str):
                arguments = raw_arguments

        events.append(
            {
                "output_index": output_index,
                "call_index": call_index,
                "call_id": call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "content": _output_text(item.get("output")),
            }
        )

    return events


def _user_context(
    items: list[Any],
    *,
    cutoff_index: int,
) -> str:
    parts: list[str] = []

    for item in items[:cutoff_index]:
        if not isinstance(item, dict):
            continue

        if item.get("role") != "user":
            continue

        text = _content_text(item.get("content"))

        if text:
            parts.append(text)

    return "\n\n".join(parts)



def _parse_legacy_v3_bridges(
    value: Any,
) -> list[str]:
    """Parse frozen bridge identifiers from historical V3 context."""
    if not isinstance(value, str):
        return []

    prefix = "Trajectory bridge identifiers:"

    for line in value.splitlines():
        stripped = line.strip()

        if not stripped.startswith(prefix):
            continue

        payload = stripped[
            len(prefix):
        ].strip()

        if not payload:
            return []

        return [
            token
            for token in (
                payload
                .replace(",", " ")
                .split()
            )
            if token
        ]

    return []

def normalize_unit(
    *,
    task_id: str,
    task: dict[str, Any],
    unit: dict[str, Any],
    request: dict[str, Any],
    wire_path: str,
) -> dict[str, Any]:
    """Normalize one Headroom matched-replay unit."""
    body = request.get("body")

    if not isinstance(body, dict):
        raise TypeError("Wire request has no body object")

    items = body.get("input")

    if not isinstance(items, list):
        raise TypeError("Wire request body.input is not a list")

    before_index = unit["before_index"]

    if not isinstance(before_index, int):
        raise TypeError("Unit before_index is not an integer")

    if not 0 <= before_index < len(items):
        raise ValueError(
            f"Invalid before_index={before_index} "
            f"for input length={len(items)}"
        )

    target_item = items[before_index]

    if not isinstance(target_item, dict):
        raise TypeError("Target item is not an object")

    if target_item.get("type") not in OUTPUT_TYPES:
        raise ValueError(
            "Target item is not a recognized tool output"
        )

    call_id = unit["call_id"]

    if target_item.get("call_id") != call_id:
        raise ValueError(
            "Target output call_id does not match replay unit"
        )

    calls = _calls_by_id(items)

    producing = calls.get(call_id)

    if producing is None:
        raise ValueError(
            f"Could not find producing call for {call_id}"
        )

    action_index, action = producing

    if action_index >= before_index:
        raise ValueError(
            "Producing action must occur before target output"
        )

    target_content = _output_text(
        target_item.get("output")
    )

    expected_hash = unit.get("target_sha256")

    if (
        isinstance(expected_hash, str)
        and _sha256(target_content) != expected_hash
    ):
        raise ValueError(
            "Target output hash does not match replay artifact"
        )

    action_name = action.get("name")
    action_arguments = action.get("arguments")

    if not isinstance(action_name, str):
        action_name = ""

    if not isinstance(action_arguments, str):
        action_arguments = ""

    records = []

    for record in unit.get("records", []):
        if not isinstance(record, dict):
            continue

        records.append(
            {
                "raw": record.get("raw", ""),
                "path": record.get("path", ""),
                "line": record.get("line"),
                "text": record.get("text", ""),
                "critical": bool(
                    record.get("fix_adjacent")
                ),
                "development_label": (
                    "historical_fix_adjacent"
                ),
                "captured_keep": bool(
                    record.get("captured_keep")
                ),
                "legacy_kept": dict(
                    record.get("kept", {})
                ),
            }
        )

    history_before_action = _history_events(
        items,
        cutoff_index=action_index,
        calls=calls,
    )

    legacy_history_before_target = _history_events(
        items,
        cutoff_index=before_index,
        calls=calls,
    )

    return {
        "unit_id": (
            f"{task_id}:{unit['request_id']}:{call_id}"
        ),
        "task_id": task_id,
        "request_id": unit["request_id"],
        "call_id": call_id,
        "action_index": action_index,
        "target_output_index": before_index,
        "current_action": {
            "tool_name": action_name,
            "arguments": action_arguments,
            "command": unit.get("command", ""),
        },
        "query_context": unit.get(
            "query_context",
            "",
        ),
        "legacy_frozen_bridges": (
            _parse_legacy_v3_bridges(
                unit.get("v3_context")
            )
        ),
        "user_context": _user_context(
            items,
            cutoff_index=action_index,
        ),
        "target_content": target_content,
        "target_sha256": unit.get(
            "target_sha256",
            "",
        ),
        "history_before_action": history_before_action,
        "legacy_history_before_target": (
            legacy_history_before_target
        ),
        "records": records,
        "replay_valid": bool(
            unit.get("replay_valid")
        ),
        "direct_replay_hash_exact": bool(
            unit.get(
                "off_unit_hash_matches_captured"
            )
        ),
        "provenance": {
            "wire_path": wire_path,
            "source_commit": task.get(
                "source_commit",
                "",
            ),
            "subject": task.get(
                "subject",
                "",
            ),
        },
    }


def _index_wire_requests(
    wire_dir: Path,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    indexed: dict[
        str,
        tuple[Path, dict[str, Any]],
    ] = {}

    # Replay units are derived from row["pre"], i.e. the inbound
    # request before Headroom compression. The upstream request is
    # row["post"] and may already contain transformed/compressed content.
    for path in wire_dir.rglob(
        "*http_inbound_request.json"
    ):
        data = json.loads(path.read_text())

        request_id = data.get("request_id")

        if not isinstance(request_id, str):
            continue

        if request_id in indexed:
            raise ValueError(
                f"Duplicate upstream request: {request_id}"
            )

        indexed[request_id] = (path, data)

    return indexed


def _legacy_unit_key(
    unit: dict[str, Any],
) -> tuple[str, str, str]:
    task = unit.get("task")
    request_id = unit.get("request_id")
    call_id = unit.get("call_id")

    if not isinstance(task, str):
        raise TypeError(
            "Legacy unit task is not a string"
        )

    if not isinstance(request_id, str):
        raise TypeError(
            "Legacy unit request_id is not a string"
        )

    if not isinstance(call_id, str):
        raise TypeError(
            "Legacy unit call_id is not a string"
        )

    return task, request_id, call_id


def _record_identity(
    record: dict[str, Any],
) -> tuple[Any, Any, Any]:
    return (
        record.get("path"),
        record.get("line"),
        record.get("text"),
    )


def build_development_corpus(
    *,
    raw_root: Path,
) -> dict[str, Any]:
    replay_path = (
        raw_root
        / "hard_matched_context_replay_results.json"
    )

    legacy_path = (
        raw_root
        / "normalized_retention_measurement.json"
    )

    replay = json.loads(
        replay_path.read_text()
    )

    legacy = json.loads(
        legacy_path.read_text()
    )

    normalized_units: list[dict[str, Any]] = []

    tasks = replay.get("tasks")

    if not isinstance(tasks, dict):
        raise TypeError(
            "Replay artifact has no tasks object"
        )

    legacy_units = legacy.get("units")

    if not isinstance(legacy_units, list):
        raise TypeError(
            "Legacy measurement has no units list"
        )

    legacy_by_key: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    for legacy_unit in legacy_units:
        if not isinstance(legacy_unit, dict):
            continue

        key = _legacy_unit_key(
            legacy_unit
        )

        if key in legacy_by_key:
            raise ValueError(
                f"Duplicate legacy unit: {key}"
            )

        legacy_by_key[key] = legacy_unit

    used_legacy_keys: set[
        tuple[str, str, str]
    ] = set()

    for task_id, task in tasks.items():
        if not isinstance(task, dict):
            continue

        wire_dir = (
            raw_root
            / "results"
            / f"{task_id}-OFF"
            / "codex-wire"
        )

        request_index = _index_wire_requests(
            wire_dir
        )

        for unit in task.get("units", []):
            if not isinstance(unit, dict):
                continue

            request_id = unit.get("request_id")

            if not isinstance(request_id, str):
                raise TypeError(
                    "Replay unit has no request_id"
                )

            call_id = unit.get("call_id")

            if not isinstance(call_id, str):
                raise TypeError(
                    "Replay unit has no call_id"
                )

            match = request_index.get(
                request_id
            )

            if match is None:
                raise ValueError(
                    "No inbound wire request for "
                    f"{task_id}/{request_id}"
                )

            wire_path, request = match

            normalized = normalize_unit(
                task_id=task_id,
                task=task,
                unit=unit,
                request=request,
                wire_path=str(
                    wire_path.relative_to(
                        raw_root
                    )
                ),
            )

            legacy_key = (
                task_id,
                request_id,
                call_id,
            )

            legacy_unit = legacy_by_key.get(
                legacy_key
            )

            if legacy_unit is None:
                raise ValueError(
                    "No legacy normalized unit for "
                    f"{legacy_key}"
                )

            used_legacy_keys.add(
                legacy_key
            )

            normalized[
                "legacy_strict_hash_valid"
            ] = bool(
                legacy_unit.get(
                    "strict_hash_valid"
                )
            )

            normalized[
                "legacy_content_valid"
            ] = bool(
                legacy_unit.get(
                    "content_valid"
                )
            )

            normalized[
                "legacy_bc_replay_stable_across_runs"
            ] = bool(
                legacy_unit.get(
                    "BC_replay_stable_across_runs"
                )
            )

            if (
                normalized[
                    "direct_replay_hash_exact"
                ]
                != normalized[
                    "legacy_strict_hash_valid"
                ]
            ):
                raise ValueError(
                    "Direct replay hash label does "
                    "not match legacy strict hash "
                    f"label for {legacy_key}"
                )

            legacy_records = (
                legacy_unit.get("records")
            )

            if not isinstance(
                legacy_records,
                list,
            ):
                raise TypeError(
                    "Legacy unit records is not a list"
                )

            new_records = normalized["records"]

            if len(new_records) != len(
                legacy_records
            ):
                raise ValueError(
                    "Record-count mismatch for "
                    f"{legacy_key}: "
                    f"{len(new_records)} != "
                    f"{len(legacy_records)}"
                )

            for new_record, legacy_record in zip(
                new_records,
                legacy_records,
                strict=True,
            ):
                if not isinstance(
                    legacy_record,
                    dict,
                ):
                    raise TypeError(
                        "Legacy record is not an object"
                    )

                if (
                    _record_identity(new_record)
                    != _record_identity(
                        legacy_record
                    )
                ):
                    raise ValueError(
                        "Record identity/order mismatch "
                        f"for {legacy_key}"
                    )

                if (
                    new_record["critical"]
                    != bool(
                        legacy_record.get(
                            "fix_adjacent"
                        )
                    )
                ):
                    raise ValueError(
                        "Critical-label mismatch for "
                        f"{legacy_key}"
                    )

                new_record[
                    "legacy_scorable"
                ] = bool(
                    legacy_record.get(
                        "scorable"
                    )
                )

                new_record[
                    "legacy_record_tokens"
                ] = legacy_record.get(
                    "record_tokens"
                )

                new_record[
                    "legacy_normalized_captured_keep"
                ] = legacy_record.get(
                    "captured_keep"
                )

                new_record[
                    "legacy_normalized_a_keep"
                ] = legacy_record.get(
                    "A_keep"
                )

                legacy_kept = (
                    legacy_record.get("kept")
                )

                new_record[
                    "legacy_normalized_kept"
                ] = (
                    dict(legacy_kept)
                    if isinstance(
                        legacy_kept,
                        dict,
                    )
                    else None
                )

            normalized_units.append(
                normalized
            )

    if set(legacy_by_key) != used_legacy_keys:
        missing = sorted(
            set(legacy_by_key)
            - used_legacy_keys
        )

        extra = sorted(
            used_legacy_keys
            - set(legacy_by_key)
        )

        raise ValueError(
            "Replay/legacy unit-set mismatch: "
            f"missing={missing}, extra={extra}"
        )

    replay_valid_units = [
        unit
        for unit in normalized_units
        if unit["replay_valid"]
    ]

    direct_hash_units = [
        unit
        for unit in normalized_units
        if unit["direct_replay_hash_exact"]
    ]

    legacy_strict_units = [
        unit
        for unit in normalized_units
        if unit["legacy_strict_hash_valid"]
    ]

    legacy_content_valid_units = [
        unit
        for unit in normalized_units
        if unit["legacy_content_valid"]
    ]

    all_records = [
        record
        for unit in normalized_units
        for record in unit["records"]
    ]

    legacy_strict_records = [
        record
        for unit in legacy_strict_units
        for record in unit["records"]
    ]

    return {
        "schema_version": 2,
        "name": (
            "headroom_hard_matched_"
            "development_v2"
        ),
        "evaluation_role": (
            "development_post_hoc"
        ),
        "held_out": False,
        "source_evaluation_type": replay.get(
            "evaluation_type"
        ),
        "source_algorithm_freeze": replay.get(
            "algorithm_freeze"
        ),
        "source_normalized_measurement": (
            legacy.get("measurement")
        ),
        "reference_history_max_tool_outputs": 8,
        "history_policy": {
            "primary": (
                "history_before_action"
            ),
            "legacy": (
                "legacy_history_before_target"
            ),
            "reason": (
                "Primary history excludes outputs "
                "that arrived after the current "
                "action was already issued."
            ),
        },
        "strata": {
            "direct_replay_hash_exact": (
                "Captured OFF output hash exactly "
                "matches the direct replay OFF "
                "output hash."
            ),
            "legacy_strict_hash_valid": (
                "Historical normalized-retention "
                "strict hash stratum."
            ),
            "legacy_content_valid": (
                "Historical normalized-retention "
                "record-content validity stratum."
            ),
            "legacy_scorable": (
                "Historical record-level "
                "measurement eligibility."
            ),
        },
        "critical_label": {
            "field": "critical",
            "source": (
                "historical_fix_adjacent"
            ),
            "development_only": True,
        },
        "summary": {
            "tasks": len(tasks),
            "units": len(normalized_units),
            "replay_valid_units": len(
                replay_valid_units
            ),
            "direct_replay_hash_exact_units": (
                len(direct_hash_units)
            ),
            "legacy_strict_units": len(
                legacy_strict_units
            ),
            "legacy_content_valid_units": len(
                legacy_content_valid_units
            ),
            "records": len(all_records),
            "legacy_scorable_records": sum(
                record["legacy_scorable"]
                for record in all_records
            ),
            "legacy_strict_records": len(
                legacy_strict_records
            ),
            "legacy_strict_scorable_records": (
                sum(
                    record["legacy_scorable"]
                    for record
                    in legacy_strict_records
                )
            ),
            "critical_records": sum(
                record["critical"]
                for record in all_records
            ),
            "legacy_strict_critical_records": (
                sum(
                    record["critical"]
                    for record
                    in legacy_strict_records
                )
            ),
            "legacy_strict_critical_scorable_records": (
                sum(
                    record["critical"]
                    and record[
                        "legacy_scorable"
                    ]
                    for record
                    in legacy_strict_records
                )
            ),
        },
        "units": normalized_units,
    }
