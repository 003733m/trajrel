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
        "strict_hash_valid": bool(
            unit.get("replay_valid")
            and unit.get(
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


def build_development_corpus(
    *,
    raw_root: Path,
) -> dict[str, Any]:
    replay_path = (
        raw_root
        / "hard_matched_context_replay_results.json"
    )

    replay = json.loads(replay_path.read_text())

    normalized_units: list[dict[str, Any]] = []

    tasks = replay.get("tasks")

    if not isinstance(tasks, dict):
        raise TypeError(
            "Replay artifact has no tasks object"
        )

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

            match = request_index.get(request_id)

            if match is None:
                raise ValueError(
                    "No upstream wire request for "
                    f"{task_id}/{request_id}"
                )

            wire_path, request = match

            normalized_units.append(
                normalize_unit(
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
            )

    strict_units = sum(
        unit["strict_hash_valid"]
        for unit in normalized_units
    )

    replay_valid_units = sum(
        unit["replay_valid"]
        for unit in normalized_units
    )

    return {
        "schema_version": 1,
        "name": "headroom_hard_matched_development_v1",
        "evaluation_role": "development_post_hoc",
        "held_out": False,
        "source_evaluation_type": replay.get(
            "evaluation_type"
        ),
        "source_algorithm_freeze": replay.get(
            "algorithm_freeze"
        ),
        "reference_history_max_tool_outputs": 8,
        "history_policy": {
            "primary": "history_before_action",
            "legacy": "legacy_history_before_target",
            "reason": (
                "Primary history excludes outputs that arrived "
                "after the current action was already issued."
            ),
        },
        "critical_label": {
            "field": "critical",
            "source": "historical_fix_adjacent",
            "development_only": True,
        },
        "summary": {
            "tasks": len(tasks),
            "units": len(normalized_units),
            "replay_valid_units": replay_valid_units,
            "strict_hash_valid_units": strict_units,
            "records": sum(
                len(unit["records"])
                for unit in normalized_units
            ),
            "strict_records": sum(
                len(unit["records"])
                for unit in normalized_units
                if unit["strict_hash_valid"]
            ),
            "critical_records": sum(
                record["critical"]
                for unit in normalized_units
                for record in unit["records"]
            ),
            "strict_critical_records": sum(
                record["critical"]
                for unit in normalized_units
                if unit["strict_hash_valid"]
                for record in unit["records"]
            ),
        },
        "units": normalized_units,
    }
