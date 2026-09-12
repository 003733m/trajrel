from trajrel.importers.headroom_development import (
    normalize_unit,
)


def _call(
    call_id: str,
    name: str,
    arguments: str,
):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _output(
    call_id: str,
    output: str,
):
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def test_primary_history_stops_before_current_action():
    items = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Fix the bug.",
                }
            ],
        },
        _call(
            "old",
            "bash",
            '{"command":"pytest"}',
        ),
        _output(
            "old",
            "historical failure evidence",
        ),
        _call(
            "search",
            "bash",
            '{"command":"rg important_symbol ."}',
        ),
        _call(
            "parallel",
            "read",
            '{"filePath":"x.py"}',
        ),
        _output(
            "parallel",
            "late evidence after action",
        ),
        _output(
            "search",
            "x.py:10: important_symbol",
        ),
    ]

    unit = {
        "request_id": "req-1",
        "call_id": "search",
        "before_index": 6,
        "command": "rg important_symbol .",
        "target": "x.py:10: important_symbol",
        "records": [
            {
                "raw": "x.py:10: important_symbol",
                "path": "x.py",
                "line": 10,
                "text": "important_symbol",
                "fix_adjacent": True,
                "captured_keep": False,
                "kept": {
                    "A": False,
                    "B": False,
                    "C": True,
                    "D": True,
                },
            }
        ],
        "replay_valid": True,
        "off_unit_hash_matches_captured": True,
    }

    request = {
        "body": {
            "input": items,
        }
    }

    normalized = normalize_unit(
        task_id="GXX",
        task={
            "source_commit": "abc",
            "subject": "synthetic",
        },
        unit=unit,
        request=request,
        wire_path="wire.json",
    )

    assert normalized["action_index"] == 3
    assert normalized["target_output_index"] == 6

    primary = [
        event["content"]
        for event in normalized[
            "history_before_action"
        ]
    ]

    legacy = [
        event["content"]
        for event in normalized[
            "legacy_history_before_target"
        ]
    ]

    assert primary == [
        "historical failure evidence"
    ]

    assert legacy == [
        "historical failure evidence",
        "late evidence after action",
    ]


def test_target_output_never_enters_either_history():
    items = [
        _call(
            "search",
            "bash",
            '{"command":"rg foo ."}',
        ),
        _output(
            "search",
            "TARGET CONTENT",
        ),
    ]

    normalized = normalize_unit(
        task_id="GXX",
        task={},
        unit={
            "request_id": "req",
            "call_id": "search",
            "before_index": 1,
            "command": "rg foo .",
            "records": [],
            "replay_valid": True,
            "off_unit_hash_matches_captured": True,
        },
        request={
            "body": {
                "input": items,
            }
        },
        wire_path="wire.json",
    )

    assert normalized["history_before_action"] == []
    assert (
        normalized[
            "legacy_history_before_target"
        ]
        == []
    )


def test_user_context_and_record_labels_are_preserved():
    items = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Investigate provider routing.",
                }
            ],
        },
        _call(
            "search",
            "bash",
            '{"command":"rg route ."}',
        ),
        _output(
            "search",
            "registry.py:10: route",
        ),
    ]

    normalized = normalize_unit(
        task_id="G01",
        task={},
        unit={
            "request_id": "req",
            "call_id": "search",
            "before_index": 2,
            "command": "rg route .",
            "query_context": "bash rg route .",
            "records": [
                {
                    "raw": "registry.py:10: route",
                    "path": "registry.py",
                    "line": 10,
                    "text": "route",
                    "fix_adjacent": True,
                    "captured_keep": False,
                    "kept": {
                        "A": False,
                        "B": False,
                        "C": True,
                        "D": True,
                    },
                }
            ],
            "replay_valid": True,
            "off_unit_hash_matches_captured": True,
        },
        request={
            "body": {
                "input": items,
            }
        },
        wire_path="wire.json",
    )

    assert (
        normalized["user_context"]
        == "Investigate provider routing."
    )

    assert (
        normalized["current_action"]["command"]
        == "rg route ."
    )

    record = normalized["records"][0]

    assert record["critical"] is True
    assert (
        record["development_label"]
        == "historical_fix_adjacent"
    )
    assert record["legacy_kept"]["C"] is True
