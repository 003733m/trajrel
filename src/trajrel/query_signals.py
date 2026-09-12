"""Current-action query signals used by TrajRel.

For shell text search actions, Q is deliberately conservative: identifiers are
extracted from the rg/grep search pattern, not from paths, working directories,
or unrelated command arguments.

This preserves the distinction between:

- what the agent is explicitly searching for; and
- where/how the agent performs that search.
"""

from __future__ import annotations

import re
import shlex

OPTS_WITH_VALUE = frozenset(
    {
        "-g",
        "--glob",
        "-C",
        "--context",
        "-A",
        "--after-context",
        "-B",
        "--before-context",
        "-m",
        "--max-count",
        "-e",
        "--regexp",
        "-f",
        "--file",
        "-t",
        "--type",
        "-T",
        "--type-not",
        "--encoding",
        "--engine",
        "--ignore-file",
        "--sort",
        "--sortr",
    }
)

IDENT_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
)


def extract_search_pattern(command: str) -> str:
    """Extract the actual rg/grep pattern from a shell command."""
    try:
        args = shlex.split(command)
    except (TypeError, ValueError):
        return ""

    start = None

    for index, arg in enumerate(args):
        if arg in {"rg", "grep"}:
            start = index + 1
            break

    if start is None:
        return ""

    explicit: list[str] = []
    index = start

    while index < len(args):
        arg = args[index]

        if (
            arg in {"-e", "--regexp"}
            and index + 1 < len(args)
        ):
            explicit.append(
                args[index + 1]
            )
            index += 2
            continue

        if arg.startswith("--regexp="):
            explicit.append(
                arg.split("=", 1)[1]
            )

        index += 1

    if explicit:
        return "|".join(explicit)

    index = start

    while index < len(args):
        arg = args[index]

        if arg == "--":
            if index + 1 < len(args):
                return args[index + 1]

            return ""

        if arg in OPTS_WITH_VALUE:
            index += 2
            continue

        if (
            arg.startswith("--")
            and "=" in arg
        ):
            index += 1
            continue

        if arg.startswith("-"):
            index += 1
            continue

        return arg

    return ""


def _is_camel_case(token: str) -> bool:
    return (
        bool(token)
        and token[0].isalpha()
        and any(
            character.islower()
            for character in token
        )
        and any(
            character.isupper()
            for character in token
        )
    )


def structured_search_identifiers(
    pattern: str,
) -> tuple[str, ...]:
    """Extract conservative structured identifiers from a search pattern.

    Keeps:
    - snake_case / _private / ALL_CAPS_WITH_UNDERSCORE
    - CamelCase
    - identifiers explicitly following ``def`` or ``class``

    Generic bare words are intentionally rejected.
    """
    selected: set[str] = set()

    for match in re.finditer(
        r"\b(?:def|class)\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)",
        pattern,
    ):
        selected.add(
            match.group(1)
        )

    for token in IDENT_RE.findall(pattern):
        if len(token) < 3:
            continue

        if (
            "_" in token
            or token.startswith("_")
            or _is_camel_case(token)
        ):
            selected.add(token)

    return tuple(
        sorted(
            selected,
            key=lambda value: (
                value.casefold(),
                value,
            ),
        )
    )


def structured_search_identifiers_from_command(
    command: str,
) -> tuple[str, ...]:
    """Extract Q directly from an rg/grep current action."""
    return structured_search_identifiers(
        extract_search_pattern(command)
    )
