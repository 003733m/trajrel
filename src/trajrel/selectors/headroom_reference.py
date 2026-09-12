"""Faithful port of the frozen KTH target-conditioned bridge selector."""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass

from trajrel.scorers.headroom_reference import (
    _MAX_NEGATIVE_CUE,
    _MAX_POSITIVE_CUE,
    TARGET_CORROBORATION_KINDS,
    ReferenceBridgeCandidate,
    ReferenceScoringConfig,
)


@dataclass(frozen=True, slots=True)
class ReferenceTargetBridgeCandidate:
    candidate: ReferenceBridgeCandidate
    score: float
    target_specificity: float

    @property
    def token(self) -> str:
        return self.candidate.token

    @property
    def kind(self) -> str:
        return self.candidate.kind


_GENERIC_BRIDGE_IDENTIFIERS = frozenset(
    {
        "bool",
        "bytes",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "object",
        "set",
        "str",
        "tuple",
        "type",
    }
)


def _normalized_candidate_text(text: str, kind: str) -> str:
    if kind == "file_path":
        return text.replace("\\", "/")
    return text


def _candidate_match_pattern(
    token: str,
    *,
    kind: str,
) -> re.Pattern[str]:
    normalized = _normalized_candidate_text(token, kind)
    escaped = re.escape(normalized)

    if kind == "file_path":
        boundary = r"A-Za-z0-9_.-"
    else:
        boundary = r"A-Za-z0-9_"

    return re.compile(
        rf"(?<![{boundary}]){escaped}(?![{boundary}])",
        re.IGNORECASE,
    )


def _contains_candidate(
    text: str,
    token: str,
    *,
    kind: str,
) -> bool:
    normalized = _normalized_candidate_text(text, kind)

    return (
        _candidate_match_pattern(
            token,
            kind=kind,
        ).search(normalized)
        is not None
    )


def _is_generic_bridge_identifier(
    candidate: ReferenceBridgeCandidate,
) -> bool:
    if candidate.kind not in {"function_name", "class_name"}:
        return False

    return candidate.token.casefold() in _GENERIC_BRIDGE_IDENTIFIERS


def _is_weak_unstructured_bridge_identifier(
    candidate: ReferenceBridgeCandidate,
) -> bool:
    if candidate.positive_evidence > 0:
        return False

    if candidate.kind == "function_name":
        return "_" not in candidate.token

    if candidate.kind == "config_key":
        return (
            "_" not in candidate.token
            and len(candidate.token) <= 3
        )

    return False


def _target_document_frequencies(
    candidates: list[ReferenceBridgeCandidate],
    target_content: str,
) -> tuple[int, dict[ReferenceBridgeCandidate, int]]:
    frequencies = dict.fromkeys(candidates, 0)

    raw_lines = target_content.splitlines(keepends=True)

    n_nonempty = sum(
        1
        for line in raw_lines
        if line.strip()
    )

    if not candidates or not raw_lines:
        return n_nonempty, frequencies

    line_starts: list[int] = []
    offset = 0

    for line in raw_lines:
        line_starts.append(offset)
        offset += len(line)

    normalized_tokens = [
        _normalized_candidate_text(
            candidate.token,
            candidate.kind,
        )
        for candidate in candidates
    ]

    folded_tokens = [
        token.casefold()
        for token in normalized_tokens
    ]

    ambiguous: set[int] = set()

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            a = folded_tokens[left]
            b = folded_tokens[right]

            if a in b or b in a:
                ambiguous.add(left)
                ambiguous.add(right)

    normalized_line_cache: dict[str, list[str]] = {}

    for index in ambiguous:
        candidate = candidates[index]

        cache_key = (
            "file"
            if candidate.kind == "file_path"
            else "identifier"
        )

        lines = normalized_line_cache.get(cache_key)

        if lines is None:
            lines = [
                _normalized_candidate_text(
                    line,
                    candidate.kind,
                )
                for line in target_content.splitlines()
                if line.strip()
            ]
            normalized_line_cache[cache_key] = lines

        pattern = _candidate_match_pattern(
            candidate.token,
            kind=candidate.kind,
        )

        frequencies[candidate] = sum(
            1
            for line in lines
            if pattern.search(line) is not None
        )

    groups: dict[str, list[int]] = {
        "file": [],
        "identifier": [],
    }

    for index, candidate in enumerate(candidates):
        if index in ambiguous:
            continue

        group = (
            "file"
            if candidate.kind == "file_path"
            else "identifier"
        )

        groups[group].append(index)

    for group, indexes in groups.items():
        if not indexes:
            continue

        if group == "file":
            boundary = r"A-Za-z0-9_.-"
            haystack = target_content.replace("\\", "/")
        else:
            boundary = r"A-Za-z0-9_"
            haystack = target_content

        token_to_index = {
            normalized_tokens[index].casefold(): index
            for index in indexes
        }

        alternatives = sorted(
            (
                re.escape(normalized_tokens[index])
                for index in indexes
            ),
            key=len,
            reverse=True,
        )

        pattern = re.compile(
            rf"(?<![{boundary}])"
            rf"(?:{'|'.join(alternatives)})"
            rf"(?![{boundary}])",
            re.IGNORECASE,
        )

        seen_lines: dict[int, set[int]] = {
            index: set()
            for index in indexes
        }

        for match in pattern.finditer(haystack):
            index = token_to_index.get(
                match.group(0).casefold()
            )

            if index is None:
                continue

            line_index = (
                bisect.bisect_right(
                    line_starts,
                    match.start(),
                )
                - 1
            )

            seen_lines[index].add(line_index)

        for index, lines in seen_lines.items():
            frequencies[candidates[index]] = len(lines)

    return n_nonempty, frequencies


def _target_specificity_from_frequency(
    *,
    n_lines: int,
    document_frequency: int,
) -> float:
    if n_lines <= 0 or document_frequency <= 0:
        return 0.0

    denominator = math.log(n_lines + 1)

    if denominator <= 0.0:
        return 0.0

    value = (
        math.log(
            (n_lines + 1)
            / (document_frequency + 1)
        )
        / denominator
    )

    return min(1.0, max(0.0, value))


def target_specificity(
    token: str,
    target_content: str,
    *,
    kind: str,
) -> float:
    lines = [
        _normalized_candidate_text(line, kind)
        for line in target_content.splitlines()
        if line.strip()
    ]

    if not lines:
        return 0.0

    pattern = _candidate_match_pattern(
        token,
        kind=kind,
    )

    document_frequency = sum(
        1
        for line in lines
        if pattern.search(line) is not None
    )

    if document_frequency == 0:
        return 0.0

    return _target_specificity_from_frequency(
        n_lines=len(lines),
        document_frequency=document_frequency,
    )


def select_headroom_reference_candidates(
    candidates: list[ReferenceBridgeCandidate],
    *,
    target_content: str,
    user_context: str,
    top_k: int = 6,
    scoring: ReferenceScoringConfig | None = None,
) -> tuple[ReferenceTargetBridgeCandidate, ...]:
    """Condition frozen historical candidates on the current target."""
    if not candidates or not target_content or top_k <= 0:
        return ()

    scoring = scoring or ReferenceScoringConfig()

    admissible: list[ReferenceBridgeCandidate] = []

    for candidate in candidates:
        if _is_generic_bridge_identifier(candidate):
            continue

        if _is_weak_unstructured_bridge_identifier(candidate):
            continue

        if user_context and _contains_candidate(
            user_context,
            candidate.token,
            kind=candidate.kind,
        ):
            continue

        admissible.append(candidate)

    if not admissible:
        return ()

    n_lines, document_frequencies = _target_document_frequencies(
        admissible,
        target_content,
    )

    if n_lines <= 0:
        return ()

    selected: list[ReferenceTargetBridgeCandidate] = []

    for candidate in admissible:
        specificity = _target_specificity_from_frequency(
            n_lines=n_lines,
            document_frequency=document_frequencies[candidate],
        )

        if specificity <= 0.0:
            continue

        evidence_score = candidate.score

        corroborated_singleton = (
            candidate.kind in TARGET_CORROBORATION_KINDS
            and candidate.distinct_tools == 1
            and candidate.positive_evidence <= 0
        )

        if corroborated_singleton:
            effective_distinct_tools = 2
            effective_occurrences = candidate.occurrences + 1

            cross_tool_steps = min(
                max(effective_distinct_tools - 1, 0),
                scoring.max_cross_tool_steps,
            )

            bounded_occurrences = min(
                effective_occurrences,
                scoring.max_occurrences,
            )

            causal_signal = (
                float(candidate.positive_evidence)
                / float(_MAX_POSITIVE_CUE)
            )

            if scoring.max_cross_tool_steps > 0:
                cross_tool_signal = (
                    float(cross_tool_steps)
                    / float(scoring.max_cross_tool_steps)
                )
            else:
                cross_tool_signal = 0.0

            if scoring.max_occurrences > 0:
                occurrence_signal = (
                    float(bounded_occurrences)
                    / float(scoring.max_occurrences)
                )
            else:
                occurrence_signal = 0.0

            speculation_signal = (
                float(candidate.negative_evidence)
                / float(_MAX_NEGATIVE_CUE)
            )

            evidence_score = (
                scoring.causal_weight * causal_signal
                + scoring.cross_tool_weight * cross_tool_signal
                + scoring.occurrence_weight * occurrence_signal
                + scoring.recency_weight
                - scoring.speculation_weight * speculation_signal
            )

            if (
                scoring.min_score is not None
                and evidence_score < scoring.min_score
            ):
                continue

        adjusted_score = evidence_score * specificity

        if adjusted_score <= 0.0:
            continue

        selected.append(
            ReferenceTargetBridgeCandidate(
                candidate=candidate,
                score=adjusted_score,
                target_specificity=specificity,
            )
        )

    selected.sort(
        key=lambda item: (
            -item.score,
            -item.candidate.score,
            -item.candidate.distinct_tools,
            -item.candidate.recency,
            item.token,
        )
    )

    return tuple(selected[:top_k])
