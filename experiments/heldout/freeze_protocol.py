"""Freeze provenance for the preregistered TrajRel held-out protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FILES = [
    "benchmarks/heldout/protocol_v1.json",
    "src/trajrel/scorers/headroom_reference.py",
    "src/trajrel/selectors/headroom_reference.py",
    "src/trajrel/query_signals.py",
    "src/trajrel/adoption.py",
    "src/trajrel/evaluation/batch.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def main() -> None:
    status = git_output(
        "status",
        "--porcelain",
    )

    if status:
        raise SystemExit(
            "Working tree is not clean. "
            "Commit current changes before freezing."
        )

    commit = git_output(
        "rev-parse",
        "HEAD",
    )

    branch = git_output(
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    )

    hashes = {}

    for relative in FILES:
        path = ROOT / relative

        if not path.exists():
            raise FileNotFoundError(
                relative
            )

        hashes[relative] = sha256(
            path
        )

    output = {
        "freeze_version": 1,
        "git_commit": commit,
        "git_branch": branch,
        "files": hashes,
    }

    target = (
        ROOT
        / "benchmarks/heldout/"
        "protocol_v1_freeze.json"
    )

    target.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("WROTE:", target)


if __name__ == "__main__":
    main()
