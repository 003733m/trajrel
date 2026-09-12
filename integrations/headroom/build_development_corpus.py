"""Build the normalized Headroom development corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajrel.importers.headroom_development import (
    build_development_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    corpus = build_development_corpus(
        raw_root=args.raw_root,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            corpus,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print(
        json.dumps(
            corpus["summary"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
