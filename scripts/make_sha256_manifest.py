#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for a deliverable directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", default="MANIFEST_SHA256.txt")
    args = parser.parse_args()

    root = args.root.resolve()
    output = (root / args.output).resolve()
    rows: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or path.resolve() == output:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(root).as_posix()
        rows.append(f"{digest}  {relative}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} hashes to {output}")


if __name__ == "__main__":
    main()
