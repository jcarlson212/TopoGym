#!/usr/bin/env python3
"""Sync the package version from pyproject.toml (the single source of
truth) into CITATION.cff, topogym/__init__.py, and the README citation.

    python scripts/sync_version.py          # rewrite files in place
    python scripts/sync_version.py --check  # exit 1 if anything is stale
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise SystemExit("no version field found in pyproject.toml")
    return match.group(1)


#: (file, pattern, replacement-template) — {v} is the version.
TARGETS = (
    ("CITATION.cff", r'(?m)^version:\s*\S+$', "version: {v}"),
    ("topogym/__init__.py", r'(?m)^__version__ = "[^"]*"$',
     '__version__ = "{v}"'),
    ("README.md", r'(?m)^(\s*version\s*=\s*)\{[^}]*\}(,?)$',
     r"\g<1>{{{v}}}\g<2>"),
)


def main() -> int:
    check = "--check" in sys.argv[1:]
    version = pyproject_version()
    stale = []
    for name, pattern, template in TARGETS:
        path = ROOT / name
        old = path.read_text()
        new = re.sub(pattern, template.format(v=version), old)
        if new != old:
            stale.append(name)
            if not check:
                path.write_text(new)
    if stale:
        verb = "out of sync" if check else "updated"
        print(f"version {version}: {verb}: {', '.join(stale)}")
        return 1 if check else 0
    print(f"version {version}: all files in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
