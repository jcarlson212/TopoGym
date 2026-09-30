#!/usr/bin/env python3
"""Which expensive checks a change needs.

The unit tests, the croissant regeneration and the golden replay cost
from minutes to runner-hours, and most changes cannot affect them: a
docs edit, a CHANGELOG entry, a benchmark artefact, or a release commit
that only moves version strings. This script looks at what actually
changed and says which checks apply, for both the pre-commit hook and
CI, so the two agree.

A change is judged line by line. Lines that only carry the package
version or release date, in the files ``scripts/sync_version.py``
manages, are *version-only* and need nothing beyond the version-sync
check. Every other changed line makes its file *meaningful*, and a
meaningful file triggers the checks whose paths it falls under:

- ``tests``: package code, tests, scripts, dependencies, or the CI and
  hook configuration;
- ``croissant``: code that can change the registry's metadata (not the
  canonical layer or the baselines), or the metadata files themselves;
- ``golden``: code that can change env behaviour, or the fixtures.

Changing this script or the workflows that call it runs everything, so
a change to the gating is always exercised by the gates.

Usage::

    # pre-commit: run CMD only if the staged change needs CHECK
    python scripts/change_scope.py --staged --need tests -- CMD ...

    # CI: write tests=/croissant=/golden= to $GITHUB_OUTPUT
    python scripts/change_scope.py --event EVENT --base SHA --head SHA \\
        --github-output

    # show the decision for the working tree, the index, or a range
    python scripts/change_scope.py --staged
    python scripts/change_scope.py --base SHA --head SHA
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys

CHECKS = ("tests", "croissant", "golden")

#: Lines that only carry the version or the release date, per file.
VERSION_LINES = {
    "pyproject.toml": (r'^version\s*=\s*"[^"]*"$',),
    "topogym/__init__.py": (r'^__version__ = "[^"]*"$',),
    "CITATION.cff": (r"^version:\s*\S+$", r"^date-released:\s*\S+$"),
    "README.md": (r"^\s*version\s*=\s*\{[^}]*\},?$",),
    "croissant.json": (r'^\s*"version":\s*"[^"]*",?$',
                       r'^\s*"citeAs":\s*".*version=\{[^}]*\}.*",?$'),
}

#: Paths whose meaningful change triggers each check (fnmatch globs).
TRIGGERS = {
    "tests": (
        "topogym/*", "tests/*", "scripts/*", "pyproject.toml",
        ".pre-commit-config.yaml", ".github/workflows/ci.yml",
    ),
    "croissant": (
        "topogym/*", "scripts/generate_croissant.py", "croissant.json",
        "docs/manifest.csv", "pyproject.toml",
    ),
    "golden": (
        "topogym/envs/*", "topogym/generation/*", "topogym/core/*",
        "topogym/rendering/*", "topogym/complexes/*",
        "topogym/registry.py", "topogym/spec.py", "topogym/__init__.py",
        "tests/compat/*",
    ),
}

#: Meaningful changes here never trigger the check (it cannot observe
#: them): croissant is generated from the registry, not from the
#: canonical layer or the baselines.
EXEMPT = {
    "croissant": ("topogym/canonical/*", "topogym/baselines/*"),
}

#: Changing the gating itself runs every check.
EVERYTHING = (
    "scripts/change_scope.py", ".pre-commit-config.yaml",
    ".github/workflows/ci.yml", ".github/workflows/golden.yml",
)


def _matches(path: str, globs) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in globs)


def parse_diff(text: str) -> dict:
    """``{path: [changed lines]}`` from ``git diff -U0`` output. Binary
    or mode-only changes carry a sentinel line, so they count as
    meaningful."""
    files: dict = {}
    current = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*) b/(.*)$", line)
            current = m.group(2) if m else None
            if current is not None:
                files.setdefault(current, [])
                if m.group(1) != m.group(2):  # rename: both sides count
                    files.setdefault(m.group(1), []).append("\0rename")
            continue
        if current is None:
            continue
        if line.startswith(("+++ ", "--- ")):
            continue
        if line.startswith(("Binary files", "old mode", "new mode",
                            "deleted file", "new file mode")):
            files[current].append("\0" + line)
        elif line.startswith(("+", "-")):
            files[current].append(line[1:])
    return files


def is_version_line(path: str, line: str) -> bool:
    return any(re.match(p, line) for p in VERSION_LINES.get(path, ()))


def meaningful_files(changes: dict) -> list:
    """Changed files with at least one line that is not version-only."""
    out = []
    for path, lines in changes.items():
        if not lines or any(not is_version_line(path, ln) for ln in lines):
            out.append(path)
    return sorted(out)


def decide(changes: dict) -> dict:
    """``{check: bool}`` plus ``reasons``: which files triggered what."""
    meaningful = meaningful_files(changes)
    result = {c: False for c in CHECKS}
    reasons = {c: [] for c in CHECKS}
    for path in meaningful:
        if _matches(path, EVERYTHING):
            for c in CHECKS:
                result[c] = True
                reasons[c].append(path)
            continue
        for c in CHECKS:
            if _matches(path, TRIGGERS[c]) and \
                    not _matches(path, EXEMPT.get(c, ())):
                result[c] = True
                reasons[c].append(path)
    result["reasons"] = reasons
    result["version_only"] = bool(changes) and not meaningful
    return result


def everything(reason: str) -> dict:
    return {**{c: True for c in CHECKS},
            "reasons": {c: [reason] for c in CHECKS}, "version_only": False}


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout


def changes_for(args) -> dict | None:
    """The change to judge, or None when it cannot be determined (then
    every check runs)."""
    diff = ["diff", "-U0", "--no-color", "--no-ext-diff", "-M"]
    if args.staged:
        return parse_diff(_git(*diff, "--cached"))
    if not args.base or set(args.base) == {"0"}:
        return None  # a new branch, or no base: judge nothing, run all
    try:
        base = _git("merge-base", args.base, args.head).strip()
        return parse_diff(_git(*diff, base, args.head))
    except subprocess.CalledProcessError:
        return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--base")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--event", default="")
    ap.add_argument("--need", choices=CHECKS)
    ap.add_argument("--github-output", action="store_true")
    args = ap.parse_args(argv)

    if args.event in ("workflow_dispatch", "schedule", "release"):
        result = everything(f"{args.event} runs everything")
    else:
        changes = changes_for(args)
        result = everything("change could not be determined") \
            if changes is None else decide(changes)

    if args.github_output:
        path = os.environ.get("GITHUB_OUTPUT")
        lines = [f"{c}={'true' if result[c] else 'false'}" for c in CHECKS]
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
    if args.need:
        if not result[args.need]:
            why = "version-only change" if result["version_only"] \
                else "no change it can observe"
            print(f"change_scope: skipping {args.need} ({why})")
            return 0
        if cmd:
            return subprocess.call(cmd)
        return 0
    for c in CHECKS:
        files = result["reasons"][c]
        shown = ", ".join(files[:4]) + (" ..." if len(files) > 4 else "")
        print(f"{c:9s} {'run ' if result[c] else 'skip'}  {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
