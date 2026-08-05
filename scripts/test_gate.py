#!/usr/bin/env python3
"""Commit gate: require at least --min-pass of the unit tests to pass.

    python scripts/test_gate.py --min-pass 0.9

Runs the suite quietly and exits nonzero when the pass ratio falls
below the threshold (or when nothing ran at all).
"""

from __future__ import annotations

import argparse
import sys

import pytest


class _Counter:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
        elif report.failed:  # setup/teardown errors count as failures
            self.failed += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pass", type=float, default=0.9)
    args = ap.parse_args()
    counter = _Counter()
    pytest.main(["-q", "--no-header", "-p", "no:cacheprovider", "tests"],
                plugins=[counter])
    total = counter.passed + counter.failed
    ratio = counter.passed / total if total else 0.0
    print(f"test gate: {counter.passed}/{total} passed ({ratio:.1%}); "
          f"threshold {args.min_pass:.0%}")
    return 0 if total and ratio >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(main())
