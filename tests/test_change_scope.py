"""scripts/change_scope.py: which expensive checks a change needs."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "change_scope", ROOT / "scripts" / "change_scope.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def diff(*files) -> str:
    """A ``git diff -U0`` for ``(path, removed, added)`` triples."""
    out = []
    for path, removed, added in files:
        out += [f"diff --git a/{path} b/{path}", f"--- a/{path}",
                f"+++ b/{path}", "@@ -1 +1 @@"]
        out += [f"-{ln}" for ln in removed] + [f"+{ln}" for ln in added]
    return "\n".join(out) + "\n"


def decide(*files) -> dict:
    return cs.decide(cs.parse_diff(diff(*files)))


def ran(result) -> set:
    return {c for c in cs.CHECKS if result[c]}


RELEASE = (
    ("pyproject.toml", ['version = "0.5.1"'], ['version = "0.6.0"']),
    ("topogym/__init__.py", ['__version__ = "0.5.1"'],
     ['__version__ = "0.6.0"']),
    ("CITATION.cff", ["version: 0.5.1", "date-released: 2026-09-29"],
     ["version: 0.6.0", "date-released: 2026-09-30"]),
    ("README.md", ["  version = {0.5.1},"], ["  version = {0.6.0},"]),
    ("croissant.json", ['  "version": "0.5.1",',
                        '  "citeAs": "@software{x, version={0.5.1}, url={u}}",'],
     ['  "version": "0.6.0",',
      '  "citeAs": "@software{x, version={0.6.0}, url={u}}",']),
    ("CHANGELOG.md", ["## 0.6.0 (unreleased)"], ["## 0.6.0 (2026-09-30)"]),
)


def test_a_release_bump_needs_nothing():
    result = decide(*RELEASE)
    assert ran(result) == set()
    assert decide(*RELEASE[:-1])["version_only"]  # without the CHANGELOG


def test_version_lines_only_count_in_their_own_file():
    # A version-looking line in a code file is still code.
    result = decide(("topogym/envs/core.py", [], ['version = "1.0"']))
    assert ran(result) == {"tests", "croissant", "golden"}


def test_a_release_bump_with_a_code_change_runs_its_checks():
    result = decide(*RELEASE, ("topogym/__init__.py",
                               ["from x import a"], ["from x import b"]))
    assert ran(result) == {"tests", "croissant", "golden"}


def test_dependency_changes_are_meaningful():
    result = decide(("pyproject.toml", ['dependencies = ["gudhi>=3.8"]'],
                     ['dependencies = ["gudhi>=3.9"]']))
    assert ran(result) == {"tests", "croissant"}


def test_docs_and_artefacts_need_nothing():
    result = decide(("docs/canonical.md", ["a"], ["b"]),
                    ("CHANGELOG.md", ["a"], ["b"]),
                    ("benchmarks/x/results/run.json", ["1"], ["2"]))
    assert ran(result) == set()


def test_canonical_layer_needs_tests_only():
    result = decide(("topogym/canonical/export.py", ["a"], ["b"]))
    assert ran(result) == {"tests"}


def test_baselines_skip_croissant_and_golden():
    result = decide(("topogym/baselines/gridworld2dv1/ppo.py", ["a"], ["b"]))
    assert ran(result) == {"tests"}


@pytest.mark.parametrize("path", ["topogym/envs/core.py",
                                  "topogym/generation/generator.py",
                                  "topogym/registry.py"])
def test_env_code_runs_everything(path):
    assert ran(decide((path, ["a"], ["b"]))) == {"tests", "croissant",
                                                 "golden"}


def test_golden_fixtures_run_golden():
    result = decide(("tests/compat/golden/TopoGym__Maze-50-v0.json",
                     ["a"], ["b"]))
    assert "golden" in ran(result)


@pytest.mark.parametrize("path", ["scripts/change_scope.py",
                                  ".github/workflows/golden.yml",
                                  ".github/workflows/ci.yml",
                                  ".pre-commit-config.yaml"])
def test_changing_the_gating_runs_every_gate(path):
    assert ran(decide((path, ["a"], ["b"]))) == set(cs.CHECKS)


def test_new_and_binary_files_are_meaningful():
    text = ("diff --git a/topogym/core/x.png b/topogym/core/x.png\n"
            "new file mode 100644\n"
            "Binary files /dev/null and b/topogym/core/x.png differ\n")
    assert ran(cs.decide(cs.parse_diff(text))) == set(cs.CHECKS)


def test_renames_count_both_sides():
    text = ("diff --git a/docs/a.md b/topogym/envs/a.py\n"
            "similarity index 100%\n")
    assert "golden" in ran(cs.decide(cs.parse_diff(text)))


def test_unknown_base_runs_everything(capsys):
    assert cs.main(["--base", "0" * 40, "--head", "HEAD",
                    "--need", "golden"]) == 0
    assert "skipping" not in capsys.readouterr().out
    assert cs.main(["--event", "workflow_dispatch", "--need", "tests"]) == 0
    assert "skipping" not in capsys.readouterr().out


def test_need_runs_the_command_only_when_needed(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cs, "changes_for", lambda args: cs.parse_diff(
        diff(*RELEASE)))
    monkeypatch.setattr(cs.subprocess, "call",
                        lambda cmd: calls.append(cmd) or 0)
    assert cs.main(["--staged", "--need", "tests", "--", "pytest"]) == 0
    assert calls == []
    assert "skipping tests" in capsys.readouterr().out
    monkeypatch.setattr(cs, "changes_for", lambda args: cs.parse_diff(
        diff(("topogym/envs/core.py", ["a"], ["b"]))))
    assert cs.main(["--staged", "--need", "tests", "--", "pytest"]) == 0
    assert calls == [["pytest"]]


def test_github_output(monkeypatch, tmp_path):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(cs, "changes_for", lambda args: cs.parse_diff(
        diff(("topogym/canonical/spec.py", ["a"], ["b"]))))
    cs.main(["--base", "a", "--head", "b", "--github-output"])
    assert out.read_text().split() == ["tests=true", "croissant=false",
                                       "golden=false"]
