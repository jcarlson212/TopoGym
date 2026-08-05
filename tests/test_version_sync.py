"""Version metadata stays in lockstep with pyproject.toml."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_sync_version_check_passes():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_version.py"),
         "--check"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_versions_match():
    import topogym

    pyproject = (ROOT / "pyproject.toml").read_text()
    assert f'version = "{topogym.__version__}"' in pyproject
    citation = (ROOT / "CITATION.cff").read_text()
    assert f"version: {topogym.__version__}" in citation
    croissant = (ROOT / "croissant.json").read_text()
    assert f'"version": "{topogym.__version__}"' in croissant
    assert f"version={{{topogym.__version__}}}" in croissant  # citeAs
    readme = (ROOT / "README.md").read_text()
    assert f"version = {{{topogym.__version__}}}" in readme
