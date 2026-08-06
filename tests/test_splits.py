"""The benchmark split manifests: disjoint, complete, and auditable."""

import csv
import json
import pathlib
import statistics

import pytest

from topogym import benchmarks

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLITS = ROOT / "docs" / "splits"
MAIN = ("tune", "train", "val", "test")


def _rows(stem):
    path = SPLITS / f"{stem}.csv"
    if not path.exists():
        pytest.skip(f"{path.name} not generated")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("stem", MAIN)
def test_split_is_populated_and_self_consistent(stem):
    rows = _rows(stem)
    assert rows
    for row in rows:
        assert row["split"] == stem
        assert benchmarks.split_of(int(row["seed"])) == stem
        assert row["canonical_config"]
        if row["optimal_actions"]:
            # Every instance is solvable with the slack factor of room.
            assert (int(row["optimal_actions"]) * 3
                    <= int(row["horizon"]))


def test_splits_share_no_instance():
    seen = {}
    for stem in MAIN:
        for row in _rows(stem):
            key = row["canonical_config"]
            assert key not in seen, f"{key} in {stem} and {seen.get(key)}"
            seen[key] = stem


def test_every_unit_appears_in_every_split():
    units = {stem: {r["unit"] for r in _rows(stem)} for stem in MAIN}
    reference = units["train"]
    for stem in MAIN:
        assert units[stem] == reference


def test_every_declared_family_size_is_covered():
    """Each GridWorld2D family must be present at every size the spec
    declares, so no split can be overfitted to one scale.

    Coverage is by *configuration*, not by label: distinct registry ids
    can name the same world (``ShapeSq`` is ``Dilution`` with the
    square shape spelled out), and such units are deliberately
    collapsed so no world is weighted twice.
    """
    import dataclasses

    from topogym import registry

    present = {row["canonical_config"].rsplit("-seed", 1)[0]
               for row in _rows("train")}
    checked = 0
    for name, cfg in registry.REGISTRY.items():
        family = benchmarks.family_of(name)
        base = cfg.size if isinstance(cfg.size, int) else max(cfg.size)
        declared = benchmarks.sizes_for(family, base)
        assert len(declared) >= 2, f"{family} declares only {declared}"
        for size in declared:
            resized = dataclasses.replace(cfg, size=size)
            jitter = benchmarks.jitter_for(size)
            key = registry.canonical_string(
                dataclasses.replace(resized, placement_jitter=jitter), 0
            ).rsplit("-seed", 1)[0]
            assert key in present, f"{family} at size {size} is missing"
            checked += 1
    assert checked > 40


def test_difficulty_distributions_are_comparable():
    """Train and test must be samples of the same task, not different
    ones: per unit, the median optimal route should be close."""
    def medians(stem):
        out = {}
        for row in _rows(stem):
            if row["optimal_actions"]:
                out.setdefault(row["unit"], []).append(
                    int(row["optimal_actions"]))
        return {u: statistics.median(v) for u, v in out.items()}

    train, test = medians("train"), medians("test")
    for unit in sorted(set(train) & set(test)):
        a, b = train[unit], test[unit]
        assert abs(a - b) <= 0.5 * max(a, b), (
            f"{unit}: train median {a} vs test median {b}"
        )


def test_extrapolation_splits_are_disjoint_in_size_and_family():
    small = _rows("size-extrapolation-train")
    large = _rows("size-extrapolation-test")
    assert small and large
    assert max(int(r["size"]) for r in small) \
        <= benchmarks.EXTRAPOLATION_TRAIN_MAX
    assert min(int(r["size"]) for r in large) \
        > benchmarks.EXTRAPOLATION_TRAIN_MAX

    held = set(benchmarks.HELD_OUT_FAMILIES)
    assert not {r["family"] for r in _rows("family-holdout-train")} & held
    assert {r["family"] for r in _rows("family-holdout-test")} <= held


def test_croissant_publishes_every_split():
    croissant = json.loads((ROOT / "croissant.json").read_text())
    files = {d["@id"] for d in croissant["distribution"]}
    record_sets = {r["@id"] for r in croissant["recordSet"]}
    for path in sorted(SPLITS.glob("*.csv")):
        assert f"splits/{path.stem}.csv" in files
        assert f"split/{path.stem}" in record_sets


def test_collapsed_units_are_recoverable_through_aliases():
    """Identical configurations are carried once, but the labels that
    collapsed into them stay discoverable -- otherwise grouping by
    family silently loses ShapeSq's square control at sizes where it
    coincides with Dilution."""
    rows = _rows("train")
    labels = {r["unit"] for r in rows}
    aliased = {a for r in rows for a in r["aliases"].split() if a}
    assert aliased, "expected at least one collapsed label"
    assert "ShapeSq-50" in aliased      # it is Dilution-50 exactly
    assert "ShapeSq-50" not in labels   # carried once, not twice
    # Every Shape variant is reachable by label or alias at both sizes.
    for shape in ("Sq", "Ci", "Tr", "St"):
        for size in (50, 100):
            name = f"Shape{shape}-{size}"
            assert name in labels or name in aliased, name
