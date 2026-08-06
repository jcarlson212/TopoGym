#!/usr/bin/env python3
"""Generate Croissant metadata for the TopoGym-v1 benchmark.

    python scripts/generate_croissant.py          # rewrite both files
    python scripts/generate_croissant.py --check  # exit 1 if stale

Writes ``docs/manifest.csv`` (one record per pinned environment id, with
its canonical configuration and certified topology at the reference
seed) and ``croissant.json`` (MLCommons Croissant 1.0 JSON-LD
describing the benchmark and that record set) — the machine-readable
metadata NeurIPS Datasets & Benchmarks submissions attach on
OpenReview. Both files are generated from the registry, so they cannot
drift from the code.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402

import topogym  # noqa: E402
from topogym import registry  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE_SEED = 0

FIELDS = (
    "env_id", "slice", "family", "canonical_config", "reference_seed",
    "size", "horizon", "optimal_actions", "n_cells", "n_free_cells",
    "betti_z2", "betti_z2_sealed", "homology",
)


def _rows() -> list:
    rows = []
    entries = (
        [("GridWorld2D", n, f"TopoGym/{n}-v0")
         for n in registry.REGISTRY]
        + [("Top", f"{n}-50", f"TopoGym/{n}-50-v0")
           for n in registry.TOP_TOPOLOGIES]
        + [("Texture", n, f"TopoGym/{n}-v0")
           for n in registry.TEXTURE_SCENARIOS]
    )
    for slice_name, name, env_id in entries:
        env = gym.make(env_id, seed=REFERENCE_SEED).unwrapped
        env.reset(seed=0)
        md = env.topology
        family = name.rstrip("0123456789").rstrip("-").rstrip(
            "0123456789"
        ) or name
        if slice_name == "GridWorld2D":
            canonical = registry.canonical_string(
                registry.get_config(env_id), REFERENCE_SEED
            )
        else:
            canonical = f"TG-{slice_name}-{name}-seed{REFERENCE_SEED}"
        rows.append({
            "env_id": env_id,
            "slice": slice_name,
            "family": family,
            "canonical_config": canonical,
            "reference_seed": REFERENCE_SEED,
            "size": max(md.size),
            "horizon": env._max_steps,
            "optimal_actions": env.optimal_actions() or "",
            "n_cells": md.n_cells,
            "n_free_cells": md.n_free_cells,
            "betti_z2": " ".join(map(str, md.betti_z2)),
            "betti_z2_sealed": " ".join(map(str, md.betti_z2_sealed)),
            "homology": "; ".join(
                f"{k}={v}" for k, v in md.homology.items()
            ),
        })
        env.close()
    return rows


def _field(name: str, dtype: str, description: str,
           record_set: str = "environments",
           file_id: str = "manifest.csv") -> dict:
    return {
        "@type": "cr:Field",
        "@id": f"{record_set}/{name}",
        "name": name,
        "description": description,
        "dataType": dtype,
        "source": {
            "fileObject": {"@id": file_id},
            "extract": {"column": name},
        },
    }


#: Column -> (type, description) for the benchmark split manifests.
SPLIT_FIELDS = {
    "split": ("sc:Text", "tune, train, val, or test."),
    "unit": ("sc:Text", "Family-size unit, e.g. Decoys4-100."),
    "aliases": ("sc:Text",
                "Space-separated registry labels naming this exact "
                "world. Identical configurations are carried once so "
                "no world is weighted twice; the aliases keep the "
                "collapsed labels discoverable."),
    "template_id": ("sc:Text",
                    "Registry id the instance is built from."),
    "slice": ("sc:Text", "GridWorld2D, Top, or Texture."),
    "family": ("sc:Text", "Environment family."),
    "size": ("sc:Integer", "Grid side length."),
    "seed": ("sc:Integer",
             "Layout seed, drawn from this split's disjoint band."),
    "placement_jitter": ("sc:Integer",
                         "Cells of placement perturbation; keeps the "
                         "family's arrangement while making every "
                         "instance distinct."),
    "canonical_config": ("sc:Text",
                         "Canonical configuration string: the "
                         "reproduction key for this instance."),
    "horizon": ("sc:Integer", "Pre-determined episode length."),
    "optimal_actions": ("sc:Integer",
                        "Fewest actions from start to goal counting "
                        "turns; makes the split's difficulty "
                        "distribution auditable."),
    "n_free_cells": ("sc:Integer", "Traversable cells."),
    "betti_z2": ("sc:Text", "Certified Betti numbers (doors walkable)."),
    "betti_z2_sealed": ("sc:Text",
                        "Certified Betti numbers (doors as walls)."),
}

#: What each split file is for, keyed by stem.
SPLIT_DESCRIPTIONS = {
    "tune": "Hyperparameter selection only; never eval or hold-out.",
    "train": "Training instances.",
    "val": "Model selection during development.",
    "test": "Hold-out; reported once.",
    "size-extrapolation-train": "Train small (size <= 100).",
    "size-extrapolation-test": "Test large (size > 100): size "
                               "extrapolation, reported separately.",
    "family-holdout-train": "Train with whole families withheld.",
    "family-holdout-test": "The withheld families: concept transfer, "
                           "reported separately.",
}


def _split_files() -> list:
    """``(stem, path, sha256, n_rows)`` per split manifest on disk."""
    out = []
    for path in sorted((ROOT / "docs" / "splits").glob("*.csv")):
        text = path.read_text()
        out.append((
            path.stem, path,
            hashlib.sha256(text.encode()).hexdigest(),
            max(0, len(text.splitlines()) - 1),
        ))
    return out


def main() -> int:
    check = "--check" in sys.argv[1:]
    manifest = ROOT / "docs" / "manifest.csv"
    rows = _rows()
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS,
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    manifest_text = buffer.getvalue()
    if not check:
        manifest.write_text(manifest_text)
    sha = hashlib.sha256(manifest_text.encode()).hexdigest()

    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "sc": "https://schema.org/",
            "dct": "http://purl.org/dc/terms/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileObject": "cr:fileObject",
            "recordSet": "cr:recordSet",
            "source": "cr:source",
        },
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "TopoGym-v1",
        "description": (
            "The pinned environment registry and benchmark splits of "
            "TopoGym: gridworld "
            "reinforcement-learning environments with certified "
            "topology for exploration research. One record per stable "
            "environment id, with its canonical generator "
            "configuration and the certified Betti numbers (in both "
            "door conventions) at the reference seed. Environments "
            "are deterministic up to seeds, end to end."
        ),
        "url": "https://github.com/jcarlson212/TopoGym",
        "version": topogym.__version__,
        "license": "https://spdx.org/licenses/MIT.html",
        "creator": {"@type": "sc:Person", "name": "Jason Carlson"},
        "citeAs": (
            "@software{carlson2026topogym, author={Carlson, Jason}, "
            "title={TopoGym: Environments and Benchmarks for "
            "Topological Exploration in Reinforcement Learning}, "
            f"year={{2026}}, version={{{topogym.__version__}}}, "
            "url={https://github.com/jcarlson212/TopoGym}}"
        ),
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "manifest.csv",
                "name": "manifest.csv",
                "description": "One record per pinned environment id.",
                "contentUrl": (
                    "https://raw.githubusercontent.com/jcarlson212/"
                    "TopoGym/main/docs/manifest.csv"
                ),
                "encodingFormat": "text/csv",
                "sha256": sha,
            },
            *[
                {
                    "@type": "cr:FileObject",
                    "@id": f"splits/{stem}.csv",
                    "name": f"{stem}.csv",
                    "description": (
                        f"Benchmark split: {SPLIT_DESCRIPTIONS.get(stem, '')}"
                        f" {rows} instances."
                    ),
                    "contentUrl": (
                        "https://raw.githubusercontent.com/jcarlson212/"
                        f"TopoGym/main/docs/splits/{stem}.csv"
                    ),
                    "encodingFormat": "text/csv",
                    "sha256": sha,
                }
                for stem, _path, sha, rows in _split_files()
            ],
            {
                "@type": "cr:FileObject",
                "@id": "repository",
                "name": "repository",
                "description": (
                    "The TopoGym source: pip install topogym; "
                    "import topogym registers every id below."
                ),
                "contentUrl": "https://github.com/jcarlson212/TopoGym",
                "encodingFormat": "git+https",
                "sha256": "main",
            },
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "environments",
                "name": "environments",
                "description": (
                    "The TopoGym-v1 registry: every stable environment "
                    "id with certified topology."
                ),
                "field": [
                    _field("env_id", "sc:Text",
                           "Stable Gymnasium id, e.g. "
                           "TopoGym/Decoys4-50-v0."),
                    _field("slice", "sc:Text",
                           "GridWorld2D, Top, or Texture."),
                    _field("family", "sc:Text", "Environment family."),
                    _field("canonical_config", "sc:Text",
                           "Canonical configuration string (run-log "
                           "key)."),
                    _field("reference_seed", "sc:Integer",
                           "Seed used for the certified values below; "
                           "any seed regenerates deterministically."),
                    _field("size", "sc:Integer", "Grid side length."),
                    _field("horizon", "sc:Integer",
                           "Pre-determined episode length: the larger "
                           "of the size floor and 3x the turn-aware "
                           "optimal route."),
                    _field("optimal_actions", "sc:Integer",
                           "Fewest actions from start to goal counting "
                           "turns; blank when the environment has no "
                           "goal. Makes difficulty auditable."),
                    _field("n_cells", "sc:Integer", "Total cells."),
                    _field("n_free_cells", "sc:Integer",
                           "Traversable cells."),
                    _field("betti_z2", "sc:Text",
                           "Certified Betti numbers (doors walkable)."),
                    _field("betti_z2_sealed", "sc:Text",
                           "Certified Betti numbers (doors count as "
                           "walls)."),
                    _field("homology", "sc:Text",
                           "Integral homology groups H0/H1/H2."),
                ],
            },
            *[
                {
                    "@type": "cr:RecordSet",
                    "@id": f"split/{stem}",
                    "name": f"split/{stem}",
                    "description": (
                        f"{SPLIT_DESCRIPTIONS.get(stem, 'Benchmark split.')}"
                        " One record per instance; seeds come from the "
                        "split's disjoint band and canonical_config "
                        "reproduces the instance exactly."
                    ),
                    "field": [
                        _field(name, dtype, description,
                               record_set=f"split/{stem}",
                               file_id=f"splits/{stem}.csv")
                        for name, (dtype, description)
                        in SPLIT_FIELDS.items()
                    ],
                }
                for stem, _path, _sha, _rows in _split_files()
            ],
        ],
    }
    croissant_text = json.dumps(croissant, indent=2) + "\n"
    croissant_path = ROOT / "croissant.json"
    if check:
        stale = [
            name for name, current, fresh in (
                ("docs/manifest.csv",
                 manifest.read_text() if manifest.exists() else "",
                 manifest_text),
                ("croissant.json",
                 croissant_path.read_text() if croissant_path.exists()
                 else "", croissant_text),
            ) if current != fresh
        ]
        if stale:
            print("croissant metadata is stale: " + ", ".join(stale)
                  + " -- run python scripts/generate_croissant.py")
            return 1
        print(f"croissant metadata is current "
              f"({len(rows)} environments, sha256 {sha[:12]}...)")
        return 0
    croissant_path.write_text(croissant_text)
    print(f"wrote docs/manifest.csv ({len(rows)} environments) "
          f"and croissant.json (sha256 {sha[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
