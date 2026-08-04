#!/usr/bin/env python3
"""Generate Croissant metadata for the TopoGym-v1 benchmark.

    python scripts/generate_croissant.py

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
    "size", "horizon", "n_cells", "n_free_cells",
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


def _field(name: str, dtype: str, description: str) -> dict:
    return {
        "@type": "cr:Field",
        "@id": f"environments/{name}",
        "name": name,
        "description": description,
        "dataType": dtype,
        "source": {
            "fileObject": {"@id": "manifest.csv"},
            "extract": {"column": name},
        },
    }


def main() -> int:
    manifest = ROOT / "docs" / "manifest.csv"
    rows = _rows()
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

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
            "The pinned environment registry of TopoGym: gridworld "
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
                           "Pre-determined episode length."),
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
            }
        ],
    }
    (ROOT / "croissant.json").write_text(
        json.dumps(croissant, indent=2) + "\n"
    )
    print(f"wrote docs/manifest.csv ({len(rows)} environments) "
          f"and croissant.json (sha256 {sha[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
