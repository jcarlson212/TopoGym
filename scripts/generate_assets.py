#!/usr/bin/env python3
"""Regenerate the SVG gallery for every benchmark entry.

Writes ``docs/envs/<collection>/<name>.svg`` plus an index page with the
certified metadata of each entry. Run from the repo root after changing
benchmark definitions:

    python scripts/generate_assets.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from topogym import benchmarks as bench  # noqa: E402
from topogym.generation import generate_2d, generate_3d  # noqa: E402
from topogym.rendering.svg import layout_to_svg  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "envs"


def main():
    index = [
        "# Environment gallery",
        "",
        "Rendered in *reveal* mode: hidden doors (purple), decoy fills",
        "(dark red), start (blue), goal (green), one-way doors (yellow,",
        "arrow = entry side), trapdoors (orange). Agents do not see any",
        "of the revealed information.",
        "",
    ]
    for collection, entries in bench.BENCHMARKS.items():
        coll_dir = OUT / collection
        coll_dir.mkdir(parents=True, exist_ok=True)
        index.append(f"## `{collection}`\n")
        index.append("| env | preview | certified topology |")
        index.append("|---|---|---|")
        for entry in entries:
            gen = generate_2d if entry.dim == 2 else generate_3d
            layout = gen(entry.config, entry.layout_seed)
            svg = layout_to_svg(layout)
            path = coll_dir / f"{entry.name}.svg"
            path.write_text(svg)
            md = layout.metadata
            topo = f"b = {list(md.betti_z2)}"
            if not md.asymmetry["is_symmetric"]:
                topo += f", SCCs = {md.asymmetry['n_sccs']}"
                if md.asymmetry["n_absorbing_sccs"]:
                    topo += f" ({md.asymmetry['n_absorbing_sccs']} absorbing)"
            index.append(
                f"| **{entry.name}**<br>{entry.description} "
                f"| <img src=\"{collection}/{entry.name}.svg\" width=\"220\"/> "
                f"| `{topo}` |"
            )
            print(f"wrote {path.relative_to(ROOT)}  {topo}")
        index.append("")
    (OUT / "README.md").write_text("\n".join(index) + "\n")
    print(f"wrote {(OUT / 'README.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
