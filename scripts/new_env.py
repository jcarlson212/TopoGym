#!/usr/bin/env python3
"""Design a TopoGym environment from the command line.

Any field of TopoGenConfig2D / TopoGenConfig3D can be set with
``--set key=value`` (values are parsed as Python literals when possible):

    python scripts/new_env.py --dim 2 --name klein-b7 --seed 42 \\
        --set base=klein size=21 target_b1=7 n_chambers=2 n_decoys=1

Writes ``docs/envs/community/<name>.svg`` and ``<name>.json`` (config +
seed + certified metadata) and prints the metadata. See
docs/contributing_environments.md for how to submit the result as a PR.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from topogym.generation import (  # noqa: E402
    TopoGenConfig2D,
    TopoGenConfig3D,
    generate_2d,
    generate_3d,
)
from topogym.rendering.svg import layout_to_svg  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_value(raw: str):
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw  # plain string, e.g. base=klein


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dim", type=int, choices=(2, 3), default=2)
    ap.add_argument("--name", required=True, help="output file stem")
    ap.add_argument("--seed", type=int, required=True,
                    help="layout seed (part of the environment's identity)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="generator config fields, e.g. base=torus size=17")
    ap.add_argument("--out", default=str(ROOT / "docs" / "envs" / "community"))
    args = ap.parse_args()

    config_cls = TopoGenConfig2D if args.dim == 2 else TopoGenConfig3D
    fields = {f.name for f in dataclasses.fields(config_cls)}
    overrides = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        if key not in fields:
            ap.error(f"unknown config field {key!r}; valid: {sorted(fields)}")
        overrides[key] = parse_value(raw)
    cfg = config_cls(**overrides)

    gen = generate_2d if args.dim == 2 else generate_3d
    layout = gen(cfg, args.seed)
    md = layout.metadata

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{args.name}.svg"
    svg_path.write_text(layout_to_svg(layout))
    json_path = out_dir / f"{args.name}.json"
    json_path.write_text(json.dumps(
        {"config": cfg.to_dict(), "seed": args.seed,
         "metadata": md.to_dict()},
        indent=2, default=str,
    ) + "\n")

    print(json.dumps(md.to_dict(), indent=2, default=str))
    print(f"\nwrote {svg_path}\nwrote {json_path}")
    print(
        "\nHappy with it? Freeze it as a benchmark entry — see "
        "docs/contributing_environments.md (step 2)."
    )


if __name__ == "__main__":
    main()
