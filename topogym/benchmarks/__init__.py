"""Benchmark collections: frozen (config, layout_seed) suites.

Each entry pins a generator config and a layout seed, so every user runs
byte-identical environments. Entries are also registered as Gymnasium ids
(``TopoGym/<Collection>-<name>-v0``).

Collections
-----------
- ``2d_bench_grid_small``: 16 environments covering every 2D base manifold,
  chambers/decoys, and two topologically-trivial controls.
- ``3d_bench_grid_small``: 8 environments covering the 3D bases, rings
  (b1), voids (b2), rooms, and a 3D maze control.
- ``2d_bench_grid_small_directed`` / ``3d_bench_grid_small_directed``:
  mirrored suites whose environments contain one-way doors and trapdoors
  (trap rooms, airlocks, trapdoor rooms) — asymmetric traversability, see
  the ``asymmetry`` metadata block.

Usage::

    import topogym.benchmarks as bench

    for entry in bench.get_benchmark("2d_bench_grid_small"):
        env = entry.make()
        obs, info = env.reset(seed=0)
        print(entry.name, info["topology"]["betti_z2"])

    rows = bench.benchmark_metadata("2d_bench_grid_small")  # list of dicts
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium.envs.registration import register

from topogym.generation import TopoGenConfig2D, TopoGenConfig3D

__all__ = [
    "BenchmarkEntry",
    "BENCHMARKS",
    "benchmark_names",
    "get_benchmark",
    "benchmark_metadata",
]


@dataclass(frozen=True)
class BenchmarkEntry:
    name: str
    collection: str
    env_id: str
    dim: int
    config: object  # TopoGenConfig2D | TopoGenConfig3D
    layout_seed: int
    description: str

    def make(self, **kwargs) -> gym.Env:
        return gym.make(self.env_id, **kwargs)

    def metadata(self):
        """Generate (or re-generate) the layout and return its certified
        :class:`TopologyMetadata`."""
        from topogym.generation import generate_2d, generate_3d

        gen = generate_2d if self.dim == 2 else generate_3d
        return gen(self.config, self.layout_seed).metadata


def _c2(**kw) -> TopoGenConfig2D:
    return TopoGenConfig2D(**kw)


def _c3(**kw) -> TopoGenConfig3D:
    return TopoGenConfig3D(**kw)


_RAW = {
    "2d_bench_grid_small": [
        ("square-holes", 101, "3 holes on a disc: b1 = 3, nothing hidden",
         _c2(base="square", size=15, n_holes=3, n_chambers=0, n_decoys=0)),
        ("square-rooms", 102, "holes + hidden chambers + a decoy",
         _c2(base="square", size=17, n_holes=2, n_chambers=2, n_decoys=1)),
        ("square-decoyfield", 103,
         "many identical-looking rooms; most are empty decoys",
         _c2(base="square", size=21, n_holes=0, n_chambers=2, n_decoys=3,
             door_tries=(3, 6))),
        ("annulus", 104, "thick annulus base + one chamber and one decoy",
         _c2(base="annulus", size=19, n_holes=0, n_chambers=1, n_decoys=1)),
        ("plane-6holes", 105, "plane with 6 large holes: b1 = 6",
         _c2(base="x_holes", size=21, n_base_holes=6, n_holes=0,
             n_chambers=0, n_decoys=0)),
        ("cylinder-rooms", 106, "cylinder: one loop is the world itself",
         _c2(base="cylinder", size=17, n_holes=1, n_chambers=1, n_decoys=1)),
        ("mobius-rooms", 107, "Mobius band: crossing the seam mirrors you",
         _c2(base="mobius", size=17, n_holes=1, n_chambers=1, n_decoys=1)),
        ("torus-holes", 108, "torus with 2 holes: b1 = 4",
         _c2(base="torus", size=15, n_holes=2, n_chambers=0, n_decoys=0)),
        ("torus-rooms", 109, "torus with chambers and a decoy",
         _c2(base="torus", size=17, n_holes=1, n_chambers=2, n_decoys=1)),
        ("torus-goal-in-chamber", 110,
         "the goal hides inside a chamber: doors must be found",
         _c2(base="torus", size=17, n_holes=1, n_chambers=2, n_decoys=1,
             goal_in_chamber=True, door_tries=(2, 4))),
        ("klein-rooms", 111, "Klein bottle: torus-like but non-orientable",
         _c2(base="klein", size=17, n_holes=1, n_chambers=1, n_decoys=1)),
        ("rp2-rooms", 112, "RP^2: the antipodal world",
         _c2(base="rp2", size=17, n_holes=1, n_chambers=1, n_decoys=1)),
        ("sphere-holes", 113, "cube-sphere with 3 holes",
         _c2(base="sphere", size=7, n_holes=3, n_chambers=0, n_decoys=0)),
        ("sphere-rooms", 114, "cube-sphere with chambers and a decoy",
         _c2(base="sphere", size=7, n_holes=1, n_chambers=1, n_decoys=1)),
        ("control-maze", 115,
         "control: perfect maze, hard to explore, b1 = 0",
         _c2(base="square", size=17, style="maze")),
        ("control-zigzag", 116,
         "control: serpentine corridor, one long path, b1 = 0",
         _c2(base="square", size=17, style="zigzag")),
    ],
    "2d_bench_grid_small_directed": [
        ("square-traproom", 201, "a one-way room: enter and you stay",
         _c2(base="square", size=19, n_holes=1, n_chambers=0, n_decoys=0,
             n_trap_rooms=1)),
        ("square-airlock", 202, "a directed circuit: in one door, out the other",
         _c2(base="square", size=19, n_holes=1, n_chambers=0, n_decoys=0,
             n_airlocks=1)),
        ("square-trapdoor", 203,
         "a trapdoor room: the way in seals, a hidden hatch leads out",
         _c2(base="square", size=19, n_holes=1, n_chambers=0, n_decoys=0,
             n_trapdoor_rooms=1, trapdoor_escape_tries=(3, 5))),
        ("cylinder-trapdoor", 204, "trapdoor room on a cylinder",
         _c2(base="cylinder", size=19, n_holes=1, n_chambers=0, n_decoys=0,
             n_trapdoor_rooms=1)),
        ("torus-traproom", 205, "trap room on a torus",
         _c2(base="torus", size=19, n_holes=1, n_chambers=1, n_decoys=0,
             n_trap_rooms=1)),
        ("torus-airlock-mix", 206, "airlock + trapdoor room on a torus",
         _c2(base="torus", size=21, n_holes=1, n_chambers=0, n_decoys=1,
             n_airlocks=1, n_trapdoor_rooms=1)),
        ("sphere-traproom", 207, "trap room on the cube-sphere",
         _c2(base="sphere", size=7, n_holes=1, n_chambers=0, n_decoys=0,
             n_trap_rooms=1)),
        ("square-gauntlet", 208,
         "one of everything: trap room, airlock, trapdoor room, decoy",
         _c2(base="square", size=25, n_holes=1, n_chambers=1, n_decoys=1,
             n_trap_rooms=1, n_airlocks=1, n_trapdoor_rooms=1)),
    ],
    "3d_bench_grid_small": [
        ("box-blobs", 301, "2 solid obstacles: b2 = 2 enclosing shells",
         _c3(base="box", size=10, n_rings=0, n_blobs=2, n_chambers=0,
             n_decoys=0)),
        ("box-ring", 302, "a ring obstacle: b1 = 1, b2 = 1",
         _c3(base="box", size=10, n_rings=1, n_blobs=0, n_chambers=0,
             n_decoys=0)),
        ("box-rooms", 303, "a hollow chamber and a sealed decoy",
         _c3(base="box", size=12, n_rings=0, n_blobs=0, n_chambers=1,
             n_decoys=1)),
        ("box-mixed", 304, "ring + void + chamber",
         _c3(base="box", size=12, n_rings=1, n_blobs=1, n_chambers=1,
             n_decoys=0)),
        ("solid-torus", 305, "solid torus: one loop is the world itself",
         _c3(base="solid_torus", size=10, n_rings=1, n_blobs=1,
             n_chambers=0, n_decoys=0)),
        ("torus3", 306, "3-torus: wraps in every direction, b1 = 3",
         _c3(base="torus3", size=8, n_rings=0, n_blobs=1, n_chambers=0,
             n_decoys=0)),
        ("shell", 307, "spherical shell: a big void you can never enter",
         _c3(base="shell", size=10, n_rings=0, n_blobs=0, n_chambers=0,
             n_decoys=0)),
        ("control-maze3d", 308, "control: perfect 3D maze, b1 = b2 = 0",
         _c3(base="box", size=9, style="maze")),
    ],
    "3d_bench_grid_small_directed": [
        ("box-traproom", 401, "a one-way room in 3D",
         _c3(base="box", size=12, n_rings=0, n_blobs=1, n_chambers=0,
             n_decoys=0, n_trap_rooms=1)),
        ("box-airlock", 402, "a 3D airlock: directed circuit through a room",
         _c3(base="box", size=12, n_rings=0, n_blobs=0, n_chambers=0,
             n_decoys=0, n_airlocks=1)),
        ("box-trapdoor", 403, "3D trapdoor room with a hidden escape hatch",
         _c3(base="box", size=12, n_rings=0, n_blobs=0, n_chambers=0,
             n_decoys=0, n_trapdoor_rooms=1)),
        ("solid-torus-traproom", 404, "trap room in a solid torus",
         _c3(base="solid_torus", size=12, n_rings=0, n_blobs=0,
             n_chambers=0, n_decoys=0, n_trap_rooms=1)),
    ],
}

_COLLECTION_ID = {
    "2d_bench_grid_small": "Bench2DSmall",
    "2d_bench_grid_small_directed": "Bench2DSmallDirected",
    "3d_bench_grid_small": "Bench3DSmall",
    "3d_bench_grid_small_directed": "Bench3DSmallDirected",
}

BENCHMARKS = {}
for _coll, _entries in _RAW.items():
    _out = []
    for _name, _seed, _desc, _cfg in _entries:
        _env_id = f"TopoGym/{_COLLECTION_ID[_coll]}-{_name}-v0"
        _dim = 2 if isinstance(_cfg, TopoGenConfig2D) else 3
        register(
            id=_env_id,
            entry_point=(
                "topogym.envs:TopoGrid2DEnv" if _dim == 2
                else "topogym.envs:TopoGrid3DEnv"
            ),
            kwargs={"config": _cfg, "layout_seed": _seed},
        )
        _out.append(BenchmarkEntry(
            name=_name, collection=_coll, env_id=_env_id, dim=_dim,
            config=_cfg, layout_seed=_seed, description=_desc,
        ))
    BENCHMARKS[_coll] = tuple(_out)


def benchmark_names():
    return list(BENCHMARKS)


def get_benchmark(name: str):
    if name not in BENCHMARKS:
        raise KeyError(
            f"unknown benchmark {name!r}; available: {benchmark_names()}"
        )
    return BENCHMARKS[name]


def benchmark_metadata(name: str):
    """Certified metadata dicts for every entry — ready for sweeping
    (each row includes the entry name and env id)."""
    rows = []
    for entry in get_benchmark(name):
        row = entry.metadata().to_dict()
        row["name"] = entry.name
        row["env_id"] = entry.env_id
        row["description"] = entry.description
        rows.append(row)
    return rows
