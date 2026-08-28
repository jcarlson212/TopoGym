"""Soak test: the door-separation guarantee, over many seeds.

The unit tests in ``tests/generation/test_door_separation.py`` check
the property on a handful of seeds because they run in the commit
gate. This checks it on thousands, which is the number that matters:
the guarantee is enforced by *rejection*, so the failure mode it could
still have is a seed whose every attempt is rejected -- a world that
simply does not exist -- and rare cases are found by looking at many
seeds, not by looking harder at eight.

Two things are asserted per seed, both re-derived from the layout
rather than read from what generation recorded:

* every pair of chamber doors is at least ``min_door_distance`` apart
  over the free-cell graph, and no two touch even diagonally;
* the metadata's ``min_door_distance`` equals the measured value, so a
  reader who trusts the certificate and one who measures agree.

Usage::

    python scripts/soak_door_separation.py --seeds 10000 --workers 8
"""

from __future__ import annotations

import argparse
import collections
import itertools
import multiprocessing as mp

import topogym  # noqa: F401  (registers the ids)
from topogym.registry import EXTRA_KWARGS, REGISTRY


def _measure(layout) -> tuple:
    """``(min pairwise door distance, closest Chebyshev gap)``."""
    free = set(map(tuple, layout.free_cells))
    doors = [tuple(spec.cell)
             for f in layout.features if f.kind == "chamber"
             for spec in f.doors]
    if len(doors) < 2:
        return None, None

    def walk(source):
        seen = {source: 0}
        queue = collections.deque([source])
        while queue:
            x, y = queue.popleft()
            for step in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if step in free and step not in seen:
                    seen[step] = seen[(x, y)] + 1
                    queue.append(step)
        return seen

    graph, plane = None, None
    for one, other in itertools.combinations(doors, 2):
        reach = walk(one).get(other)
        if reach is not None:
            graph = reach if graph is None else min(graph, reach)
        touch = max(abs(one[0] - other[0]), abs(one[1] - other[1]))
        plane = touch if plane is None else min(plane, touch)
    return graph, plane


def _check(job) -> tuple:
    name, seed = job
    from topogym.generation.generator import generate_2d

    cfg = REGISTRY[name]
    try:
        layout = generate_2d(cfg, seed=seed)
    except Exception as exc:                       # a seed with no world
        return name, seed, f"generation failed: {type(exc).__name__}"
    graph, plane = _measure(layout)
    if graph is None:
        return name, seed, None
    if graph < cfg.min_door_distance:
        return name, seed, f"gap {graph} < {cfg.min_door_distance}"
    if plane is not None and plane <= 1:
        return name, seed, f"doors touch (Chebyshev {plane})"
    recorded = layout.metadata.connectivity.get("min_door_distance")
    if recorded != graph:
        return name, seed, f"metadata {recorded} != measured {graph}"
    horizon = EXTRA_KWARGS.get(name, {}).get("max_steps")
    if horizon is not None and graph <= horizon:
        return name, seed, f"gap {graph} <= horizon {horizon}"
    return name, seed, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--family", default="EnlargedChamberCount")
    args = parser.parse_args()

    names = sorted(n for n in REGISTRY
                   if n.startswith(args.family)
                   and REGISTRY[n].min_door_distance > 0)
    if not names:
        print(f"no constrained worlds match {args.family!r}")
        return 1
    per_world = max(1, args.seeds // len(names))
    jobs = [(name, seed) for name in names for seed in range(per_world)]
    print(f"{len(names)} worlds x {per_world} seeds = {len(jobs)} checks "
          f"on {args.workers} workers")

    failures, done = [], 0
    with mp.Pool(args.workers) as pool:
        for name, seed, problem in pool.imap_unordered(_check, jobs,
                                                       chunksize=8):
            done += 1
            if problem:
                failures.append((name, seed, problem))
                print(f"FAIL {name} seed={seed}: {problem}", flush=True)
            if done % 250 == 0:
                print(f"  {done}/{len(jobs)} checked, "
                      f"{len(failures)} failures", flush=True)
    if failures:
        print(f"\n{len(failures)} FAILURES of {len(jobs)}")
        return 1
    print(f"\nALL PASS: {len(jobs)} seeds, every pair of doors beyond "
          "the horizon, metadata agrees with geometry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
