"""Run per-instance work across processes.

Ray parallelises PPO's *rollouts*, but everything else the benchmark
does is a loop over instances in the driver: the hold-out evaluation,
and the archive-selection sweeps a method like Go-Explore needs. Both
are embarrassingly parallel -- an instance is independent of every
other, and each is separately seeded -- so they belong on all the
cores rather than one.

``work`` must be a *picklable* callable -- typically an object holding
a factory, which builds what it needs (a policy, an archive) once
inside each worker. Passing a live torch module per task would cost
more than the work itself; passing the thing that constructs it does
not, and a module-level class with plain attributes pickles where a
closure or a lambda does not.

Results come back in submission order regardless of completion order,
so a parallel run and a serial one produce identical output. That is
not a nicety: the benchmark's determinism guarantee has to survive the
scheduler.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

logger = logging.getLogger("topogym")


def default_workers() -> int:
    """Leave a couple of cores for the driver and the OS."""
    return max(1, (os.cpu_count() or 4) - 2)


def _run_one(task: tuple):
    """Top-level so it is picklable: build if needed, then apply."""
    work, argument = task
    return work(argument)


def map_instances(work: Callable, arguments: list,
                  workers: int | None = None,
                  chunksize: int = 1) -> list:
    """Apply ``factory()`` to every argument, in parallel.

    ``workers`` of 1 (or 0) runs serially in this process, which is
    what a debugger and a smoke run want. Anything else uses a process
    pool. Order always follows ``arguments``.
    """
    if not arguments:
        return []
    workers = default_workers() if workers is None else workers
    if workers <= 1 or len(arguments) == 1:
        return [work(argument) for argument in arguments]

    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    workers = min(workers, len(arguments))
    logger.info("running %d tasks across %d processes", len(arguments),
                workers)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_run_one,
                                 [(work, a) for a in arguments],
                                 chunksize=chunksize))
    except (BrokenProcessPool, OSError, ValueError) as exc:
        # A pool can fail to start for reasons that have nothing to do
        # with the work -- exhausted descriptors, a closed handle left
        # by an earlier subprocess-heavy stage. Results are identical
        # either way (a test pins that), so losing the pool should cost
        # time, not the evaluation.
        logger.warning("process pool unusable (%s); running serially",
                       exc)
        return [work(argument) for argument in arguments]
