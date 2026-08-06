"""Reference baselines, one package per benchmark version.

A benchmark version fixes its splits, its evaluation protocol, and the
shape of what a policy sees. Baselines are written against those, so
they live under the version they were run for --
:mod:`topogym.baselines.gridworld2dv1` -- and a later benchmark can
change any of it without disturbing results already published.

Requires the optional extra::

    pip install topogym[benchmarks]

Nothing here is imported by ``topogym`` itself, and Ray and torch are
imported lazily inside the methods that need them, so a core install
never pays for them.
"""

#: Benchmark version -> the package holding its baselines.
BENCHMARK_PACKAGES = {
    "gridworld2d-v1": "topogym.baselines.gridworld2dv1",
}

DEFAULT_BENCHMARK = "gridworld2d-v1"

__all__ = ["BENCHMARK_PACKAGES", "DEFAULT_BENCHMARK", "baselines_for"]


def baselines_for(benchmark: str = DEFAULT_BENCHMARK):
    """The baselines package for a benchmark version."""
    import importlib

    if benchmark not in BENCHMARK_PACKAGES:
        raise KeyError(
            f"no baselines for benchmark {benchmark!r}; known: "
            f"{sorted(BENCHMARK_PACKAGES)}"
        )
    return importlib.import_module(BENCHMARK_PACKAGES[benchmark])
