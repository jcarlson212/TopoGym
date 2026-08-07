"""Reference baselines for the **gridworld2d-v1** benchmark.

Baselines live under the benchmark version they were run against, so a
later benchmark can change its splits, protocol, or observation shape
without disturbing published results. Nothing here is shared with
another version by accident: if a future benchmark wants this
protocol, it imports it deliberately or forks it.

The algorithms come from Ray RLlib -- the library does not reimplement
PPO -- so what lives here is the protocol every baseline shares (how it
consumes each split, how it is evaluated, how results are published)
and thin subclasses per algorithm.

Requires the optional extra::

    pip install topogym[benchmarks]

Nothing here is imported by ``topogym`` itself, and Ray and torch are
imported lazily inside the methods that need them, so a core install
never pays for them. Importing a specific baseline is likewise lazy:
:func:`get_baseline` resolves by name.
"""

from topogym.baselines.gridworld2dv1.protocol import (
    SPLIT_USAGE,
    Baseline,
    BaselineConfig,
    BaselineResult,
    Hyperparameters,
    TrainingReport,
)

#: name -> "module:attribute", resolved on demand so that listing the
#: baselines never imports Ray.
BASELINES = {
    "random": "topogym.baselines.gridworld2dv1.concrete_baselines.random_walk:RandomBaseline",
    "ppo": "topogym.baselines.gridworld2dv1.concrete_baselines.ppo:PPOBaseline",
    "icm-ppo":
        "topogym.baselines.gridworld2dv1.concrete_baselines.intrinsic"
        ":ICMBaseline",
    "rnd-ppo":
        "topogym.baselines.gridworld2dv1.concrete_baselines.intrinsic"
        ":RNDBaseline",
    "go-explore-phase1":
        "topogym.baselines.gridworld2dv1.concrete_baselines.goexplore_phase1:GoExplorePhase1Baseline",
    "go-explore-phase1and2":
        "topogym.baselines.gridworld2dv1.concrete_baselines"
        ".goexplore_phase1_and_phase2:GoExplorePhase12Baseline",
}

__all__ = [
    "BASELINES",
    "SPLIT_USAGE",
    "Baseline",
    "BaselineConfig",
    "BaselineResult",
    "Hyperparameters",
    "TrainingReport",
    "get_baseline",
]


def get_baseline(name: str) -> type:
    """The baseline class registered under ``name``."""
    import importlib

    if name not in BASELINES:
        raise KeyError(
            f"unknown baseline {name!r}; expected one of "
            f"{sorted(BASELINES)}"
        )
    module_name, attribute = BASELINES[name].split(":")
    return getattr(importlib.import_module(module_name), attribute)
