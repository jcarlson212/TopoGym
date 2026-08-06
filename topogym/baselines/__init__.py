"""Reference baselines for the TopoGym benchmark.

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

from topogym.baselines.protocol import (
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
    "random": "topogym.baselines.random_walk:RandomBaseline",
    "ppo": "topogym.baselines.ppo:PPOBaseline",
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
