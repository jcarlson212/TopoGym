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

#: The baselines shipping in this repository, captured before any
#: private module is merged in. Artefact paths are named after the
#: algorithm, so running an unpublished method would write its name
#: into the working tree; anything not in here is routed under a
#: ``private/`` subtree that .gitignore covers.
PUBLIC_BASELINES = dict(BASELINES)


def is_public(name: str) -> bool:
    """Whether results for ``name`` may be written where git can see."""
    return name in PUBLIC_BASELINES


def _register_private_baselines() -> None:
    """Merge in baselines that ship outside the repository.

    A method under review has no business being public, but it should
    not need a fork to run inside the harness either -- it has to face
    exactly the protocol every published baseline faces, or its numbers
    are not comparable. Modules matching
    ``concrete_baselines/_private_*.py`` and declaring their own
    ``BASELINES`` mapping are folded in when present and ignored when
    absent, so a clone without them is a working library with fewer
    baselines rather than an import error.

    The pattern is generic on purpose: naming a method here would put
    it in a tracked file, which is what keeping it out of the
    repository was for.
    """
    import importlib
    import pathlib

    folder = pathlib.Path(__file__).parent / "concrete_baselines"
    for module in sorted(folder.glob("_private_*.py")):
        name = (f"topogym.baselines.gridworld2dv1.concrete_baselines"
                f".{module.stem}")
        try:
            BASELINES.update(importlib.import_module(name).BASELINES)
        except Exception as exc:  # never break the public library
            import logging

            logging.getLogger("topogym").warning(
                "private baseline %s did not load: %s", module.stem, exc)


_register_private_baselines()

__all__ = [
    "BASELINES",
    "PUBLIC_BASELINES",
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
