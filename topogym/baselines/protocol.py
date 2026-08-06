"""The baseline interface.

Every reference algorithm implements :class:`Baseline`, whether or not
it learns anything: Go-Explore's exploration is random by default, PPO
takes gradients, and RND or ICM are PPO with an intrinsic reward bolted
on. They differ in :meth:`Baseline.fit` and :meth:`Baseline.policy`;
they do not differ in how they are allowed to touch the splits, which
is the point of putting the protocol here rather than in each
algorithm.

The shared dataclasses -- :class:`BaselineConfig`,
:class:`Hyperparameters`, :class:`TrainingReport` -- are what a
subclass reuses so that a variant of an existing algorithm is a
subclass and an override, not a copy.
"""

from __future__ import annotations

import abc
import logging
import pathlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

logger = logging.getLogger("topogym")

#: How every baseline must consume the splits. The hold-out rule is not
#: per-algorithm: ``test`` is touched once, after training has stopped,
#: and never informs a decision.
SPLIT_USAGE = {
    "tune": "hyperparameter selection only",
    "train": "policy updates",
    "val": "early stopping; no gradient is ever taken on it",
    "test": "final evaluation, once, after training has stopped",
}

#: Only ``test`` is constrained. How a method spends ``tune``,
#: ``train`` and ``val`` is its own business: a gradient method takes
#: updates on ``train`` and stops on ``val``, while an archive method
#: may reasonably pool all three, since what it is fitting is a
#: selection strategy rather than a policy. What is *not* negotiable is
#: that every method faces the same hold-out under the same conditions
#: -- the same instances, the same contiguous episode budget on each,
#: and the same offer of an archive reset at every episode boundary.
HOLD_OUT_RULE = (
    "test is read once, after training has stopped; every method gets "
    "the same instances, the same contiguous per-instance episode "
    "budget, and the same episode-boundary reset probe"
)


@dataclass
class BaselineConfig:
    """Knobs every baseline honours, whatever it does internally."""

    seed: int = 0
    #: Training iterations per candidate during hyperparameter search.
    tune_iterations: int = 2
    max_iterations: int = 200
    #: Consecutive validation checks without improvement before stopping.
    patience: int = 5
    val_every: int = 5
    #: Episodes per validation check, swept across the split rather
    #: than run per instance: this runs every ``val_every`` iterations,
    #: so a per-instance sweep would cost far more than the training it
    #: is supposed to supervise.
    val_episodes: int = 50
    #: Episodes per hold-out instance at final evaluation. The hold-out
    #: is read once, so it can afford the thorough version -- and a real
    #: budget per world is what makes "how well does it explore"
    #: answerable at all.
    eval_episodes: int = 50
    #: Rollout parallelism. Environment stepping is the bottleneck for
    #: these worlds -- the policy is a small MLP over a 49-dim vector --
    #: so cores matter and accelerators mostly do not.
    num_env_runners: int = 2
    #: Environments vectorised inside each runner; multiplies throughput
    #: without another process.
    num_envs_per_runner: int = 1
    #: Consecutive training episodes on one instance. One suits
    #: gradient methods; archive methods need a contiguous run for an
    #: archive to accumulate on a given world.
    train_episodes_per_instance: int = 1
    num_learners: int = 0
    #: CUDA devices per learner. Apple MPS is not a Ray GPU resource, so
    #: leave this at 0 on Apple silicon.
    gpus_per_learner: int = 0
    train_batch_size: int = 4000
    #: Where checkpoints and logs go; gitignored by convention.
    run_dir: pathlib.Path | None = None

    def to_dict(self) -> dict:
        out = asdict(self)
        out["run_dir"] = str(self.run_dir) if self.run_dir else None
        return out


@dataclass
class Hyperparameters:
    """A chosen configuration and the tuning score that chose it."""

    values: dict = field(default_factory=dict)
    tuning_score: float | None = None
    #: Every candidate considered, for reproducibility.
    searched: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"values": dict(self.values),
                "tuning_score": self.tuning_score,
                "searched": list(self.searched)}


@dataclass
class TrainingReport:
    """What training cost and why it stopped."""

    iterations: int = 0
    stopped_early: bool = False
    #: Why training ended, in words -- "budget exhausted", "validation
    #: plateaued", or "no learning signal". A run that never moved its
    #: objective has not converged, and must not be reported as if it
    #: had.
    stopped_because: str = "budget exhausted"
    best_val_return: float | None = None
    best_checkpoint: str | None = None
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BaselineResult:
    """Everything a baseline reports, ready to serialize."""

    algorithm: str
    config: dict = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    instances: list = field(default_factory=list)
    aggregates: dict = field(default_factory=dict)
    curves: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "split_usage": SPLIT_USAGE,
            "config": self.config,
            "hyperparameters": self.hyperparameters,
            "training": self.training,
            "aggregates": self.aggregates,
            "curves": self.curves,
            "instances": self.instances,
        }


class Baseline(abc.ABC):
    """A reference algorithm, learned or not.

    Subclasses implement :meth:`fit` and :meth:`policy`. Overriding
    :meth:`select_hyperparameters` is optional -- the default declines
    to tune, which is the honest answer for an algorithm with nothing
    to tune.
    """

    #: Short identifier used in filenames, plots, and result JSON.
    name: str = "baseline"

    #: Set when a baseline is trained on one group of the split rather
    #: than all of it; recorded with the results.
    group: str | None = None

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig()

    # -- the three things a baseline may customise --------------------

    def select_hyperparameters(self, tune_rows: list) -> Hyperparameters:
        """Choose hyperparameters on the tuning split, and only there."""
        return Hyperparameters()

    @abc.abstractmethod
    def fit(self, train_rows: list, val_rows: list,
            hyperparameters: Hyperparameters) -> TrainingReport:
        """Train on ``train_rows``, stopping on ``val_rows``."""

    @abc.abstractmethod
    def policy(self) -> Callable:
        """``policy(observation, env) -> action`` after fitting."""

    def choose_reset(self, env, info: dict):
        """Where the next episode should resume, or ``None`` to start
        where the world says.

        Called at every episode boundary, which is the only place the
        environment allows the choice. Archive methods override this to
        return a previously visited cell; everything else inherits the
        default and never notices. Returning a cell costs no step --
        the reset lands directly on it.
        """
        return None

    def close(self) -> None:
        """Release anything :meth:`fit` acquired."""

    # -- the protocol, which subclasses do not override ---------------

    def run(self, splits: dict) -> BaselineResult:
        """Tune, train, stop, and evaluate -- in that order.

        ``test`` is read once, here, after training has finished. No
        subclass gets to change that.
        """
        from topogym.baselines.evaluate import evaluate_split

        logger.info("[%s] selecting hyperparameters on tune (%d rows)",
                    self.name, len(splits["tune"]))
        hyperparameters = self.select_hyperparameters(splits["tune"])

        logger.info("[%s] training on train (%d rows), stopping on val "
                    "(%d rows)", self.name, len(splits["train"]),
                    len(splits["val"]))
        report = self.fit(splits["train"], splits["val"], hyperparameters)

        logger.info("[%s] evaluating on test (%d rows) -- first and only "
                    "look", self.name, len(splits["test"]))
        instances = evaluate_split(
            splits["test"], self.policy(),
            episodes=self.config.eval_episodes, seed=self.config.seed,
            choose_reset=self.choose_reset,
        )
        return BaselineResult(
            algorithm=self.name,
            config=self.config.to_dict(),
            hyperparameters=hyperparameters.to_dict(),
            training=report.to_dict(),
            instances=instances,
        )


#: How split rows may be partitioned before training. A baseline is
#: trained once per group, which is the axis that decides what the
#: benchmark is measuring: ``all`` asks for one general explorer,
#: ``unit`` is the Procgen-style setting where a policy sees one world
#: family at one size and generalises across seeds.
GROUPINGS = ("all", "slice", "family", "unit")


def group_rows(rows: list, grouping: str) -> dict:
    """Partition split rows by the chosen grouping, in a stable order."""
    if grouping not in GROUPINGS:
        raise ValueError(
            f"unknown grouping {grouping!r}; expected one of {GROUPINGS}"
        )
    if grouping == "all":
        return {"all": list(rows)}
    grouped: dict = {}
    for row in rows:
        grouped.setdefault(row[grouping], []).append(row)
    return dict(sorted(grouped.items()))
