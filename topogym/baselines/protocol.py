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
    val_episodes: int = 10
    #: Episodes per hold-out instance at final evaluation.
    eval_episodes: int = 5
    num_env_runners: int = 2
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
        )
        return BaselineResult(
            algorithm=self.name,
            config=self.config.to_dict(),
            hyperparameters=hyperparameters.to_dict(),
            training=report.to_dict(),
            instances=instances,
        )
