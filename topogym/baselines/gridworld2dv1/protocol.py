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

from topogym.core.constants import ActionMode

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
    #: Episodes per instance during hyperparameter search. Defaults to
    #: the evaluation budget; lower it when a grid is wide.
    tune_episodes: int | None = None
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
    #: Processes for the hold-out sweep and for archive-selection
    #: sweeps. Instances are independent, so this is close to linear.
    eval_workers: int = 1
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

    #: Which splits feed hyperparameter selection. Only ``test`` is
    #: constrained, so a method that fits a selection strategy rather
    #: than a policy may pool everything else -- it declares that here
    #: instead of quietly reaching for it.
    tuning_splits: tuple = ("tune",)

    #: Set when a baseline is trained on one group of the split rather
    #: than all of it; recorded with the results.
    group: str | None = None

    #: The action space this baseline drives, declared rather than
    #: assumed. "egocentric" (turn left / turn right / forward) is the
    #: default; "fourway" is screen directions. Whatever a baseline
    #: declares is used identically in training and evaluation, so a
    #: policy never meets a space it was not trained on -- and it is
    #: recorded with the results, because an agent that pays for turns
    #: and one that does not are not directly comparable.
    actions: ActionMode = ActionMode.EGOCENTRIC

    #: Observation mode, or None to follow the action mode. Baselines
    #: that want the universal (x, y) + texture vector set "vector".
    obs_mode: str | None = None

    #: Reward mode, or None for the environment default (``sparse``:
    #: +1 on reaching the goal). Declared here so a method that wants a
    #: denser signal has to say so, and the choice is recorded with the
    #: results rather than inferred from the numbers.
    reward_mode: str | None = None

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig()

    # -- the three things a baseline may customise --------------------

    def select_hyperparameters(self, tuning: dict) -> Hyperparameters:
        """Choose hyperparameters from the declared tuning splits.

        ``tuning`` maps each name in :attr:`tuning_splits` to its rows,
        so a method can use them as one pool or as successive rungs --
        its choice, and never including the hold-out.
        """
        return Hyperparameters()

    @abc.abstractmethod
    def fit(self, train_rows: list, val_rows: list,
            hyperparameters: Hyperparameters) -> TrainingReport:
        """Train on ``train_rows``, stopping on ``val_rows``."""

    @abc.abstractmethod
    def policy(self) -> Callable:
        """``policy(observation, env) -> action`` after fitting."""

    @staticmethod
    def observation_codes() -> int:
        """How many distinct observation codes exist -- the size an
        embedding over symbolic observations must have.

        Constant across every slice, and deliberately *not* derived
        from what a run has seen. Hazard (8) and wormhole (9) codes
        occur only in Texture worlds, so a policy trained on
        GridWorld2D meets them for the first time on the hold-out; a
        table sized to the training codes would error there or, worse,
        alias them onto something familiar.
        """
        from topogym.core.constants import OBS_CODE_COUNT

        return OBS_CODE_COUNT

    def env_options(self) -> dict:
        """Keyword arguments applied to every instance this baseline
        sees, in training and in evaluation alike.

        Override to ask for something other than the defaults -- a
        different action space, the universal vector observation, a
        wider view radius. Anything declared here is recorded with the
        results, so a run states the terms it was measured on instead
        of leaving them to be inferred.
        """
        options = {"actions": ActionMode(self.actions).value}
        if self.obs_mode is not None:
            options["obs_mode"] = self.obs_mode
        if self.reward_mode is not None:
            options["reward_mode"] = self.reward_mode
        return options

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

    def policy_factory(self):
        """A picklable, zero-argument callable building :meth:`policy`
        inside a worker process, or ``None`` when the policy cannot
        cross a process boundary.

        Returning ``None`` is not a failure -- a policy wrapping a
        torch module is legitimately process-bound -- it simply keeps
        that baseline's evaluation serial.
        """
        return None

    def choose_reset_factory(self):
        """A picklable builder for :meth:`choose_reset`, or ``None``.

        Needed only for parallel evaluation, where the hook has to be
        constructed inside each worker. A bound method is not a
        factory, and an archive that lives in the driver cannot be
        shared across processes -- so an archive method supplies one
        here and rebuilds its archive per instance, which is what it
        does anyway.
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
        from topogym.baselines.gridworld2dv1.evaluate import evaluate_split

        assert "test" not in self.tuning_splits, (
            "the hold-out is not a tuning split"
        )
        tuning = {name: splits.get(name, [])
                  for name in self.tuning_splits}
        logger.info("[%s] selecting hyperparameters on %s (%d rows)",
                    self.name, "+".join(self.tuning_splits),
                    sum(len(rows) for rows in tuning.values()))
        hyperparameters = self.select_hyperparameters(tuning)

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
            choose_reset_factory=self.choose_reset_factory(),
            policy_factory=self.policy_factory(),
            workers=self.config.eval_workers,
            env_options=self.env_options(),
        )
        return BaselineResult(
            algorithm=self.name,
            config={**self.config.to_dict(),
                    "env_options": self.env_options()},
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
