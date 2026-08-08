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

    #: Training budget in environment *steps*. When set it is the
    #: authority, and the episode counts are derived from it per layout
    #: rather than taken as given. Steps are the only currency every
    #: method spends the same way: an episode is worth 130 steps on one
    #: layout and 6,760 on another, and a method counted in gradient
    #: iterations spends whatever its batch size multiplies out to.
    train_steps: int | None = None
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
    #: Best validation coverage. Training stops only when neither this
    #: nor the return has improved for ``patience`` checks.
    best_val_coverage: float | None = None
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


#: What a tuning sweep may rank candidates by, best signal first.
TUNING_SIGNALS = ("return", "coverage")


def choose_tuning_signal(measurements: list) -> str:
    """Which signal a sweep should rank by, decided once for all of it.

    Return is the objective and is used whenever it carries
    information. It often does not: with a sparse goal and batches
    spread over many parallel environments, every candidate can score
    zero -- or nothing at all, if no episode finished -- and picking a
    winner from that is a coin flip dressed as a choice. Coverage is
    the fallback, because a method that reached more of the world did
    something a method that reached less did not.

    Chosen once per sweep rather than per candidate: ranking one
    candidate by return and another by coverage would compare
    incomparable scales.
    """
    import math

    informative = [
        m.get("return") for m in measurements
        if m.get("return") is not None
        and not math.isnan(m.get("return"))
        and m.get("return") != 0.0
    ]
    return "return" if informative else "coverage"


def rank_candidates(measurements: list) -> tuple:
    """``(ranked measurements, signal)``, best first."""
    signal = choose_tuning_signal(measurements)
    import math

    def key(measurement):
        value = measurement.get(signal)
        if value is None or math.isnan(value):
            return float("inf")
        return -value

    return sorted(measurements, key=key), signal


class Baseline(abc.ABC):
    """A reference algorithm, learned or not.

    Subclasses implement :meth:`fit` and :meth:`policy`. Overriding
    :meth:`select_hyperparameters` is optional -- the default declines
    to tune, which is the honest answer for an algorithm with nothing
    to tune.
    """

    #: Short identifier used in filenames, plots, and result JSON.
    name: str = "baseline"

    #: Whether the method adapts *within* a hold-out instance.
    #:
    #: Almost always ``False``: a policy is fitted on ``train`` and the
    #: hold-out measures whether it transfers. Go-Explore's phase 2 is
    #: the exception the paper describes -- it robustifies a trajectory
    #: in the world that produced it, so it improves per world rather
    #: than transferring a fixed policy.
    #:
    #: The rule is asymmetric by split, and that asymmetry is the whole
    #: basis of the claim:
    #:
    #: - On ``train``, robustification updates the shared weights. One
    #:   policy accumulates across every training world, and what it
    #:   becomes is the checkpoint.
    #: - On ``test``, every instance starts from that same frozen
    #:   checkpoint and its adapted weights are discarded with it.
    #:   Nothing learned on one hold-out world may reach another.
    #:
    #: So the hold-out still measures what the checkpoint transfers,
    #: plus what the method can do online inside the stated episode
    #: budget -- not what it could learn by training on the hold-out.
    #: Declaring it puts the distinction in the result JSON and the
    #: report rather than leaving a reader to infer it from a
    #: suspiciously good number.
    adapts_per_instance: bool = False

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

    #: Candidate hyperparameter settings, searched by
    #: :meth:`select_hyperparameters`. Empty for methods with nothing
    #: to tune; the first entry is the declared default (see
    #: :meth:`defaults`).
    tune_grid: tuple = ()

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

    def steps_per_iteration(self) -> int | None:
        """Environment steps one training iteration consumes, or None
        for a method not counted in iterations at all.

        The one thing a step budget needs to know about a method's
        training loop. Declaring it here lets
        :meth:`apply_step_budget` enforce the budget for every method
        from one place, rather than each reimplementing the arithmetic
        and one of them getting it wrong.
        """
        return None

    @staticmethod
    def episodes_in(steps: int, horizon: int) -> int:
        """Episodes that fit in a step budget at this horizon. Rounded
        down and never zero: overrunning a budget is worse than
        underrunning it, but a run of no episodes is not a run."""
        return max(1, int(steps) // max(1, int(horizon)))

    def apply_step_budget(self, step_budget: int | None,
                          horizon: int | None = None) -> int | None:
        """Make ``step_budget`` the authority for this run.

        Every method gets the same number of environment steps, and
        "the same" has to be enforced rather than announced. Two things
        follow, and both happen here so no method can honour one and
        forget the other: the episode count for methods that train
        episode by episode, and the iteration cap for methods counted
        in iterations -- which would otherwise take whatever
        ``steps_per_iteration`` times their cap multiplies out to. A
        100k-step study left at 40 iterations of 4,000 hands PPO 160k,
        60% more than the archive methods, and every comparison drawn
        from it measures that discrepancy rather than the methods.

        Returns the derived episode count, or None without a budget or
        horizon. The cap is a ceiling: a run that asked for fewer
        iterations keeps them.
        """
        if not step_budget:
            return None
        self.config.train_steps = int(step_budget)
        per_iteration = self.steps_per_iteration()
        if per_iteration:
            affordable = max(1, int(step_budget) // int(per_iteration))
            if affordable < self.config.max_iterations:
                logger.info(
                    "[%s] step budget %d / %d per iteration caps "
                    "training at %d iterations (was %d)",
                    self.name, step_budget, per_iteration, affordable,
                    self.config.max_iterations,
                )
            self.config.max_iterations = min(self.config.max_iterations,
                                             affordable)
        if not horizon:
            return None
        episodes = self.episodes_in(step_budget, horizon)
        self.config.eval_episodes = episodes
        return episodes

    def bind_env(self, env) -> None:
        """Offer the one world a single-layout study runs in.

        Training and evaluation happen in the same world there, and
        "the same" means the same live environment: rebuilding it
        between the two resets the visit history the teleport guard is
        checked against, and an archive method arrives holding cells
        its environment has never heard of. A method that explores in
        its own loop should keep this; one that trains through Ray
        workers can ignore it, as the default does.
        """
        return None

    def bind_telemetry(self, root: str | None, stride: int = 1) -> None:
        """Offer a telemetry destination for the *training* phase.

        Evaluation is recorded by the harness, which owns that loop. A
        method exploring in a loop of its own owns the more interesting
        half: on a single-layout study almost all the exploring happens
        during training, so recording only evaluation leaves the
        coverage curve invisible exactly where it was earned.
        """
        self._telemetry = (root, stride) if root else None

    def restorable_cells(self) -> tuple:
        """Cells this method may return to, having reached them during
        training on this layout.

        Empty for a method carrying no archive. It matters only where
        training and evaluation share a world and the environment is
        rebuilt between them -- the teleport guard is per instance, so
        a fresh env refuses cells the archive knows about. These are
        cells the agent genuinely stood on; returning to what it has
        already found is the entire claim an archive method makes.
        """
        return ()

    def default_hyperparameters(self) -> dict:
        """Values to use when no tuning sweep is available.

        A single-layout study has no hold-out to tune against, so it
        needs a stated starting point rather than an implicit one. The
        first entry of :attr:`tune_grid` is that point by convention;
        override to declare something else.

        (Not ``defaults`` -- concrete baselines already use that name
        for their fixed, un-searched settings.)
        """
        return dict(self.tune_grid[0]) if self.tune_grid else {}

    def single_layout_train_test_run(self, row: dict, **kwargs):
        """Learn on one layout for a step budget, then evaluate frozen.

        The benchmark's question is whether a policy *transfers*; this
        one is how much of a single world a method uncovers given a
        long budget in it. Both matter, and they are not the same
        question -- Go-Explore is a single-game algorithm that the
        transfer protocol flatters least.

        The default implementation runs the protocol's own pieces
        (:meth:`fit`, then a frozen evaluation) and suits every method
        that learns one policy. Override when the method's shape
        differs -- Go-Explore's phase 1 and phase 2 do.
        """
        from topogym.baselines.gridworld2dv1.single_layout import (
            run_single_layout,
        )

        return run_single_layout(self, row, **kwargs)

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
                    "env_options": self.env_options(),
                    "adapts_per_instance": self.adapts_per_instance},
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
