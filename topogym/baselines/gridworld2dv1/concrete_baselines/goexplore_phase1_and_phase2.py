"""Go-Explore, both phases: explore and archive, then robustify.

Phase 1 is :mod:`~...concrete_baselines.goexplore_phase1` with one
addition -- the archive stores the *path* to each cell, not just the
cell -- because phase 2 has nothing to robustify without it.

Phase 2 is the **Backward Algorithm** of Salimans and Chen, which
Ecoffet et al. adopt for robustification:

    "It works by starting the agent near the last state in the
    trajectory, and then running an ordinary RL algorithm from there
    (in this case Proximal Policy Optimization). Once the agent is able
    to obtain the same or a better reward than the example trajectory
    in a certain fraction of the rollouts, the algorithm backs the
    agent's starting point up to a slightly earlier place along the
    trajectory, and repeats the process until eventually the agent has
    learned to obtain a score greater than or equal to the example
    trajectory all the way from the initial state."

References:
    A. Ecoffet, J. Huizinga, J. Lehman, K. O. Stanley, J. Clune.
    "Go-Explore: a New Approach for Hard-Exploration Problems."
    arXiv:1901.10995.
    T. Salimans and R. Chen. "Learning Montezuma's Revenge from a
    Single Demonstration." arXiv:1812.03381.

The demonstration reaches the agent through the environment's
``demonstration`` argument, which makes its cells legal reset targets
without marking them visited -- so the restart is declared rather than
smuggled in, and it never inflates coverage.

Fairness
--------
Phase 2 improves a policy *in the world that produced the trajectory*,
which is a different claim from "this policy transfers", so the rule is
asymmetric by split and the asymmetry is the whole basis of the claim:

- On ``train``, robustification updates the shared weights. One policy
  accumulates across every training world, and what it becomes is the
  checkpoint.
- On ``test``, every instance starts from that same frozen checkpoint
  and its adapted weights are discarded with it. Nothing learned on one
  hold-out world may reach another.

``adapts_per_instance = True`` records this in the result JSON, and the
report marks the row with a dagger, so a reader is never left to infer
it from a suspiciously good number.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from topogym.baselines.gridworld2dv1.archive import (
    DEFAULTS,
    LayoutArchive,
)
from topogym.baselines.gridworld2dv1.concrete_baselines.goexplore_phase1 import (
    GoExplorePhase1Baseline,
)
from topogym.baselines.gridworld2dv1.concrete_baselines.ppo import PPOBaseline
from topogym.baselines.gridworld2dv1.protocol import TrainingReport

logger = logging.getLogger("topogym")

__all__ = ["TrajectoryArchive", "GoExplorePhase12Baseline"]

#: Fraction of the stage's rollouts that must reach the goal before the
#: start point moves further back. The paper's "a certain fraction".
SUCCESS_THRESHOLD = 0.6

#: Cells the start point retreats by when a stage passes at exactly the
#: threshold. One cell at a time would be faithful and unaffordable; a
#: stride keeps the number of stages proportional to the trajectory
#: rather than equal to it. The actual retreat scales with how easily
#: the stage passed -- Salimans and Chen's success counter "used to
#: decrease the central starting point at the right speed".
BACKUP_STRIDE = 8

#: The window ``{tau - D, ..., tau}`` each rollout samples its local
#: start from. Sampling rather than fixing the restart cell is the
#: diversity term of Algorithm 1; without it a stage learns one
#: position rather than a stretch of the trajectory.
LOCAL_START_WINDOW = 4

#: Salimans and Chen also prime the policy by replaying K demonstration
#: actions before each rollout, with those steps masked out of the
#: gradient. That exists to initialise a *recurrent* policy's hidden
#: state so the agent is not dropped mid-trajectory with no memory of
#: how it got there. Every baseline here is feedforward over a Markov
#: observation, so there is no hidden state to prime and the mechanism
#: is a no-op by construction -- not an omission, but it stops being
#: one the moment a recurrent module appears, which is why it is
#: written down rather than left silent.
DEMONSTRATION_PRIMING_STEPS = 0


class TrajectoryArchive(LayoutArchive):
    """A phase-1 archive that also remembers how it got to each cell.

    Phase 2 restarts the agent partway along a trajectory, so an
    archive of cells alone -- which is all phase 1 needs -- leaves it
    with nothing to restart *along*. Each entry keeps the shortest path
    seen to that cell, since a shorter demonstration is a shorter
    curriculum.
    """

    def new_entry(self, cell: tuple) -> dict:
        entry = super().new_entry(cell)
        entry["trajectory"] = ()
        return entry

    def observe(self, visited, chosen_from=None, trajectory=(),
                reached_goal: bool = False, **kwargs):
        """Fold in a finished episode, recording the path it took.

        ``trajectory`` is the ordered cell path of the episode. The
        stored path for a cell is the whole route from the layout's
        start, so a prefix of it is always a legal curriculum: cells
        reached after an archive reset inherit the path to the cell the
        episode resumed at.
        """
        count = super().observe(visited, chosen_from, **kwargs)
        if not trajectory:
            return count
        prefix = ()
        if chosen_from is not None:
            entry = self.cells.get(tuple(chosen_from))
            prefix = tuple(entry.get("trajectory") or ()) if entry else ()
            # The resumed episode starts *at* the cell the prefix ends
            # on -- restoring to a point of a trajectory is being at
            # that point, not stepping to it again -- so the join must
            # not double the cell.
            if (prefix and trajectory
                    and prefix[-1] == tuple(trajectory[0])):
                prefix = prefix[:-1]
        seen: set = set()
        for index, cell in enumerate(trajectory):
            cell = tuple(cell)
            if cell in seen:
                continue
            seen.add(cell)
            entry = self.cells.get(cell)
            if entry is None:
                continue
            route = prefix + tuple(map(tuple, trajectory[:index + 1]))
            current = entry.get("trajectory") or ()
            if not current or len(route) < len(current):
                entry["trajectory"] = route
        if reached_goal and trajectory:
            # The episode ended on the goal, so the last cell of the
            # path is it. Marking the cell rather than storing the
            # route separately keeps one source of truth: the shortest
            # route to this cell *is* the shortest demonstration.
            goal = self.cells.get(tuple(trajectory[-1]))
            if goal is not None:
                goal["reached_goal"] = True
        return count

    def best_goal_trajectory(self) -> tuple:
        """The shortest recorded route to a goal cell, or ``()``.

        Phase 2 needs one demonstration, not the archive: the whole
        point is to turn a single lucky trajectory into a reliable
        policy.
        """
        routes = [tuple(entry["trajectory"])
                  for entry in self.cells.values()
                  if entry.get("reached_goal") and entry.get("trajectory")]
        return min(routes, key=len) if routes else ()


class _Session:
    """Per-instance state shared by the policy and the reset probe.

    The harness builds those two from separate factories, but phase 1
    needs them to agree: only the policy sees the agent step by step
    (so only it can record a trajectory), and only the reset probe is
    asked where to resume. They meet here, keyed by the seed both
    factories are handed, which is derived from the instance -- so a
    session belongs to exactly one (instance, worker).
    """

    def __init__(self, params: dict, seed: int):
        self.params = dict(params)
        self.seed = seed
        self.archive: TrajectoryArchive | None = None
        self.trajectory: list = []
        self.chosen_from = None
        self._layout = None
        self._rng = np.random.default_rng(seed)

    def _ensure(self, env) -> None:
        layout = getattr(env, "layout", None)
        if layout is not self._layout:
            # An archive of another world's cells is meaningless, and a
            # trajectory through it is worse than meaningless.
            self.archive = TrajectoryArchive(
                self.params, self.seed, neighbors=layout.base.neighbors)
            self._layout = layout
            self.chosen_from = None
            self.trajectory = []

    # -- the policy side ----------------------------------------------

    def act(self, _observation, env):
        self._ensure(env)
        cell = env._state.cell
        if not self.trajectory or self.trajectory[-1] != cell:
            self.trajectory.append(cell)
        return int(self._rng.integers(env.action_space.n))

    # -- the episode-boundary side ------------------------------------

    def choose_reset(self, env, info: dict):
        self._ensure(env)
        self.archive.observe(env._visited, self.chosen_from,
                             trajectory=tuple(self.trajectory),
                             reached_goal=bool(info.get("goal_reached")))
        self.trajectory = []
        self.chosen_from = self.archive.select()
        return self.chosen_from


#: seed -> session, so the two factories below hand back the same
#: object inside one worker process.
_SESSIONS: dict = {}


def _session(params: dict, seed: int) -> _Session:
    existing = _SESSIONS.get(seed)
    if existing is None or existing.params != dict(params):
        existing = _Session(params, seed)
        _SESSIONS[seed] = existing
    return existing


class TrajectoryPolicyFactory:
    """Builds the phase-1 explorer, sharing state with the probe."""

    def __init__(self, params: dict, seed: int = 0):
        self.params = dict(params)
        self.seed = seed

    def __call__(self, seed: int | None = None):
        return _session(self.params,
                        self.seed if seed is None else seed).act


class TrajectoryResetFactory:
    """Builds the phase-1 probe, sharing state with the explorer."""

    def __init__(self, params: dict, seed: int = 0):
        self.params = dict(params)
        self.seed = seed

    def __call__(self, seed: int | None = None):
        return _session(self.params,
                        self.seed if seed is None else seed).choose_reset


class GoExplorePhase12Baseline(PPOBaseline):
    """Explore and archive, then robustify the best trajectory.

    Inherits PPO wholesale -- phase 2 *is* "an ordinary RL algorithm",
    and the paper says so -- and adds the phase-1 exploration that
    produces the demonstration, plus the backward curriculum that
    consumes it.
    """

    name = "go-explore-phase1and2"

    #: Phase 2 improves the policy in the world that produced the
    #: trajectory. See the module docstring: on ``train`` those updates
    #: are shared, on the hold-out they are discarded per instance.
    adapts_per_instance = True

    #: Like phase 1, the selection strategy is fitted against every
    #: non-hold-out split as one pool.
    tuning_splits = ("tune", "train", "val")

    def __init__(self, config=None):
        super().__init__(config)
        self._archive_params = dict(DEFAULTS)
        self._demonstration: tuple = ()

    # -- phase 1 ------------------------------------------------------

    def explore(self, rows: list, episodes: int, seed: int = 0) -> tuple:
        """Run phase 1 and return ``(records, demonstration)``.

        The demonstration is the shortest goal-reaching route any
        instance produced. There may be none -- these worlds are hard,
        and that is the finding rather than a failure -- in which case
        phase 2 has nothing to robustify and says so.
        """
        from topogym.baselines.gridworld2dv1.evaluate import evaluate_split

        _SESSIONS.clear()  # a fresh archive per phase-1 run
        records = evaluate_split(
            rows, None, episodes=episodes, seed=seed, trace=False,
            policy_factory=TrajectoryPolicyFactory(
                self._archive_params, seed),
            choose_reset_factory=TrajectoryResetFactory(
                self._archive_params, seed),
            workers=1,  # the session is per-process; keep it in ours
            env_options=self.env_options(),
        )
        demonstrations = [
            session.archive.best_goal_trajectory()
            for session in _SESSIONS.values() if session.archive
        ]
        demonstrations = [d for d in demonstrations if d]
        best = min(demonstrations, key=len) if demonstrations else ()
        logger.info(
            "[%s] phase 1: %d instances, %d goal trajectories, "
            "best route %d cells",
            self.name, len(records), len(demonstrations), len(best),
        )
        return records, best

    # -- phase 2 ------------------------------------------------------

    def backward_stages(self, demonstration: tuple) -> list:
        """Restart points, from near the goal back to the start.

        The paper's curriculum: begin next to the last state, and back
        up only once the agent can finish from where it stands.
        """
        if not demonstration:
            return []
        indices = list(range(len(demonstration) - 1, -1, -BACKUP_STRIDE))
        if indices and indices[-1] != 0:
            indices.append(0)  # always finish from the true start
        return [demonstration[i] for i in indices]

    def local_starts(self, demonstration: tuple, index: int) -> list:
        """The window ``{tau - D, ..., tau}`` a stage samples from.

        Every cell in it is part of the demonstration, so every one is
        a legal reset target; the window only ever reaches backward, so
        it never leaks a position the curriculum has not got to yet.
        """
        low = max(0, index - LOCAL_START_WINDOW)
        return list(demonstration[low:index + 1])

    def robustify(self, rows: list, demonstration: tuple, values: dict,
                  iterations: int, seed: int = 0) -> dict:
        """Train PPO backward along ``demonstration``.

        One stage per central restart point ``tau``, each sampling its
        local start from :meth:`local_starts`. A stage ends when the
        agent reaches the goal in :data:`SUCCESS_THRESHOLD` of its
        rollouts -- the paper's "same or better reward than the example
        trajectory", which under a sparse goal is simply reaching it --
        and ``tau`` then retreats at a speed set by how easily it
        passed. A stage that exhausts its iteration budget stops the
        curriculum and says so, rather than backing up to a position
        the agent has not earned.
        """
        from topogym.baselines.gridworld2dv1.concrete_baselines.ppo import (
            mean_return,
        )

        if not demonstration:
            return {"stages": [], "reached_start": False,
                    "why": "no goal trajectory to robustify"}

        planned = max(1, len(self.backward_stages(demonstration)))
        per_stage = max(1, iterations // planned)
        log: list = []
        tau = len(demonstration) - 1
        reached_start = False
        stage = 0
        while tau >= 0:
            window = self.local_starts(demonstration, tau)
            config = self.algorithm_config(rows, values, seed + stage)
            config.env_config["start_cells"] = [tuple(c) for c in window]
            config.env_config["demonstration"] = tuple(demonstration)
            algorithm = config.build_algo()
            success, iteration = 0.0, 0
            try:
                for iteration in range(1, per_stage + 1):
                    success = mean_return(algorithm.train())
                    if success >= SUCCESS_THRESHOLD:
                        break
                self._algorithm = algorithm
                self._checkpoint_path = self._checkpoint()
            finally:
                if algorithm is not self._algorithm:
                    algorithm.stop()
            passed = success >= SUCCESS_THRESHOLD
            log.append({
                "stage": stage, "tau": tau,
                "start_window": [list(c) for c in window],
                "cells_from_goal": len(demonstration) - 1 - tau,
                "iterations": iteration, "success_rate": success,
                "passed": passed,
            })
            logger.info(
                "[%s] phase 2 stage %d at tau=%d (%d from the goal), "
                "window of %d: %.2f after %d iterations (%s)",
                self.name, stage + 1, tau, len(demonstration) - 1 - tau,
                len(window), success, iteration,
                "passed" if passed else "budget exhausted",
            )
            if not passed:
                # Backing up to a position the agent has not earned
                # would report a curriculum it never completed.
                break
            if tau == 0:
                reached_start = True
                break
            # "Decrease the central starting point at the right speed":
            # an easy pass earns a longer retreat, a bare one a shorter.
            retreat = max(1, round(BACKUP_STRIDE * success
                                   / max(SUCCESS_THRESHOLD, 1e-9)))
            tau = max(0, tau - min(retreat, 4 * BACKUP_STRIDE))
            stage += 1
        return {"stages": log, "reached_start": reached_start,
                "why": ("robustified all the way to the start"
                        if reached_start else
                        "curriculum stopped before the layout's start")}

    # -- the protocol -------------------------------------------------

    def select_hyperparameters(self, tuning: dict):
        """Phase 1's archive sweep, then PPO's own defaults.

        Tuning both halves jointly would multiply two grids for a
        method whose interesting behaviour is the handoff between them.
        The archive strategy is what phase 1 is, so that is what gets
        searched.
        """
        phase1 = GoExplorePhase1Baseline(self.config)
        chosen = phase1.select_hyperparameters(tuning)
        self._archive_params = {**DEFAULTS, **(chosen.values or {})}
        chosen.values = {**self._archive_params, **self.defaults}
        return chosen

    def fit(self, train_rows: list, val_rows: list,
            hyperparameters) -> TrainingReport:
        """Phase 1 on ``train``, then robustify -- weights shared.

        This is the ``train`` half of the asymmetry in the module
        docstring: one policy accumulates across every training world,
        and what it becomes is the checkpoint the hold-out starts from.
        """
        values = dict(hyperparameters.values or {})
        self._archive_params = {**DEFAULTS,
                                **{k: v for k, v in values.items()
                                   if k in DEFAULTS}}
        episodes = (self.config.train_episodes_per_instance
                    or self.config.eval_episodes)
        _, demonstration = self.explore(train_rows, episodes,
                                        self.config.seed)
        self._demonstration = demonstration
        if not demonstration:
            return TrainingReport(
                iterations=0, stopped_early=True,
                stopped_because="phase 1 found no goal trajectory on "
                                "train, so phase 2 has nothing to "
                                "robustify",
            )
        outcome = self.robustify(train_rows, demonstration, values,
                                 self.config.max_iterations,
                                 self.config.seed)
        return TrainingReport(
            iterations=sum(s["iterations"] for s in outcome["stages"]),
            stopped_early=not outcome["reached_start"],
            stopped_because=outcome["why"],
            best_checkpoint=self._checkpoint(),
        )

    def single_layout_train_test_run(self, row: dict, *,
                                     step_budget: int = 1_000_000,
                                     eval_episodes: int = 100,
                                     telemetry_root: str | None = None,
                                     step_stride: int = 1,
                                     hyperparameters: dict | None = None,
                                     **kwargs):
        """Explore until the goal is found, then robustify, then freeze.

        Budget split follows the paper's own shape rather than a fixed
        ratio: phase 1 runs until it has a goal-reaching trajectory,
        because that is its exit condition, and phase 2 gets what is
        left. A cap keeps a world that never yields a trajectory from
        consuming the whole budget in phase 1 -- on those worlds the
        honest result is "phase 1 never found it", and spending the
        second half proving it again adds nothing.
        """
        import time

        from topogym.baselines.gridworld2dv1.evaluate import evaluate_split
        from topogym.baselines.gridworld2dv1.protocol import Hyperparameters
        from topogym.baselines.gridworld2dv1.single_layout import (
            SingleLayoutResult,
            episodes_for,
        )

        started = time.time()
        horizon = int(row["horizon"])
        total_episodes = episodes_for(step_budget, horizon)
        values = dict(hyperparameters if hyperparameters is not None
                      else self.default_hyperparameters())
        self._archive_params = {**DEFAULTS,
                                **{k: v for k, v in values.items()
                                   if k in DEFAULTS}}

        # Phase 1, in chunks, stopping the moment a route exists.
        cap = total_episodes // 2
        chunk = max(1, cap // 10)
        demonstration, spent = (), 0
        while spent < cap and not demonstration:
            _, demonstration = self.explore(
                [row], min(chunk, cap - spent), self.config.seed + spent)
            spent += chunk
        logger.info(
            "[%s] phase 1 spent %d of %d episodes; %s",
            self.name, min(spent, cap), total_episodes,
            f"route of {len(demonstration)} cells"
            if demonstration else "no route found",
        )
        self._demonstration = demonstration

        remaining = max(0, total_episodes - min(spent, cap))
        outcome = self.robustify(
            [row], demonstration, values,
            max(1, self.config.max_iterations), self.config.seed,
        ) if demonstration else {
            "stages": [], "reached_start": False,
            "why": "phase 1 never reached the goal within its half of "
                   "the budget",
        }

        self.config.eval_episodes = eval_episodes
        records = evaluate_split(
            [row], self.policy(), episodes=eval_episodes, seed=0,
            trace=True, choose_reset=self.choose_reset,
            env_options=self.env_options(),
            telemetry_root=telemetry_root, algorithm=self.name,
            step_stride=step_stride, split="single-eval",
        )
        return SingleLayoutResult(
            algorithm=self.name, layout=row["unit"],
            env_id=row["template_id"], seed=int(row["seed"]),
            horizon=horizon,
            optimal_actions=(int(row["optimal_actions"])
                             if row["optimal_actions"] else None),
            step_budget=step_budget,
            train_episodes=total_episodes,
            eval_episodes=eval_episodes,
            evaluation=records[0] if records else {},
            training={
                "phase1_episodes": min(spent, cap),
                "phase2_episodes_available": remaining,
                "demonstration_cells": len(demonstration),
                **outcome,
            },
            hyperparameters=Hyperparameters(
                values=values, tuning_score=None, searched=[]).to_dict(),
            wall_seconds=time.time() - started,
            config={**self.config.to_dict(),
                    "env_options": self.env_options(),
                    "adapts_per_instance": self.adapts_per_instance},
        )

    def policy(self) -> Callable:
        """The robustified policy, or phase 1's explorer if there is
        none -- never a silently untrained network."""
        if getattr(self, "_algorithm", None) is not None:
            return super().policy()
        logger.info("[%s] no robustified policy; evaluating with the "
                    "phase-1 explorer", self.name)
        return _session(self._archive_params, self.config.seed).act

    def choose_reset(self, env, info: dict):
        return _session(self._archive_params,
                        self.config.seed).choose_reset(env, info)

    def choose_reset_factory(self):
        return TrajectoryResetFactory(self._archive_params,
                                      self.config.seed)
