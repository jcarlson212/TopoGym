"""Evaluate a policy on hold-out instances.

The hold-out is touched once, after training has stopped. For each
instance we run a fixed number of episodes, record the standard
metric set, and trace the discovery curves the published figures are
drawn from -- unique states, chambers entered, and how much of the
negatively curved structure (doorways, corridors, bottlenecks) the
agent has actually reached, all against step count.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable

import numpy as np

from topogym.baselines.instances import instance_key, make_instance
from topogym.stats import StatsRecorder

logger = logging.getLogger("topogym")

#: Curves traced during evaluation, sampled every ``CURVE_STRIDE``
#: interactions. They are *cumulative across the whole evaluation
#: budget*, not per-episode: with a 50-episode budget the question is
#: how much of a world an explorer uncovers given that budget, which a
#: per-episode average would flatten into "how much does one episode
#: see".
CURVE_METRICS = ("unique_states", "chambers_entered", "curvature_reached")
CURVE_STRIDE = 25


#: Worlds larger than this skip the curvature trace: the field is
#: exact and cached, but it is quadratic-ish in the free space and a
#: 400-grid would dominate the evaluation budget.
MAX_CELLS_FOR_CURVATURE = 60_000


def _negative_curvature_cells(core) -> set:
    """Cells whose Ollivier-Ricci curvature is negative -- the
    doorways, corridors, and bottlenecks of the world.

    Must be called *after* the first reset: the layout does not exist
    before one.
    """
    if len(core.layout.free_cells) > MAX_CELLS_FOR_CURVATURE:
        logger.debug("curvature trace skipped: %d free cells",
                     len(core.layout.free_cells))
        return set()
    return {c for c, k in core.ollivier_ricci().items() if k < 0}


def evaluate_instance(row: dict, policy: Callable, episodes: int = 5,
                      trace: bool = True, seed: int = 0,
                      track_topology: bool = False,
                      choose_reset: Callable | None = None) -> dict:
    """Run ``policy`` on one hold-out instance.

    ``policy(observation, env) -> action``. Returns the instance's
    record: the headline numbers at the top level and the *complete*
    native metric set under ``"metrics"``, so anything TopoGym tracks
    is available for later analysis even when no figure plots it.

    ``track_topology`` additionally enables the per-step homology and
    curvature trackers, which timestamp hole discoveries
    (``steps_to_h0_holes`` / ``steps_to_h1_holes``). They run GUDHI on
    every step, so they are off by default and belong to focused
    studies rather than a full sweep.

    Evaluation is single-process by construction: Ray parallelises
    training rollouts, but every reported number is produced here, in
    the driver, from one ``StatsRecorder``. There is nothing to
    aggregate across workers.
    """
    env = StatsRecorder(make_instance(row),
                        track_holes=track_topology,
                        track_curvature=track_topology)
    core = env.unwrapped
    curves = {name: [] for name in CURVE_METRICS}
    steps_to_goal = []
    negative: set = set()

    interactions, chambers_seen, archive_resets = 0, set(), 0
    info: dict = {}
    for episode in range(episodes):
        # The episode-boundary probe: every method gets the same offer,
        # and the same contiguous budget on this one world, whatever it
        # chose to do during training.
        target = choose_reset(core, info) if choose_reset and episode \
            else None
        options = None
        if target is not None:
            options = {"teleport": tuple(int(v) for v in target)}
            archive_resets += 1
        obs, info = env.reset(seed=seed + episode, options=options)
        if trace and episode == 0:  # the layout exists only after reset
            negative = _negative_curvature_cells(core)
        reached, step = None, 0
        while True:
            obs, reward, terminated, truncated, info = env.step(
                policy(obs, core)
            )
            step += 1
            interactions += 1
            if reached is None and info.get("goal_reached"):
                reached = step
            if trace and interactions % CURVE_STRIDE == 0:
                lifetime = core._ever_visited | core._visited
                chambers_seen |= set(core.chamber_entry_steps)
                curves["unique_states"].append(
                    (interactions, len(lifetime)))
                curves["chambers_entered"].append(
                    (interactions, len(chambers_seen)))
                curves["curvature_reached"].append((
                    interactions,
                    len(lifetime & negative) / len(negative)
                    if negative else 0.0,
                ))
            if terminated or truncated:
                break
        chambers_seen |= set(core.chamber_entry_steps)
        steps_to_goal.append(reached)

    import dataclasses

    metrics = env.metrics()
    solved = [s for s in steps_to_goal if s is not None]
    record = {
        "instance": instance_key(row),
        "unit": row["unit"],
        "slice": row["slice"],
        "family": row["family"],
        "size": int(row["size"]),
        "seed": int(row["seed"]),
        "optimal_actions": (int(row["optimal_actions"])
                            if row["optimal_actions"] else None),
        "horizon": int(row["horizon"]),
        "episodes": episodes,
        "interactions": interactions,
        "archive_resets": archive_resets,
        "success_rate": len(solved) / episodes,
        "chambers_entered": len(chambers_seen),
        "median_steps_to_goal": (float(np.median(solved))
                                 if solved else None),
        "mean_episode_coverage": metrics.mean_episode_coverage,
        "lifetime_coverage": metrics.state_coverage,
        "unique_states": metrics.unique_states,
        "visitation_entropy": metrics.visitation_entropy_normalized,
        "interactions_to_first_success":
            metrics.interactions_to_first_success,
        "mean_regret": metrics.mean_regret,
        "planning_efficiency": metrics.planning_efficiency,
        "curvature_coverage_below_zero": (
            len((core._ever_visited | core._visited) & negative)
            / len(negative) if negative else None
        ),
        # The complete native metric set, kept whether or not a figure
        # uses it: coverage milestones, entropy, regret, planning
        # efficiency, and -- with track_topology -- the steps at which
        # each hole was discovered.
        "metrics": dataclasses.asdict(metrics),
        "steps_to_goal": steps_to_goal,
    }
    if trace:
        record["curves"] = {
            name: _mean_curve(points) for name, points in curves.items()
        }
    env.close()
    return record


def _mean_curve(points: list) -> list:
    """Average repeated (step, value) samples into one curve."""
    grouped: dict = {}
    for step, value in points:
        grouped.setdefault(step, []).append(value)
    return [[step, float(np.mean(grouped[step]))]
            for step in sorted(grouped)]


def instance_seed(base_seed: int, row: dict) -> int:
    """A seed determined by the instance, not by evaluation order.

    A stateful policy built once and reused across a sweep carries its
    random stream from one instance to the next, so results depend on
    how the work was sharded -- serial and parallel runs disagree, and
    neither is reproducible. Deriving each instance's seed from its own
    canonical configuration makes an evaluation a pure function of
    ``(row, base seed)``, which is the property the benchmark's
    determinism guarantee needs.
    """
    digest = hashlib.sha256(
        f"{base_seed}:{row.get('canonical_config', row.get('unit'))}"
        .encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


class InstanceTask:
    """One instance's evaluation, as a picklable unit of work.

    Holds the *factory* for a policy rather than a policy: a live
    policy may wrap a torch module, and shipping one per task would
    cost more than the evaluation. Module-level with plain attributes,
    because a process pool has to pickle it.
    """

    def __init__(self, policy_factory: Callable, episodes: int,
                 seed: int, track_topology: bool = False,
                 choose_reset_factory: Callable | None = None):
        self.policy_factory = policy_factory
        self.episodes = episodes
        self.seed = seed
        self.track_topology = track_topology
        self.choose_reset_factory = choose_reset_factory

    def __call__(self, row: dict) -> dict:
        # Rebuilt per instance, seeded from the instance: see
        # instance_seed. Cheap for a policy, and the only way the
        # result does not depend on which worker took the row.
        derived = instance_seed(self.seed, row)
        policy = _build(self.policy_factory, derived)
        choose_reset = (_build(self.choose_reset_factory, derived)
                        if self.choose_reset_factory else None)
        return evaluate_instance(
            row, policy, episodes=self.episodes, seed=self.seed,
            track_topology=self.track_topology,
            choose_reset=choose_reset,
        )


def _build(factory: Callable, seed: int):
    """Call a factory, passing the seed if it accepts one."""
    try:
        return factory(seed)
    except TypeError:
        return factory()


def evaluate_split(rows: list, policy: Callable, episodes: int = 5,
                   seed: int = 0, track_topology: bool = False,
                   choose_reset: Callable | None = None,
                   choose_reset_factory: Callable | None = None,
                   policy_factory: Callable | None = None,
                   workers: int = 1) -> list:
    """Evaluate every hold-out instance, in manifest order.

    Instances are independent and separately seeded, so the loop
    parallelises exactly. Doing so needs ``policy_factory`` -- a
    picklable, zero-argument callable that builds the policy inside
    each worker; without one the run stays serial, which is correct
    for a policy that cannot cross a process boundary.
    """
    if policy_factory is not None:
        # Same code path serial or parallel, so the two cannot drift.
        from topogym.baselines.parallel import map_instances

        task = InstanceTask(policy_factory, episodes, seed,
                            track_topology, choose_reset_factory)
        return map_instances(task, rows, workers=workers)

    records = []
    for i, row in enumerate(rows):
        records.append(evaluate_instance(
            row, policy, episodes=episodes, seed=seed,
            track_topology=track_topology, choose_reset=choose_reset,
        ))
        if (i + 1) % 25 == 0:
            logger.info("evaluated %d/%d instances", i + 1, len(rows))
    return records


def random_policy(rng: np.random.Generator) -> Callable:
    """The floor every learned baseline has to clear."""

    def act(_obs, env):
        return int(rng.integers(env.action_space.n))

    return act
