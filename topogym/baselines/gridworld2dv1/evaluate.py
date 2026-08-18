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
import re
from collections.abc import Callable

import numpy as np

from topogym.baselines.gridworld2dv1 import telemetry
from topogym.baselines.gridworld2dv1.instances import instance_key, make_instance
from topogym.baselines.utilities import SplitBudget
from topogym.stats import StatsRecorder

logger = logging.getLogger("topogym")

#: Curves traced during evaluation, sampled every ``CURVE_STRIDE``
#: interactions. Two properties matter.
#:
#: They are *cumulative across the whole evaluation budget*, not
#: per-episode: with a 50-episode budget the question is how much of a
#: world an explorer uncovers given that budget, which a per-episode
#: average would flatten into "how much does one episode see".
#:
#: They are *fractions*, not counts. Hold-out instances range from
#: ~1,000 to ~160,000 free cells, so averaging raw counts across them
#: would report little more than which worlds are large. Every curve
#: is in [0, 1] and therefore comparable across sizes and averageable
#: across instances.
CURVE_METRICS = ("coverage", "chambers_entered", "curvature_reached",
                 "cumulative_return")
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
                      choose_reset: Callable | None = None,
                      env_options: dict | None = None,
                      telemetry=None, step_stride: int = 1,
                      split: str | None = None, env=None,
                      step_budget: int | None = None) -> dict:
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

    ``telemetry`` is a :mod:`~topogym.baselines.gridworld2dv1.telemetry`
    writer. Given one, the run also emits a row per environment step
    (every ``step_stride``th) and a row per episode, which is what
    makes questions nobody asked in advance answerable afterwards
    without re-running. Off by default: the summary record below is
    what the published result needs.

    Evaluation is single-process by construction: Ray parallelises
    training rollouts, but every reported number is produced here, in
    the driver, from one ``StatsRecorder``. There is nothing to
    aggregate across workers.
    """
    # ``env`` lets a caller supply a world that is already alive. A
    # single-layout study trains and evaluates in the *same* world, so
    # rebuilding it between the two resets the visit history the
    # teleport guard is checked against, and an archive method arrives
    # holding cells its environment has never heard of. Ownership
    # follows creation: a supplied env is not closed here.
    # A step budget is per *layout*: horizons across the registry span
    # 130 to 6,760, so a flat episode count hands one world fifty times
    # the experience of another and the comparison measures the horizon
    # rather than the method. Episodes are derived here, per instance.
    if step_budget:
        episodes = (SplitBudget(steps=int(step_budget))
                    .resolve(int(row["horizon"])).episodes)
    borrowed = env is not None
    if not borrowed:
        env = StatsRecorder(make_instance(row, **(env_options or {})),
                            track_holes=track_topology,
                            track_curvature=track_topology)
    core = env.unwrapped
    curves = {name: [] for name in CURVE_METRICS}
    steps_to_goal = []
    negative: set = set()

    # A borrowed env carries its lifetime step count into the x-axis;
    # a fresh one starts at zero. A chunked training loop evaluates
    # the same live world across many calls, and a counter restarting
    # each call filed every chunk's curves on top of one another at
    # the origin -- a curve that began at two thousand discovered
    # states was the second half of a run wearing the first half's
    # x-positions.
    start = (sum(getattr(core, "lifetime_visit_counts", {}).values())
             if borrowed else 0)
    interactions, chambers_seen, archive_resets = start, set(), 0
    total_return = 0.0
    keys = {
        # The caller's label wins: a row knows which manifest split it
        # came from, but only the caller knows which *phase* of a run
        # this is, and the three tables have to agree on one answer or
        # they cannot be joined.
        "split": split or row.get("split") or "test",
        "instance": instance_key(row), "family": row["family"],
        "size": int(row["size"]), "seed": int(row["seed"]),
    }
    step_rows: list = []
    episode_rows: list = []
    n_free = max(1, len(core.layout.free_cells) if core.layout else 1)
    n_chambers = 0
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
        if episode == 0:  # the layout exists only after reset
            # Not gated on ``trace``: these are the denominators every
            # fraction is divided by, and leaving them at 1 does not
            # disable those fields, it silently turns them into raw
            # counts. Only the curvature set is expensive enough to gate.
            n_free = max(1, len(core.layout.free_cells))
            n_chambers = sum(1 for f in core.layout.features
                             if f.kind == "chamber")
            if trace:
                negative = _negative_curvature_cells(core)
        reached, step = None, 0
        episode_return = 0.0
        while True:
            action = policy(obs, core)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            interactions += 1
            total_return += float(reward)
            episode_return += float(reward)
            if telemetry is not None and step % step_stride == 0:
                cell = info["position"]
                # Everything here is already computed by the step: the
                # per-step table must not cost more than the stepping
                # it records, or it changes what it measures.
                step_rows.append({
                    "episode": episode, "step": step,
                    "interaction": interactions,
                    "action": int(action),
                    "x": int(cell[0]), "y": int(cell[1]),
                    "facing": str(core._state.frame),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "new_cell": core.visit_counts.get(cell, 0) <= 1,
                    "visit_count": core.visit_counts.get(cell, 0),
                    "unique_states": len(core._ever_visited
                                         | core._visited),
                    "h0_components": info["known_components"],
                })
            if reached is None and info.get("goal_reached"):
                reached = step
            if trace and interactions % CURVE_STRIDE == 0:
                lifetime = core._ever_visited | core._visited
                chambers_seen |= set(core.chamber_entry_steps)
                curves["coverage"].append(
                    (interactions, len(lifetime) / n_free))
                curves["chambers_entered"].append((
                    interactions,
                    len(chambers_seen) / n_chambers if n_chambers else 0.0,
                ))
                curves["curvature_reached"].append((
                    interactions,
                    len(lifetime & negative) / len(negative)
                    if negative else 0.0,
                ))
                # Unnormalised on purpose: reward does not scale with
                # world size, so the raw total is already comparable --
                # under the sparse default it counts goals reached.
                curves["cumulative_return"].append(
                    (interactions, total_return))
            if terminated or truncated:
                break
        # Lifetime, not per-phase. The environment clears
        # ``chamber_entry_steps`` each episode, and this loop may be the
        # *evaluation* half of a world already explored -- so counting
        # only what it saw reports 1 chamber beside a coverage figure
        # that includes the 6 found earlier. Two numbers on different
        # denominators in one record is worse than either being wrong.
        chambers_seen |= {
            index for cell, index in core._chamber_of.items()
            if cell in core.lifetime_visit_counts
        }
        steps_to_goal.append(reached)
        if telemetry is not None:
            lifetime = core._ever_visited | core._visited
            # One homology call per episode: what the agent has
            # actually discovered by now, certified the same way the
            # environment certifies itself.
            observed = core.homology_stats("observed")
            episode_rows.append({
                "episode": episode, "length": step,
                "interactions": interactions,
                "episode_return": episode_return,
                "steps_to_goal": reached,
                "reached_goal": reached is not None,
                "episode_coverage": len(core._visited) / n_free,
                "lifetime_coverage": len(lifetime) / n_free,
                "unique_states": len(lifetime),
                "visit_entropy": info.get("visitation_entropy"),
                "chambers_entered": len(chambers_seen),
                "chambers_total": n_chambers,
                "decoys_entered": _decoys_found(core, lifetime),
                "decoys_total": sum(1 for f in core.layout.features
                                    if f.kind == "decoy"),
                "observed_h0": observed.h0,
                "observed_h1": observed.h1,
                "observed_frac": info.get("observed_frac"),
                "archive_reset": target is not None,
                "reset_cell": str(target) if target else None,
            })
            telemetry.add_steps(step_rows, **keys)
            telemetry.add_episodes(episode_rows, **keys)
            step_rows, episode_rows = [], []

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
        # What *this call* spent, not the borrowed env's lifetime: the
        # offset above places curves on a shared axis, while a record
        # still answers "how many steps did this evaluation take".
        "interactions": interactions - start,
        "archive_resets": archive_resets,
        # Summed over the instance's whole episode budget. Aggregates
        # then average across *instances*, not across instance-episodes
        # -- so under the sparse default this reads as "goals reached
        # in the budget". The per-episode form is the interpretable
        # one (it is the success rate when every goal pays 1).
        "cumulative_return": total_return,
        "return_per_episode": total_return / max(1, episodes),
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
    if not borrowed:
        env.close()
    return record


def _decoys_found(core, lifetime: set) -> int:
    """Decoys the agent has been adjacent to.

    A decoy is solid -- there is nothing to enter -- so "found" can
    only mean having stood next to it and seen that it encloses
    nothing. That is exactly the moment a topological explorer should
    stop being interested, which makes the count worth recording.
    """
    found = 0
    for feature in core.layout.features:
        if feature.kind != "decoy":
            continue
        if any(nb in lifetime for cell in feature.cells
               for nb in core.layout.base.neighbors(cell)):
            found += 1
    return found


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
                 choose_reset_factory: Callable | None = None,
                 env_options: dict | None = None, trace: bool = True,
                 telemetry_root: str | None = None,
                 algorithm: str = "unknown", step_stride: int = 1,
                 split: str | None = None,
                 step_budget: int | None = None):
        self.policy_factory = policy_factory
        self.episodes = episodes
        self.seed = seed
        self.track_topology = track_topology
        self.choose_reset_factory = choose_reset_factory
        self.env_options = env_options or {}
        self.trace = trace
        # The *root*, not a writer: a writer holds a filesystem handle
        # and does not survive pickling. Each worker opens its own.
        self.telemetry_root = telemetry_root
        self.algorithm = algorithm
        self.step_stride = step_stride
        self.split = split
        self.step_budget = step_budget

    def __call__(self, row: dict) -> dict:
        # Rebuilt per instance, seeded from the instance: see
        # instance_seed. Cheap for a policy, and the only way the
        # result does not depend on which worker took the row.
        derived = instance_seed(self.seed, row)
        policy = _build(self.policy_factory, derived)
        choose_reset = (_build(self.choose_reset_factory, derived)
                        if self.choose_reset_factory else None)
        key = instance_key(row)
        with telemetry.open_writer(
            self.telemetry_root, self.algorithm,
            part_prefix=f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}-",
        ) as writer:  # noqa: E501
            record = evaluate_instance(
                row, policy, episodes=self.episodes, seed=self.seed,
                trace=self.trace, track_topology=self.track_topology,
                choose_reset=choose_reset, env_options=self.env_options,
                telemetry=(writer if self.telemetry_root else None),
                step_stride=self.step_stride, split=self.split,
                step_budget=self.step_budget,
            )
            if self.telemetry_root:
                writer.add_instance(
                    record,
                    split=self.split or row.get("split") or "test",
                    instance=key, family=row["family"],
                    size=int(row["size"]), seed=int(row["seed"]),
                )
        return record


def _build(factory: Callable, seed: int):
    """Call a factory, passing the seed if it accepts one."""
    try:
        return factory(seed)
    except TypeError:
        return factory()


def evaluate_split(rows: list, policy: Callable, episodes: int = 5,
                   seed: int = 0, trace: bool = True,
                   track_topology: bool = False,
                   choose_reset: Callable | None = None,
                   choose_reset_factory: Callable | None = None,
                   policy_factory: Callable | None = None,
                   workers: int = 1,
                   env_options: dict | None = None,
                   telemetry_root: str | None = None,
                   algorithm: str = "unknown", step_stride: int = 1,
                   split: str | None = None, env=None,
                   step_budget: int | None = None) -> list:
    """Evaluate every hold-out instance, in manifest order.

    Instances are independent and separately seeded, so the loop
    parallelises exactly. Doing so needs ``policy_factory`` -- a
    picklable, zero-argument callable that builds the policy inside
    each worker; without one the run stays serial, which is correct
    for a policy that cannot cross a process boundary.
    """
    if env is not None and len(rows) != 1:
        raise ValueError(
            f"a supplied env belongs to one instance; got {len(rows)} rows"
        )
    if policy_factory is not None and env is None:
        # Same code path serial or parallel, so the two cannot drift.
        from topogym.baselines.gridworld2dv1.parallel import map_instances

        task = InstanceTask(policy_factory, episodes, seed,
                            track_topology, choose_reset_factory,
                            env_options, trace,
                            telemetry_root=telemetry_root,
                            algorithm=algorithm,
                            step_stride=step_stride, split=split,
                            step_budget=step_budget)
        return map_instances(task, rows, workers=workers)

    records = []
    for i, row in enumerate(rows):
        key = instance_key(row)
        with telemetry.open_writer(
            telemetry_root, algorithm,
            part_prefix=f"{re.sub(r'[^A-Za-z0-9_.-]', '_', key)}-",
        ) as writer:
            record = evaluate_instance(
                row, policy, episodes=episodes, seed=seed, trace=trace,
                track_topology=track_topology, choose_reset=choose_reset,
                env_options=env_options,
                telemetry=(writer if telemetry_root else None),
                step_stride=step_stride, split=split, env=env,
                step_budget=step_budget,
            )
            if telemetry_root:
                writer.add_instance(
                    record, split=split or row.get("split") or "test",
                    instance=key, family=row["family"],
                    size=int(row["size"]), seed=int(row["seed"]),
                )
        records.append(record)
        if (i + 1) % 25 == 0:
            logger.info("evaluated %d/%d instances", i + 1, len(rows))
    return records


def random_policy(rng: np.random.Generator) -> Callable:
    """The floor every learned baseline has to clear."""

    def act(_obs, env):
        return int(rng.integers(env.action_space.n))

    return act
