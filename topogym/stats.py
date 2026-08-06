"""Cross-episode statistics recording.

:class:`StatsRecorder` wraps any TopoGym env and accumulates the metric
rows an exploration experiment needs, at two granularities:

- **per episode** (always): return, length, final coverage, lifetime
  coverage, chambers entered (and each chamber's first-entry step),
  doors opened, H0 merges, whether the goal was reached;
- **per step** (``record_steps=True``): reward and coverage over time —
  reward curves, entry-time plots.

Rows are plain dicts, ready for pandas::

    env = StatsRecorder(gym.make("TopoGym/Decoys4-50-v0", seed=1))
    ... run episodes ...
    df = pandas.DataFrame(env.episodes)
    env.summary()
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gymnasium as gym

logger = logging.getLogger("topogym")

if TYPE_CHECKING:
    from topogym.envs.core import TopoEnvCore




_MILESTONES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


def _entropy_bits(counts: dict) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    ent = 0.0
    for n in counts.values():
        p = n / total
        ent -= p * math.log2(p)
    return ent


@dataclass(frozen=True)
class Metrics:
    """Standardized exploration metrics, computed from a
    :class:`StatsRecorder`'s records. ``to_dict()`` is logging-ready."""

    episodes: int
    #: fraction of episodes that reached the goal
    success_rate: float
    #: total env interactions until the first success (None if never)
    interactions_to_first_success: int | None
    #: lifetime unique cells visited on the layout
    unique_states: int
    #: lifetime fraction of the free space visited
    state_coverage: float
    #: Shannon entropy (bits) of the lifetime visitation distribution
    visitation_entropy: float
    #: entropy / log2(n_free): 1.0 = perfectly uniform visitation
    visitation_entropy_normalized: float
    #: mean (steps-to-goal - shortest-path) over successful episodes
    mean_regret: float | None
    #: mean shortest-path/steps over successes *after* the first —
    #: 1.0 means optimal replay once the goal has been discovered
    planning_efficiency: float | None
    #: lifetime-coverage milestones: fraction -> global step reached
    steps_to_coverage: dict
    #: components discovered: k -> global step the observed region
    #: first had h0 >= k (requires track_holes=True)
    steps_to_h0_holes: dict
    #: loops discovered: k -> global step the observed region first
    #: had h1 >= k (requires track_holes=True)
    steps_to_h1_holes: dict
    #: mean per-episode final coverage
    mean_episode_coverage: float
    #: fraction of negatively curved cells (Ollivier-Ricci < 0 —
    #: corridors, doorways, bottlenecks) the agent has reached; None
    #: unless track_curvature=True
    curvature_coverage_below_zero: float | None = None

    @property
    def sample_efficiency(self) -> int | None:
        """Alias of ``interactions_to_first_success``."""
        return self.interactions_to_first_success

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["sample_efficiency"] = self.sample_efficiency
        return d


class StatsRecorder(gym.Wrapper):
    """Record per-episode (and optionally per-step) exploration stats."""

    def __init__(self, env: gym.Env, record_steps: bool = False,
                 track_holes: bool = False,
                 track_curvature: bool = False):
        super().__init__(env)
        self.record_steps = record_steps
        #: compute observed homology every step to timestamp hole
        #: discoveries (opt-in: it runs GUDHI per step)
        self.track_holes = track_holes
        #: include Ollivier-Ricci curvature coverage in metrics()
        #: (opt-in: one exact-W1 solve per free edge, once per layout)
        self.track_curvature = track_curvature
        self.episodes: list = []
        self.steps: list = []
        self._episode_index = -1
        self._last_info: dict = {}
        self._goal_reached = False
        self._global_step = 0
        self._first_success_step: int | None = None
        self.lifetime_milestones: dict = {}
        #: dim -> {k: global step the observed region first had
        #: h_dim >= k}; gridworlds track dims 0 and 1
        self.hole_steps: dict = {0: {}, 1: {}}
        self._optimal: int | None = None
        self._ep_milestones: dict = {}
        self._steps_to_success: int | None = None

    @property
    def _core(self) -> TopoEnvCore:
        return self.env.unwrapped

    def reset(self, **kwargs) -> tuple:
        self._flush()
        obs, info = self.env.reset(**kwargs)
        self._episode_index += 1
        self._last_info = info
        self._goal_reached = False
        self._ep_milestones = {}
        self._steps_to_success = None
        core = self._core
        if core.goal_exists:
            try:
                path = core.shortest_path()
                self._optimal = len(path) - 1 if path else None
            except ValueError:
                self._optimal = None
        else:
            self._optimal = None
        return obs, info

    def step(self, action: int) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        core = self._core
        self._global_step += 1
        for frac in _MILESTONES:
            if info["coverage"] >= frac and frac not in self._ep_milestones:
                self._ep_milestones[frac] = info["steps"]
            if info["lifetime_coverage"] >= frac \
                    and frac not in self.lifetime_milestones:
                self.lifetime_milestones[frac] = self._global_step
        if self.track_holes:
            stats = core.homology_stats("observed")
            for dim, count in ((0, stats.h0), (1, stats.h1)):
                found = self.hole_steps[dim]
                for k in range(len(found) + 1, count + 1):
                    found[k] = self._global_step
        if terminated and core.goal_exists \
                and info.get("position") == core.layout.goal:
            self._goal_reached = True
            self._steps_to_success = info["steps"]
            if self._first_success_step is None:
                self._first_success_step = self._global_step
        if self.record_steps:
            self.steps.append({
                "episode": self._episode_index,
                "global_step": self._global_step,
                "step": info["steps"],
                "reward": reward,
                "coverage": info["coverage"],
                "lifetime_coverage": info["lifetime_coverage"],
                "chambers_entered": info["chambers_entered"],
            })
        if terminated or truncated:
            self._flush()
        return obs, reward, terminated, truncated, info

    def _flush(self) -> None:
        if not self._last_info or self._episode_index < 0:
            return
        info = self._last_info
        self.episodes.append({
            "episode": self._episode_index,
            "length": info["steps"],
            "return": info["episode_return"],
            "coverage": info["coverage"],
            "lifetime_coverage": info["lifetime_coverage"],
            "chambers_entered": info["chambers_entered"],
            "chamber_entry_steps": dict(self._core.chamber_entry_steps),
            "doors_opened": info["doors_opened"],
            "h0_merges": info["h0_merges"],
            "goal_reached": self._goal_reached,
            # True when the agent chose this episode's start from the
            # archive at the previous episode's boundary.
            "teleport_start": info.get("teleport_start", False),
            "unique_states": len(self._core._visited),
            "steps_to_success": self._steps_to_success,
            "optimal_steps": self._optimal,
            "regret": (self._steps_to_success - self._optimal
                       if self._steps_to_success is not None
                       and self._optimal is not None else None),
            "coverage_milestones": dict(self._ep_milestones),
            "visitation_entropy":
                _entropy_bits(self._core.visit_counts),
        })
        if logger.isEnabledFor(logging.INFO):
            row = self.episodes[-1]
            logger.info(
                "episode=%d length=%d return=%.4f coverage=%.3f "
                "lifetime=%.3f chambers=%d goal=%s regret=%s",
                row["episode"], row["length"], row["return"],
                row["coverage"], row["lifetime_coverage"],
                row["chambers_entered"], row["goal_reached"],
                row["regret"],
            )
        self._last_info = {}

    def coverage_at(self, global_step: int) -> float:
        """Lifetime coverage reached by the given global step (requires
        ``record_steps=True``)."""
        if not self.record_steps:
            raise RuntimeError(
                "coverage_at needs StatsRecorder(record_steps=True)"
            )
        best = 0.0
        for row in self.steps:
            if row["global_step"] > global_step:
                break
            best = row["lifetime_coverage"]
        return best

    def metrics(self) -> Metrics:
        """The standardized metric set, as a frozen value object."""
        self._flush()
        eps = self.episodes
        n = len(eps)
        core = self._core
        n_free = len(core.layout.free_cells) if core.layout else 1
        lifetime_counts = core.lifetime_visit_counts
        entropy = _entropy_bits(lifetime_counts)
        successes = [e for e in eps if e["goal_reached"]
                     and e["regret"] is not None]
        later = [e for e in successes[1:]
                 if e["steps_to_success"]]
        return Metrics(
            episodes=n,
            success_rate=(sum(e["goal_reached"] for e in eps) / n
                          if n else 0.0),
            interactions_to_first_success=self._first_success_step,
            unique_states=len(lifetime_counts),
            state_coverage=len(lifetime_counts) / n_free,
            visitation_entropy=entropy,
            visitation_entropy_normalized=(
                entropy / math.log2(n_free) if n_free > 1 else 0.0
            ),
            mean_regret=(sum(e["regret"] for e in successes)
                         / len(successes) if successes else None),
            planning_efficiency=(
                sum(e["optimal_steps"] / e["steps_to_success"]
                    for e in later) / len(later) if later else None
            ),
            steps_to_coverage=dict(self.lifetime_milestones),
            steps_to_h0_holes=dict(self.hole_steps[0]),
            steps_to_h1_holes=dict(self.hole_steps[1]),
            mean_episode_coverage=(
                sum(e["coverage"] for e in eps) / n if n else 0.0
            ),
            curvature_coverage_below_zero=(
                self.curvature_coverage(0.0)
                if self.track_curvature else None
            ),
        )

    def curvature_coverage(self, threshold: float) -> float:
        """Fraction of cells with Ollivier-Ricci curvature below the
        threshold that the agent has reached (lifetime). Curvature is
        computed on demand and cached per layout, so explicit calls
        work regardless of the ``track_curvature`` toggle."""
        core = self._core
        ricci = core.ollivier_ricci()
        low = [c for c, k in ricci.items() if k < threshold]
        if not low:
            return 1.0
        reached = core.lifetime_visit_counts
        return sum(1 for c in low if c in reached) / len(low)

    def run_key(self) -> str:
        """The canonical run-log key for this env (configuration and
        layout seed serialized to the spec's canonical string)."""
        from topogym.registry import canonical_string

        core = self._core
        return canonical_string(core.cfg, core.layout_seed or 0,
                                p_slip=core.p_slip)

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write the standardized run log as JSON: a header (run key,
        library version, topology, horizon, reward mode), the episode
        rows, the metric set, and — with ``record_steps`` — the step
        rows. Content is a pure function of the run (no wall-clock
        timestamps), preserving determinism up to seeds."""
        import topogym

        core = self._core
        self._flush()
        payload = {
            "run": {
                "key": self.run_key(),
                "topogym_version": topogym.__version__,
                "layout_seed": core.layout_seed,
                "reward_mode": core.reward_mode,
                "horizon": getattr(core, "_max_steps", None),
                "topology": core.layout.metadata.to_dict()
                if core.layout else None,
            },
            "metrics": self.metrics().to_dict(),
            "episodes": self.episodes,
        }
        if self.record_steps:
            payload["steps"] = self.steps
        path = pathlib.Path(path)
        path.write_text(json.dumps(payload, indent=2, default=repr)
                        + "\n")
        return path

    def summary(self) -> dict:
        """Aggregates over all recorded episodes."""
        eps = self.episodes or [{}]
        n = len(self.episodes)
        def mean(key):
            return (sum(e.get(key, 0) for e in self.episodes) / n
                    if n else 0.0)
        return {
            "episodes": n,
            "mean_return": mean("return"),
            "mean_coverage": mean("coverage"),
            "lifetime_coverage": eps[-1].get("lifetime_coverage", 0.0),
            "mean_chambers_entered": mean("chambers_entered"),
            "goal_rate": mean("goal_reached"),
        }
