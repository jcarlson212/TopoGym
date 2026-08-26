"""Training telemetry from inside RLlib env-runners.

The archive methods drive their own episode loop, so the harness
records their training curves directly. The RLlib baselines train
inside Ray env-runner actors where that recorder never sees the
environment -- historically they published evaluation telemetry only,
and their discovery curves were invisible exactly where they were
earned (the point of ``Baseline.bind_telemetry``'s docstring).

This module closes the gap with an :class:`RLlibCallback` (new API
stack, Ray 2.5x signatures) that runs on every env-runner and streams
one row per finished episode into the same Parquet schema the archive
methods use, under ``split=single-train``.

Semantics that differ from the single-env loop, by necessity:

* Runners are parallel: with N runners x M vector envs there are N*M
  independent exploration streams. Rows carry a ``runner`` column
  (``worker_index * 100 + env_index``) and per-stream ``interactions``
  counters; readers reconstruct a global axis by merging streams
  rather than assuming one.
* ``lifetime_coverage`` is per underlying environment instance (each
  vector env owns its world copy), not a cross-runner union -- the
  same meaning "lifetime" has for a single-env method, replicated per
  stream.
* No steps table: per-step rows from every runner would multiply the
  fleet's telemetry volume for little figure value; episodes carry the
  curves.

Each runner writes its own part files (``TelemetryWriter`` was built
for concurrent writers -- the ``part_prefix`` keeps them from
overwriting each other), so no cross-process coordination exists at
all.
"""

from __future__ import annotations

import atexit


def telemetry_callbacks(root: str, algorithm: str,
                        split: str = "single-train"):
    """A callbacks class streaming per-episode rows from each runner.

    Built by a factory so the destination travels inside the class the
    config pickles to the runners; ``RLlibCallback`` subclasses are
    instantiated there without arguments.
    """
    from ray.rllib.callbacks.callbacks import RLlibCallback

    class _RunnerTelemetry(RLlibCallback):

        def __init__(self):
            super().__init__()
            self._writer = None
            self._streams: dict = {}    # (worker, env_index) -> counters
            self._chambers: dict = {}   # instance key -> set of indices

        # -- plumbing --------------------------------------------------

        def _split_env(self, env, env_index):
            """Walk the wrapper stack down to the SplitEnv instance."""
            candidate = getattr(env, "unwrapped", env)
            envs = getattr(candidate, "envs", None)
            if envs is not None:
                candidate = envs[env_index]
            seen = set()
            while candidate is not None and id(candidate) not in seen:
                seen.add(id(candidate))
                if hasattr(candidate, "rows") and hasattr(candidate, "row"):
                    return candidate
                candidate = getattr(candidate, "env", None) or (
                    getattr(candidate, "unwrapped", None)
                    if getattr(candidate, "unwrapped", None)
                    is not candidate else None)
            return None

        def _ensure_writer(self, worker_index):
            if self._writer is None:
                from topogym.baselines.gridworld2dv1.telemetry import TelemetryWriter, is_available
                if not is_available():
                    return None
                self._writer = TelemetryWriter(
                    root, algorithm, batch_size=100,
                    part_prefix=f"w{worker_index:02d}-")
                atexit.register(self._writer.close)
            return self._writer

        # -- the hook --------------------------------------------------

        def on_episode_end(self, *, episode, env_runner=None,
                           metrics_logger=None, env=None, env_index,
                           rl_module=None, **kwargs) -> None:
            split_env = self._split_env(env, env_index)
            if split_env is None or split_env.env is None:
                return
            worker = getattr(env_runner, "worker_index", 0) or 0
            writer = self._ensure_writer(worker)
            if writer is None:
                return
            core = split_env.env.unwrapped
            row = split_env.row
            from topogym.baselines.gridworld2dv1.evaluate import _decoys_found
            from topogym.baselines.gridworld2dv1.instances import instance_key

            key = (worker, env_index)
            stream = self._streams.setdefault(
                key, {"episode": 0, "interactions": 0})
            length = len(episode)
            stream["episode"] += 1
            stream["interactions"] += length

            instance = instance_key(row)
            lifetime = core._ever_visited | core._visited
            n_free = max(1, len(core.layout.free_cells)
                         if core.layout else 1)
            chambers = self._chambers.setdefault(instance, set())
            chambers |= {
                index for cell, index in core._chamber_of.items()
                if cell in core.lifetime_visit_counts
            }
            try:
                last_info = episode.get_infos(-1) or {}
            except Exception:
                last_info = {}
            reached = bool(last_info.get("goal_reached"))
            observed = core.homology_stats("observed")
            writer.add_episodes([{
                "episode": stream["episode"],
                "length": length,
                "interactions": stream["interactions"],
                "episode_return": float(episode.get_return()),
                "steps_to_goal": length if reached else None,
                "reached_goal": reached,
                "episode_coverage": len(core._visited) / n_free,
                "lifetime_coverage": len(lifetime) / n_free,
                "unique_states": len(lifetime),
                "visit_entropy": last_info.get("visitation_entropy"),
                "chambers_entered": len(chambers),
                "chambers_total": sum(
                    1 for f in core.layout.features
                    if f.kind == "chamber"),
                "decoys_entered": _decoys_found(core, lifetime),
                "decoys_total": sum(
                    1 for f in core.layout.features
                    if f.kind == "decoy"),
                "observed_h0": observed.h0,
                "observed_h1": observed.h1,
                "observed_frac": last_info.get("observed_frac"),
                "archive_reset": False,
                "reset_cell": None,
                "runner": worker * 100 + int(env_index),
            }], split=split, instance=instance,
                family=row["family"], size=int(row["size"]),
                seed=int(row["seed"]))

    return _RunnerTelemetry
