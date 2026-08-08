"""Long-form run telemetry: three Parquet tables per run.

The published JSON result is a *summary* -- one row per hold-out
instance plus a downsampled curve -- and summaries answer only the
questions you thought to ask before running. This module writes the
raw thing alongside it, so a question raised after a ten-hour sweep
does not require another ten-hour sweep:

- ``steps``     one row per environment step (optionally strided)
- ``episodes``  one row per episode
- ``instances`` one row per (instance, split), the JSON record flattened

Every table carries ``algorithm``, ``split``, ``instance``, ``family``,
``size`` and ``seed``, so the four benchmark splits and the ``all``
rollup are ordinary predicates rather than separate files. Tables land
under ``<root>/<table>/algorithm=<name>/split=<split>/part-*.parquet``:
Hive-style partitioning, which ``pandas.read_parquet`` and every query
engine understand without being told the layout.

``root`` may be a local path or a ``gs://`` URI -- a GKE job writes
straight to Cloud Storage, since a pod's disk does not outlive it.

Parquet needs ``pyarrow`` (in the ``benchmarks`` extra). Without it,
telemetry disables itself with a warning rather than failing a run:
losing the analysis file is bad, losing the run is worse.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

logger = logging.getLogger("topogym")

__all__ = ["TelemetryWriter", "is_available", "open_writer"]

#: Columns every table carries, so any table can be filtered by split,
#: family or size without a join.
KEYS = ("algorithm", "split", "instance", "family", "size", "seed")

#: Per-step columns. Deliberately narrow: this table is by far the
#: largest, and anything derivable (coverage fractions, returns to go)
#: is cheaper to compute at analysis time than to store 30 million
#: times.
#:
#: No homology here on purpose. Certified homology of the observed
#: region means a GUDHI call, and one per step would cost more than
#: the stepping it is measuring -- it would change the thing it
#: records. ``h0_components`` is the exception: the environment
#: maintains it in a union-find as it goes, so it is already paid for.
#: Everything topological is per *episode*, in the table below.
STEP_FIELDS = (
    "episode", "step", "interaction", "action", "x", "y", "facing",
    "reward", "terminated", "truncated", "new_cell", "visit_count",
    "unique_states", "h0_components",
)

#: Per-episode columns: the interpretable unit. ``steps_to_goal`` is
#: None when the episode did not reach it, which is the common case.
#: The expensive, interesting ones live here: what the agent had
#: *discovered* by the end of each episode -- the certified homology of
#: the region it has observed, and how many chambers and decoys it has
#: found. One GUDHI call per episode rather than per step, which is
#: affordable at any budget.
EPISODE_FIELDS = (
    "episode", "length", "interactions", "episode_return",
    "steps_to_goal", "reached_goal", "episode_coverage",
    "lifetime_coverage", "unique_states", "visit_entropy",
    "chambers_entered", "chambers_total", "decoys_entered",
    "decoys_total", "observed_h0", "observed_h1", "observed_frac",
    "archive_reset", "reset_cell",
)


#: Column types, declared rather than inferred.
#:
#: Inference is per *file*, and a column can be legitimately all-null in
#: one part and populated in another -- ``reset_cell`` is null for every
#: row of an evaluation that takes no archive resets, and a string
#: during training. Arrow then infers ``null`` for one part and
#: ``string`` for the other and refuses to merge them, so the whole
#: dataset becomes unreadable because of a column nobody was querying.
FIELD_TYPES = {
    "episode": "int64", "step": "int64", "interaction": "int64",
    "action": "int64", "x": "int64", "y": "int64", "facing": "string",
    "reward": "float64", "terminated": "bool_", "truncated": "bool_",
    "new_cell": "bool_", "visit_count": "int64",
    "unique_states": "int64", "h0_components": "int64",
    "length": "int64", "interactions": "int64",
    "episode_return": "float64", "steps_to_goal": "int64",
    "reached_goal": "bool_", "episode_coverage": "float64",
    "lifetime_coverage": "float64", "visit_entropy": "float64",
    "chambers_entered": "int64", "chambers_total": "int64",
    "decoys_entered": "int64", "decoys_total": "int64",
    "observed_h0": "int64", "observed_h1": "int64",
    "observed_frac": "float64", "archive_reset": "bool_",
    "reset_cell": "string",
    "instance": "string", "family": "string", "size": "int64",
    "seed": "int64",
}


def _schema_for(rows: list):
    """An Arrow schema for these columns: declared where known, inferred
    where not (the instances table is open-ended). A null-typed column
    is never left as such -- it could not merge with a populated one."""
    import pyarrow as pa

    fields = []
    for field in pa.Table.from_pylist(rows).schema:
        declared = FIELD_TYPES.get(field.name)
        if declared is None:
            kind = (pa.string() if pa.types.is_null(field.type)
                    else field.type)
        else:
            kind = getattr(pa, declared)()
        fields.append(pa.field(field.name, kind, nullable=True))
    return pa.schema(fields)


def is_available() -> bool:
    """Whether Parquet telemetry can be written in this environment."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _filesystem(root: str):
    """``(filesystem, path)`` for a local path or ``gs://`` URI.

    ``pyarrow.fs.FileSystem.from_uri`` resolves ``gs://`` through the
    GCS driver using ambient credentials -- the Workload Identity a GKE
    pod already has, so a cloud run needs a URI and nothing else.
    """
    import pyarrow.fs as pafs

    if "://" in root:
        return pafs.FileSystem.from_uri(root)
    return pafs.LocalFileSystem(), os.path.abspath(root)


class TelemetryWriter:
    """Buffered Parquet writer for one run's three tables.

    Rows accumulate in memory and flush in batches, because a Parquet
    file of ten-row row-groups is slower to read than the JSON it was
    meant to improve on. Use as a context manager, or call
    :meth:`close`; either way the tail gets flushed.
    """

    def __init__(self, root: str, algorithm: str,
                 batch_size: int = 50_000, compression: str = "zstd",
                 part_prefix: str = ""):
        self.root = root
        self.algorithm = algorithm
        # Distinguishes the parts written by concurrent workers. A
        # sweep evaluates instances in separate processes, each with
        # its own writer, so part numbering alone would collide and one
        # worker would silently overwrite another's rows.
        self.part_prefix = part_prefix
        self.batch_size = batch_size
        self.compression = compression
        self._buffers: dict = {}
        self._writers: dict = {}
        self._counts: dict = {}
        self._fs, self._base = _filesystem(root)

    # -- writing ------------------------------------------------------

    def add_steps(self, rows: Iterable[dict], **keys) -> None:
        self._add("steps", rows, STEP_FIELDS, keys)

    def add_episodes(self, rows: Iterable[dict], **keys) -> None:
        self._add("episodes", rows, EPISODE_FIELDS, keys)

    def add_instance(self, row: dict, **keys) -> None:
        """One flattened per-instance record.

        Its ``metrics`` sub-dict is flattened to ``metric_*`` columns
        and list-valued metrics are dropped: they belong to the
        per-episode table, where they are one row each rather than one
        cell holding a hundred numbers.
        """
        flat = {k: v for k, v in row.items()
                if k not in ("metrics", "curves", "steps_to_goal")
                and not isinstance(v, (list, dict))}
        for name, value in (row.get("metrics") or {}).items():
            if not isinstance(value, (list, dict, tuple)):
                flat[f"metric_{name}"] = value
        self._add("instances", [flat], tuple(flat), keys)

    def _add(self, table: str, rows, fields: tuple, keys: dict) -> None:
        buffer = self._buffers.setdefault(table, [])
        for row in rows:
            record = {field: row.get(field) for field in fields}
            record.update(keys)
            record.setdefault("algorithm", self.algorithm)
            buffer.append(record)
        if len(buffer) >= self.batch_size:
            self.flush(table)

    # -- flushing -----------------------------------------------------

    def flush(self, table: str | None = None) -> None:
        for name in ([table] if table else list(self._buffers)):
            rows = self._buffers.get(name)
            if rows:
                self._write(name, rows)
                self._buffers[name] = []

    def _write(self, table: str, rows: list) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Partition by split so a query for one split reads one
        # directory; write within it under a per-part name, since a
        # single run may flush many times.
        by_split: dict = {}
        for row in rows:
            by_split.setdefault(row.get("split") or "unknown", []).append(row)
        for split, group in by_split.items():
            folder = (f"{self._base.rstrip('/')}/{table}/"
                      f"algorithm={self.algorithm}/split={split}")
            self._fs.create_dir(folder, recursive=True)
            index = self._counts.get((table, split), 0)
            self._counts[(table, split)] = index + 1
            path = f"{folder}/part-{self.part_prefix}{index:05d}.parquet"
            # The two partition columns live in the *path*, not the
            # file. Writing them in both places makes readers infer a
            # dictionary type from the directory and a plain string
            # from the body, and refuse to merge the two.
            body = [{k: v for k, v in row.items()
                     if k not in ("algorithm", "split")}
                    for row in group]
            table_data = pa.Table.from_pylist(body,
                                              schema=_schema_for(body))
            with self._fs.open_output_stream(path) as sink:
                pq.write_table(table_data, sink,
                               compression=self.compression)
            logger.debug("telemetry: %d rows -> %s", len(group), path)

    def close(self) -> None:
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _NullWriter:
    """Stands in when Parquet is unavailable or telemetry is off, so
    callers never branch on whether it exists."""

    def add_steps(self, rows, **keys) -> None:
        pass

    def add_episodes(self, rows, **keys) -> None:
        pass

    def add_instance(self, row, **keys) -> None:
        pass

    def flush(self, table=None) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


def open_writer(root: str | None, algorithm: str, **kwargs):
    """A writer for ``root``, or a no-op stand-in when telemetry is off
    or ``pyarrow`` is missing."""
    if not root:
        return _NullWriter()
    if not is_available():
        logger.warning(
            "telemetry requested at %s but pyarrow is not installed; "
            "continuing without it (pip install 'topogym[benchmarks]')",
            root,
        )
        return _NullWriter()
    return TelemetryWriter(root, algorithm, **kwargs)
