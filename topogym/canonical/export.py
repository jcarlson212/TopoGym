"""Per-episode export in the LeRobot dataset v3.0 layout.

Each episode is written as a self-contained one-episode dataset::

    <dir>/meta/info.json
    <dir>/meta/tasks.parquet
    <dir>/meta/episodes/chunk-000/file-000.parquet
    <dir>/meta/stats.json
    <dir>/meta/topo.json                 # the canonical extension
    <dir>/data/chunk-000/file-000.parquet
    <dir>/videos/<key>/chunk-000/file-000.mp4    # video features only
    <dir>/native/<stream>/chunk-000/file-000.parquet   # side streams only

and :func:`assemble` merges any number of them into one dataset, one
data (video, side-stream) file per episode, without re-encoding.

**Rows are decisions.** Row ``t`` holds the observation an action was
chosen from, that action (at every level it was expressed in), and what
followed it (``next.reward``, ``next.done``, ``next.success``). What
happens *between* decisions -- per-tick native controls, sensor ticks --
goes in side streams (:meth:`EpisodeWriter.add_side`), keyed by the
decision row they belong to.

**The clock is regular**: ``timestamp = frame_index / fps``, as LeRobot
requires. Simulated or wall time, if you record it, is a declared
feature of its own (e.g. ``observation.native.sim_time``).

**Missing values**: float features use NaN, never None; integer
features declare a sentinel in their ``info`` (``{"missing": -1}``);
JSON metadata uses null. The writer refuses None in a numeric feature.

Two ways in:

- :class:`EpisodeWriter` takes a list of
  :class:`~topogym.canonical.spec.FeatureSpec` and frames as dicts. It
  knows nothing about environments.
- ``EpisodeWriter(root, env)`` records a TopoGym grid env in canonical
  mode from its step results (:class:`GridEpisodeWriter`, a subclass
  written against the public hooks only).

Needs the ``[export]`` extra (pyarrow). PNG images and 16-bit depth and
segmentation are encoded with the standard library; JPEG needs the
``[jpeg]`` extra (Pillow), video the ``[video]`` extra (PyAV).
"""

from __future__ import annotations

import contextlib
import json
import math
import pathlib
import shutil
import warnings

import numpy as np

from topogym.canonical import codecs, spec
from topogym.canonical.spec import FeatureSpec

CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = ("videos/{video_key}/chunk-{chunk_index:03d}/"
              "file-{file_index:03d}.mp4")
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
TASKS_PATH = "meta/tasks.parquet"
SIDE_PATH = "native/{stream}/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}

#: Frame keys the writer fills itself.
_AUTOMATIC = frozenset(DEFAULT_FEATURES)

#: How each storage is declared in meta/info.json.
_IMAGE_STORAGES = ("png", "jpeg", "video") + spec.ARRAY_STORAGES

#: Licence fields every exported dataset declares.
LICENCE_FIELDS = ("profile", "class", "spdx", "internal_only",
                  "attributions")

#: The licence of data this package's own environments generate.
TOPOGYM_LICENCE = {
    "profile": "topogym", "class": "open", "spdx": "MIT",
    "internal_only": False, "attributions": [],
    # 0.5 field names, kept so existing readers still find them.
    "data": "MIT", "assets": "procedural, MIT",
}


def _pa():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "episode export needs pyarrow: pip install 'topogym[export]'"
        ) from exc
    return pa, pq


# Kept for callers of 0.5: the RGB encoder is the general one.
encode_png = codecs.encode_png


def _stats(values: np.ndarray, image: bool = False,
           channels: int = 3) -> dict:
    """LeRobot-style per-feature stats: per channel for images (scaled
    to [0, 1], shape (C, 1, 1)), per dimension otherwise. NaNs are
    ignored; an all-NaN dimension reports NaN (JSON null)."""
    if image:
        x = values.reshape(-1, channels).astype(np.float64) / 255.0
        shape = (channels, 1, 1)
    else:
        x = values.reshape(len(values), -1).astype(np.float64)
        shape = x.shape[1:]
    with np.errstate(invalid="ignore"), _quiet():
        out = {
            "min": np.nanmin(x, axis=0), "max": np.nanmax(x, axis=0),
            "mean": np.nanmean(x, axis=0), "std": np.nanstd(x, axis=0),
        }
        for q in QUANTILES:
            out[f"q{int(q * 100):02d}"] = np.nanquantile(x, q, axis=0)
    out = {k: [None if isinstance(v, float) and math.isnan(v) else v
               for v in np.asarray(v).reshape(-1).tolist()]
           for k, v in out.items()}
    out = {k: np.asarray(v, dtype=object).reshape(shape).tolist()
           for k, v in out.items()}
    out["count"] = [int(len(values))]
    return out


@contextlib.contextmanager
def _quiet():
    """Silence numpy's all-NaN RuntimeWarnings inside a block."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None  # JSON has no NaN: metadata uses null
    return value


# -- the writer ------------------------------------------------------------------


def _is_features(arg) -> bool:
    return isinstance(arg, (list, tuple)) and all(
        isinstance(f, FeatureSpec) for f in arg)


class EpisodeWriter:
    """Write one episode as a one-episode LeRobot v3.0 dataset.

    ::

        writer = EpisodeWriter(out_dir, features, fps=10,
                               licence={...}, split="train")
        for t in range(n):
            writer.add_frame({...one value per feature...}, task=text)
            for tick in ticks_of_decision_t:
                writer.add_side("controls", {...})
        writer.close()

    ``features`` are :class:`~topogym.canonical.spec.FeatureSpec`. Each
    feature's ``storage`` decides how it is written: ``None`` for a
    parquet column (any fixed shape; ``None`` in the shape for a
    variable-length axis), ``"png"``/``"jpeg"``/``"video"`` for images,
    ``"depth_png16"``/``"segmentation_png16"``/``"array_png16"`` for
    2-D arrays (see :mod:`topogym.canonical.codecs`).

    ``licence`` is required: ``{"profile", "class", "spdx",
    "internal_only", "attributions"}``. Only this package's own envs
    default to MIT (:class:`GridEpisodeWriter`).

    **Extension points** (public, for subclasses):
    :attr:`features`, :attr:`frames`, :attr:`episode_record` (merged
    into ``meta/topo.json``'s per-episode record), :attr:`ext`
    (``{producer: {...}}``, uninterpreted), and the hooks
    :meth:`lerobot_features`, :meth:`episode_metadata`,
    :meth:`topo_record` and :meth:`on_close`.

    ``EpisodeWriter(root, env, ...)`` (an env instead of features)
    returns a :class:`GridEpisodeWriter`, as in 0.5.
    """

    def __new__(cls, root=None, features=None, *args, **kwargs):
        if cls is EpisodeWriter and features is not None and \
                not _is_features(features):
            return super().__new__(GridEpisodeWriter)
        return super().__new__(cls)

    def __init__(self, root, features, *, fps: int = 10,
                 episode_index: int = 0, robot_type: str | None = None,
                 licence: dict | None = None, split: str | None = None,
                 tags=(), seed=None, manifest: dict | None = None,
                 ext: dict | None = None):
        _pa()
        if not _is_features(features):
            raise TypeError("features must be a list of FeatureSpec")
        if licence is None:
            raise ValueError(
                "licence is required: {profile, class, spdx, internal_only, "
                "attributions}; this package's MIT applies only to its own "
                "environments")
        missing = [k for k in LICENCE_FIELDS if k not in licence]
        if missing:
            raise ValueError(f"licence lacks {missing}")
        if split is not None and split not in spec.SPLITS:
            raise ValueError(f"split must be one of {spec.SPLITS} or None")
        problems = spec.validate_features(features)
        if problems:
            raise ValueError("invalid features: " + "; ".join(problems))
        self.root = pathlib.Path(root)
        self.fps = int(fps)
        self.episode_index = int(episode_index)
        self.robot_type = robot_type
        self.licence = dict(licence)
        self.split = {"split": split, "tags": list(tags), "seed": seed}
        self.manifest = manifest
        #: ``{producer: {...}}``, written to topo.json uninterpreted.
        self.ext: dict = {k: dict(v) for k, v in (ext or {}).items()}
        #: ``key -> FeatureSpec``, in column order.
        self.features: dict = {f.key: f for f in features}
        #: One dict per decision row, as given to :meth:`add_frame`.
        self.frames: list = []
        #: Tasks, one per frame.
        self.tasks: list = []
        #: Per-episode record for topo.json (instruction, topology,
        #: segmentation tables, ...). Subclasses and callers add to it.
        self.episode_record: dict = {}
        #: ``stream -> list of tick records``.
        self.side: dict = {}

    # -- recording ------------------------------------------------------------

    def add_frame(self, frame: dict, *, task: str) -> None:
        """Record decision row ``len(frames)``. ``frame`` holds one
        value per declared feature; ``timestamp`` and the indices are
        the writer's."""
        clash = _AUTOMATIC & set(frame)
        if clash:
            raise ValueError(
                f"{sorted(clash)} are set by the writer (timestamp is "
                "frame_index / fps; record simulated or wall time as a "
                "declared feature such as observation.native.sim_time)")
        unknown = set(frame) - set(self.features)
        if unknown:
            raise ValueError(f"undeclared features {sorted(unknown)}")
        absent = set(self.features) - set(frame)
        if absent:
            raise ValueError(f"frame lacks {sorted(absent)}")
        for key, value in frame.items():
            problem = spec.check_value(self.features[key], value)
            if problem:
                raise ValueError(f"{key}: {problem}")
        if not isinstance(task, str) or not task:
            raise ValueError("every frame needs a task string")
        self.frames.append(dict(frame))
        self.tasks.append(task)

    def add_side(self, stream: str, record: dict) -> None:
        """Record one tick of a side stream (e.g. per-tick native
        controls) under the decision row most recently added."""
        if not self.frames:
            raise RuntimeError("add the decision frame before its ticks")
        if not stream or "/" in stream:
            raise ValueError("stream names are non-empty, without '/'")
        for key, value in record.items():
            if value is None:
                raise ValueError(f"{stream}.{key}: None; use NaN")
        ticks = self.side.setdefault(stream, [])
        if ticks and set(ticks[0]["record"]) != set(record):
            raise ValueError(f"{stream}: every tick needs the same fields")
        ticks.append({"frame_index": len(self.frames) - 1,
                      "record": dict(record)})

    def set_segmentation_table(self, key: str, table: dict) -> None:
        """What each instance id of segmentation feature ``key`` means:
        ``{id: {"category": ..., "label": ...}}``."""
        f = self.features.get(key)
        if f is None or f.storage != "segmentation_png16":
            raise ValueError(f"{key} is not a segmentation feature")
        seg = self.episode_record.setdefault("segmentation", {})
        seg[key] = {str(int(k)): {"category": v.get("category"),
                                  "label": v.get("label")}
                    for k, v in table.items()}

    # -- hooks ----------------------------------------------------------------

    def lerobot_features(self) -> dict:
        """The ``features`` block of meta/info.json."""
        out = {}
        for key, f in self.features.items():
            out[key] = _lerobot_feature(f, self.fps)
        return {**out, **DEFAULT_FEATURES}

    def episode_metadata(self) -> dict:
        """Extra columns for this episode's row in meta/episodes (the
        LeRobot bookkeeping columns are added by the writer)."""
        return {}

    def topo_record(self) -> dict:
        """meta/topo.json, the canonical extension."""
        return {
            "spec_version": spec.CANONICAL_SPEC_VERSION,
            "manifest": self.manifest,
            "split": self.split,
            "licence": self.licence,
            "ext": self.ext,
            "side_streams": {
                name: _side_schema(ticks) for name, ticks in self.side.items()
            },
            "episodes": {str(self.episode_index): self.episode_record},
        }

    def on_close(self, root: pathlib.Path) -> None:
        """Called after everything is written (e.g. to add files)."""

    # -- writing --------------------------------------------------------------

    def close(self) -> pathlib.Path:
        """Write the dataset; returns its directory."""
        if not self.frames:
            raise RuntimeError("no frames recorded")
        pa, pq = _pa()
        root = self.root
        if root.exists():
            shutil.rmtree(root)
        (root / "meta").mkdir(parents=True)
        n = len(self.frames)
        feats = self.lerobot_features()
        tasks = list(dict.fromkeys(self.tasks))
        task_index = {t: i for i, t in enumerate(tasks)}

        columns = {}
        for key, f in self.features.items():
            if f.storage == "video":
                continue
            values = [fr[key] for fr in self.frames]
            columns[key] = _column(pa, f, values)
        columns["timestamp"] = pa.array(
            [np.float32(t / self.fps) for t in range(n)], pa.float32())
        columns["frame_index"] = pa.array(range(n), pa.int64())
        columns["episode_index"] = pa.array([self.episode_index] * n,
                                            pa.int64())
        columns["index"] = pa.array(range(n), pa.int64())
        columns["task_index"] = pa.array(
            [task_index[t] for t in self.tasks], pa.int64())
        order = [k for k in feats if k in columns]
        table = pa.Table.from_arrays([columns[k] for k in order], names=order)
        data_path = root / DATA_PATH.format(chunk_index=0, file_index=0)
        data_path.parent.mkdir(parents=True)
        pq.write_table(table, data_path, compression="snappy")

        episode = {
            "episode_index": self.episode_index, "tasks": tasks,
            "length": n, "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": 0, "dataset_to_index": n,
        }
        for key, f in self.features.items():
            if f.storage != "video":
                continue
            path = root / VIDEO_PATH.format(video_key=key, chunk_index=0,
                                            file_index=0)
            path.parent.mkdir(parents=True)
            codec = _encode_mp4(path, [fr[key] for fr in self.frames],
                                self.fps)
            feats[key]["info"] = {
                **feats[key].get("info", {}),
                "video.height": f.shape[0], "video.width": f.shape[1],
                "video.codec": codec, "video.pix_fmt": "yuv420p",
                "video.fps": self.fps, "video.channels": 3,
                "has_audio": False, "is_depth_map": False,
            }
            episode.update({
                f"videos/{key}/chunk_index": 0,
                f"videos/{key}/file_index": 0,
                f"videos/{key}/from_timestamp": 0.0,
                f"videos/{key}/to_timestamp": n / self.fps,
            })
        for stream, ticks in self.side.items():
            path = root / SIDE_PATH.format(stream=stream, chunk_index=0,
                                           file_index=0)
            path.parent.mkdir(parents=True)
            pq.write_table(_side_table(pa, ticks, self.episode_index), path)

        stats = self._stats(feats)
        episode.update(self.episode_metadata())
        episode.update({"meta/episodes/chunk_index": 0,
                        "meta/episodes/file_index": 0})
        write_episodes(root, [episode], stats=[stats])
        write_tasks(root, tasks)
        (root / "meta" / "stats.json").write_text(
            json.dumps(_jsonable(stats)))
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "features": feats,
            "total_episodes": 1, "total_frames": n,
            "total_tasks": len(tasks),
            "chunks_size": CHUNKS_SIZE,
            "data_files_size_in_mb": 100, "video_files_size_in_mb": 200,
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if any(
                f.storage == "video" for f in self.features.values())
            else None,
            "splits": _splits([self.split["split"]])[0],
        }
        (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
        (root / "meta" / "topo.json").write_text(
            json.dumps(_jsonable(self.topo_record()), indent=2))
        self.on_close(root)
        return root

    def _stats(self, feats: dict) -> dict:
        n = len(self.frames)
        stats = {}
        for key, f in self.features.items():
            if f.dtype == "string" or f.variable_length:
                continue
            values = np.stack([np.asarray(fr[key]) for fr in self.frames])
            if f.storage in ("png", "jpeg", "video"):
                channels = f.shape[2] if len(f.shape) == 3 else 1
                stats[key] = _stats(values, image=True, channels=channels)
            elif f.storage in spec.ARRAY_STORAGES:
                # In the feature's own units (metres, ids), per image.
                stats[key] = _stats(values.reshape(n, -1).astype(
                    np.float64).mean(axis=1, keepdims=True))
            else:
                stats[key] = _stats(values.astype(np.float64))
        defaults = {"timestamp": np.arange(n) / self.fps,
                    "frame_index": np.arange(n), "index": np.arange(n),
                    "episode_index": np.full(n, self.episode_index),
                    "task_index": np.asarray(
                        [self.tasks.index(t) for t in self.tasks])}
        for key, values in defaults.items():
            stats[key] = _stats(np.asarray(values, dtype=np.float64))
        return stats


def _lerobot_feature(f: FeatureSpec, fps: int) -> dict:
    """How a FeatureSpec appears in meta/info.json."""
    shape = [s if s is not None else None for s in f.shape]
    if f.storage in ("png", "jpeg", "video"):
        if len(shape) == 2:
            shape = shape + [1]
        out = {"dtype": "video" if f.storage == "video" else "image",
               "shape": shape, "names": ["height", "width", "channels"]}
        info = {"storage": f.storage}
        if f.storage == "video":
            info = {}
        out["info"] = {**info, **f.info} if (info or f.info) else None
        if out["info"] is None:
            del out["info"]
        return out
    if f.storage in spec.ARRAY_STORAGES:
        h, w = shape[:2]
        info = {"storage": f.storage, **spec.array_storage_info(f.storage),
                **f.info}
        return {"dtype": "image", "shape": [h, w, 1],
                "names": ["height", "width", "channels"], "info": info}
    out = {"dtype": f.dtype, "shape": shape,
           "names": list(f.names) if f.names is not None else None}
    if f.info:
        out["info"] = dict(f.info)
    return out


def _image_struct(pa):
    return pa.struct([("bytes", pa.binary()), ("path", pa.string())])


_PA_SCALARS = ("float16", "float32", "float64", "int8", "int16", "int32",
               "int64", "uint8", "uint16", "uint32", "uint64", "bool")


def _pa_type(pa, dtype: str):
    if dtype == "bool":
        return pa.bool_()
    if dtype == "string":
        return pa.string()
    return pa.from_numpy_dtype(np.dtype(dtype))


def _column(pa, f: FeatureSpec, values: list):
    if f.storage in ("png", "jpeg"):
        enc = codecs.encode_png if f.storage == "png" else codecs.encode_jpeg
        return pa.array([{"bytes": enc(np.asarray(v)), "path": None}
                         for v in values], _image_struct(pa))
    if f.storage == "depth_png16":
        return pa.array([{"bytes": codecs.encode_depth_png(v), "path": None}
                         for v in values], _image_struct(pa))
    if f.storage == "segmentation_png16":
        return pa.array([{"bytes": codecs.encode_png(
            codecs.encode_segmentation(v)), "path": None}
            for v in values], _image_struct(pa))
    if f.storage == "array_png16":
        return pa.array([{"bytes": codecs.encode_png(
            codecs.encode_array16(v)), "path": None}
            for v in values], _image_struct(pa))
    if f.dtype == "string":
        return pa.array(values, pa.string())
    typ = _pa_type(pa, f.dtype)
    shape = tuple(f.shape)
    if shape == (1,):
        return pa.array([np.asarray(v).reshape(()).item() for v in values],
                        typ)
    for dim in reversed(shape):
        typ = pa.list_(typ) if dim is None else pa.list_(typ, int(dim))
    return pa.array([np.asarray(v).tolist() for v in values], typ)


def _side_schema(ticks: list) -> dict:
    first = ticks[0]["record"]
    return {k: {"dtype": str(np.asarray(v).dtype) if not isinstance(v, str)
                else "string", "shape": list(np.asarray(v).shape) or [1]}
            for k, v in first.items()}


def _side_table(pa, ticks: list, episode_index: int):
    cols = {"episode_index": pa.array([episode_index] * len(ticks),
                                      pa.int64()),
            "frame_index": pa.array([t["frame_index"] for t in ticks],
                                    pa.int64()),
            "tick_index": pa.array(range(len(ticks)), pa.int64())}
    for key in ticks[0]["record"]:
        values = [t["record"][key] for t in ticks]
        if isinstance(values[0], str):
            cols[key] = pa.array(values, pa.string())
        else:
            cols[key] = pa.array([np.asarray(v).tolist() for v in values])
    return pa.table(cols)


def _encode_mp4(path: pathlib.Path, frames: list, fps: int) -> str:
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "video=True needs PyAV: pip install 'topogym[video]'") from exc
    for codec in ("libsvtav1", "h264", "mpeg4"):
        try:
            av.codec.Codec(codec, "w")
        except Exception:  # codec not built into this PyAV
            continue
        break
    with av.open(str(path), "w") as out:
        stream = out.add_stream(codec, rate=fps)
        h, w, _ = frames[0].shape
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        for img in frames:
            frame = av.VideoFrame.from_ndarray(np.asarray(img, np.uint8),
                                               format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return codec


# -- metadata files ----------------------------------------------------------------


def _pandas_meta(index: str, columns: list) -> bytes:
    """The schema metadata pandas writes, so ``pd.read_parquet`` restores
    ``index`` as the index (LeRobot reads tasks that way)."""
    return json.dumps({
        "index_columns": [index],
        "column_indexes": [{"name": None, "field_name": None,
                            "pandas_type": "unicode", "numpy_type": "object",
                            "metadata": {"encoding": "UTF-8"}}],
        "columns": [{"name": name, "field_name": name, "pandas_type": pt,
                     "numpy_type": nt, "metadata": None}
                    for name, pt, nt in columns],
        "creator": {"library": "pyarrow", "version": "topogym"},
        "pandas_version": "2.0.0",
    }).encode()


def write_tasks(root, tasks: list) -> pathlib.Path:
    """Write ``meta/tasks.parquet``: one row per task, ``task_index``
    in list order, indexed by the task string (how LeRobot reads it,
    ``pd.read_parquet(...).index``). Returns the file's path."""
    pa, pq = _pa()
    root = pathlib.Path(root)
    tasks = list(tasks)
    if len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be unique")
    table = pa.table({"task_index": pa.array(range(len(tasks)), pa.int64()),
                      "task": pa.array(tasks, pa.string())})
    table = table.replace_schema_metadata({b"pandas": _pandas_meta(
        "task", [("task_index", "int64", "int64"),
                 ("task", "unicode", "object")])})
    path = root / TASKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def _flatten_stats(stats: dict) -> dict:
    return {f"stats/{key}/{name}": value
            for key, per in stats.items() for name, value in per.items()}


def write_episodes(root, episodes: list, stats: list | None = None,
                   *, chunk_index: int = 0, file_index: int = 0
                   ) -> pathlib.Path:
    """Write ``meta/episodes/chunk-XXX/file-XXX.parquet``: one row per
    episode, with LeRobot's per-feature ``stats/<key>/<stat>`` columns
    when ``stats`` (one dict per episode) is given. Returns the path."""
    pa, pq = _pa()
    root = pathlib.Path(root)
    rows = [dict(ep) for ep in episodes]
    if stats is not None:
        for row, st in zip(rows, stats):
            row.update(_flatten_stats(_jsonable(st)))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    table = pa.Table.from_pylist(
        [{k: row.get(k) for k in keys} for row in rows])
    path = root / EPISODES_PATH.format(chunk_index=chunk_index,
                                       file_index=file_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


# 0.5 names.
_write_tasks = write_tasks
_write_episodes = write_episodes


# -- assembling --------------------------------------------------------------------


def _splits(names: list) -> tuple:
    """``meta/info.json`` splits for episodes whose splits are ``names``
    (in episode order): each split maps to its ``"start:end"`` range.
    Episodes in no split (``None``) are listed under none. Returns the
    map and the splits left out because their episodes are not
    contiguous, which a range cannot express."""
    ranges: dict = {}
    broken: list = []
    for i, name in enumerate(names):
        if name is None or name in broken:
            continue
        start, end = ranges.get(name, (i, i))
        if name in ranges and end != i:
            broken.append(name)
            del ranges[name]
            continue
        ranges[name] = (start, i + 1)
    return {name: f"{a}:{b}" for name, (a, b) in ranges.items()}, broken


def _split_of(src: pathlib.Path):
    topo = json.loads((src / "meta" / "topo.json").read_text())
    split = topo.get("split")
    return split.get("split") if isinstance(split, dict) else split


def _feature_signature(f: dict) -> dict:
    """What must agree for two episodes' feature to share a column."""
    info = f.get("info") or {}
    keep = {k: v for k, v in info.items()
            if not k.startswith("video.") and k != "has_audio"}
    return {"dtype": f["dtype"], "shape": list(f["shape"]),
            "names": f.get("names"), "info": keep}


def _check_compatible(first: dict, other: dict, src) -> None:
    a, b = first["features"], other["features"]
    if a.keys() != b.keys():
        missing = sorted(set(a) - set(b))
        extra = sorted(set(b) - set(a))
        raise ValueError(f"{src}: features differ from the first episode "
                         f"(missing {missing}, extra {extra})")
    for key in a:
        sa, sb = _feature_signature(a[key]), _feature_signature(b[key])
        if sa != sb:
            diff = {k: (sa[k], sb[k]) for k in sa if sa[k] != sb[k]}
            raise ValueError(f"{src}: feature {key!r} differs from the first "
                             f"episode: {diff}")
    if first["fps"] != other["fps"]:
        raise ValueError(f"{src}: fps {other['fps']} != {first['fps']}")


def assemble(episode_dirs: list, out, *,
             group_by_split: bool = True) -> pathlib.Path:
    """Merge one-episode datasets into one LeRobot v3.0 dataset.

    Episodes keep their files: episode ``i`` becomes data file ``i``
    (and video and side-stream file ``i``) in chunk ``i // 1000``, so
    nothing is re-encoded. Episode, frame and task indices are
    renumbered. Every episode must declare the same features -- names,
    dtypes, shapes and storage -- at the same fps; a mismatch is
    refused, naming the feature.

    ``meta/topo.json`` keeps every episode's own record: its split,
    licence, ``ext`` entries and (if it differs from the first) its
    manifest, alongside the per-episode record it was written with.

    ``meta/info.json`` records each split's episode range, which needs
    each split's episodes to be contiguous. By default
    (``group_by_split=True``) episodes are grouped by split -- stably,
    splits in order of first appearance, episodes in no split last --
    with a warning when that changes their order; each episode's record
    keeps its original position (``source_position``). With
    ``group_by_split=False`` the given order is kept, and a split whose
    episodes are not contiguous is left out of ``splits`` with a
    warning (each episode's record still names its split).
    """
    pa, pq = _pa()
    episode_dirs = [pathlib.Path(d) for d in episode_dirs]
    if not episode_dirs:
        raise ValueError("no episodes to assemble")
    names = [_split_of(d) for d in episode_dirs]
    positions = list(range(len(episode_dirs)))
    if group_by_split:
        order = list(dict.fromkeys(n for n in names if n is not None))
        rank = {n: k for k, n in enumerate(order)}
        ranked = sorted(zip(names, episode_dirs, positions),
                        key=lambda p: rank.get(p[0], len(order)))
        if [p for _, _, p in ranked] != positions:
            warnings.warn(
                "assemble: episodes were reordered to group them by split "
                "(each record keeps its source_position); pass "
                "group_by_split=False to keep the given order",
                stacklevel=2)
        names = [n for n, _, _ in ranked]
        episode_dirs = [d for _, d, _ in ranked]
        positions = [p for _, _, p in ranked]
    splits, broken = _splits(names)
    if broken:
        warnings.warn(
            f"assemble: splits {broken} are not contiguous in the given "
            "order, so meta/info.json cannot express them as ranges and "
            "leaves them out; each episode's record still names its split "
            "(group_by_split=True groups them)", stacklevel=2)
    out = pathlib.Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    tasks: dict = {}
    episodes, all_stats, topo_eps = [], [], {}
    manifests: list = []
    licences: list = []
    side_schema: dict | None = None
    info = None
    total = 0
    for i, src in enumerate(episode_dirs):
        src_info = json.loads((src / "meta" / "info.json").read_text())
        if info is None:
            info = src_info
        else:
            _check_compatible(info, src_info, src)
        topo = json.loads((src / "meta" / "topo.json").read_text())
        streams = topo.get("side_streams") or {}
        if side_schema is None:
            side_schema = streams
        elif streams != side_schema:
            raise ValueError(f"{src}: side streams differ from the first "
                             f"episode ({sorted(streams)} vs "
                             f"{sorted(side_schema)})")
        chunk, file = divmod(i, CHUNKS_SIZE)
        table = pq.read_table(src / DATA_PATH.format(chunk_index=0,
                                                     file_index=0))
        n = table.num_rows
        ep_src = pq.read_table(src / EPISODES_PATH.format(
            chunk_index=0, file_index=0)).to_pylist()[0]
        src_tasks = _read_task_list(src)
        remap = {k: tasks.setdefault(t, len(tasks))
                 for k, t in enumerate(src_tasks)}

        def replace(tbl, name, values, typ):
            return tbl.set_column(tbl.schema.get_field_index(name), name,
                                  pa.array(values, typ))

        old_tasks = table.column("task_index").to_pylist()
        table = replace(table, "episode_index", [i] * n, pa.int64())
        table = replace(table, "index", range(total, total + n), pa.int64())
        table = replace(table, "task_index", [remap[t] for t in old_tasks],
                        pa.int64())
        dst = out / DATA_PATH.format(chunk_index=chunk, file_index=file)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst, compression="snappy")
        ep = {k: v for k, v in ep_src.items() if not k.startswith("stats/")}
        ep.update({"episode_index": i, "data/chunk_index": chunk,
                   "data/file_index": file, "dataset_from_index": total,
                   "dataset_to_index": total + n,
                   "meta/episodes/chunk_index": 0,
                   "meta/episodes/file_index": 0})
        for key, f in src_info["features"].items():
            if f["dtype"] != "video":
                continue
            vsrc = src / VIDEO_PATH.format(video_key=key, chunk_index=0,
                                           file_index=0)
            vdst = out / VIDEO_PATH.format(video_key=key, chunk_index=chunk,
                                           file_index=file)
            vdst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(vsrc, vdst)
            ep[f"videos/{key}/chunk_index"] = chunk
            ep[f"videos/{key}/file_index"] = file
        for stream in streams:
            ssrc = src / SIDE_PATH.format(stream=stream, chunk_index=0,
                                          file_index=0)
            side = pq.read_table(ssrc)
            side = replace(side, "episode_index", [i] * side.num_rows,
                           pa.int64())
            sdst = out / SIDE_PATH.format(stream=stream, chunk_index=chunk,
                                          file_index=file)
            sdst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(side, sdst)
        episodes.append(ep)
        all_stats.append(json.loads((src / "meta" / "stats.json").read_text()))
        manifest = topo.get("manifest")
        if manifest not in manifests:
            manifests.append(manifest)
        licence = topo.get("licence")
        if licence not in licences:
            licences.append(licence)
        for record in topo.get("episodes", {}).values():
            topo_eps[str(i)] = {
                **record, "source": src.name,
                "source_position": positions[i], "split": topo.get("split"),
                "licence": licence,
                "ext": _merge_ext(topo.get("ext"), record.get("ext")),
                "manifest_index": manifests.index(manifest),
            }
        total += n
    write_episodes(out, episodes, stats=all_stats)
    write_tasks(out, list(tasks))
    (out / "meta" / "stats.json").write_text(
        json.dumps(_jsonable(_merge_stats(all_stats))))
    info.update({"total_episodes": len(episodes), "total_frames": total,
                 "total_tasks": len(tasks), "splits": splits})
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    topo = json.loads((episode_dirs[0] / "meta" / "topo.json").read_text())
    topo.pop("split", None)
    topo.pop("ext", None)
    topo["licence"] = licences[0] if len(licences) == 1 else {
        "mixed": True, "note": "see each episode's licence"}
    if len(manifests) > 1:
        topo["manifests"] = manifests
    topo["side_streams"] = side_schema or {}
    topo["episodes"] = topo_eps
    (out / "meta" / "topo.json").write_text(
        json.dumps(_jsonable(topo), indent=2))
    return out


def _merge_ext(dataset: dict | None, episode: dict | None) -> dict:
    """One episode's ``ext`` records: the dataset-level entries it was
    written with, overlaid per producer by its own."""
    out = {k: dict(v) for k, v in (dataset or {}).items()}
    for producer, fields in (episode or {}).items():
        out.setdefault(producer, {}).update(fields)
    return out


def _read_task_list(src: pathlib.Path) -> list:
    """Tasks of a dataset in task_index order."""
    _, pq = _pa()
    rows = pq.read_table(src / TASKS_PATH).to_pylist()
    return [r["task"] for r in sorted(rows, key=lambda r: r["task_index"])]


def _merge_stats(all_stats: list) -> dict:
    """Count-weighted mean/std, global min/max; quantiles are averaged
    (an approximation, as LeRobot's own aggregation does). Nulls (all
    NaN) are skipped."""
    out = {}
    for key in all_stats[0]:
        per = [s[key] for s in all_stats if key in s]
        counts = np.array([p["count"][0] for p in per], np.float64)
        w = counts / counts.sum()

        def arr(p, name):
            return np.asarray(p[name], dtype=np.float64)

        with np.errstate(invalid="ignore"), _quiet():
            mean = sum(wi * arr(p, "mean") for wi, p in zip(w, per))
            var = sum(wi * (arr(p, "std") ** 2 + (arr(p, "mean") - mean) ** 2)
                      for wi, p in zip(w, per))
            merged = {
                "min": np.nanmin([arr(p, "min") for p in per], axis=0),
                "max": np.nanmax([arr(p, "max") for p in per], axis=0),
                "mean": np.asarray(mean), "std": np.sqrt(var),
            }
            for q in QUANTILES:
                name = f"q{int(q * 100):02d}"
                merged[name] = sum(wi * arr(p, name)
                                   for wi, p in zip(w, per))
        merged = {k: np.asarray(v).tolist() for k, v in merged.items()}
        merged["count"] = [int(counts.sum())]
        out[key] = merged
    return out


# -- the grid envs -------------------------------------------------------------------


class GridEpisodeWriter(EpisodeWriter):
    """Record one episode of a TopoGym grid env in canonical mode.

    ::

        writer = EpisodeWriter(out_dir, env)   # or GridEpisodeWriter
        obs, info = env.reset(seed=0)
        writer.start(obs, info)
        while True:
            action = policy(obs)
            obs, r, term, trunc, info = env.step(action)
            writer.add(action, r, term, trunc, info, next_obs=obs)
            if term or trunc:
                break
        writer.close()

    Built entirely on :class:`EpisodeWriter`'s public hooks. The data
    is this package's own, so the licence defaults to MIT.
    """

    def __init__(self, root, env, *, fps: int = 8, video: bool = False,
                 episode_index: int = 0, robot_type: str = "topogym_grid",
                 licence: dict | None = None, ext: dict | None = None):
        from topogym.canonical.manifest import manifest

        m = manifest(env)
        self.level = m["actions"]["level"]
        image = "video" if video else "png"
        feats = [FeatureSpec(k, "uint8", tuple(f["shape"]),
                             ("height", "width", "channels"), storage=image)
                 for k, f in m["features"].items()
                 if f["dtype"] == "uint8" and len(f["shape"]) == 3]
        feats += _GRID_FEATURES_HEAD
        if self.level == "waypoint":
            feats.append(FeatureSpec("action", "float32", (4,),
                                     spec.WAYPOINT_NAMES))
        else:
            feats.append(FeatureSpec("action", "int64", (1,), ("word",)))
        feats += _GRID_FEATURES_TAIL
        split = m["split"]
        super().__init__(root, feats, fps=fps, episode_index=episode_index,
                         robot_type=robot_type,
                         licence=licence or TOPOGYM_LICENCE,
                         split=split["split"], tags=split["tags"],
                         seed=split["seed"], manifest=m, ext=ext)
        self.image_keys = [f.key for f in feats
                           if f.storage in ("png", "video")]
        self._obs = None
        self._priv: dict = {}

    def start(self, obs: dict, info: dict) -> None:
        if not isinstance(obs, dict) or spec.INSTRUCTION not in obs:
            raise ValueError("EpisodeWriter records canonical observations")
        self._obs = obs
        self.episode_record.update({
            "instruction": info.get("instruction"),
            "privileged": {k: info["privileged"][k]
                           for k in spec.PRIVILEGED_EPISODE_FIELDS
                           if k in info.get("privileged", {})},
        })
        self._priv = info.get("privileged", {})

    def add(self, action, reward, terminated, truncated, info,
            next_obs=None) -> None:
        if self._obs is None:
            raise RuntimeError("call start() with the reset observation, "
                               "and pass next_obs to every add()")
        obs, priv = self._obs, self._priv
        t = len(self.frames)
        canon = info.get("canonical", {})
        if self.level == "waypoint":
            act = np.asarray(action, np.float32).reshape(4)
        else:
            if isinstance(action, str):
                action = spec.WORDS.index(action)
            act = int(action)
        frame = {key: np.asarray(obs[key], np.uint8)
                 for key in self.image_keys}
        frame.update({
            spec.STATE: np.asarray(obs[spec.STATE], np.float32),
            spec.INSTRUCTION: obs[spec.INSTRUCTION],
            spec.TEXT: obs[spec.TEXT],
            spec.STRUCTURED: obs[spec.STRUCTURED],
            "action.words": " ".join(canon.get("words", [])),
            "action": act,
            "action.waypoint": np.asarray(
                canon.get("waypoint", [0, 0, 0, 0]), np.float32),
            "next.reward": np.float32(reward),
            "next.done": bool(terminated or truncated),
            "next.success": bool(info.get("success", False)),
            "is_first": t == 0,
            "is_last": bool(terminated or truncated),
            "is_terminal": bool(terminated),
            "privileged.goal_visible": bool(priv.get("goal_visible")),
            "privileged.world_pose": np.asarray(priv["world_pose"],
                                                np.float32),
            "privileged.goal_pose": np.asarray(
                priv["goal_pose"] if priv.get("goal_pose") is not None
                else [np.nan, np.nan], np.float32),
            "privileged.d_goal": int(-1 if priv.get("d_goal") is None
                                     else priv["d_goal"]),
            "privileged.region_id": int(priv.get("region_id", 0)),
        })
        self.add_frame(frame, task=obs[spec.INSTRUCTION])
        # What the step returned is the next row's input.
        self._obs = next_obs
        self._priv = info.get("privileged", {})


_GRID_FEATURES_HEAD = [
    FeatureSpec(spec.STATE, "float32", (3,), spec.STATE_NAMES_2D),
    FeatureSpec(spec.INSTRUCTION, "string", (1,)),
    FeatureSpec(spec.TEXT, "string", (1,)),
    FeatureSpec(spec.STRUCTURED, "string", (1,)),
    FeatureSpec("action.words", "string", (1,)),
]

_GRID_FEATURES_TAIL = [
    FeatureSpec("action.waypoint", "float32", (4,), spec.WAYPOINT_NAMES),
    FeatureSpec("next.reward", "float32", (1,)),
    FeatureSpec("next.done", "bool", (1,)),
    FeatureSpec("next.success", "bool", (1,)),
    FeatureSpec("is_first", "bool", (1,)),
    FeatureSpec("is_last", "bool", (1,)),
    FeatureSpec("is_terminal", "bool", (1,)),
    FeatureSpec("privileged.goal_visible", "bool", (1,)),
    FeatureSpec("privileged.world_pose", "float32", (3,),
                spec.WORLD_POSE_NAMES_2D),
    FeatureSpec("privileged.goal_pose", "float32", (2,),
                spec.GOAL_POSE_NAMES_2D),
    FeatureSpec("privileged.d_goal", "int64", (1,),
                info={"missing": -1, "units": "cell"}),
    FeatureSpec("privileged.region_id", "int64", (1,)),
]
