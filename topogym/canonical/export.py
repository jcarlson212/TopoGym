"""Per-episode export in the LeRobot dataset v3.0 layout.

Each episode is written as a self-contained one-episode dataset::

    <dir>/meta/info.json
    <dir>/meta/tasks.parquet
    <dir>/meta/episodes/chunk-000/file-000.parquet
    <dir>/meta/stats.json
    <dir>/meta/topo.json                 # the TopoGym extension
    <dir>/data/chunk-000/file-000.parquet
    <dir>/videos/<key>/chunk-000/file-000.mp4   # only with video=True

and :func:`assemble` merges any number of them into one dataset, one
data (and video) file per episode, without re-encoding. Rows are
decisions: row ``t`` holds the observation the action was chosen from,
the action at every level it was expressed in, and what followed
(``next.reward``, ``next.done``, ``next.success``). The privileged
record rides along as ``privileged.*`` columns; loaders select features
by name, so a policy never sees them unless asked.

Needs the ``[export]`` extra (pyarrow). Images are embedded PNGs,
encoded with the standard library and lossless; ``video=True`` writes
MP4 (AV1 where the encoder is available, lossy) and needs the
``[video]`` extra (PyAV).
"""

from __future__ import annotations

import json
import pathlib
import shutil
import struct
import warnings
import zlib

import numpy as np

from topogym.canonical import spec

CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = ("videos/{video_key}/chunk-{chunk_index:03d}/"
              "file-{file_index:03d}.mp4")
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
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


def encode_png(img: np.ndarray) -> bytes:
    """A (H, W, 3) uint8 image as PNG bytes (no dependencies)."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w, _ = img.shape
    raw = np.empty((h, 1 + 3 * w), dtype=np.uint8)
    raw[:, 0] = 0  # filter type: none
    raw[:, 1:] = img.reshape(h, 3 * w)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
            + chunk(b"IEND", b""))


def _stats(values: np.ndarray, image: bool = False) -> dict:
    """LeRobot-style per-feature stats: per channel for images (scaled
    to [0, 1], shape (3, 1, 1)), per dimension otherwise."""
    if image:
        x = values.reshape(-1, 3).astype(np.float64) / 255.0
        shape = (3, 1, 1)
    else:
        x = values.reshape(len(values), -1).astype(np.float64)
        shape = x.shape[1:]
    out = {
        "min": x.min(axis=0), "max": x.max(axis=0),
        "mean": x.mean(axis=0), "std": x.std(axis=0),
    }
    for q in QUANTILES:
        out[f"q{int(q * 100):02d}"] = np.quantile(x, q, axis=0)
    out = {k: np.asarray(v).reshape(shape).tolist() for k, v in out.items()}
    out["count"] = [int(len(values))]
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class EpisodeWriter:
    """Record one episode of a canonical env and write it as a
    one-episode LeRobot v3.0 dataset.

    ::

        writer = EpisodeWriter(out_dir, env)
        obs, info = env.reset(seed=0)
        writer.start(obs, info)
        while True:
            action = policy(obs)
            obs, r, term, trunc, info = env.step(action)
            writer.add(action, r, term, trunc, info, next_obs=obs)
            if term or trunc:
                break
        writer.close()

    Each row pairs an action with the observation (and privileged
    record) it was chosen from, so the observation after the final
    action is not stored (the dataset convention).
    """

    def __init__(self, root, env, *, fps: int = 8, video: bool = False,
                 episode_index: int = 0, robot_type: str = "topogym_grid"):
        from topogym.canonical.manifest import manifest

        _pa()
        self.root = pathlib.Path(root)
        self.fps = int(fps)
        self.video = bool(video)
        self.episode_index = int(episode_index)
        self.robot_type = robot_type
        self.manifest = manifest(env)
        self.level = self.manifest["actions"]["level"]
        self.image_keys = [k for k, f in self.manifest["features"].items()
                           if f["dtype"] == "uint8" and len(f["shape"]) == 3]
        self._rows: list = []
        self._frames: dict = {k: [] for k in self.image_keys}
        self._obs = None
        self._episode: dict = {}

    def start(self, obs: dict, info: dict) -> None:
        if not isinstance(obs, dict) or spec.INSTRUCTION not in obs:
            raise ValueError("EpisodeWriter records canonical observations")
        self._obs = obs
        self._episode = {
            "instruction": info.get("instruction"),
            "privileged": {k: info["privileged"][k]
                           for k in spec.PRIVILEGED_EPISODE_FIELDS
                           if k in info.get("privileged", {})},
        }
        self._priv = info.get("privileged", {})

    def add(self, action, reward, terminated, truncated, info,
            next_obs=None) -> None:
        if self._obs is None:
            raise RuntimeError("call start() with the reset observation, "
                               "and pass next_obs to every add()")
        obs, priv = self._obs, self._priv
        t = len(self._rows)
        canon = info.get("canonical", {})
        row = {
            "timestamp": np.float32(t / self.fps),
            "frame_index": t,
            "task": obs[spec.INSTRUCTION],
            spec.STATE: np.asarray(obs[spec.STATE], np.float32),
            spec.INSTRUCTION: obs[spec.INSTRUCTION],
            spec.TEXT: obs[spec.TEXT],
            spec.STRUCTURED: obs[spec.STRUCTURED],
            "action.words": " ".join(canon.get("words", [])),
            "action.waypoint": np.asarray(
                canon.get("waypoint", [0, 0, 0, 0]), np.float32),
            "next.reward": np.float32(reward),
            "next.done": bool(terminated or truncated),
            "next.success": bool(info.get("success", False)),
            "is_first": t == 0,
            "is_last": bool(terminated or truncated),
            "is_terminal": bool(terminated),
            "privileged.world_pose": np.asarray(priv["world_pose"],
                                                np.float32),
            "privileged.goal_pose": np.asarray(
                priv["goal_pose"] if priv.get("goal_pose") is not None
                else [np.nan, np.nan], np.float32),
            "privileged.d_goal": int(-1 if priv.get("d_goal") is None
                                     else priv["d_goal"]),
            "privileged.goal_visible": bool(priv.get("goal_visible")),
            "privileged.region_id": int(priv.get("region_id", 0)),
        }
        if self.level == "waypoint":
            row["action"] = np.asarray(action, np.float32).reshape(4)
        else:
            if isinstance(action, str):
                action = spec.WORDS.index(action)
            row["action"] = int(action)
        for key in self.image_keys:
            self._frames[key].append(np.asarray(obs[key], np.uint8))
        self._rows.append(row)
        # What the step returned is the next row's input.
        self._obs = next_obs
        self._priv = info.get("privileged", {})

    # -- writing --------------------------------------------------------------

    def _features(self) -> dict:
        m = self.manifest["features"]
        feats = {}
        for key in self.image_keys:
            feats[key] = {"dtype": "video" if self.video else "image",
                          "shape": m[key]["shape"],
                          "names": ["height", "width", "channels"]}
        feats[spec.STATE] = {"dtype": "float32", "shape": [3],
                             "names": list(spec.STATE_NAMES_2D)}
        for key in (spec.INSTRUCTION, spec.TEXT, spec.STRUCTURED,
                    "action.words"):
            feats[key] = {"dtype": "string", "shape": [1], "names": None}
        if self.level == "waypoint":
            feats["action"] = {"dtype": "float32", "shape": [4],
                               "names": list(spec.WAYPOINT_NAMES)}
        else:
            feats["action"] = {"dtype": "int64", "shape": [1],
                               "names": ["word"]}
        feats["action.waypoint"] = {"dtype": "float32", "shape": [4],
                                    "names": list(spec.WAYPOINT_NAMES)}
        feats["next.reward"] = {"dtype": "float32", "shape": [1],
                                "names": None}
        for key in ("next.done", "next.success", "is_first", "is_last",
                    "is_terminal", "privileged.goal_visible"):
            feats[key] = {"dtype": "bool", "shape": [1], "names": None}
        feats["privileged.world_pose"] = {"dtype": "float32", "shape": [3],
                                          "names": ["x", "y", "yaw"]}
        feats["privileged.goal_pose"] = {"dtype": "float32", "shape": [2],
                                         "names": ["x", "y"]}
        for key in ("privileged.d_goal", "privileged.region_id"):
            feats[key] = {"dtype": "int64", "shape": [1], "names": None}
        return {**feats, **DEFAULT_FEATURES}

    def close(self) -> pathlib.Path:
        """Write the dataset; returns its directory."""
        if not self._rows:
            raise RuntimeError("no steps recorded")
        pa, pq = _pa()
        root = self.root
        if root.exists():
            shutil.rmtree(root)
        (root / "meta").mkdir(parents=True)
        n = len(self._rows)
        task = self._rows[0]["task"]
        feats = self._features()

        cols: dict = {}
        for key, f in feats.items():
            if f["dtype"] == "video":
                continue
            if key in ("episode_index", "index", "task_index"):
                continue
            if f["dtype"] == "image":
                cols[key] = [{"bytes": encode_png(img), "path": None}
                             for img in self._frames[key]]
                continue
            cols[key] = [r[key] for r in self._rows]
        cols["episode_index"] = [self.episode_index] * n
        cols["index"] = list(range(n))
        cols["task_index"] = [0] * n
        arrays, fields = [], []
        for key, values in cols.items():
            f = feats[key]
            if f["dtype"] == "image":
                typ = pa.struct([("bytes", pa.binary()),
                                 ("path", pa.string())])
            elif f["dtype"] == "string":
                typ = pa.string()
            elif f["shape"] == [1]:
                typ = {"float32": pa.float32(), "int64": pa.int64(),
                       "bool": pa.bool_()}[f["dtype"]]
            else:
                typ = pa.list_({"float32": pa.float32(),
                                "int64": pa.int64()}[f["dtype"]],
                               f["shape"][0])
                values = [np.asarray(v).tolist() for v in values]
            arrays.append(pa.array(values, type=typ))
            fields.append(pa.field(key, typ))
        table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
        data_path = root / DATA_PATH.format(chunk_index=0, file_index=0)
        data_path.parent.mkdir(parents=True)
        pq.write_table(table, data_path, compression="snappy")

        episode = {
            "episode_index": self.episode_index, "tasks": [task],
            "length": n, "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": 0, "dataset_to_index": n,
        }
        if self.video:
            for key in self.image_keys:
                path = root / VIDEO_PATH.format(video_key=key, chunk_index=0,
                                                file_index=0)
                path.parent.mkdir(parents=True)
                codec = _encode_mp4(path, self._frames[key], self.fps)
                feats[key]["info"] = {
                    "video.height": feats[key]["shape"][0],
                    "video.width": feats[key]["shape"][1],
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

        stats = self._compute_stats(feats)
        episode.update({"meta/episodes/chunk_index": 0,
                        "meta/episodes/file_index": 0})
        _write_episodes(root, [episode], stats=[stats])
        _write_tasks(root, [task])
        (root / "meta" / "stats.json").write_text(json.dumps(stats))
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "features": feats,
            "total_episodes": 1, "total_frames": n, "total_tasks": 1,
            "chunks_size": CHUNKS_SIZE,
            "data_files_size_in_mb": 100, "video_files_size_in_mb": 200,
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if self.video else None,
            "splits": _splits([self.manifest["split"]["split"]])[0],
        }
        (root / "meta" / "info.json").write_text(
            json.dumps(info, indent=2))
        topo = {
            "spec_version": spec.CANONICAL_SPEC_VERSION,
            "manifest": self.manifest,
            "split": self.manifest["split"],
            "licence": {"data": "MIT", "assets": "procedural, MIT",
                        "attributions": []},
            "episodes": {str(self.episode_index): _jsonable(self._episode)},
        }
        (root / "meta" / "topo.json").write_text(
            json.dumps(_jsonable(topo), indent=2))
        return root

    def _compute_stats(self, feats: dict) -> dict:
        stats = {}
        for key, f in feats.items():
            if f["dtype"] in ("string",):
                continue
            if f["dtype"] in ("image", "video"):
                stats[key] = _stats(np.stack(self._frames[key]), image=True)
                continue
            if key in ("episode_index", "index", "task_index"):
                values = {"episode_index": [self.episode_index],
                          "index": range(len(self._rows)),
                          "task_index": [0]}[key]
                values = np.asarray(list(values), np.float64)
            else:
                values = np.asarray([r[key] for r in self._rows],
                                    dtype=np.float64)
            with np.errstate(invalid="ignore"):
                stats[key] = _stats(np.nan_to_num(values, nan=0.0))
        return stats


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
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return codec


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


def _write_tasks(root: pathlib.Path, tasks: list) -> None:
    pa, pq = _pa()
    table = pa.table({"task_index": pa.array(range(len(tasks)), pa.int64()),
                      "task": pa.array(tasks, pa.string())})
    table = table.replace_schema_metadata({b"pandas": _pandas_meta(
        "task", [("task_index", "int64", "int64"),
                 ("task", "unicode", "object")])})
    pq.write_table(table, root / "meta" / "tasks.parquet")


def _flatten_stats(stats: dict) -> dict:
    return {f"stats/{key}/{name}": value
            for key, per in stats.items() for name, value in per.items()}


def _write_episodes(root: pathlib.Path, episodes: list,
                    stats: list | None = None) -> None:
    pa, pq = _pa()
    rows = [dict(ep) for ep in episodes]
    if stats is not None:
        for row, st in zip(rows, stats):
            row.update(_flatten_stats(st))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    table = pa.Table.from_pylist(
        [{k: row.get(k) for k in keys} for row in rows])
    path = root / EPISODES_PATH.format(chunk_index=0, file_index=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


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


def assemble(episode_dirs: list, out, *,
             group_by_split: bool = True) -> pathlib.Path:
    """Merge one-episode datasets into one LeRobot v3.0 dataset.

    Episodes keep their files: episode ``i`` becomes data file ``i``
    (and video file ``i``) in chunk ``i // 1000``, so nothing is
    re-encoded. Episode, frame and task indices are renumbered; the
    TopoGym extension (``meta/topo.json``) keeps every episode's record.

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
    info = None
    total = 0
    for i, src in enumerate(episode_dirs):
        src_info = json.loads((src / "meta" / "info.json").read_text())
        if info is None:
            info = src_info
        elif src_info["features"].keys() != info["features"].keys() or \
                src_info["fps"] != info["fps"]:
            raise ValueError(f"{src}: features or fps differ from the first "
                             "episode")
        chunk, file = divmod(i, CHUNKS_SIZE)
        table = pq.read_table(src / DATA_PATH.format(chunk_index=0,
                                                     file_index=0))
        n = table.num_rows
        ep_src = pq.read_table(src / EPISODES_PATH.format(
            chunk_index=0, file_index=0)).to_pylist()[0]
        task = ep_src["tasks"][0]
        t_index = tasks.setdefault(task, len(tasks))

        def replace(tbl, name, values, typ):
            return tbl.set_column(tbl.schema.get_field_index(name), name,
                                  pa.array(values, typ))

        table = replace(table, "episode_index", [i] * n, pa.int64())
        table = replace(table, "index", range(total, total + n), pa.int64())
        table = replace(table, "task_index", [t_index] * n, pa.int64())
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
        episodes.append(ep)
        all_stats.append(json.loads((src / "meta" / "stats.json").read_text()))
        topo = json.loads((src / "meta" / "topo.json").read_text())
        for record in topo.get("episodes", {}).values():
            topo_eps[str(i)] = {**record, "source": src.name,
                                "source_position": positions[i],
                                "split": topo.get("split")}
        total += n
    if info is None:
        raise ValueError("no episodes to assemble")
    _write_episodes(out, episodes, stats=all_stats)
    _write_tasks(out, list(tasks))
    (out / "meta" / "stats.json").write_text(
        json.dumps(_merge_stats(all_stats)))
    info.update({"total_episodes": len(episodes), "total_frames": total,
                 "total_tasks": len(tasks),
                 "splits": splits})
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    topo = json.loads((episode_dirs[0] / "meta" / "topo.json").read_text())
    topo.pop("split", None)
    topo["episodes"] = topo_eps
    (out / "meta" / "topo.json").write_text(json.dumps(topo, indent=2))
    return out


def _merge_stats(all_stats: list) -> dict:
    """Count-weighted mean/std, global min/max; quantiles are averaged
    (an approximation, as LeRobot's own aggregation does)."""
    out = {}
    for key in all_stats[0]:
        per = [s[key] for s in all_stats if key in s]
        counts = np.array([p["count"][0] for p in per], np.float64)
        w = counts / counts.sum()
        mean = sum(wi * np.asarray(p["mean"]) for wi, p in zip(w, per))
        var = sum(wi * (np.asarray(p["std"]) ** 2
                        + (np.asarray(p["mean"]) - mean) ** 2)
                  for wi, p in zip(w, per))
        merged = {
            "min": np.min([p["min"] for p in per], axis=0).tolist(),
            "max": np.max([p["max"] for p in per], axis=0).tolist(),
            "mean": np.asarray(mean).tolist(),
            "std": np.sqrt(var).tolist(),
            "count": [int(counts.sum())],
        }
        for q in QUANTILES:
            name = f"q{int(q * 100):02d}"
            merged[name] = sum(wi * np.asarray(p[name])
                               for wi, p in zip(w, per)).tolist()
        out[key] = merged
    return out
