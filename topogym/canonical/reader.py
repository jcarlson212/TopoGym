"""Read datasets written by :mod:`topogym.canonical.export`.

:func:`read_dataset` is the inverse of the writer and of
:func:`~topogym.canonical.export.assemble`: it returns every row with
its features decoded back to their declared types (images to uint8
arrays, depth to float32 metres with NaN for no return, segmentation to
int32 ids, fixed-shape arrays to numpy arrays), the tasks, the
per-episode metadata and records, and the side streams.

Needs pyarrow (``[export]``); JPEG frames need Pillow and video frames
PyAV (``[jpeg]``, ``[video]``), and only when those features are read.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from topogym.canonical import codecs, spec
from topogym.canonical.export import (
    SIDE_PATH,
    VIDEO_PATH,
    _pa,
    _read_task_list,
)


def _decode(feature: dict, value, *, key: str):
    info = feature.get("info") or {}
    dtype = feature["dtype"]
    if dtype == "image":
        data = value["bytes"] if isinstance(value, dict) else value
        storage = info.get("storage", "png")
        if storage in ("depth_png16", "segmentation_png16", "array_png16"):
            if storage == "depth_png16":
                # The producer's declared live value for "no depth"
                # (e.g. 0.0); NaN when undeclared, as in 0.6.0.
                invalid = info.get(spec.DEPTH_INVALID_VALUE_KEY)
                arr = codecs.decode_depth_png(
                    data, float("nan") if invalid is None else invalid)
            elif storage == "segmentation_png16":
                arr = codecs.decode_segmentation(codecs.decode_png(data))
            else:
                arr = codecs.decode_array16(codecs.decode_png(data))
            # Back to the shape the producer declared: (H, W) or
            # (H, W, 1). Datasets from 0.6.0 did not record it: (H, W).
            declared = info.get("array.declared_shape")
            return arr.reshape(declared) if declared else arr
        img = codecs.decode_image(data)
        if img.ndim == 2 and feature["shape"][-1] == 1:
            img = img[..., None]
        return img
    if dtype == "string":
        return value
    shape = feature["shape"]
    if list(shape) == [1]:
        return value
    if any(s is None for s in shape):
        arr = np.asarray(value, dtype=dtype)
        if shape[0] is None and all(s is not None for s in shape[1:]):
            # A variable-length list of fixed-shape items keeps the
            # item shape even when empty.
            arr = arr.reshape((-1,) + tuple(shape[1:]))
        return arr
    return np.asarray(value, dtype=dtype).reshape(shape)


def _video_frames(path: pathlib.Path) -> list:
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "reading video features needs PyAV: pip install "
            "'topogym[video]'") from exc
    with av.open(str(path)) as container:
        return [f.to_ndarray(format="rgb24")
                for f in container.decode(video=0)]


def read_dataset(path, *, decode_video: bool = True) -> dict:
    """Everything in a dataset, decoded.

    Returns ``{"info", "features", "tasks", "episodes", "topo",
    "rows", "side"}``: ``tasks`` in task_index order; ``episodes`` one
    metadata row per episode (without the stats columns), each with its
    canonical ``record`` from meta/topo.json; ``rows`` one dict per
    frame in index order, including ``task`` (the string); ``side``
    ``{stream: [tick dicts]}``.
    """
    pa, pq = _pa()
    root = pathlib.Path(path)
    info = json.loads((root / "meta" / "info.json").read_text())
    topo_path = root / "meta" / "topo.json"
    topo = json.loads(topo_path.read_text()) if topo_path.exists() else {}
    features = info["features"]
    tasks = _read_task_list(root)
    episodes = []
    for p in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        episodes.extend(pq.read_table(p).to_pylist())
    episodes.sort(key=lambda e: e["episode_index"])
    records = topo.get("episodes", {})
    for ep in episodes:
        for k in [k for k in ep if k.startswith("stats/")]:
            del ep[k]
        ep["record"] = records.get(str(ep["episode_index"]))

    rows = []
    for ep in episodes:
        chunk, file = ep["data/chunk_index"], ep["data/file_index"]
        table = pq.read_table(root / info["data_path"].format(
            chunk_index=chunk, file_index=file))
        data = table.to_pylist()
        videos = {}
        if decode_video:
            for key, f in features.items():
                if f["dtype"] == "video":
                    videos[key] = _video_frames(root / VIDEO_PATH.format(
                        video_key=key,
                        chunk_index=ep[f"videos/{key}/chunk_index"],
                        file_index=ep[f"videos/{key}/file_index"]))
        for t, raw in enumerate(data):
            row = {}
            for key, f in features.items():
                if f["dtype"] == "video":
                    if decode_video:
                        row[key] = videos[key][t]
                    continue
                row[key] = _decode(f, raw[key], key=key)
            row["task"] = tasks[raw["task_index"]]
            rows.append(row)
    rows.sort(key=lambda r: r["index"])

    side = {}
    for stream in (topo.get("side_streams") or {}):
        ticks = []
        for ep in episodes:
            chunk, file = ep["data/chunk_index"], ep["data/file_index"]
            p = root / SIDE_PATH.format(stream=stream, chunk_index=chunk,
                                        file_index=file)
            if p.exists():
                ticks.extend(pq.read_table(p).to_pylist())
        side[stream] = ticks
    return {"info": info, "features": features, "tasks": tasks,
            "episodes": episodes, "topo": topo, "rows": rows, "side": side}


def read_episode(path, *, decode_video: bool = True) -> dict:
    """A one-episode dataset (as the writer produces): the
    :func:`read_dataset` result plus ``record``, the episode's canonical
    record (instruction, topology, segmentation tables, ext, ...)."""
    out = read_dataset(path, decode_video=decode_video)
    if len(out["episodes"]) != 1:
        raise ValueError(f"{path} holds {len(out['episodes'])} episodes; "
                         "use read_dataset")
    out["record"] = out["episodes"][0]["record"]
    return out
