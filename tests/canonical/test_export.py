"""Per-episode LeRobot v3.0 export (the ``[export]`` extra)."""

from __future__ import annotations

import json
import zlib

import gymnasium as gym
import numpy as np
import pytest

import topogym  # noqa: F401
from topogym.canonical import spec
from topogym.canonical.export import EpisodeWriter, assemble, encode_png

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

ACTIONS = [2, 2, 0, 2, 2, 1, 2, 2, 2, 0, 2]


def record(tmp_path, name, seed, actions="words", **kw):
    env = gym.make("TopoGym/Dilution-50-v0", seed=seed, obs_mode="canonical",
                   actions=actions, **kw)
    writer = EpisodeWriter(tmp_path / name, env)
    obs, info = env.reset(seed=seed)
    writer.start(obs, info)
    for a in ACTIONS:
        if actions == "waypoint":
            a = np.array([1, 0, 0, 0], np.float32)
        obs, r, term, trunc, info = env.step(a)
        writer.add(a, r, term, trunc, info, next_obs=obs)
        if term or trunc:
            break
    return writer.close()


def test_png_roundtrip_structure():
    img = (np.arange(4 * 5 * 3) % 256).astype(np.uint8).reshape(4, 5, 3)
    png = encode_png(img)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # IDAT decompresses to filter byte + RGB per row.
    start = png.index(b"IDAT") + 4
    length = int.from_bytes(png[start - 8:start - 4], "big")
    raw = zlib.decompress(png[start:start + length])
    rows = np.frombuffer(raw, np.uint8).reshape(4, 1 + 15)
    assert (rows[:, 0] == 0).all()
    assert np.array_equal(rows[:, 1:].reshape(4, 5, 3), img)


def test_episode_layout(tmp_path):
    root = record(tmp_path, "ep0", seed=0)
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 1
    n = info["total_frames"]
    assert n == len(ACTIONS)
    for key in ("timestamp", "frame_index", "episode_index", "index",
                "task_index"):
        assert key in info["features"]
    assert info["features"][spec.HEAD]["dtype"] == "image"
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    assert table.num_rows == n
    assert set(info["features"]) == set(table.column_names) - set()
    row = table.slice(0, 1).to_pylist()[0]
    assert row[spec.HEAD]["bytes"][:4] == b"\x89PNG"
    assert row["is_first"] and row["frame_index"] == 0
    assert row[spec.INSTRUCTION].startswith("Go to")
    assert row["privileged.region_id"] == 0
    eps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet")
    ep = eps.to_pylist()[0]
    assert ep["length"] == n and ep["dataset_to_index"] == n
    assert "stats/observation.state/mean" in ep
    topo = json.loads((root / "meta" / "topo.json").read_text())
    assert topo["spec_version"] == spec.CANONICAL_SPEC_VERSION
    assert topo["manifest"]["actions"]["level"] == "words"
    stats = json.loads((root / "meta" / "stats.json").read_text())
    assert np.asarray(stats[spec.HEAD]["mean"]).shape == (3, 1, 1)


def test_tasks_read_back_with_pandas(tmp_path):
    pd = pytest.importorskip("pandas")
    root = record(tmp_path, "ep0", seed=0)
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    assert tasks.index.name == "task"
    assert list(tasks["task_index"]) == [0]
    assert tasks.index[0].startswith("Go to")


def test_waypoint_level(tmp_path):
    root = record(tmp_path, "wp", seed=1, actions="waypoint")
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["features"]["action"]["shape"] == [4]


def test_assemble(tmp_path):
    a = record(tmp_path, "a", seed=0)
    b = record(tmp_path, "b", seed=1, phrasing="paraphrase:1")
    out = assemble([a, b], tmp_path / "all")
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2 and info["total_tasks"] == 2
    t0 = pq.read_table(out / "data/chunk-000/file-000.parquet")
    t1 = pq.read_table(out / "data/chunk-000/file-001.parquet")
    assert t1.column("episode_index").to_pylist() == [1] * t1.num_rows
    assert t1.column("index").to_pylist()[0] == t0.num_rows
    assert t1.column("task_index").to_pylist()[0] == 1
    eps = pq.read_table(
        out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [e["data/file_index"] for e in eps] == [0, 1]
    assert eps[1]["dataset_from_index"] == t0.num_rows
    topo = json.loads((out / "meta" / "topo.json").read_text())
    assert set(topo["episodes"]) == {"0", "1"}


# -- splits ---------------------------------------------------------------------------


def test_episode_records_its_split(tmp_path):
    """Split-band seeds carry their split; a world in no band (the
    canonical seed) is listed under none rather than mislabelled."""
    for seed, want in ((2001, {"train": "0:1"}), (3001, {"val": "0:1"}),
                       (0, {})):
        root = record(tmp_path, f"s{seed}", seed=seed)
        info = json.loads((root / "meta" / "info.json").read_text())
        assert info["splits"] == want


def test_assemble_writes_split_ranges(tmp_path):
    eps = [record(tmp_path, f"e{s}", seed=s) for s in (2000, 2001, 3000, 0)]
    out = assemble(eps, tmp_path / "all")
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["splits"] == {"train": "0:2", "val": "2:3"}


def test_assemble_interleaved_splits(tmp_path):
    """0.5.0 assembled interleaved splits; a patch release must too.
    By default they are grouped (with a warning, keeping each episode's
    original position); on request the order is kept and the split that
    cannot be a range is left out, with a warning."""
    eps = [record(tmp_path, f"e{s}", seed=s) for s in (2000, 3000, 2001)]
    with pytest.warns(UserWarning, match="reordered"):
        out = assemble(eps, tmp_path / "grouped")
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["splits"] == {"train": "0:2", "val": "2:3"}
    topo = json.loads((out / "meta" / "topo.json").read_text())
    records = [topo["episodes"][str(i)] for i in range(3)]
    assert [r["split"]["split"] for r in records] == ["train", "train", "val"]
    assert [r["source_position"] for r in records] == [0, 2, 1]
    with pytest.warns(UserWarning, match="not contiguous"):
        out = assemble(eps, tmp_path / "kept", group_by_split=False)
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["splits"] == {"val": "1:2"}
    topo = json.loads((out / "meta" / "topo.json").read_text())
    assert [topo["episodes"][str(i)]["split"]["split"] for i in range(3)] \
        == ["train", "val", "train"]


def test_assemble_grouped_input_is_not_reordered(tmp_path):
    import warnings

    eps = [record(tmp_path, f"e{s}", seed=s) for s in (2000, 2001, 3000)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = assemble(eps, tmp_path / "all")
    topo = json.loads((out / "meta" / "topo.json").read_text())
    assert [topo["episodes"][str(i)]["source_position"] for i in range(3)] \
        == [0, 1, 2]
