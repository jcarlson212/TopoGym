"""The feature-driven writer, the reader, and assemble (0.6)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from topogym.canonical import codecs, spec
from topogym.canonical.export import (
    EpisodeWriter,
    assemble,
    write_episodes,
    write_tasks,
)
from topogym.canonical.reader import read_dataset, read_episode
from topogym.canonical.spec import FeatureSpec

pytest.importorskip("pyarrow")

LICENCE_A = {"profile": "producer-a", "class": "open", "spdx": "CC-BY-4.0",
             "internal_only": False,
             "attributions": ["Scene meshes by A, CC-BY-4.0"]}
LICENCE_B = {"profile": "producer-b", "class": "restricted", "spdx": None,
             "internal_only": True, "attributions": []}

H, W = 12, 16


def features_3d():
    return [
        FeatureSpec(spec.HEAD, "uint8", (H, W, 3),
                    ("height", "width", "channels"), storage="png"),
        FeatureSpec(spec.depth_key("head"), "float32", (H, W),
                    storage="depth_png16"),
        FeatureSpec(spec.segmentation_key("head"), "int32", (H, W),
                    storage="segmentation_png16"),
        FeatureSpec(spec.INSTRUCTION, "string", (1,)),
        FeatureSpec(spec.STATE, "float32", (5,), spec.STATE_NAMES_3D,
                    units=dict(spec.STATE_UNITS_3D)),
        FeatureSpec("action", "float32", (4,), spec.WAYPOINT_NAMES),
        FeatureSpec("next.reward", "float32", (1,)),
        FeatureSpec("next.done", "bool", (1,)),
        FeatureSpec("privileged.world_pose", "float32", (4,),
                    spec.WORLD_POSE_NAMES_3D),
        FeatureSpec("privileged.camera_pose", "float32", (7,),
                    spec.POSE7_NAMES),
        FeatureSpec("privileged.d_goal", "float32", (1,),
                    info={"units": "m"}),
        FeatureSpec("privileged.region_id", "int64", (1,),
                    info={"missing": spec.REGION_NONE}),
        FeatureSpec(spec.privileged_ext_key("acme", "wind"), "float32", (3,)),
        FeatureSpec("observation.native.sim_time", "float64", (1,)),
        FeatureSpec("observation.native.contacts", "float32", (None, 3)),
    ]


def frame(t, rng):
    depth = rng.uniform(0.3, 80.0, (H, W)).astype(np.float32)
    depth[0, 0] = np.nan  # no return
    seg = rng.integers(0, 5, (H, W)).astype(np.int32)
    return {
        spec.HEAD: rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
        spec.depth_key("head"): depth,
        spec.segmentation_key("head"): seg,
        spec.INSTRUCTION: "Find the cave.",
        spec.STATE: rng.normal(size=5).astype(np.float32),
        "action": rng.normal(size=4).astype(np.float32),
        "next.reward": np.float32(t == 3),
        "next.done": t == 3,
        "privileged.world_pose": rng.normal(size=4).astype(np.float32),
        "privileged.camera_pose": np.array([1, 2, 3, 0, 0, 0, 1],
                                           np.float32),
        "privileged.d_goal": np.float32(math.nan if t == 0 else 4.5 - t),
        "privileged.region_id": -1 if t < 2 else 3,
        spec.privileged_ext_key("acme", "wind"): np.float32([t, 0, 1]),
        "observation.native.sim_time": 0.013 * t,
        "observation.native.contacts": np.ones((t, 3), np.float32),
    }


def write(tmp_path, name, *, licence=LICENCE_A, split="train", seed=0,
          episode_index=0, ext=None, n=4):
    rng = np.random.default_rng(seed)
    w = EpisodeWriter(tmp_path / name, features_3d(), fps=5,
                      licence=licence, split=split, seed=seed,
                      episode_index=episode_index,
                      ext=ext or {"acme": {"map": f"level-{seed}"}})
    frames = []
    for t in range(n):
        fr = frame(t, rng)
        frames.append(fr)
        w.add_frame(fr, task="Find the cave.")
        for tick in range(3):
            w.add_side("controls", {"thrust": np.float32([tick, t]),
                                    "sim_time": 0.013 * t + 0.004 * tick})
    w.set_segmentation_table(spec.segmentation_key("head"), {
        k: {"category": "cave" if k else None, "label": f"id{k}"}
        for k in range(5)})
    w.episode_record["ext"] = {"acme": {"episode_note": seed}}
    return w.close(), frames


def assert_row(row, fr):
    assert np.array_equal(row[spec.HEAD], fr[spec.HEAD])
    depth = fr[spec.depth_key("head")]
    got = row[spec.depth_key("head")]
    ok = np.isfinite(depth)
    assert np.isnan(got[~ok]).all()
    assert np.abs(got[ok] - np.clip(depth[ok], 0, 65.534)).max() <= 1e-3
    assert np.array_equal(row[spec.segmentation_key("head")],
                          fr[spec.segmentation_key("head")])
    for key in (spec.STATE, "action", "privileged.world_pose",
                "privileged.camera_pose",
                spec.privileged_ext_key("acme", "wind")):
        assert np.allclose(row[key], fr[key])
    assert np.array_equal(row["observation.native.contacts"],
                          fr["observation.native.contacts"])
    d = fr["privileged.d_goal"]
    assert (math.isnan(row["privileged.d_goal"]) if math.isnan(d)
            else row["privileged.d_goal"] == pytest.approx(d))
    assert row["privileged.region_id"] == fr["privileged.region_id"]
    assert row["next.done"] == fr["next.done"]


def test_round_trip_one_episode(tmp_path):
    root, frames = write(tmp_path, "ep", seed=1)
    ep = read_episode(root)
    assert len(ep["rows"]) == 4
    for t, (row, fr) in enumerate(zip(ep["rows"], frames)):
        assert_row(row, fr)
        assert row["timestamp"] == pytest.approx(t / 5)
        assert row["task"] == "Find the cave."
    info = ep["info"]
    head_depth = info["features"][spec.depth_key("head")]
    assert head_depth["dtype"] == "image" and head_depth["info"][
        "is_depth_map"] and head_depth["info"]["depth.unit_m"] == 0.002
    assert info["splits"] == {"train": "0:1"}
    assert ep["record"]["segmentation"][spec.segmentation_key("head")][
        "1"]["category"] == "cave"
    assert ep["topo"]["licence"] == LICENCE_A
    assert ep["topo"]["ext"]["acme"]["map"] == "level-1"
    assert len(ep["side"]["controls"]) == 12
    assert ep["side"]["controls"][4]["frame_index"] == 1


def test_assemble_round_trip_keeps_each_episode(tmp_path):
    a, fa = write(tmp_path, "a", seed=1, split="train")
    b, fb = write(tmp_path, "b", seed=2, split="train", licence=LICENCE_B)
    c, fc = write(tmp_path, "c", seed=3, split="val")
    out = assemble([a, b, c], tmp_path / "all")
    ds = read_dataset(out)
    assert ds["info"]["splits"] == {"train": "0:2", "val": "2:3"}
    for row, fr in zip(ds["rows"], fa + fb + fc):
        assert_row(row, fr)
    assert [r["episode_index"] for r in ds["rows"]] == \
        [0] * 4 + [1] * 4 + [2] * 4
    assert [r["index"] for r in ds["rows"]] == list(range(12))
    recs = [ep["record"] for ep in ds["episodes"]]
    assert [r["licence"]["profile"] for r in recs] == \
        ["producer-a", "producer-b", "producer-a"]
    assert ds["topo"]["licence"]["mixed"] is True
    assert [r["ext"]["acme"]["map"] for r in recs] == \
        ["level-1", "level-2", "level-3"]
    assert [r["ext"]["acme"]["episode_note"] for r in recs] == [1, 2, 3]
    side = ds["side"]["controls"]
    assert len(side) == 36
    assert sorted({t["episode_index"] for t in side}) == [0, 1, 2]
    assert (tmp_path / "all" / "native" / "controls" / "chunk-000" /
            "file-002.parquet").exists()


def test_assemble_refuses_type_and_storage_mismatches(tmp_path):
    a, _ = write(tmp_path, "a", seed=1)
    feats = features_3d()
    feats[0] = FeatureSpec(spec.HEAD, "uint8", (H, W, 3), storage="jpeg")
    pytest.importorskip("PIL")
    rng = np.random.default_rng(0)
    w = EpisodeWriter(tmp_path / "b", feats, fps=5, licence=LICENCE_A)
    for t in range(2):
        w.add_frame(frame(t, rng), task="x")
    b = w.close()
    with pytest.raises(ValueError, match="observation.images.head"):
        assemble([a, b], tmp_path / "all")


def test_assemble_refuses_dtype_mismatch(tmp_path):
    a, _ = write(tmp_path, "a", seed=1)
    feats = [f if f.key != spec.STATE else
             FeatureSpec(spec.STATE, "float64", (5,), spec.STATE_NAMES_3D,
                         units=dict(spec.STATE_UNITS_3D))
             for f in features_3d()]
    rng = np.random.default_rng(0)
    w = EpisodeWriter(tmp_path / "b", feats, fps=5, licence=LICENCE_A)
    w.add_frame(frame(1, rng), task="x")
    b = w.close()
    with pytest.raises(ValueError, match="observation.state"):
        assemble([a, b], tmp_path / "all")


# -- conventions the writer enforces ---------------------------------------------------


def test_none_is_refused_in_numeric_features(tmp_path):
    w = EpisodeWriter(tmp_path / "x", features_3d(), licence=LICENCE_A)
    fr = frame(1, np.random.default_rng(0))
    fr["privileged.d_goal"] = None
    with pytest.raises(ValueError, match="NaN"):
        w.add_frame(fr, task="x")


def test_timestamp_is_the_writers(tmp_path):
    w = EpisodeWriter(tmp_path / "x", features_3d(), licence=LICENCE_A)
    fr = frame(1, np.random.default_rng(0))
    fr["timestamp"] = 0.5
    with pytest.raises(ValueError, match="frame_index / fps"):
        w.add_frame(fr, task="x")


def test_licence_is_required(tmp_path):
    with pytest.raises(ValueError, match="licence is required"):
        EpisodeWriter(tmp_path / "x", features_3d())


def test_privileged_keys_outside_the_namespace_are_refused(tmp_path):
    feats = features_3d() + [FeatureSpec("privileged.wind", "float32", (3,))]
    with pytest.raises(ValueError, match="privileged.ext"):
        EpisodeWriter(tmp_path / "x", feats, licence=LICENCE_A)


def test_depth_codec_convention():
    d = np.array([[0.0, 0.002, 1.0], [65.534, 100.0, np.nan]], np.float32)
    codes = codecs.encode_depth(d)
    assert codes.tolist() == [[0, 1, 500], [32767, 32767, 0]]
    back = codecs.decode_depth(codes.astype(np.int16))  # as LeRobot decodes
    assert np.isnan(back[0, 0]) and np.isnan(back[1, 2])
    assert back[1, 0] == pytest.approx(65.534)
    png = codecs.encode_depth_png(d)
    assert np.array_equal(codecs.decode_png(png), codes)


def test_segmentation_ids_are_capped():
    with pytest.raises(ValueError, match="32767"):
        codecs.encode_segmentation(np.array([[40000]]))


def test_public_metadata_writers(tmp_path):
    pd = pytest.importorskip("pandas")
    write_tasks(tmp_path, ["a", "b"])
    tasks = pd.read_parquet(tmp_path / "meta" / "tasks.parquet")
    assert tasks.index.name == "task" and list(tasks.index) == ["a", "b"]
    assert list(tasks["task_index"]) == [0, 1]
    path = write_episodes(tmp_path, [{"episode_index": 0, "length": 3}])
    assert path.name == "file-000.parquet"
    with pytest.raises(ValueError, match="unique"):
        write_tasks(tmp_path, ["a", "a"])


def test_writer_hooks_are_public(tmp_path):
    """A subclass extends the writer through public hooks only."""

    class Tagged(EpisodeWriter):
        def episode_metadata(self):
            return {"producer/scene": "kitchen"}

        def topo_record(self):
            out = super().topo_record()
            out["ext"]["acme"]["hooked"] = True
            return out

        def on_close(self, root):
            (root / "native" / "readme.txt").parent.mkdir(exist_ok=True)
            (root / "native" / "readme.txt").write_text("hi")

    w = Tagged(tmp_path / "x", features_3d(), licence=LICENCE_A,
               ext={"acme": {}})
    w.add_frame(frame(1, np.random.default_rng(0)), task="x")
    root = w.close()
    ep = read_episode(root)
    assert ep["episodes"][0]["producer/scene"] == "kitchen"
    assert ep["topo"]["ext"]["acme"]["hooked"] is True
    assert (root / "native" / "readme.txt").read_text() == "hi"
    assert isinstance(w.frames, list) and isinstance(w.features, dict)


def test_json_metadata_uses_null(tmp_path):
    root, _ = write(tmp_path, "ep", seed=1)
    stats = json.loads((root / "meta" / "stats.json").read_text())
    assert "NaN" not in (root / "meta" / "stats.json").read_text()
    assert stats["privileged.d_goal"]["count"] == [4]


def test_video_and_jpeg_round_trip(tmp_path):
    """Lossy storages round-trip approximately and stay aligned."""
    pytest.importorskip("av")
    pytest.importorskip("PIL")
    feats = [FeatureSpec(spec.HEAD, "uint8", (32, 32, 3), storage="video"),
             FeatureSpec("observation.images.wrist", "uint8", (32, 32, 3),
                         storage="jpeg"),
             FeatureSpec("next.reward", "float32", (1,))]
    rng = np.random.default_rng(0)
    frames = []
    w = EpisodeWriter(tmp_path / "v", feats, fps=5, licence=LICENCE_A)
    for t in range(6):
        img = np.zeros((32, 32, 3), np.uint8)
        img[:, : 5 * (t + 1)] = 200  # a bar that grows each frame
        fr = {spec.HEAD: img, "observation.images.wrist": img,
              "next.reward": np.float32(rng.random())}
        frames.append(fr)
        w.add_frame(fr, task="x")
    ds = read_episode(w.close())
    assert ds["info"]["features"][spec.HEAD]["dtype"] == "video"
    for row, fr in zip(ds["rows"], frames):
        for key in (spec.HEAD, "observation.images.wrist"):
            err = np.abs(row[key].astype(int) - fr[key].astype(int)).mean()
            assert err < 12, (key, err)
