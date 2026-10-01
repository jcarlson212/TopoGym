"""0.7.0: follow-ups from migrating a continuous-world producer."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from topogym.canonical import spec
from topogym.canonical.export import EpisodeWriter, _quiet_stderr
from topogym.canonical.manifest import manifest, manifest_from_features
from topogym.canonical.reader import read_episode
from topogym.canonical.spec import FeatureSpec

pytest.importorskip("pyarrow")

LICENCE = {"profile": "t", "class": "open", "spdx": "MIT",
           "internal_only": False, "attributions": []}


def depth_writer(tmp_path, name, shape, info=None):
    feats = [FeatureSpec(spec.depth_key("head"), "float32", shape,
                         storage="depth_png16", info=info or {}),
             FeatureSpec("next.reward", "float32", (1,))]
    return EpisodeWriter(tmp_path / name, feats, licence=LICENCE)


def depth_frame(shape):
    d = np.full(shape, 2.0, np.float32)
    d[(0,) * len(shape)] = 0.0  # no return, as the live sensor says it
    return {spec.depth_key("head"): d, "next.reward": np.float32(0)}


# -- depth reads back as observed ----------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 5), (4, 5, 1)])
def test_depth_keeps_its_declared_shape(tmp_path, shape):
    w = depth_writer(tmp_path, "d", shape)
    w.add_frame(depth_frame(shape), task="x")
    row = read_episode(w.close())["rows"][0]
    assert row[spec.depth_key("head")].shape == shape


def test_datasets_without_a_declared_shape_read_as_before(tmp_path):
    w = depth_writer(tmp_path, "d", (4, 5, 1))
    w.add_frame(depth_frame((4, 5, 1)), task="x")
    root = w.close()
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["features"][spec.depth_key("head")]["info"][
        "array.declared_shape"]  # as 0.6.0 wrote it
    info_path.write_text(json.dumps(info))
    assert read_episode(root)["rows"][0][spec.depth_key("head")].shape == \
        (4, 5)


def test_missing_depth_reads_back_as_the_declared_live_value(tmp_path):
    live = {spec.DEPTH_INVALID_VALUE_KEY: 0.0}
    w = depth_writer(tmp_path, "zero", (4, 5), info=live)
    frame = depth_frame((4, 5))
    w.add_frame(frame, task="x")
    got = read_episode(w.close())["rows"][0][spec.depth_key("head")]
    assert np.array_equal(got, frame[spec.depth_key("head")])
    w = depth_writer(tmp_path, "nan", (4, 5))  # undeclared: NaN, as 0.6.0
    w.add_frame(depth_frame((4, 5)), task="x")
    got = read_episode(w.close())["rows"][0][spec.depth_key("head")]
    assert np.isnan(got[0, 0]) and got[1, 1] == pytest.approx(2.0)


# -- close is atomic, abandon discards ----------------------------------------------


def test_a_failed_close_leaves_the_previous_dataset(tmp_path):
    w = depth_writer(tmp_path, "ep", (4, 5))
    w.add_frame(depth_frame((4, 5)), task="first")
    root = w.close()
    before = sorted(p.relative_to(root) for p in root.rglob("*"))

    class Broken(EpisodeWriter):
        def topo_record(self):
            raise OSError("disk full")

    b = Broken(root, list(w.features.values()), licence=LICENCE)
    b.add_frame(depth_frame((4, 5)), task="second")
    with pytest.raises(OSError, match="disk full"):
        b.close()
    assert sorted(p.relative_to(root) for p in root.rglob("*")) == before
    assert read_episode(root)["tasks"] == ["first"]
    assert [p.name for p in tmp_path.iterdir()] == ["ep"]  # no temp dirs


def test_close_replaces_an_existing_dataset(tmp_path):
    for task in ("first", "second"):
        w = depth_writer(tmp_path, "ep", (4, 5))
        w.add_frame(depth_frame((4, 5)), task=task)
        root = w.close()
    assert read_episode(root)["tasks"] == ["second"]
    assert [p.name for p in tmp_path.iterdir()] == ["ep"]


def test_abandon(tmp_path):
    w = depth_writer(tmp_path, "ep", (4, 5))
    w.add_frame(depth_frame((4, 5)), task="x")
    w.abandon()
    assert not (tmp_path / "ep").exists()
    assert w.state == "abandoned"
    with pytest.raises(RuntimeError, match="abandoned"):
        w.add_frame(depth_frame((4, 5)), task="x")
    with pytest.raises(RuntimeError, match="abandoned"):
        w.close()


def test_a_closed_writer_refuses_more(tmp_path):
    w = depth_writer(tmp_path, "ep", (4, 5))
    w.add_frame(depth_frame((4, 5)), task="x")
    w.close()
    with pytest.raises(RuntimeError, match="closed"):
        w.add_frame(depth_frame((4, 5)), task="x")


# -- clock --------------------------------------------------------------------------


def test_decision_clock_is_declared(tmp_path):
    w = depth_writer(tmp_path, "ep", (4, 5))
    assert w.fps == spec.DECISION_FPS and w.clock == "decision"
    w.add_frame(depth_frame((4, 5)), task="x")
    topo = read_episode(w.close())["topo"]
    assert topo["clock"] == {"kind": "decision", "fps": spec.DECISION_FPS,
                             "timestamp": "frame_index / fps"}
    with pytest.raises(ValueError, match="clock"):
        EpisodeWriter(tmp_path / "y", list(w.features.values()),
                      licence=LICENCE, clock="wall")


# -- video -------------------------------------------------------------------------


def test_quiet_stderr_silences_native_writes(capfd):
    with _quiet_stderr():
        os.write(2, b"svt noise\n")
    os.write(2, b"after\n")
    err = capfd.readouterr().err
    assert "svt noise" not in err and "after" in err


def test_video_codec_is_selectable(tmp_path):
    pytest.importorskip("av")
    feats = [FeatureSpec(spec.HEAD, "uint8", (16, 16, 3), storage="video")]
    w = EpisodeWriter(tmp_path / "v", feats, licence=LICENCE,
                      video_codec="mpeg4", video_options={"qscale": 2})
    for t in range(3):
        w.add_frame({spec.HEAD: np.full((16, 16, 3), 40 * t, np.uint8)},
                    task="x")
    info = read_episode(w.close())["info"]
    assert info["features"][spec.HEAD]["info"]["video.codec"] == "mpeg4"
    bad = EpisodeWriter(tmp_path / "b", feats, licence=LICENCE,
                        video_codec="no-such-codec")
    bad.add_frame({spec.HEAD: np.zeros((16, 16, 3), np.uint8)}, task="x")
    with pytest.raises(ValueError, match="not available"):
        bad.close()


# -- spec additions ---------------------------------------------------------------


def test_headingless_goals_stay_comparable():
    from gymnasium.utils.env_checker import data_equivalence

    pose, mask = spec.goal_pose_3d(1.0, 2.0, 0.5)
    assert pose == (1.0, 2.0, 0.5, spec.GOAL_YAW_NONE) and mask == (False,)
    a = np.asarray(pose, np.float32)
    assert data_equivalence({"goal": a}, {"goal": a.copy()})
    nan = np.asarray([1.0, 2.0, 0.5, np.nan], np.float32)
    assert not data_equivalence({"goal": nan}, {"goal": nan.copy()})
    assert spec.goal_pose_3d(1, 2, 3, 0.5)[1] == (True,)
    ok = [FeatureSpec("privileged.goal_pose", "float32", (4,),
                      spec.GOAL_POSE_NAMES_3D),
          FeatureSpec("privileged.goal_pose_mask", "bool", (1,),
                      spec.GOAL_POSE_MASK_NAMES)]
    assert spec.validate_features(ok) == []


def test_names_for_goto_velocities_and_native_outcomes():
    assert spec.GOTO_NAMES_3D == ("x", "y", "z", "yaw")
    assert set(spec.GOTO_NAMES_2D) <= set(spec.GOTO_UNITS)
    assert spec.STATE_VELOCITY_NAMES_3D == ("ego.vx", "ego.vy", "ego.vz",
                                            "ego.wz")
    assert set(spec.STATE_VELOCITY_NAMES_3D) <= set(
        spec.STATE_VELOCITY_UNITS)
    assert spec.STATE_NAMES_3D == ("ego.x", "ego.y", "ego.z", "ego.yaw",
                                   "ego.pitch")  # unchanged
    assert spec.check_key(spec.NEXT_NATIVE_PREFIX + "collision") is None
    assert spec.check_key(spec.GOTO) is None


def test_manifest_from_features_needs_no_env():
    feats = [FeatureSpec(spec.HEAD, "uint8", (8, 8, 3), storage="png"),
             FeatureSpec(spec.STATE, "float32", (9,),
                         spec.STATE_NAMES_3D + spec.STATE_VELOCITY_NAMES_3D),
             FeatureSpec("privileged.world_pose", "float32", (4,),
                         spec.WORLD_POSE_NAMES_3D),
             FeatureSpec(spec.privileged_ext_key("acme", "wind"), "float32",
                         (3,))]
    m = manifest(feats, producer={"name": "acme-sim", "version": "2.1"},
                 body={"kind": "drone", "can_fly": True},
                 actions={"level": "waypoint"},
                 ext={"acme": {"wind": "m/s"}})
    assert m is not None and m == manifest_from_features(
        feats, producer={"name": "acme-sim", "version": "2.1"},
        body={"kind": "drone", "can_fly": True},
        actions={"level": "waypoint"}, ext={"acme": {"wind": "m/s"}})
    assert set(m["features"]) == {spec.HEAD, spec.STATE}
    assert m["privileged"]["world_pose"]["names"] == \
        list(spec.WORLD_POSE_NAMES_3D)
    assert m["privileged"]["ext.acme"] == {"wind": "m/s"}
    assert m["body"]["can_fly"] is True
    json.dumps(m)
    with pytest.raises(ValueError, match="producer"):
        manifest(feats, producer={})
