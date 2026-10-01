"""Spec 1.2: conventions, registries, grammar, transforms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from topogym.canonical import spec
from topogym.canonical import transforms as tf
from topogym.canonical.spec import Category, FeatureSpec, SeenCell

# -- transforms -------------------------------------------------------------------------


def test_rep103_opencv_axes():
    # Body forward is the camera's optical axis; body left is camera -x;
    # body up is camera -y.
    assert tf.rep103_to_opencv([1, 0, 0]).tolist() == [0, 0, 1]
    assert tf.rep103_to_opencv([0, 1, 0]).tolist() == [-1, 0, 0]
    assert tf.rep103_to_opencv([0, 0, 1]).tolist() == [0, -1, 0]
    v = np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(tf.opencv_to_rep103(tf.rep103_to_opencv(v)), v)
    assert np.allclose(tf.quat_to_matrix(tf.Q_REP103_FROM_OPENCV),
                       tf.R_REP103_FROM_OPENCV)


def test_quaternions():
    q = tf.quat_from_euler(0.3, -0.2, 1.1)
    assert np.allclose(tf.quat_to_euler(q), [0.3, -0.2, 1.1])
    assert np.allclose(tf.matrix_to_quat(tf.quat_to_matrix(q)), q)
    yaw = tf.quat_from_yaw(math.pi / 2)
    assert np.allclose(tf.quat_rotate(yaw, [1, 0, 0]), [0, 1, 0])
    assert np.allclose(tf.quat_multiply(q, tf.quat_conjugate(q)),
                       [0, 0, 0, 1])


def test_pose7_composition():
    rng = np.random.default_rng(1)
    a = tf.pose7(rng.normal(size=3), tf.quat_from_euler(*rng.normal(size=3)))
    b = tf.pose7(rng.normal(size=3), tf.quat_from_euler(*rng.normal(size=3)))
    ab = tf.pose7_compose(a, b)
    assert np.allclose(tf.pose7_to_matrix(ab),
                       tf.pose7_to_matrix(a) @ tf.pose7_to_matrix(b))
    assert np.allclose(tf.pose7_compose(a, tf.pose7_inverse(a)),
                       tf.IDENTITY_POSE7)
    p = rng.normal(size=(4, 3))
    assert np.allclose(tf.pose7_apply(ab, p),
                       tf.pose7_apply(a, tf.pose7_apply(b, p)))
    assert np.allclose(tf.pose7_relative(a, ab), b)
    assert np.allclose(tf.matrix_to_pose7(tf.pose7_to_matrix(a)), a)


def test_intrinsics_and_backprojection():
    k = tf.intrinsics_from_fov(64, 48, hfov=math.radians(90))
    assert k["fx"] == pytest.approx(32.0) and k["fy"] == k["fx"]
    assert (k["cx"], k["cy"]) == (31.5, 23.5)
    depth = np.full((48, 64), 2.0)
    depth[0, 0] = 0.0
    pts = tf.backproject_depth(depth, k)
    assert np.isnan(pts[0, 0]).all()
    uv = tf.project(k, pts[10, 20])
    assert np.allclose(uv, [20, 10])
    body = tf.backproject_depth(depth, k, frame="rep103")
    assert np.allclose(body[..., 0][1:, 1:], 2.0)  # forward = depth
    cam = tf.pose7([1, 2, 3], tf.Q_REP103_FROM_OPENCV)
    world = tf.backproject_depth(depth, k, T_world_cam=cam)
    centre = world[24, 32]
    assert centre[0] == pytest.approx(3.0, abs=0.1)  # 2 m ahead of x=1


# -- keys, words, conventions --------------------------------------------------------------


def test_new_keys_and_words():
    assert spec.depth_key("head") == "observation.depth.head"
    assert spec.segmentation_key("head") == "observation.segmentation.head"
    assert spec.CHAT == "observation.language.chat"
    assert spec.MAP_POSE == "observation.map_pose"
    assert spec.GOTO == "action.goto"
    assert spec.OPTIONAL_WORDS[-1] == "attack"
    assert spec.vocabulary()[:4] == spec.WORDS
    assert spec.word_id("attack") == len(spec.vocabulary()) - 1


def test_3d_conventions():
    assert spec.STATE_NAMES_3D[:3] == ("ego.x", "ego.y", "ego.z")
    assert spec.POSE7_NAMES == ("x", "y", "z", "qx", "qy", "qz", "qw")
    assert spec.WORLD_POSE_NAMES_3D == ("x", "y", "z", "yaw")
    assert spec.REGION_NONE == -1


def test_validate_features():
    ok = [FeatureSpec("observation.native.x", "float32", (None, 12)),
          FeatureSpec(spec.privileged_ext_key("acme", "wind"), "float32",
                      (3,)),
          FeatureSpec(spec.depth_key("h"), "float32", (4, 4),
                      storage="depth_png16")]
    assert spec.validate_features(ok) == []
    bad = [FeatureSpec("wind", "float32", (3,)),
           FeatureSpec("privileged.wind", "float32", (3,)),
           FeatureSpec(spec.depth_key("h"), "int32", (4, 4),
                       storage="depth_png16"),
           FeatureSpec(spec.HEAD, "float32", (4, 4, 3), storage="png")]
    assert len(spec.validate_features(bad)) == 4
    with pytest.raises(ValueError):
        spec.privileged_ext_key("bad producer", "x")


def test_check_value_missing_convention():
    f = FeatureSpec("next.reward", "float32", (1,))
    assert spec.check_value(f, float("nan")) is None
    assert "NaN" in spec.check_value(f, None)
    g = FeatureSpec("observation.state", "float32", (3,))
    assert "NaN" in spec.check_value(g, [1.0, None, 2.0])
    assert "values" in spec.check_value(g, [1.0, 2.0])


# -- registries -----------------------------------------------------------------------


def test_categories_are_append_only_and_namespaced():
    assert spec.CATEGORIES["cave"].synset == "cave.n.01"
    assert spec.CATEGORIES["body_of_water"].label == "body of water"
    key = spec.register_category("sofa", Category("sofa", "sofa.n.01"),
                                 producer="testprod")
    assert key == "testprod:sofa"
    assert spec.register_category("sofa", Category("sofa", "sofa.n.01"),
                                  producer="testprod") == key
    with pytest.raises(ValueError, match="append-only"):
        spec.register_category("sofa", Category("couch", "sofa.n.01"),
                               producer="testprod")


def test_templates_are_append_only_and_namespaced():
    key = spec.register_templates("search", ["Search for the {target}."],
                                  producer="testprod")
    assert spec.instruction(key, 0, target="beacon") == \
        "Search for the beacon."
    spec.register_templates("search", ["Search for the {target}.",
                                       "Look for the {target}."],
                            producer="testprod")
    assert spec.instruction(key, 1, target="x") == "Look for the x."
    with pytest.raises(ValueError, match="append-only"):
        spec.register_templates("search", ["Hunt the {target}."],
                                producer="testprod")
    assert spec.TEMPLATES["goto"][0] == "Go to the {target}."


# -- grammar --------------------------------------------------------------------------


def test_grid_text_unchanged():
    cells = [SeenCell(1, 0, "wall"), SeenCell(0, -1, "floor", ("water",)),
             SeenCell(0, 1, "doorway"), SeenCell(-1, 0, "floor"),
             SeenCell(2, 2, "floor", (), ("key",)),
             SeenCell(0, 0, "floor", ("ladder",))]
    assert spec.render_text(cells, here=("ladder",)) == (
        "There is a wall ahead and a doorway to the right. "
        "You are on a ladder. You see the key two cells ahead and two to "
        "the right and water one cell to the left.")


def test_blockers_water_units_and_cover():
    spec.register_category("sofa", Category("sofa", "sofa.n.01"),
                           producer="testprod")
    cells = [SeenCell(1, 0, "obstacle", blocker="testprod:sofa"),
             SeenCell(0, 1, "obstacle", blocker="armchair"),
             SeenCell(0, -1, "floor"), SeenCell(-1, 0, "floor"),
             SeenCell(2.5, -1, "water"),
             SeenCell(3, 0, "obstacle")]
    text = spec.render_text(cells, here=("covered",), units="metres")
    assert text.startswith("There is a sofa ahead and an armchair to the "
                           "right.")
    assert "You are under cover." in text
    assert "water 2.5 metres ahead and one to the left" in text
    assert "an obstacle three metres ahead" in text
    assert "way ahead" not in text  # the open-run sentence is cells-only
    assert spec.offset_phrase(1, 0, "metres") == "one metre ahead"


def test_manifest_declares_ext_entries():
    import gymnasium as gym

    import topogym  # noqa: F401
    from topogym import canonical

    env = gym.make("TopoGym/Dilution-50-v0", seed=0, obs_mode="canonical",
                   actions="words")
    m = canonical.manifest(env, ext={"acme": {"wind": "gust vector, m/s"}})
    assert m["privileged"]["ext.acme"] == {"wind": "gust vector, m/s"}
    assert m["spec_version"] == spec.CANONICAL_SPEC_VERSION
    with pytest.raises(ValueError):
        canonical.manifest(env, ext={"bad name": {}})
