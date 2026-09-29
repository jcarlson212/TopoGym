"""The opt-in canonical layer: observations, actions, goals, manifest."""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys
from collections import deque

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import topogym  # noqa: F401
from topogym import canonical
from topogym.canonical import spec
from topogym.canonical.manifest import features, manifest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "compat"))
import golden_harness as gh  # noqa: E402

#: Small ids covering every slice: plain, textured, glued, teleporting.
IDS = ("TopoGym/Dilution-50-v0", "TopoGym/Decoys2-50-v0",
       "TopoGym/IceShip-v0", "TopoGym/SpaceWarp-v0", "TopoGym/ClownChase-v0",
       "TopoGym/TopKlein-50-v0", "TopoGym/TopTorus-50-v0")

EGO = gh.ACTIONS["egocentric"]


def make(env_id="TopoGym/Dilution-50-v0", **kw):
    kw.setdefault("obs_mode", "canonical")
    return gym.make(env_id, seed=kw.pop("seed", 0), **kw)


def plan_to(env, target) -> list:
    """Egocentric primitives to ``target`` over the full map (a test
    oracle, deliberately privileged)."""
    base = env.unwrapped
    lay = base.layout.base
    blocked = base._planning_blocked()
    start = base._state
    parents = {start: None}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        if s.cell == target:
            out = []
            while parents[s] is not None:
                s, a = parents[s]
                out.append(a)
            return out[::-1]
        nxt = [(lay.turn_left(s), 0), (lay.turn_right(s), 1)]
        f = lay.forward(s)
        if f is not None and f.cell not in blocked:
            nxt.append((f, 2))
        for n, a in nxt:
            if n not in parents:
                parents[n] = (s, a)
                queue.append(n)
    raise AssertionError(f"no route to {target}")


# -- the gym contract ----------------------------------------------------------


@pytest.mark.parametrize("actions", ["words", "waypoint", "egocentric",
                                     "fourway"])
def test_check_env(actions):
    env = make(actions=actions)
    check_env(env.unwrapped, skip_render_check=True)


@pytest.mark.parametrize("env_id", IDS)
def test_check_env_every_slice(env_id):
    check_env(make(env_id, actions="words").unwrapped,
              skip_render_check=True)


def test_topdown_and_image_size():
    env = make(actions="words", topdown=True, image_size=128)
    obs, _ = env.reset(seed=0)
    assert obs[spec.HEAD].shape == (128, 128, 3)
    assert obs[spec.TOPDOWN].shape == (128, 128, 3)
    check_env(env.unwrapped, skip_render_check=True)


# -- what is and is not observed ----------------------------------------------


@pytest.mark.parametrize("env_id", IDS)
def test_no_privileged_keys_and_valid_features(env_id):
    env = make(env_id, actions="words")
    obs, info = env.reset(seed=0)
    feats = features(env.unwrapped._canonical)
    for t, a in enumerate(EGO[:30]):
        assert not [k for k in obs if "privileged" in k]
        assert spec.validate_observation(obs, feats) == []
        assert env.observation_space.contains(obs)
        assert set(info["privileged"]) >= set(spec.PRIVILEGED_STEP_FIELDS)
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            obs, info = env.reset()


@pytest.mark.parametrize("env_id", IDS)
def test_instruction_every_step(env_id):
    env = make(env_id, actions="words")
    obs, info = env.reset(seed=1)
    first = obs[spec.INSTRUCTION]
    assert first and first == info["instruction"]["text"]
    for a in EGO:
        obs, _, term, trunc, info = env.step(a)
        assert obs[spec.INSTRUCTION] == first
        if term or trunc:
            break


def test_occluded_semantics_stay_hidden():
    """The text reports only what the patch shows: nothing behind a
    wall, and no hidden door (bump doors read as walls)."""
    env = make("TopoGym/Dilution-50-v0", actions="words")
    env.reset(seed=0)
    adapter = env.unwrapped._canonical
    cells, _ = adapter._seen_cells()
    patch = env.unwrapped._sight_patch()
    r = env.unwrapped.view_radius
    unseen = {(r - i, j - r) for i, j in zip(*np.nonzero(patch == 6))}
    assert unseen.isdisjoint({(c.forward, c.right) for c in cells})


# -- determinism ----------------------------------------------------------------


def test_text_grammar_is_a_pure_function():
    cells = [spec.SeenCell(1, 0, "wall"), spec.SeenCell(0, -1, "floor",
                                                        ("water",)),
             spec.SeenCell(0, 1, "doorway"), spec.SeenCell(-1, 0, "floor"),
             spec.SeenCell(2, 2, "floor", (), ("key",)),
             spec.SeenCell(0, 0, "floor", ("ladder",))]
    text = spec.render_text(cells, here=("ladder",))
    assert text == (
        "There is a wall ahead and a doorway to the right. "
        "You are on a ladder. You see the key two cells ahead and two to "
        "the right and water one cell to the left.")
    assert spec.render_text(list(reversed(cells)), here=("ladder",)) == text
    assert json.loads(spec.render_structured(cells, ("ladder",)))["here"] \
        == ["ladder"]


_PROBE = r"""
import json, sys, gymnasium as gym, topogym
env = gym.make("TopoGym/IceShip-v0", seed=2, obs_mode="canonical",
               actions="words", n_goals=3)
obs, info = env.reset(seed=2)
out = [obs["observation.text"], obs["observation.structured"],
       obs["observation.language.instruction"]]
for a in [2, 2, 0, 2, 2, 1, 2, 2, 2]:
    obs, r, t, tr, info = env.step(a)
    out += [obs["observation.text"], obs["observation.structured"],
            float(obs["observation.state"].sum()),
            int(obs["observation.images.head"].sum())]
print(json.dumps(out))
"""


def test_rendering_identical_across_processes():
    def run():
        return subprocess.run([sys.executable, "-c", _PROBE], check=True,
                              capture_output=True, text=True).stdout

    assert run() == run()


# -- backward compatibility ------------------------------------------------------


@pytest.mark.parametrize("obs_mode", gh.OBS_MODES)
@pytest.mark.parametrize("actions", gh.ACTION_MODES)
def test_new_kwargs_at_defaults_reproduce_golden(obs_mode, actions):
    """Passing every new kwarg at its default is the 0.4.2 env."""
    env_id = "TopoGym/Decoys2-50-v0"
    want = json.loads(gh.fixture_path(env_id).read_text())["rollouts"]
    for seed in gh.SEEDS:
        got = gh.rollout(env_id, seed, obs_mode, actions, n_goals=1,
                         stop_to_succeed=False, image_size=256,
                         topdown=False, phrasing="canonical")
        exp = want[gh.combo_key(seed, obs_mode, actions)]
        assert got == {k: exp[k] for k in got}


@pytest.mark.parametrize("env_id", IDS)
def test_canonical_dynamics_match_native(env_id):
    """Single-goal canonical mode moves, pays and ends exactly like the
    native env: the layer only re-presents it."""
    native = gym.make(env_id, seed=1, obs_mode="dict")
    canon = make(env_id, seed=1, actions="words")
    _, ni = native.reset(seed=1)
    _, ci = canon.reset(seed=1)
    extra = {"instruction", "privileged", "canonical", "success"}
    assert {k: v for k, v in ci.items() if k not in extra} == ni
    for a in EGO:
        n = native.step(a)
        c = canon.step(a)
        assert n[1:4] == c[1:4]
        assert {k: v for k, v in c[4].items() if k not in extra} == n[4]
        if n[2] or n[3]:
            assert native.reset()[1] == {
                k: v for k, v in canon.reset()[1].items() if k not in extra}


def test_new_kwargs_need_the_canonical_layer():
    with pytest.raises(ValueError, match="canonical layer"):
        gym.make("TopoGym/Dilution-50-v0", n_goals=2)
    with pytest.raises(ValueError, match="canonical layer"):
        gym.make("TopoGym/Dilution-50-v0", topdown=True)


def test_existing_errors_unchanged():
    with pytest.raises(ValueError, match="unknown obs_mode"):
        gym.make("TopoGym/Dilution-50-v0", obs_mode="pixels")
    with pytest.raises(ValueError, match="actions must be"):
        gym.make("TopoGym/Dilution-50-v0", actions="teleport")


# -- wrap ------------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", IDS[:4])
def test_wrap_matches_obs_mode(env_id):
    a = make(env_id, seed=2, actions="words")
    b = canonical.wrap(gym.make(env_id, seed=2), actions="words")
    oa, ia = a.reset(seed=2)
    ob, ib = b.reset(seed=2)
    for act in [None] + EGO[:25]:
        if act is not None:
            oa, ra, ta, ua, ia = a.step(act)
            ob, rb, tb, ub, ib = b.step(act)
            assert (ra, ta, ua) == (rb, tb, ub)
        for k in oa:
            assert np.array_equal(oa[k], ob[k]) if isinstance(oa[k], np.ndarray) \
                else oa[k] == ob[k]
        assert ia["privileged"] == ib["privileged"]
        if act is not None and (ta or ua):
            break


def test_wrap_rejects_global_and_fourway_words():
    with pytest.raises(ValueError, match="partial view"):
        canonical.wrap(gym.make("TopoGym/Dilution-50-v0", obs_mode="global"))
    with pytest.raises(ValueError, match="egocentric body"):
        canonical.wrap(gym.make("TopoGym/Dilution-50-v0", actions="fourway"),
                       actions="words")


# -- pose ------------------------------------------------------------------------------


def test_pose_is_relative_to_the_episode_start():
    env = make(actions="words")
    obs, _ = env.reset(seed=0)
    assert obs[spec.STATE].tolist() == [0.0, 0.0, 0.0]
    obs, *_ = env.step("turn_left")
    assert obs[spec.STATE][2] == pytest.approx(math.pi / 2)
    obs, *_ = env.step("turn_right")
    obs, *_ = env.step("turn_right")
    assert obs[spec.STATE][2] == pytest.approx(-math.pi / 2)


def test_pose_on_a_torus_is_dead_reckoned():
    """Walking once around an empty torus returns to the start cell, but
    the ego pose keeps counting: it lives in the covering space."""
    from topogym.generation import TopoGenConfig2D

    cfg = TopoGenConfig2D(base="torus", size=9, n_holes=0, n_chambers=0,
                          n_decoys=0)
    env = gym.make("TopoGym/Grid2D-v0", config=cfg, layout_seed=0,
                   obs_mode="canonical", actions="words", goal=False)
    env.reset(seed=0)
    start = env.unwrapped._state.cell
    for _ in range(9):
        obs, *_ = env.step("move_forward")
    assert env.unwrapped._state.cell == start
    assert obs[spec.STATE].tolist() == [9.0, 0.0, 0.0]


def test_waypoint_frame_matches_state():
    """The realized waypoint of a word step is the pose change in the
    frame it was issued from."""
    env = make(actions="words")
    obs, _ = env.reset(seed=3)
    for a in EGO[:20]:
        before = obs[spec.STATE].astype(float)
        obs, _, term, trunc, info = env.step(a)
        dx, dy, _, dyaw = info["canonical"]["waypoint"]
        yaw = before[2]
        x = before[0] + math.cos(yaw) * dx - math.sin(yaw) * dy
        y = before[1] + math.sin(yaw) * dx + math.cos(yaw) * dy
        assert (x, y) == pytest.approx(tuple(obs[spec.STATE][:2]), abs=1e-5)
        if term or trunc:
            break


# -- actions ---------------------------------------------------------------------------


def test_words_accept_names_and_ids():
    a, b = make(actions="words"), make(actions="words")
    a.reset(seed=0)
    b.reset(seed=0)
    for w in ["move_forward", "turn_left", "move_forward"]:
        oa = a.step(w)[0]
        ob = b.step(spec.WORDS.index(w))[0]
        assert oa[spec.TEXT] == ob[spec.TEXT]
    with pytest.raises(ValueError, match="unknown word"):
        a.step("jump")


def test_stop_without_stop_to_succeed_is_a_give_up():
    env = make(actions="words")
    env.reset(seed=0)
    _, r, term, trunc, info = env.step("stop")
    assert (r, term, trunc, info["success"], info["stopped"]) == \
        (0.0, True, False, False, True)


def test_stop_to_succeed():
    env = make(actions="words", stop_to_succeed=True)
    env.reset(seed=0)
    goal = env.unwrapped.layout.goal
    for a in plan_to(env, goal):
        _, r, term, trunc, info = env.step(a)
        assert not term, "arriving must not end the episode"
    assert info["goal_reached"] and r == 0.0
    _, r, term, _, info = env.step("stop")
    assert (r, term, info["success"]) == (1.0, True, True)


def test_auto_success_is_the_default():
    env = make(actions="words")
    env.reset(seed=0)
    path = plan_to(env, env.unwrapped.layout.goal)
    for a in path:
        _, r, term, _, info = env.step(a)
    assert (r, term, info["success"]) == (1.0, True, True)


def test_waypoint_reaches_a_visible_cell():
    env = make(actions="waypoint")
    obs, _ = env.reset(seed=0)
    base = env.unwrapped
    base._sight_patch()
    r = base.view_radius
    # Any observed-free cell in view is a legal target.
    for (i, j), cell in sorted(base._cell_at.items()):
        if cell in base._observed_free and cell != base._state.cell:
            dx, dy = r - i, r - j
            break
    _, _, _, _, info = env.step(np.array([dx, dy, 0, 0], np.float32))
    assert info["canonical"]["waypoint_reached"]
    assert base._state.cell == cell
    assert info["canonical"]["waypoint"][:2] == [pytest.approx(dx),
                                                  pytest.approx(dy)]


def test_waypoint_to_a_wall_is_not_reached_and_time_passes():
    env = make(actions="waypoint")
    env.reset(seed=0)
    base = env.unwrapped
    steps = base._steps
    _, _, _, _, info = env.step(np.zeros(4, np.float32))
    assert base._steps == steps + 1  # a wait
    r = base.view_radius
    blocked = [(r - i, r - j) for (i, j), c in base._cell_at.items()
               if c not in base._observed_free]
    if blocked:
        _, _, _, _, info = env.step(
            np.array([*blocked[0], 0, 0], np.float32))
        assert not info["canonical"]["waypoint_reached"]


def test_slip_never_samples_stop():
    env = make(actions="words", p_slip=1.0)
    env.reset(seed=0)
    for _ in range(50):
        _, _, term, trunc, info = env.step("move_forward")
        assert not info.get("stopped")
        if term or trunc:
            break


# -- goals -----------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", ["TopoGym/Dilution-50-v0",
                                    "TopoGym/ChamberCount4-200-v0",
                                    "TopoGym/IceShip-v0"])
def test_multi_goal_placement(env_id):
    env = make(env_id, actions="words", n_goals=4)
    _, info = env.reset(seed=0)
    goals = info["privileged"]["goals"]
    cells = [tuple(g["cell"]) for g in goals]
    cats = [g["category"] for g in goals]
    assert len(set(cells)) == 4 and len(set(cats)) == 4
    assert cells[0] == env.unwrapped.layout.goal
    base = env.unwrapped
    for c in cells:
        assert base.actions_between(base.layout.start, c) is not None
    # Same world, same goals, every episode.
    _, again = env.reset(seed=5)
    assert again["privileged"]["goals"] == goals


def test_multi_goal_only_the_instructed_goal_pays():
    env = make("TopoGym/Dilution-50-v0", actions="words", n_goals=3)
    _, info = env.reset(seed=0, options={"goal_index": 2})
    goals = [tuple(g["cell"]) for g in info["privileged"]["goals"]]
    assert info["instruction"]["category"] == \
        info["privileged"]["goals"][2]["category"]
    assert spec.CATEGORIES[info["instruction"]["category"]].label in \
        info["instruction"]["text"]
    for a in plan_to(env, goals[0]):
        _, r, term, _, _ = env.step(a)
        assert not term and r == 0.0
    for a in plan_to(env, goals[2]):
        _, r, term, _, info = env.step(a)
    assert (r, term, info["success"]) == (1.0, True, True)


def test_multi_goal_horizon_covers_every_goal():
    env = make("TopoGym/ChamberCount4-200-v0", actions="words", n_goals=5)
    _, info = env.reset(seed=0)
    base = env.unwrapped
    for g in info["privileged"]["goals"]:
        assert 3 * base.actions_between(base.layout.start,
                                        tuple(g["cell"])) <= base._max_steps


def test_phrasing_is_recorded():
    env = make(actions="words", phrasing="paraphrase:2")
    obs, info = env.reset(seed=0)
    assert info["instruction"]["phrasing"] == "paraphrase:2"
    assert obs[spec.INSTRUCTION].startswith("Navigate to the")
    env = make(actions="words", phrasing="random")
    tags = {env.reset(seed=s)[1]["instruction"]["phrasing"]
            for s in range(20)}
    assert len(tags) > 1


def test_describe_goal_uses_the_same_templates():
    env = make(actions="words")
    obs, info = env.reset(seed=0)
    goal = env.unwrapped.layout.goal
    d = canonical.describe_goal(env, goal)
    assert d["instruction"] == obs[spec.INSTRUCTION]
    assert d["goal"]["category"] == "treasure"
    here = canonical.describe_goal(env)
    assert here["instruction"].startswith("Go to")
    assert here["phrasing"] == "canonical"


def test_regions_and_topology():
    env = make("TopoGym/ChamberCount4-200-v0", actions="words")
    _, info = env.reset(seed=0)
    topo = info["privileged"]["topology"]
    kinds = [r["kind"] for r in topo["regions"]]
    assert kinds.count("chamber") == 4 and "door" in kinds
    assert topo["betti_z2"] == env.unwrapped.topology.betti_z2 or \
        topo["betti_z2"] == list(env.unwrapped.topology.betti_z2)
    base = env.unwrapped
    for a in plan_to(env, base.layout.goal)[:-1]:
        _, _, _, _, info = env.step(a)
    rid = info["privileged"]["region_id"]
    assert topo["regions"][rid]["kind"] in ("door", "chamber")


# -- manifest -----------------------------------------------------------------------


def test_manifest():
    env = make(actions="words")
    m = manifest(env)
    assert m["spec_version"] == spec.CANONICAL_SPEC_VERSION
    assert set(m["features"]) == {spec.HEAD, spec.INSTRUCTION, spec.STATE,
                                  spec.TEXT, spec.STRUCTURED}
    assert m["features"][spec.STATE]["names"] == list(spec.STATE_NAMES_2D)
    assert m["actions"]["waypoint"]["uses_privileged_map"] is False
    assert m["frames"]["episode_frame"] == "per_episode"
    assert m["actions"]["words"]["vocab"] == list(spec.WORDS)
    json.dumps(m)
    fourway = make(actions="fourway")
    assert manifest(fourway)["frames"]["episode_frame"] == "per_world"


def test_split_tags():
    env = make("TopoGym/GiveUp1-50-v0", seed=2001, actions="words")
    env.reset(seed=0)
    split = manifest(env)["split"]
    assert split["split"] == "train"
    assert "holdout:family=GiveUp" in split["tags"]
    assert manifest(make(seed=0, actions="words"))["split"]["split"] is None


def test_spec_module_is_standard_library_only():
    src = pathlib.Path(spec.__file__).read_text()
    imports = [line for line in src.splitlines()
               if line.startswith(("import ", "from "))]
    assert all(line.split()[1].split(".")[0] in
               ("__future__", "json", "dataclasses") for line in imports)
