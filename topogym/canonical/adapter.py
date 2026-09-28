"""The canonical layer's engine: one adapter per environment.

The adapter owns everything the canonical format adds on top of a grid
env -- the observation dict, the word and waypoint action levels,
``stop``, multi-goal conditioning, the ego pose and the privileged
record -- and drives the env only through its primitive step. It is
used two ways, with identical results:

- ``gym.make(id, obs_mode="canonical")`` / ``actions="words"`` builds it
  inside the env;
- :func:`topogym.canonical.wrap` builds it around an existing env.

Nothing here runs unless one of those is asked for, which is what
keeps every existing mode byte-identical (see COMPATIBILITY.md).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from gymnasium import spaces

from topogym.canonical import render, spec, world
from topogym.core import constants as C
from topogym.core.basemap import Boundary, RectGluing2D

#: word -> the egocentric primitive that executes it
WORD_PRIMITIVES = {"turn_left": C.TURN_LEFT, "turn_right": C.TURN_RIGHT,
                   "move_forward": C.FORWARD}

OBS_MODES = ("canonical",)
ACTION_LEVELS = ("words", "waypoint")

#: observation code -> terrain name of the text grammar
_TERRAIN = {
    C.OBS_EMPTY: "floor", C.OBS_WALL: "wall", C.OBS_HOLE: "pit",
    C.OBS_DOOR_OPEN: "doorway", C.OBS_GOAL: "floor",
    C.OBS_OUT_OF_WORLD: "edge", C.OBS_UNSEEN: "unseen",
    C.OBS_HAZARD: "drop", C.OBS_WORMHOLE: "wormhole",
}

#: Texture slots the text channel reports as semantics. Blocker slots
#: are world-directional and the patch already shows walls; the goal
#: and wormhole slots are reported as objects and terrain.
_SEMANTIC_SLOTS = tuple(
    (slot, name) for slot, name in enumerate(C.TEXTURE_SLOTS.names())
    if slot >= 4 and name not in ("on_treasure", "on_wormhole")
)

#: A stop within this graph distance of the goal counts as arriving.
GOAL_RADIUS = 1

_QUARTER = math.pi / 2


def _wrap_angle(a: float) -> float:
    """Into (-pi, pi]."""
    a = math.fmod(a, 2 * math.pi)
    if a <= -math.pi:
        a += 2 * math.pi
    elif a > math.pi:
        a -= 2 * math.pi
    return a


class CanonicalAdapter:
    """Canonical observations and actions for one grid env.

    ``env`` is the unwrapped env. ``obs`` is ``"canonical"`` or
    ``"native"`` (keep the env's own observation, e.g. a word-action
    policy on the symbolic patch); ``actions`` is ``"words"``,
    ``"waypoint"`` or ``"native"``.
    """

    def __init__(self, env, *, obs: str = "canonical",
                 actions: str = "native", n_goals: int = 1,
                 stop_to_succeed: bool = False,
                 image_size: int = spec.DEFAULT_IMAGE_SIZE,
                 topdown: bool = False, phrasing="canonical"):
        if obs not in ("canonical", "native"):
            raise ValueError(f"obs must be 'canonical' or 'native', got {obs!r}")
        if actions not in ("native",) + ACTION_LEVELS:
            raise ValueError(f"unknown action level {actions!r}")
        if env.obs_mode == "global":
            raise ValueError(
                'the canonical layer needs a partial view; obs_mode="global" '
                "marks the whole world observed")
        if actions in ACTION_LEVELS and env._motion != "egocentric":
            raise ValueError(
                f"actions={actions!r} drives an egocentric body; the env "
                "was built with fourway actions")
        if n_goals < 1:
            raise ValueError(f"n_goals must be >= 1, got {n_goals}")
        if n_goals > 1 and obs != "canonical":
            raise ValueError(
                "n_goals > 1 needs obs_mode='canonical': only the "
                "instruction says which goal counts")
        if n_goals > 1 and not env.goal_exists:
            raise ValueError("n_goals > 1 needs a goal (goal=True)")
        if stop_to_succeed and actions != "words":
            raise ValueError("stop_to_succeed needs actions='words' (the "
                             "vocabulary with stop)")
        if stop_to_succeed and env.reward_mode not in (
                "sparse", "goal", "deceptive"):
            raise ValueError(
                "stop_to_succeed needs a goal-paying reward_mode "
                "(sparse, goal or deceptive)")
        if phrasing != "random":
            k = spec.parse_phrasing(phrasing)
            if not 0 <= k < len(spec.TEMPLATES["goto"]):
                raise ValueError(f"no phrasing {phrasing!r}")
        self.env = env
        self.obs_kind = obs
        self.level = actions
        self.n_goals = int(n_goals)
        self.stop_to_succeed = bool(stop_to_succeed)
        self.image_size = int(image_size)
        self.topdown = bool(topdown)
        self.phrasing = phrasing
        self._rng = np.random.default_rng([0xCA0])
        self._goals: tuple = ()
        self._goal_index = 0
        self._instruction = ""
        self._phrasing_tag = "canonical"

    # -- spaces ---------------------------------------------------------------

    def observation_space(self):
        if self.obs_kind != "canonical":
            return None
        s = self.image_size
        image = spaces.Box(0, 255, shape=(s, s, 3), dtype=np.uint8)
        text = spec.TEXT_CHARSET
        entries = {
            spec.HEAD: image,
            spec.INSTRUCTION: spaces.Text(
                spec.MAX_INSTRUCTION_LENGTH, charset=text),
            spec.STATE: spaces.Box(
                np.array([-np.inf, -np.inf, -np.pi], np.float32),
                np.array([np.inf, np.inf, np.pi], np.float32),
                dtype=np.float32),
            spec.TEXT: spaces.Text(spec.MAX_TEXT_LENGTH, charset=text),
            spec.STRUCTURED: spaces.Text(
                spec.MAX_STRUCTURED_LENGTH, charset=text),
        }
        if self.topdown:
            entries[spec.TOPDOWN] = spaces.Box(
                0, 255, shape=(s, s, 3), dtype=np.uint8)
        return spaces.Dict(entries)

    def action_space(self):
        if self.level == "words":
            return spaces.Discrete(len(spec.WORDS))
        if self.level == "waypoint":
            r = float(self.env.view_radius)
            # dz is part of the shared 4-vector; a planar body ignores
            # it (a degenerate [0, 0] bound trips Gymnasium's checker).
            return spaces.Box(
                np.array([-r, -r, -r, -np.pi], np.float32),
                np.array([r, r, r, np.pi], np.float32),
                dtype=np.float32)
        return None

    # -- episode --------------------------------------------------------------

    def reset(self, native_obs, info: dict, seed=None, options=None):
        env = self.env
        if seed is not None:
            self._rng = np.random.default_rng([int(seed), 0xCA0])
        options = options or {}
        self._goals = world.place_goals(env, self.n_goals) \
            if env.goal_exists else ()
        index = options.get("goal_index")
        if index is None:
            index = int(self._rng.integers(len(self._goals))) \
                if len(self._goals) > 1 else 0
        if self._goals and not 0 <= index < len(self._goals):
            raise ValueError(f"goal_index {index} out of range")
        self._goal_index = int(index)
        # Goal 0 with auto-success is the env's own behaviour; only
        # touch the hooks when something differs, so the default
        # episode runs the exact code path of every other mode.
        if self._goals and self._goal_index != 0:
            env._goal_override = self._goals[self._goal_index][0]
            self._retarget_horizon()
        else:
            env._goal_override = None
        env._auto_goal = not self.stop_to_succeed
        self._set_instruction()
        self._start_pose()
        self._done = False
        info = dict(info)
        info["instruction"] = {
            "text": self._instruction, "phrasing": self._phrasing_tag,
            "goal_index": self._goal_index if self._goals else None,
            "category": (self._goals[self._goal_index][1]
                         if self._goals else None),
        }
        info["privileged"] = {
            **self._privileged_step(),
            "topology": world.topology(env.layout),
            "goals": [{"cell": list(c), "category": k}
                      for c, k in self._goals],
        }
        info["canonical"] = {"level": self.level, "words": [],
                             "waypoint": [0.0, 0.0, 0.0, 0.0]}
        info["success"] = False
        return self._observe(native_obs), info

    def _retarget_horizon(self) -> None:
        """With several goals, the budget must cover the farthest of
        them, or some instructions would be unsatisfiable by
        construction. Only when the env derives its own horizon."""
        from topogym.envs.core import HORIZON_SLACK

        env = self.env
        if env._max_steps_cfg:
            return
        w, h = env.layout.base.layout_size()
        floor = max(1, (6 * max(w, h)) // 5)
        need = 0
        for cell, _ in self._goals:
            found = env.actions_between(env.layout.start, cell)
            if found:
                need = max(need, math.ceil(HORIZON_SLACK * found / 10) * 10)
        env._max_steps = max(floor, need)

    def _set_instruction(self) -> None:
        env = self.env
        if self.phrasing == "random":
            k = int(self._rng.integers(len(spec.TEMPLATES["goto"])))
        else:
            k = spec.parse_phrasing(self.phrasing)
        self._phrasing_tag = spec.phrasing_tag(k)
        if not self._goals or env.reward_mode in ("none", "coverage"):
            self._instruction = spec.instruction("explore", k)
            return
        cell, category = self._goals[self._goal_index]
        self._instruction = spec.instruction(
            "goto", k, target=world.goal_target(env.layout, cell, category))

    # -- pose -----------------------------------------------------------------

    def _square(self) -> bool:
        base = self.env.layout.base
        return isinstance(base, RectGluing2D) and \
            base.rule_x == Boundary.WALL and base.rule_y == Boundary.WALL

    def _start_pose(self) -> None:
        state = self.env._state
        self._p0 = self.env.layout.base.layout_coords(state.cell)
        self._f0 = state.frame[:2]
        self._pose = [0.0, 0.0, 0.0]

    def _update_pose(self, before, executed) -> None:
        """Advance the ego pose over one primitive.

        On square worlds the pose is exact, read from layout
        coordinates, so teleports (wormholes) and slips are honest. On
        glued worlds layout coordinates wrap, so the pose is dead
        reckoned from the motion actually executed: the agent's own
        frame, which stays consistent across flip seams.
        """
        env = self.env
        after = env._state
        if self._square():
            x, y = env.layout.base.layout_coords(after.cell)
            fx, fy = self._f0
            lx, ly = fy, -fx  # left of forward, screen y down
            dx, dy = x - self._p0[0], y - self._p0[1]
            cx, cy = after.frame[:2]
            self._pose = [float(dx * fx + dy * fy), float(dx * lx + dy * ly),
                          _wrap_angle(math.atan2(cx * lx + cy * ly,
                                                 cx * fx + cy * fy))]
            return
        base = env.layout.base
        x, y, yaw = self._pose
        if env._motion == "fourway":
            if after.cell != before.cell and executed is not None:
                # Screen directions in the start frame (forward = up).
                step = {C.MOVE_UP: (1, 0), C.MOVE_DOWN: (-1, 0),
                        C.MOVE_LEFT: (0, 1), C.MOVE_RIGHT: (0, -1)}
                ex, ey = step[executed]
                x, y = x + ex, y + ey
        elif after.cell == before.cell:
            if after.frame == base.turn_left(before).frame:
                yaw += _QUARTER
            elif after.frame == base.turn_right(before).frame:
                yaw -= _QUARTER
        else:
            x += round(math.cos(yaw))
            y += round(math.sin(yaw))
        self._pose = [float(x), float(y), _wrap_angle(yaw)]

    # -- observation ----------------------------------------------------------

    def _objects(self) -> dict:
        return {cell: category for cell, category in self._goals}

    def _seen_cells(self) -> tuple:
        env = self.env
        patch = env._sight_patch()
        textures = env._texture_patch()
        r = env.view_radius
        objects = self._objects()
        cells = []
        here = ()
        n = 2 * r + 1
        for i in range(n):
            for j in range(n):
                # The occluded patch decides what is seen, including the
                # world's edge: beyond-the-edge cells have no cell to
                # annotate, and occlusion can hide them like any other.
                code = int(patch[i, j])
                if code == C.OBS_UNSEEN:
                    continue
                cell = env._cell_at.get((i, j))
                sem = tuple(name for slot, name in _SEMANTIC_SLOTS
                            if textures[i, j, slot] > 0)
                forward, right = r - i, j - r
                obj = (objects[cell],) if cell in objects and \
                    cell in env._visible else ()
                if (forward, right) == (0, 0):
                    here = sem
                cells.append(spec.SeenCell(forward, right, _TERRAIN[code],
                                           sem, obj))
        return cells, here

    def _observe(self, native_obs):
        if self.obs_kind != "canonical":
            return native_obs
        env = self.env
        objects = self._objects()
        cells, here = self._seen_cells()
        obs = {
            spec.HEAD: render.render_head(env, objects, self.image_size),
            spec.INSTRUCTION: self._instruction,
            spec.STATE: np.array(self._pose, dtype=np.float32),
            spec.TEXT: spec.render_text(cells, here),
            spec.STRUCTURED: spec.render_structured(cells, here),
        }
        if self.topdown:
            obs[spec.TOPDOWN] = render.render_topdown(
                env, objects, self.image_size)
        return obs

    # -- privileged -----------------------------------------------------------

    def active_goal(self):
        return self._goals[self._goal_index][0] if self._goals else None

    def _privileged_step(self) -> dict:
        env = self.env
        base = env.layout.base
        cell = env._state.cell
        x, y = base.layout_coords(cell)
        fx, fy = env._state.frame[:2]
        of, _ = world.regions(env.layout)
        goal = self.active_goal()
        out = {
            # World frame: x right, y down, yaw measured from +x
            # toward +y (the frame's own axes; clockwise on screen).
            "world_pose": [float(x), float(y), float(math.atan2(fy, fx))],
            "region_id": int(of.get(cell, world.OPEN)),
            "goal_pose": None, "d_goal": None, "goal_visible": False,
        }
        if goal is not None:
            gx, gy = base.layout_coords(goal)
            d = world.distance_field(env.layout, goal).get(cell)
            out.update(goal_pose=[float(gx), float(gy)],
                       d_goal=None if d is None else int(d),
                       goal_visible=goal in getattr(env, "_visible", ()))
        return out

    # -- actions --------------------------------------------------------------

    def _primitive(self, action):
        """One env primitive (None = wait), with the pose kept."""
        env = self.env
        before = env._state
        native = env._step_primitive(action)
        self._update_pose(before, env._last_executed)
        return native

    def _word(self, action) -> str:
        if isinstance(action, str):
            if action not in spec.WORDS:
                raise ValueError(f"unknown word {action!r}; vocabulary is "
                                 f"{spec.WORDS}")
            return action
        a = int(action)
        if not 0 <= a < len(spec.WORDS):
            raise ValueError(f"invalid action {action!r}")
        return spec.WORDS[a]

    def step(self, action):
        env = self.env
        start_pose = list(self._pose)
        record = {"level": self.level, "words": []}
        if self.level == "words":
            word = self._word(action)
            record["words"] = [word]
            if word == "stop":
                return self._stop(record, start_pose)
            native, reward, term, trunc, info = self._primitive(
                WORD_PRIMITIVES[word])
        elif self.level == "waypoint":
            native, reward, term, trunc, info = self._waypoint(
                action, record)
        else:
            a = int(action)
            if not 0 <= a < env._n_primitive:
                raise ValueError(f"invalid action {action!r}")
            native, reward, term, trunc, info = self._primitive(a)
            names = (C.FOURWAY_ACTION_NAMES if env._motion == "fourway"
                     else C.EGOCENTRIC_ACTION_NAMES)
            record["words"] = [names[a]]
        return self._finish(native, reward, term, trunc, info, record,
                            start_pose, success=bool(
                                term and info.get("goal_reached")
                                and env._auto_goal))

    def _finish(self, native, reward, term, trunc, info, record,
                start_pose, success):
        dx, dy = self._pose[0] - start_pose[0], self._pose[1] - start_pose[1]
        yaw0 = start_pose[2]
        c, s = math.cos(yaw0), math.sin(yaw0)
        record["waypoint"] = [round(c * dx + s * dy, 6),
                              round(-s * dx + c * dy, 6), 0.0,
                              round(_wrap_angle(self._pose[2] - yaw0), 6)]
        info = dict(info)
        info["canonical"] = record
        info["privileged"] = self._privileged_step()
        info["success"] = success
        return self._observe(native), float(reward), term, trunc, info

    def _stop(self, record, start_pose):
        """``stop``: ends the episode where the agent stands. With
        ``stop_to_succeed`` it is how the goal is claimed; otherwise the
        goal pays on arrival, so a stop is always a give-up."""
        env = self.env
        cell = env._state.cell
        reward, success = 0.0, False
        goal = self.active_goal()
        if self.stop_to_succeed and goal is not None:
            d = world.distance_field(env.layout, goal).get(cell)
            if d is not None and d <= GOAL_RADIUS:
                success = True
                if env.reward_mode == "goal":
                    reward = 1.0 - 0.9 * (env._steps / env._max_steps)
                else:
                    reward = 1.0
        env._episode_return += reward
        info = env._step_info(cell)
        info["stopped"] = True
        native = env._obs() if self.obs_kind == "native" else None
        return self._finish(native, reward, True, False, info, record,
                            start_pose, success)

    # -- waypoint -------------------------------------------------------------

    def _plan(self, target_cell) -> list | None:
        """Fewest primitives to ``target_cell`` through cells the agent
        has observed to be free: the waypoint never plans through the
        unknown, so it uses no privileged map."""
        env = self.env
        base = env.layout.base
        known = env._observed_free
        start = env._state
        limit = 4 * (2 * env.view_radius + 1)
        parents = {start: None}
        queue = deque([(start, 0)])
        while queue:
            state, depth = queue.popleft()
            if state.cell == target_cell:
                path = []
                while parents[state] is not None:
                    state, prim = parents[state]
                    path.append(prim)
                return path[::-1]
            if depth >= limit:
                continue
            nxt = [(base.turn_left(state), C.TURN_LEFT),
                   (base.turn_right(state), C.TURN_RIGHT)]
            ahead = base.forward(state)
            if ahead is not None and ahead.cell in known:
                nxt.append((ahead, C.FORWARD))
            for s, prim in nxt:
                if s not in parents:
                    parents[s] = (state, prim)
                    queue.append((s, depth + 1))
        return None

    def _expected(self, prim):
        base = self.env.layout.base
        state = self.env._state
        if prim == C.TURN_LEFT:
            return base.turn_left(state)
        if prim == C.TURN_RIGHT:
            return base.turn_right(state)
        return base.forward(state)

    def _waypoint(self, action, record):
        """Execute ``(dx, dy, dz, dyaw)``: walk to the cell ``dx`` ahead
        and ``dy`` to the left (rounded; within the view), then turn by
        ``dyaw`` (rounded to quarter turns) relative to the heading at
        issue. ``dz`` is ignored (a planar body). Replans when a
        primitive does not do what the plan expected (slip, a bumped
        door, a teleport). An unreachable or unseen target skips the
        walk and reports ``waypoint_reached: False``; a waypoint that
        asks for nothing waits one step, so time always passes."""
        env = self.env
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (4,):
            raise ValueError("waypoint actions are (dx, dy, dz, dyaw)")
        r = env.view_radius
        dx = int(np.clip(round(a[0]), -r, r))
        dy = int(np.clip(round(a[1]), -r, r))
        target_yaw = self._pose[2] + round(float(a[3]) / _QUARTER) * _QUARTER
        env._sight_patch()
        moving = (dx, dy) != (0, 0)
        target = env._cell_at.get((r - dx, r - dy)) if moving else None
        if target is not None and target not in env._observed_free:
            target = None
        budget = 4 * (2 * r + 1) + 4
        total, out, prims = 0.0, None, []

        def run(prim):
            nonlocal total, out
            out = self._primitive(prim)
            prims.append(prim)
            total += out[1]
            return out[2] or out[3]

        reached = not moving
        if target is not None:
            plan = self._plan(target)
            while plan and len(prims) < budget:
                prim = plan.pop(0)
                expected = self._expected(prim)
                if run(prim):
                    break
                if env._state.cell == target:
                    break
                if env._state != expected:
                    plan = self._plan(target) or []
            reached = env._state.cell == target
        if out is None or not (out[2] or out[3]):
            turns = round(_wrap_angle(target_yaw - self._pose[2]) / _QUARTER)
            prim = C.TURN_LEFT if turns > 0 else C.TURN_RIGHT
            for _ in range(abs(int(turns))):
                if run(prim):
                    break
        if out is None:
            run(None)  # nothing to do: time still passes
        native, _, term, trunc, info = out
        names = {C.TURN_LEFT: "turn_left", C.TURN_RIGHT: "turn_right",
                 C.FORWARD: "move_forward", None: "wait"}
        record["words"] = [names[p] for p in prims]
        record["waypoint_reached"] = bool(reached)
        return native, total, term, trunc, info
