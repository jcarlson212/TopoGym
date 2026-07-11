"""TopoGrid2D: an egocentric agent on a 2D base manifold.

Actions (``Discrete(3)``): 0 = turn left, 1 = turn right, 2 = forward.

The agent carries a local frame that is parallel-transported as it moves:
crossing a Mobius/Klein/RP^2 seam mirrors its view, walking over a
cube-sphere edge is seamless, and walking around a cube corner reveals the
curvature. Observations are egocentric ``(2r+1, 2r+1)`` patches (agent at
the center, facing "up"), occluded by walls, so chamber interiors — and
whether a suspicious room is a decoy — must be discovered by interaction.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from topogym.core import constants as C
from topogym.envs.core import TopoEnvCore
from topogym.generation.config import TopoGenConfig2D
from topogym.generation.generator import _translate
from topogym.rendering.rgb import render_rgb_2d


class TopoGrid2DEnv(TopoEnvCore):
    DIM = 2

    ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD = 0, 1, 2

    def _default_config(self):
        return TopoGenConfig2D()

    def _config_class(self):
        return TopoGenConfig2D

    def _build_spaces(self):
        self.action_space = spaces.Discrete(3)
        r = self.view_radius
        if self.obs_mode == "local":
            self.observation_space = spaces.Box(
                0, C.OBS_MAX, shape=(2 * r + 1, 2 * r + 1), dtype=np.uint8
            )
        elif self.obs_mode == "global":
            probe = self._generate(self.layout_seed or 0)
            w, h = probe.base.layout_size()
            if self.layout_seed is not None:
                self._fixed_layout = probe
            self.observation_space = spaces.Box(
                0, C.OBS_MAX, shape=(2, h, w), dtype=np.uint8
            )
        else:
            raise ValueError(f"unknown obs_mode {self.obs_mode!r}")

    # -- gym API --------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.layout = self._obtain_layout()
        self._reset_runtime()
        base = self.layout.base
        self._state = base.initial_state(self.layout.start)
        for _ in range(int(self.np_random.integers(4))):
            self._state = base.turn_left(self._state)
        self._visited.add(self._state.cell)
        return self._obs(), self._reset_info(self._state.cell)

    def step(self, action):
        base = self.layout.base
        if action == self.ACTION_LEFT:
            self._state = base.turn_left(self._state)
        elif action == self.ACTION_RIGHT:
            self._state = base.turn_right(self._state)
        elif action == self.ACTION_FORWARD:
            nxt = base.forward(self._state)
            if nxt is not None and self._try_enter(self._state.cell, nxt.cell):
                self._on_leave(self._state.cell)
                self._state = nxt
        else:
            raise ValueError(f"invalid action {action!r}")
        reward, terminated, truncated = self._step_outcome(self._state.cell)
        return self._obs(), reward, terminated, truncated, self._step_info(
            self._state.cell
        )

    # -- observations -----------------------------------------------------------

    def _obs(self):
        if self.obs_mode == "global":
            return self._global_obs()
        r = self.view_radius
        base = self.layout.base
        view = np.full((2 * r + 1, 2 * r + 1), C.OBS_OUT_OF_WORLD, np.uint8)
        cell_at = {}
        for a in range(-r, r + 1):  # forward steps
            s = _translate(base, self._state, a)
            if s is None:
                continue
            s = base.turn_right(s)
            for b in range(-r, r + 1):  # right steps
                t = _translate(base, s, b)
                if t is None:
                    continue
                view[r - a, r + b] = self._obs_code(t.cell)
                cell_at[(r - a, r + b)] = t.cell
        out = self._occlude(view, (r, r), self._BLOCKING)
        for idx, cell in cell_at.items():
            if out[idx] != C.OBS_UNSEEN:
                self._note_observed(cell, int(out[idx]))
        return out

    def _global_obs(self):
        base = self.layout.base
        w, h = base.layout_size()
        grid = np.full((h, w), C.OBS_OUT_OF_WORLD, np.uint8)
        agent = np.zeros((h, w), np.uint8)
        for cell in base.cells():
            x, y = base.layout_coords(cell)
            code = self._obs_code(cell)
            grid[y, x] = code
            self._note_observed(cell, code)
        ax, ay = base.layout_coords(self._state.cell)
        agent[ay, ax] = C.OBS_AGENT
        return np.stack([grid, agent])

    # -- rendering -----------------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            return render_rgb_2d(self)
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    _ANSI = {
        C.OBS_EMPTY: "·", C.OBS_WALL: "#", C.OBS_HOLE: "O",
        C.OBS_DOOR_OPEN: "/", C.OBS_GOAL: "G", C.OBS_OUT_OF_WORLD: " ",
        C.OBS_UNSEEN: "?", C.OBS_DOOR_ONEWAY: ">", C.OBS_TRAPDOOR: "v",
    }

    def _render_ansi(self):
        base = self.layout.base
        w, h = base.layout_size()
        rows = [[" "] * w for _ in range(h)]
        for cell in base.cells():
            x, y = base.layout_coords(cell)
            rows[y][x] = self._ANSI[self._obs_code(cell)]
        ax, ay = base.layout_coords(self._state.cell)
        rows[ay][ax] = "@"
        return "\n".join("".join(r) for r in rows)
