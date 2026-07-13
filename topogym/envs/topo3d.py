"""TopoGrid3D: a free agent in a 3D base manifold.

Actions (``Discrete(6)``): moves along ±x, ±y, ±z. All v1 3D bases are
orientable with wall/wrap boundaries only, so a global frame is
well-defined and no local orientation is needed. Observations are centered
``(2r+1)^3`` patches, occluded by walls.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from topogym.core import constants as C
from topogym.envs.core import TopoEnvCore
from topogym.generation.config import TopoGenConfig3D
from topogym.rendering.rgb import render_rgb_3d

_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


class TopoGrid3DEnv(TopoEnvCore):
    DIM = 3

    def _default_config(self):
        return TopoGenConfig3D()

    def _config_class(self):
        return TopoGenConfig3D

    def _build_spaces(self):
        self.action_space = spaces.Discrete(6)
        r = self.view_radius
        if self.obs_mode == "local":
            self.observation_space = spaces.Box(
                0, C.OBS_MAX, shape=(2 * r + 1,) * 3, dtype=np.uint8
            )
        elif self.obs_mode == "global":
            probe = self._fixed_layout
            if probe is None:
                probe = self._generate(self.layout_seed or 0)
                if self.layout_seed is not None:
                    self._fixed_layout = probe
            w, h, d = probe.base.size
            self.observation_space = spaces.Box(
                0, C.OBS_MAX, shape=(2, w, h, d), dtype=np.uint8
            )
        else:
            raise ValueError(f"unknown obs_mode {self.obs_mode!r}")

    # -- gym API --------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.layout = self._obtain_layout()
        self._reset_runtime()
        self._agent_cell = self.layout.start
        self._visited.add(self._agent_cell)
        return self._obs(), self._reset_info(self._agent_cell)

    def step(self, action):
        if not 0 <= int(action) < 6:
            raise ValueError(f"invalid action {action!r}")
        nxt = self.layout.base.step_dir(self._agent_cell, _DIRS[int(action)])
        if nxt is not None and self._try_enter(self._agent_cell, nxt):
            self._on_leave(self._agent_cell)
            self._agent_cell = nxt
        reward, terminated, truncated = self._step_outcome(self._agent_cell)
        return self._obs(), reward, terminated, truncated, self._step_info(
            self._agent_cell
        )

    # -- observations -----------------------------------------------------------

    def _offset_cell(self, off):
        base = self.layout.base
        out = []
        for k in range(3):
            v = self._agent_cell[k] + off[k]
            if v < 0 or v >= base.size[k]:
                if base.rules[k] != "wrap":
                    return None
                v %= base.size[k]
            out.append(v)
        return tuple(out)

    def _obs(self):
        if self.obs_mode == "global":
            return self._global_obs()
        r = self.view_radius
        n = 2 * r + 1
        view = np.full((n, n, n), C.OBS_OUT_OF_WORLD, np.uint8)
        cell_at = {}
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    cell = self._offset_cell((dx, dy, dz))
                    if cell is not None:
                        idx = (dx + r, dy + r, dz + r)
                        view[idx] = self._obs_code(cell)
                        cell_at[idx] = cell
        out = self._occlude(view, (r, r, r), self._BLOCKING)
        for idx, cell in cell_at.items():
            if out[idx] != C.OBS_UNSEEN:
                self._note_observed(cell, int(out[idx]))
        return out

    def _global_obs(self):
        base = self.layout.base
        grid = np.full(base.size, C.OBS_OUT_OF_WORLD, np.uint8)
        agent = np.zeros(base.size, np.uint8)
        for cell in base.cells():
            code = self._obs_code(cell)
            grid[cell] = code
            self._note_observed(cell, code)
        agent[self._agent_cell] = C.OBS_AGENT
        return np.stack([grid, agent])

    # -- rendering -----------------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            return render_rgb_3d(self)
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self):
        base = self.layout.base
        w, h, d = base.size
        chars = {
            C.OBS_EMPTY: "·", C.OBS_WALL: "#", C.OBS_HOLE: "O",
            C.OBS_DOOR_OPEN: "/", C.OBS_GOAL: "G", C.OBS_UNSEEN: "?",
            C.OBS_DOOR_ONEWAY: ">", C.OBS_TRAPDOOR: "v",
            C.OBS_OUT_OF_WORLD: " ",
        }
        slices = []
        for z in range(d):
            rows = []
            for y in range(h):
                row = []
                for x in range(w):
                    if (x, y, z) == self._agent_cell:
                        row.append("@")
                    else:
                        row.append(chars[self._obs_code((x, y, z))])
                rows.append("".join(row))
            slices.append(f"z={z}\n" + "\n".join(rows))
        return "\n\n".join(slices)
