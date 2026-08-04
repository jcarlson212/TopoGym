"""TopoGrid2D: an agent on a 2D base manifold.

Two agent interfaces (``actions=``):

- ``"fourway"`` (default, the spec's universal action space) —
  ``Discrete(4)``: 0 = up, 1 = down, 2 = left, 3 = right, in *screen*
  directions (up decreases y). Directions are taken in the agent's
  parallel-transported grid frame, so on plain bases they are absolute
  grid directions, and crossing an identified edge applies the
  identification map (coordinate and orientation reversal where the
  gluing flips) without changing the action set. Moving into an obstacle
  leaves the agent in place.
- ``"egocentric"`` — ``Discrete(3)``: 0 = turn left, 1 = turn right,
  2 = forward. The agent's local frame is parallel-transported as it
  moves: a Mobius/Klein/RP^2 seam mirrors its view.

Observations (``obs_mode=``): ``"vector"`` (default for fourway, the
spec's universal observation) — the agent's integer cell coordinates
``(x, y)`` followed by a 16-slot texture block in ``[0, 1]``, identically
zero outside the Texture variants; ``"local"`` (default for egocentric) —
occluded egocentric ``(2r+1, 2r+1)`` patches (agent at the center, facing
"up"), so chamber interiors — and whether a suspicious room is a decoy —
must be discovered by interaction; ``"global"`` — the full symbolic grid
plus an agent mask.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from topogym.core import constants as C
from topogym.envs.core import TopoEnvCore
from topogym.generation.config import TopoGenConfig2D
from topogym.generation.generator import Layout, _translate
from topogym.rendering.rgb import render_rgb_2d


class TopoGrid2DEnv(TopoEnvCore):
    DIM = 2

    # egocentric actions
    ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD = 0, 1, 2
    # fourway actions (the spec's universal action space)
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT = 0, 1, 2, 3

    #: quarter-turns right to face each fourway direction from "up"
    _FOURWAY_TURNS = {MOVE_UP: 0, MOVE_DOWN: 2, MOVE_LEFT: 3, MOVE_RIGHT: 1}

    def __init__(self, config: TopoGenConfig2D | dict | None = None, *,
                 actions: str = "fourway", **kwargs):
        if actions not in ("fourway", "egocentric"):
            raise ValueError(
                f'actions must be "fourway" or "egocentric", got {actions!r}'
            )
        self.actions = actions
        super().__init__(config, **kwargs)

    def _default_config(self) -> TopoGenConfig2D:
        return TopoGenConfig2D()

    def _config_class(self) -> type:
        return TopoGenConfig2D

    def _default_obs_mode(self) -> str:
        return "vector" if self.actions == "fourway" else "local"

    def _probe_layout(self) -> Layout:
        probe = self._fixed_layout
        if probe is None:
            probe = self._generate(self.layout_seed or 0)
            if self.layout_seed is not None:
                self._fixed_layout = probe
        return probe

    def _build_spaces(self) -> None:
        self.action_space = spaces.Discrete(
            4 if self.actions == "fourway" else 3
        )
        r = self.view_radius
        if self.obs_mode == "vector":
            w, h = self._probe_layout().base.layout_size()
            high = np.array(
                [w - 1, h - 1] + [1.0] * C.TEXTURE_DIM, dtype=np.float32
            )
            self.observation_space = spaces.Box(
                np.zeros_like(high), high, dtype=np.float32
            )
        elif self.obs_mode == "local":
            self.observation_space = spaces.Box(
                0, C.OBS_MAX, shape=(2 * r + 1, 2 * r + 1), dtype=np.uint8
            )
        elif self.obs_mode == "global":
            w, h = self._probe_layout().base.layout_size()
            self.observation_space = spaces.Box(
                0, C.OBS_MAX, shape=(2, h, w), dtype=np.uint8
            )
        else:
            raise ValueError(f"unknown obs_mode {self.obs_mode!r}")

    # -- gym API --------------------------------------------------------------

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple:
        super().reset(seed=seed)
        self._note_episode_end()
        self.layout = self._obtain_layout()
        self._reset_runtime()
        base = self.layout.base
        self._state = base.initial_state(self._resolve_start(options))
        if self.actions == "egocentric":
            for _ in range(int(self.np_random.integers(4))):
                self._state = base.turn_left(self._state)
        else:
            # The canonical frame faces +x; a left turn makes "forward"
            # screen-up so fourway actions are screen directions.
            self._state = base.turn_left(self._state)
        self._visited.add(self._state.cell)
        return self._obs(), self._reset_info(self._state.cell)

    def step(self, action: int) -> tuple:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")
        action = self._maybe_slip(action)
        if self.actions == "fourway":
            self._step_fourway(action)
        else:
            self._step_egocentric(action)
        self._post_move_hook()
        reward, terminated, truncated = self._step_outcome(self._state.cell)
        self._episode_return += reward  # the final reward, bonuses included
        info = self._step_info(self._state.cell)
        if self._debug:
            self._debug_step(action, reward, terminated, truncated, info)
        return self._obs(), reward, terminated, truncated, info

    def _step_egocentric(self, action: int) -> None:
        base = self.layout.base
        if action == self.ACTION_LEFT:
            self._state = base.turn_left(self._state)
        elif action == self.ACTION_RIGHT:
            self._state = base.turn_right(self._state)
        else:  # forward
            nxt = base.forward(self._state)
            if nxt is not None and self._try_enter(self._state.cell, nxt.cell):
                self._on_leave(self._state.cell)
                self._state = nxt

    def _step_fourway(self, action: int) -> None:
        """Move one cell in the given *screen* direction. The frame is
        re-canonicalized (screen-up) after every move: crossing an
        identified edge remaps the agent's position per the gluing, but
        never reinterprets its actions — up stays up on every base.
        (The egocentric interface keeps true frame transport; there the
        Möbius mirror is the point.)"""
        base = self.layout.base
        turns = self._FOURWAY_TURNS[action]
        t = self._state
        for _ in range(turns):
            t = base.turn_right(t)
        nxt = base.forward(t)
        if nxt is not None and self._try_enter(t.cell, nxt.cell):
            self._on_leave(self._state.cell)
            self._state = base.turn_left(base.initial_state(nxt.cell))

    def _post_move_hook(self) -> None:
        """Cell mechanics that trigger after movement (hazards, wormholes,
        NPCs); the Texture variants override this."""

    # -- observations -----------------------------------------------------------

    def _texture_block(self, cell: tuple) -> np.ndarray:
        """The 16-slot texture block of the universal observation.
        Identically zero outside the Texture variants, which override it."""
        return np.zeros(C.TEXTURE_DIM, dtype=np.float32)

    def _obs(self) -> np.ndarray:
        if self.obs_mode == "global":
            return self._global_obs()
        patch = self._sight_patch()
        if self.obs_mode == "local":
            return patch
        x, y = self.layout.base.layout_coords(self._state.cell)
        vec = np.empty(2 + C.TEXTURE_DIM, dtype=np.float32)
        vec[0], vec[1] = x, y
        vec[2:] = self._texture_block(self._state.cell)
        return vec

    def visible_cells(self) -> set:
        """The cells the agent can currently see (line of sight within
        the view radius); everything else renders dimmed."""
        if self.obs_mode == "global":
            return set(self.layout.base.cells())
        self._sight_patch()
        return self._visible

    def _sight_patch(self) -> np.ndarray:
        """The occluded egocentric patch; also feeds the observed-region
        filtration (what the agent can currently see counts as observed in
        every observation mode)."""
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
        visible = set()
        for idx, cell in cell_at.items():
            if out[idx] != C.OBS_UNSEEN:
                visible.add(cell)
                self._note_observed(cell, int(out[idx]))
        self._visible = visible
        return out

    def _global_obs(self) -> np.ndarray:
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

    def render(self) -> np.ndarray | str | None:
        if self.render_mode == "rgb_array":
            return render_rgb_2d(self)
        if self.render_mode == "human":
            if getattr(self, "_window", None) is None:
                from topogym.rendering.window import Window

                self._window = Window(
                    title=f"TopoGym ({self.cfg.base})",
                    fps=self.metadata["render_fps"],
                )
            w, h = self.layout.base.layout_size()
            tile = max(3, 720 // max(w, h))
            self._window.show(render_rgb_2d(self, tile=tile))
            return None
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def close(self) -> None:
        window = getattr(self, "_window", None)
        if window is not None:
            window.close()
            self._window = None

    _ANSI = {
        C.OBS_EMPTY: "·", C.OBS_WALL: "#", C.OBS_HOLE: "O",
        C.OBS_DOOR_OPEN: "/", C.OBS_GOAL: "G", C.OBS_OUT_OF_WORLD: " ",
        C.OBS_UNSEEN: "?", C.OBS_HAZARD: "!", C.OBS_WORMHOLE: "*",
    }

    def _render_ansi(self) -> str:
        base = self.layout.base
        w, h = base.layout_size()
        rows = [[" "] * w for _ in range(h)]
        for cell in base.cells():
            x, y = base.layout_coords(cell)
            rows[y][x] = self._ANSI[self._obs_code(cell)]
        ax, ay = base.layout_coords(self._state.cell)
        rows[ay][ax] = "@"
        return "\n".join("".join(r) for r in rows)
