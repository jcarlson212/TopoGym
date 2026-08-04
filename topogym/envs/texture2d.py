"""GridWorld2D-Texture: scenario environments with a live texture block.

The texture block of the universal observation is populated here: slots
0-3 carry directional blocker adjacency (obstacle to the left, right,
above, below of the current cell), slots 4-15 the scenario's semantic
features (:mod:`topogym.core.constants`). Scenario mechanics:

- **hazard** cells (DontFall's drop) end the episode when stepped on;
- **wormhole** cells (SpaceWarp) teleport the agent to their partner —
  recognizable on sight, destinations only by trying;
- the **clown** (ClownChase) wanders near the decoys and pays a tiny
  reward for every step that closes the distance to it, from a budget
  that runs out after a few thousand rewarding steps.
"""

from __future__ import annotations

import numpy as np

from topogym.core import constants as C
from topogym.envs.topo2d import TopoGrid2DEnv
from topogym.generation.config import TopoGenConfig2D
from topogym.generation.generator import Layout
from topogym.generation.scenarios import (
    SCENARIO_SIZES,
    SCENARIOS,
    build_scenario,
)
from topogym.rendering import tiles


class TextureGrid2DEnv(TopoGrid2DEnv):
    """A :class:`TopoGrid2DEnv` running a named Texture scenario."""

    def __init__(self, scenario: str, *, warp_sep: int = 2,
                 season: str | None = None, **kwargs):
        if scenario not in SCENARIOS:
            raise ValueError(
                f"unknown scenario {scenario!r}; choose from "
                f"{sorted(SCENARIOS)}"
            )
        if season not in (None, "summer", "winter"):
            raise ValueError('season must be "summer" or "winter"')
        self.scenario = scenario
        self.forced_season = season
        self._scenario_knobs = (
            {"warp_sep": warp_sep} if scenario == "space_warp" else {}
        )
        size = SCENARIO_SIZES[scenario]
        kwargs.setdefault("config", TopoGenConfig2D(base="square", size=size))
        super().__init__(**kwargs)

    def _generate(self, seed: int) -> Layout:
        return build_scenario(self.scenario, seed, **self._scenario_knobs)

    # -- episode state -------------------------------------------------------

    def _reset_runtime(self) -> None:
        super()._reset_runtime()
        self._fell = False
        self._decoy_cells = frozenset(
            c for f in self.layout.features if f.kind == "decoy"
            for c in f.cells
        )
        # Seasonal state (EnvironmentalIceShip): the season is drawn per
        # episode; winter freezes the channel shut cell by cell, summer
        # melts its flanks open. Frozen/melted sets are episode-local
        # overrides — the shared layout is never mutated.
        self._frozen: set = set()
        self._melted: set = set()
        seasonal = self.layout.extras.get("seasonal")
        if seasonal is not None:
            if self.forced_season is not None:
                self.season = self.forced_season
            else:
                self.season = ("summer" if self.np_random.random() < 0.5
                               else "winter")
        else:
            self.season = None
        clown = self.layout.extras.get("clown")
        if clown is not None:
            self._clown_pos = clown["anchor"]
            # The distractor budget spans the agent's lifetime on this
            # layout, not one episode: it runs out after a few thousand
            # rewarding steps in total.
            if getattr(self, "_clown_layout", None) is not self.layout:
                self._clown_budget = clown["budget"]
                self._clown_layout = self.layout
            self._clown_prev = self._dist_to_clown(self.layout.start)
        else:
            self._clown_pos = None

    # -- mechanics -----------------------------------------------------------

    def _advance_season(self) -> None:
        seasonal = self.layout.extras["seasonal"]
        t = self._steps + 1  # the step being taken
        if t < seasonal["start_step"]:
            return
        frontier = (t - seasonal["start_step"]) // seasonal["interval"] + 1
        channel = seasonal["channel"]
        waves = [
            i for i, at in enumerate(seasonal["wave_steps"]) if t >= at
        ]
        if self.season == "winter":
            self._frozen.update(channel[:frontier])
            for i in waves:  # the icescape freezes outward in waves
                self._frozen.update(seasonal["grow_layers"][i])
            cell = self._state.cell
            channel_closed = self._frozen >= set(channel)
            if cell in self._frozen:
                self._fell = True  # crushed by the growing ice
            elif channel_closed and cell in seasonal["inside"]:
                self._fell = True  # trapped behind the frozen channel
        else:  # summer: the flanks melt, and every ice rim recedes
            for pair in seasonal["flanks"][:frontier]:
                self._melted.update(pair)
            for i in waves:
                self._melted.update(seasonal["melt_layers"][i])

    def _post_move_hook(self) -> None:
        extras = self.layout.extras
        if self.season is not None:
            self._advance_season()
        cell = self._state.cell
        wormholes = extras.get("wormholes", {})
        if cell in wormholes:
            # Teleport; the frame re-canonicalizes (screen-up) on arrival.
            base = self.layout.base
            self._state = base.turn_left(
                base.initial_state(wormholes[cell])
            )
            self._visited.add(cell)
        if self._state.cell in extras.get("hazards", ()):
            self._fell = True
        if self._clown_pos is not None:
            self._move_clown()

    def _move_clown(self) -> None:
        clown = self.layout.extras["clown"]
        base = self.layout.base
        free = set(self.layout.free_cells)
        anchor, radius = clown["anchor"], clown["radius"]
        options = [
            n for n in base.neighbors(self._clown_pos)
            if n in free
            and max(abs(n[0] - anchor[0]), abs(n[1] - anchor[1])) <= radius
        ] + [self._clown_pos]
        self._clown_pos = options[int(self.np_random.integers(len(options)))]

    def _try_enter(self, frm: tuple, target: tuple) -> bool:
        if target in self._frozen:
            return False  # winter ice
        if target in self._melted:
            return True  # summer melt
        return super()._try_enter(frm, target)

    def _obs_code(self, cell: tuple) -> int:
        if cell in self._frozen:
            return C.OBS_WALL
        if cell in self._melted:
            return C.OBS_EMPTY
        return super()._obs_code(cell)

    def _dist_to_clown(self, cell: tuple) -> int:
        return (abs(cell[0] - self._clown_pos[0])
                + abs(cell[1] - self._clown_pos[1]))

    def _step_outcome(self, agent_cell: tuple) -> tuple:
        reward, terminated, truncated = super()._step_outcome(agent_cell)
        if self._fell:
            # The drop: fatal, rewardless, episode over.
            self._fell = False
            return 0.0, True, False
        if self._clown_pos is not None and self.reward_mode != "none":
            d = self._dist_to_clown(agent_cell)
            if d < self._clown_prev and self._clown_budget > 0:
                step_reward = self.layout.extras["clown"]["step_reward"]
                pay = min(step_reward, self._clown_budget)
                reward += pay
                self._clown_budget -= pay
            self._clown_prev = d
        return reward, terminated, truncated

    # -- observation textures --------------------------------------------------

    _ADJ = (((-1, 0), C.TEX_BLOCK_LEFT), ((1, 0), C.TEX_BLOCK_RIGHT),
            ((0, -1), C.TEX_BLOCK_ABOVE), ((0, 1), C.TEX_BLOCK_BELOW))

    def _texture_block(self, cell: tuple) -> np.ndarray:
        vec = np.zeros(C.TEXTURE_DIM, dtype=np.float32)
        types = self.layout.cell_types
        w, h = self.layout.base.layout_size()
        x, y = cell
        for (dx, dy), slot in self._ADJ:
            nx, ny = x + dx, y + dy
            nbr = (nx, ny)
            blocked = (
                not (0 <= nx < w and 0 <= ny < h)
                or nbr in self._frozen
                or (nbr not in self._melted and types.get(nbr, C.EMPTY)
                    in (C.WALL, C.HOLE))
            )
            if blocked:
                vec[slot] = 1.0
        for slot in self.layout.extras.get("textures", {}).get(cell, ()):
            vec[slot] = 1.0
        if self.season == "winter" and vec[C.TEX_WATER] == 1.0:
            vec[C.TEX_WATER] = 0.5  # cold water: the season is sensible
        if self._clown_pos is not None:
            near = max(abs(x - self._clown_pos[0]),
                       abs(y - self._clown_pos[1])) <= 1
            if near:
                vec[C.TEX_CLOWN_NEAR] = 1.0
        if self.goal_exists and cell == self.layout.goal:
            vec[C.TEX_TREASURE] = 1.0
        return vec

    # -- rendering -------------------------------------------------------------

    #: semantic texture slot -> floor tile, in display priority order
    _SLOT_TILES = (
        (C.TEX_LADDER, "ladder"),
        (C.TEX_BRIDGE, "bridge"),
        (C.TEX_WATER, "water"),
        (C.TEX_PLATFORM, "slab"),
        (C.TEX_HALLWAY, "hall"),
        (C.TEX_INTERIOR, "carpet"),
        (C.TEX_DIRT, "dirt"),
    )

    def _tile_name(self, cell: tuple, code: int) -> str:
        if code in (C.OBS_UNSEEN, C.OBS_OUT_OF_WORLD):
            return "unseen" if code == C.OBS_UNSEEN else "out"
        space = self.scenario == "space_warp"
        if code == C.OBS_WALL:
            if self.season is not None:
                return "ice_sun" if self.season == "summer" else "ice_cold"
            if self.scenario == "ice_ship":
                return "ice"
            if self.scenario == "clown_chase" \
                    and cell in self._decoy_cells:
                return "tent"  # the carnival grounds
            return "hull" if space else "stone"
        if code == C.OBS_HOLE and self.scenario == "search_rescue":
            return "shrapnel"
        if code == C.OBS_GOAL and self.scenario == "search_rescue":
            return "person"
        if code == C.OBS_HAZARD:
            return "drop"
        if code == C.OBS_WORMHOLE:
            return "wormhole"
        if code == C.OBS_DOOR_OPEN:
            return "hatch" if space else "door"
        if code == C.OBS_GOAL:
            return "chest"
        slots = self.layout.extras.get("textures", {}).get(cell, ())
        if self.season is not None and C.TEX_WATER in slots:
            return ("water_sun" if self.season == "summer"
                    else "water_cold")
        if space:  # outer space; station interiors are deck plating
            return "deck" if C.TEX_INTERIOR in slots else "space"
        for slot, name in self._SLOT_TILES:
            if slot in slots:
                return name
        return "floor"

    def _agent_tile(self) -> str | None:
        return "boat" if self.layout.extras.get("boat") else None

    def _render_overlay(self, img: np.ndarray, tile: int) -> None:
        if self._clown_pos is None or self._clown_pos == self._state.cell:
            return
        x, y = self.layout.base.layout_coords(self._clown_pos)
        img[y * tile:(y + 1) * tile, x * tile:(x + 1) * tile] = tiles.tile(
            "clown", tile
        )
