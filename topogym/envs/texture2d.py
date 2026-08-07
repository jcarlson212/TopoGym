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
                 season: str | None = None, n_clowns: int = 2,
                 **kwargs):
        if scenario not in SCENARIOS:
            raise ValueError(
                f"unknown scenario {scenario!r}; choose from "
                f"{sorted(SCENARIOS)}"
            )
        if season not in (None, "summer", "winter"):
            raise ValueError('season must be "summer" or "winter"')
        self.scenario = scenario
        self.forced_season = season
        if scenario == "space_warp":
            self._scenario_knobs = {"warp_sep": warp_sep}
        elif scenario == "clown_chase":
            self._scenario_knobs = {"n_clowns": n_clowns}
        else:
            self._scenario_knobs = {}
        size = SCENARIO_SIZES[scenario]
        kwargs.setdefault("config", TopoGenConfig2D(base="square", size=size))
        if scenario == "search_rescue":
            # The rescue traversal earns 56% more steps by default
            # (a 30% and a further 20% extension).
            pass  # horizon derives from the optimal route (core)
        super().__init__(**kwargs)

    def _generate(self, seed: int) -> Layout:
        from topogym.generation.cache import cached_layout

        return cached_layout(
            ("texture", self.scenario,
             tuple(sorted(self._scenario_knobs.items())), seed),
            lambda: build_scenario(self.scenario, seed,
                                   **self._scenario_knobs),
        )

    # -- episode state -------------------------------------------------------

    def _reset_runtime(self) -> None:
        super()._reset_runtime()
        self._fell = False
        self._hit_ice = False
        self._decoy_cells = frozenset(
            c for f in self.layout.features if f.kind == "decoy"
            for c in f.cells
        )
        self._chamber_walls = frozenset(
            c for f in self.layout.features if f.kind == "chamber"
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
            self._clowns = list(clown["anchors"])
            # The troupe's shared budget spans the agent's lifetime on
            # this layout, not one episode: it runs out after a few
            # thousand rewarding steps in total.
            if getattr(self, "_clown_layout", None) is not self.layout:
                self._clown_budget = clown["budget"]
                self._clown_layout = self.layout
            self._clown_prev = self._dist_to_clown(self.layout.start)
        else:
            self._clowns = []

    # -- mechanics -----------------------------------------------------------

    def _planning_teleport(self, cell: tuple) -> tuple:
        """Wormholes move the agent on entry, so a route to a chamber
        reachable only through one is a real route (SpaceWarp's treasure
        interior is spatially its own component by construction)."""
        return self.layout.extras.get("wormholes", {}).get(cell, cell)

    def _planning_blocked(self) -> set:
        """Worst case over the season schedule: every cell any wave can
        freeze counts as blocked, so a route that survives this survives
        the whole episode (ice only ever shrinks from the maximum) and
        every season draw is solvable, not just the lucky one."""
        blocked = super()._planning_blocked()
        seasonal = self.layout.extras.get("seasonal")
        if seasonal:
            for layer in seasonal.get("grow_layers", ()):
                blocked |= set(layer)
        return blocked

    def _sight_state(self) -> tuple:
        return super()._sight_state() + (
            self.season, len(self._frozen), len(self._melted)
        )

    def _advance_season(self) -> None:
        """Winter grows the floating bergs (their water fringe freezes
        in waves; being there when it freezes ends the episode); summer
        melts their rims. Channels and the landmass never change."""
        seasonal = self.layout.extras["seasonal"]
        t = self._steps + 1  # the step being taken
        waves = [
            i for i, at in enumerate(seasonal["wave_steps"]) if t >= at
        ]
        if self.season == "winter":
            for i in waves:
                self._frozen.update(seasonal["grow_layers"][i])
            if self._state.cell in self._frozen:
                self._fell = True  # frozen over by the growing berg
        else:
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
        if self._clowns:
            self._move_clowns()

    def _move_clowns(self) -> None:
        clown = self.layout.extras["clown"]
        base = self.layout.base
        free = set(self.layout.free_cells)
        radius = clown["radius"]
        for i, (pos, anchor) in enumerate(
            zip(self._clowns, clown["anchors"])
        ):
            options = [
                n for n in base.neighbors(pos)
                if n in free
                and max(abs(n[0] - anchor[0]),
                        abs(n[1] - anchor[1])) <= radius
            ] + [pos]
            self._clowns[i] = options[
                int(self.np_random.integers(len(options)))
            ]

    def _try_enter(self, frm: tuple, target: tuple) -> bool:
        boat = self.layout.extras.get("boat", False)
        if target in self._frozen:
            if boat:
                self._hit_ice = True  # ramming seasonal ice
            return False
        if target in self._melted:
            return True  # summer melt
        ok = super()._try_enter(frm, target)
        if not ok and boat and self.layout.cell_types.get(
            target, C.EMPTY
        ) in (C.WALL, C.HOLE):
            self._hit_ice = True  # hitting ice hurts the sailboat
        return ok

    def _obs_code(self, cell: tuple) -> int:
        if cell in self._frozen:
            return C.OBS_WALL
        if cell in self._melted:
            return C.OBS_EMPTY
        return super()._obs_code(cell)

    def _dist_to_clown(self, cell: tuple) -> int:
        """Manhattan distance to the nearest clown of the troupe."""
        return min(
            abs(cell[0] - cx) + abs(cell[1] - cy)
            for cx, cy in self._clowns
        )

    def _step_outcome(self, agent_cell: tuple) -> tuple:
        reward, terminated, truncated = super()._step_outcome(agent_cell)
        if self._fell or self._hit_ice:
            # The drop, growing ice, or an ice collision: episode over.
            self._fell = False
            self._hit_ice = False
            return 0.0, True, False
        if self._clowns and self.reward_mode != "none":
            d = self._dist_to_clown(agent_cell)
            if d < self._clown_prev and self._clown_budget > 0:
                step_reward = self.layout.extras["clown"]["step_reward"]
                pay = min(step_reward, self._clown_budget)
                reward += pay
                self._clown_budget -= pay
            self._clown_prev = d
        return reward, terminated, truncated

    def _debug_extras(self) -> dict:
        out: dict = {}
        if self.season is not None:
            out["season"] = self.season
            out["frozen"] = len(self._frozen)
            out["melted"] = len(self._melted)
        if self._clowns:
            out["clowns"] = list(self._clowns)
            out["clown_budget"] = round(self._clown_budget, 4)
        hazards = self.layout.extras.get("hazards")
        if hazards:
            out["hazards"] = len(hazards)
        return out

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
        if self._clowns and any(
            max(abs(x - cx), abs(y - cy)) <= 1 for cx, cy in self._clowns
        ):
            vec[C.TEX_CLOWN_NEAR] = 1.0
        if self.goal_exists and cell == self.layout.goal:
            vec[C.TEX_TREASURE] = 1.0
        return vec

    def _texture_patch(self) -> np.ndarray:
        """Per-cell blocks over the field of view.

        Only visible cells are annotated: the blocker slots read
        ``layout.cell_types`` directly, so filling occluded cells would
        hand the agent wall structure it has not seen -- information no
        other observation mode grants.
        """
        n = 2 * self.view_radius + 1
        out = np.zeros((n, n, C.TEXTURE_SLOTS.dim), dtype=np.float32)
        for index, cell in self._cell_at.items():
            if cell in self._visible:
                out[index] = self._texture_block(cell)
        return out

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
            if self.scenario == "search_rescue":
                # Intact chamber walls read as stone; everything else
                # collapsed into rubble.
                return ("stone" if cell in self._chamber_walls
                        else "rubble")
            return "hull" if space else "stone"
        if self.scenario == "search_rescue":
            if code == C.OBS_GOAL:
                return "person"
            if code == C.OBS_HAZARD:
                return "barrel"
            if code == C.OBS_EMPTY:
                return "concrete"
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
        if self.scenario == "dont_fall":
            if C.TEX_INTERIOR in slots:
                return "carpet"
            if cell in self.layout.extras.get("forest", ()):
                x, y = cell
                return "tree" if (x * 7 + y * 13) % 5 < 3 \
                    else "dirt_dark"
            return "dirt_dark"
        for slot, name in self._SLOT_TILES:
            if slot in slots:
                return name
        return "floor"

    def _agent_tile(self) -> str | None:
        return "boat" if self.layout.extras.get("boat") else None

    def _render_overlay(self, img: np.ndarray, tile: int) -> None:
        for pos in self._clowns:
            if pos == self._state.cell:
                continue
            x, y = self.layout.base.layout_coords(pos)
            img[y * tile:(y + 1) * tile,
                x * tile:(x + 1) * tile] = tiles.tile("clown", tile)
