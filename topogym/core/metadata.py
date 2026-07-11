"""Machine-readable, certified topology metadata attached to every env.

Every generated environment carries a :class:`TopologyMetadata` record so
that experiments can be swept, filtered, and analyzed programmatically
(``env.unwrapped.topology`` and the ``info`` dict at ``reset``).

Certification levels
--------------------
- ``betti_z2`` is always **certified**: computed from the actual free-space
  cubical complex by :mod:`topogym.core.homology` at generation time.
- ``betti_q`` (integral/rational Betti numbers) is certified for all 2D
  environments (a compact surface's homology is determined by its Z/2 data
  plus orientability) and for obstacle-free 3D bases; for 3D environments
  with obstacles we report the analytic expectation in
  ``betti_q_expected`` and leave ``betti_q`` as ``None``.
- ``h1_torsion`` follows the same rule (e.g. ``("Z/2",)`` for a fully-free
  RP^2 or Klein bottle; puncturing a closed surface removes the torsion).

Note that the homology of the free space is invariant to door state: a
chamber's wall footprint blocks the same loop whether its hidden door is
open or closed. What doors gate is *coverage*, not homology.

Directional asymmetry
---------------------
Betti numbers describe the *undirected* shape of the free space. Mechanics
like one-way doors and trapdoors make the transition graph *directed*, which
homology cannot see. Every environment therefore also carries a canonical
``asymmetry`` block, certified by analyzing the actual directed transition
graph (with bump-doors treated as open and trapdoors as not yet used — the
"optimistic" graph):

- ``is_symmetric``: no directed mechanics at all.
- ``mechanisms``: canonical door-level tags, from {"one_way_door",
  "trapdoor"}.
- ``feature_counts``: counts of directed feature kinds, from
  {"trap_room", "airlock", "trapdoor_room"}.
- ``n_sccs`` / ``largest_scc_frac``: strongly connected components of the
  optimistic transition graph over free cells.
- ``n_absorbing_sccs``: SCCs with no outgoing edges (trap regions).
- ``goal_in_start_scc``: whether the task is completable without ever
  making an irreversible move.
- ``n_consumable_transitions``: trapdoor passages (edges that exist only
  once per episode; SCC stats treat them optimistically).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def homology_strings(betti_q, h1_torsion, betti_z2) -> dict:
    """Human-readable homology groups, e.g. ``{"H1": "Z^2 + Z/2"}``."""
    out = {}
    if betti_q is not None:
        for k, b in enumerate(betti_q):
            parts = []
            if b == 1:
                parts.append("Z")
            elif b > 1:
                parts.append(f"Z^{b}")
            if k == 1 and h1_torsion:
                parts.extend(h1_torsion)
            out[f"H{k}"] = " + ".join(parts) if parts else "0"
    else:
        for k, b in enumerate(betti_z2):
            val = "0" if b == 0 else ("Z/2" if b == 1 else f"(Z/2)^{b}")
            out[f"H{k}"] = f"{val} (Z/2 coefficients)"
    return out


@dataclass(frozen=True)
class TopologyMetadata:
    """Everything an experiment needs to know about one environment."""

    # -- identity ----------------------------------------------------------
    dim: int
    base_map: str  # includes presets: "annulus", "x_holes", "shell"
    base: dict  # BaseMapInfo of the underlying manifold, as a dict
    size: tuple
    style: str  # "rooms" | "maze" | "zigzag"
    layout_seed: int

    # -- composition -------------------------------------------------------
    n_holes: int  # solid obstacles (incl. preset base holes)
    n_chambers: int  # enclosed rooms with hidden doors
    n_decoys: int  # chamber look-alikes with no entrance
    door_tries: tuple  # bumps required per door, sorted
    n_cells: int
    n_free_cells: int

    # -- certified topology of the free space ------------------------------
    betti_z2: tuple
    euler_characteristic: int
    orientable: bool | None  # 2D only
    genus: int | None  # 2D, orientable free space
    demigenus: int | None  # 2D, non-orientable free space
    n_boundary_components: int | None  # 2D only

    # -- integral homology (see module docstring for certification) --------
    betti_q: tuple | None
    betti_q_expected: tuple
    h1_torsion: tuple | None

    # -- directional asymmetry (see module docstring) -----------------------
    asymmetry: dict = field(default_factory=dict)

    certified: dict = field(default_factory=dict)
    homology: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable dict (for logging / sweeping / pandas)."""
        d = asdict(self)
        for key, val in d.items():
            if isinstance(val, tuple):
                d[key] = list(val)
        return d
