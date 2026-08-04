"""Machine-readable, certified topology metadata attached to every env.

Every generated environment carries a :class:`TopologyMetadata` record so
that experiments can be swept, filtered, and analyzed programmatically
(``env.unwrapped.topology`` and the ``info`` dict at ``reset``).

Certification levels
--------------------
- ``betti_z2`` is always **certified**: computed from the actual free-space
  cubical complex by :mod:`topogym.core.homology` at generation time.
- ``betti_q`` (integral/rational Betti numbers) is certified for every
  environment (a compact surface's homology is determined by its Z/2 data
  plus orientability).
- ``h1_torsion`` follows the same rule (e.g. ``("Z/2",)`` for a fully-free
  RP^2 or Klein bottle; puncturing a closed surface removes the torsion).

Note that the homology of the free space is invariant to door state: a
chamber's wall footprint blocks the same loop whether its hidden door is
open or closed. What doors gate is *coverage*, not homology.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def homology_strings(betti_q: tuple | None, h1_torsion: tuple,
                     betti_z2: tuple) -> dict:
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
    base_map: str  # includes presets: "annulus", "x_holes"
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
    #: Betti numbers of the *traversable* free space: door cells count as
    #: free, so homology is door-state invariant and b0 = 1.
    betti_z2: tuple
    #: The sealed-world convention: doors count as walls. Each doored
    #: chamber's interior becomes its own component (b0 grows) and its
    #: enclosing wall reads as one closed class. Same complex, second
    #: certified reading.
    betti_z2_sealed: tuple
    euler_characteristic: int
    orientable: bool | None
    genus: int | None  # orientable free space
    demigenus: int | None  # non-orientable free space
    n_boundary_components: int | None

    # -- integral homology (see module docstring for certification) --------
    betti_q: tuple | None
    betti_q_expected: tuple
    h1_torsion: tuple | None

    # -- bottleneck structure (certified difficulty descriptors) -----------
    # Bridges are not extra topology: discovering a passage is frontier
    # growth, an H0 merge, or an H1 birth of the observed-region filtration.
    # This block says how bottlenecked the free-cell graph is, i.e. how rare
    # and late those events are under naive exploration. Doors count as
    # passable.
    connectivity: dict = field(default_factory=dict)

    n_partitions: int = 0  # dividing walls/moats with bridge passages

    certified: dict = field(default_factory=dict)
    homology: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable dict (for logging / sweeping / pandas)."""
        d = asdict(self)
        for key, val in d.items():
            if isinstance(val, tuple):
                d[key] = list(val)
        return d
