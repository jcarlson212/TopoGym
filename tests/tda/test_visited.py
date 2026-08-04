"""VisitedComplex: incremental visited-state topology, three backends."""

import logging
import math

import gymnasium as gym
import pytest

import topogym  # noqa: F401
from topogym.core import make_base_map_2d
from topogym.tda import VisitedComplex
from topogym.tda.chains import smith_invariants


def _ring(x0, y0, x1, y1):
    return [(x, y) for x in range(x0, x1 + 1) for y in (y0, y1)] + \
        [(x, y) for y in range(y0, y1 + 1) for x in (x0, x1)]


def test_smith_invariants_known_matrix():
    cols = [{0: 2, 1: -6, 2: 10}, {0: 4, 1: 6, 2: 4},
            {0: 4, 1: 12, 2: 16}]
    assert smith_invariants(cols) == [2, 2, 156]


def test_cubical_ring_incremental():
    vc = VisitedComplex("cubical", base=make_base_map_2d("square", 12))
    vc.add(_ring(2, 2, 8, 8))
    assert vc.betti() == (1, 1)
    (rep,) = vc.representatives()
    assert set(rep) <= set(vc.points)  # a loop of visited cells
    (cls,) = vc.rims()
    assert cls["rim"] <= cls["cycle"]
    # Fill the pocket: the class dies, incrementally.
    vc.add([(x, y) for x in range(2, 9) for y in range(2, 9)])
    assert vc.betti() == (1, 0)
    assert vc.representatives() == []
    assert vc.rims() == []


def test_cubical_diagonal_touch_stays_disconnected():
    vc = VisitedComplex("cubical", base=make_base_map_2d("square", 8))
    vc.add([(1, 1), (2, 2)])  # movement cannot cross a pinch
    assert vc.betti()[0] == 2


def test_fields_and_torsion_on_klein():
    kb = make_base_map_2d("klein", 9)
    results = {}
    for coeff in (2, 3, "Z"):
        vc = VisitedComplex("cubical", base=kb, coefficients=coeff,
                            max_dim=2)
        vc.add(list(kb.cells()))
        results[coeff] = vc.betti()
    assert results[2] == (1, 2, 1)      # F2 sees the torsion class
    assert results[3] == (1, 1, 0)      # F3 does not
    assert results["Z"] == (1, 1, 0)    # free part
    assert vc.torsion(1) == (2,)        # H1 = Z + Z/2


def test_torus_integral():
    tb = make_base_map_2d("torus", 9)
    vc = VisitedComplex("cubical", base=tb, coefficients="Z", max_dim=2)
    vc.add(list(tb.cells()))
    assert vc.betti() == (1, 2, 1)
    assert vc.torsion(1) == ()
    assert len(vc.representatives()) == 2


def test_vr_circle_and_sphere():
    pts = [(round(10 + 6 * math.cos(2 * math.pi * i / 24), 3),
            round(10 + 6 * math.sin(2 * math.pi * i / 24), 3))
           for i in range(24)]
    vr = VisitedComplex("vr", epsilon=2.0)
    vr.add(pts)
    assert vr.betti() == (1, 1)
    (rep,) = vr.representatives()
    assert len(rep) >= 3
    # H2 with max_dim=2: the octahedron sphere.
    s2 = VisitedComplex("vr", epsilon=1.5, max_dim=2)
    s2.add([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)])
    assert s2.betti() == (1, 0, 1)


def test_witness_ring_default_policy():
    dense = [(round(20 + 8 * math.cos(2 * math.pi * i / 120), 3),
              round(20 + 8 * math.sin(2 * math.pi * i / 120), 3))
             for i in range(120)]
    w = VisitedComplex("witness", landmark_radius=3.0, relaxation=0.4)
    w.add(dense)
    assert len(w.landmarks) < len(w.points) // 4  # genuinely sparse
    assert w.betti() == (1, 1)
    (rep,) = w.representatives()
    assert set(rep) <= set(w.landmarks)


def test_witness_custom_policy_admit_and_evict():
    def budget(point, landmarks, dist):
        if len(landmarks) < 6:
            return True, None
        return True, landmarks[0]  # ring buffer of 6

    w = VisitedComplex("witness", landmark_policy=budget)
    w.add([(float(i), 0.0) for i in range(30)])
    assert len(w.landmarks) == 6


def test_from_env_seeds_lifetime_visits():
    env = gym.make("TopoGym/Dilution-50-v0", seed=1,
                   actions="fourway").unwrapped
    env.reset(seed=0)
    for a in (0, 3, 1, 2):
        env.step(a)
    vc = VisitedComplex.from_env(env)
    assert set(vc.points) == set(env.lifetime_visit_counts)
    assert vc.betti()[0] == 1  # one trail: connected


def test_deterministic_and_logged(caplog):
    def run():
        vc = VisitedComplex("cubical",
                            base=make_base_map_2d("square", 10))
        vc.add(_ring(1, 1, 7, 7))
        return vc.betti(), vc.representatives()

    with caplog.at_level(logging.DEBUG, logger="topogym"):
        a = run()
    assert "visited-complex" in caplog.text
    assert a == run()


def test_validation():
    with pytest.raises(ValueError):
        VisitedComplex("cubical")  # no base
    with pytest.raises(ValueError):
        VisitedComplex("vr", coefficients=4)  # not prime
    with pytest.raises(ValueError):
        VisitedComplex("delaunay")
