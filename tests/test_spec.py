"""The compositional spec API: primitives, modifiers, products, compile."""

import pytest

from topogym.spec import (
    Annulus,
    Circle,
    Interval,
    Klein,
    Mobius,
    Product,
    Sphere,
    Square,
    Torus,
    XHoles,
)


def test_primitives_are_bare():
    # A primitive is pure topology: no default holes/chambers/decoys.
    md = Torus(8).metadata(seed=1)
    assert md.betti_z2 == (1, 2, 1)
    assert md.n_holes == md.n_chambers == md.n_decoys == 0
    assert Sphere(4).metadata(seed=1).betti_z2 == (1, 0, 1)
    assert Annulus(10).metadata(seed=1).betti_z2 == (1, 1, 0)
    assert XHoles(14, 3).metadata(seed=1).betti_z2 == (1, 3, 0)


def test_fluent_modifiers_are_immutable():
    base = Torus(12)
    with_holes = base.holes(3)
    assert base.cfg.n_holes == 0
    assert with_holes.cfg.n_holes == 3
    md = with_holes.chambers(1).metadata(seed=2)
    # Torus loops (2) + 4 obstacles - 1 (the first puncture of a closed
    # surface kills b2 instead of adding a loop) = 5.
    assert md.betti_z2 == (1, 5, 0)
    assert md.certified["betti_z2"]


def test_compile_produces_working_env():
    env = Square(8).holes(1).compile(seed=3, max_steps=50)
    obs, info = env.reset(seed=0)
    assert info["topology"]["betti_z2"] == [1, 1, 0]
    for action in (2, 0, 2, 1, 2):
        obs, reward, terminated, truncated, info = env.step(action)
    assert info["steps"] == 5


def test_1d_products_normalize_to_surfaces():
    assert (Circle(6) * Circle(5)).name == "torus"
    assert (Circle(6) * Interval(5)).name == "cylinder"
    assert (Interval(5) * Circle(6)).name == "cylinder"
    assert (Interval(5) * Interval(6)).name == "square"
    assert (Circle(6) * Circle(5)).metadata().betti_z2 == (1, 2, 1)
    # The result is a full 2D spec: modifiers apply.
    md = (Circle(8) * Circle(8)).holes(2).metadata(seed=1)
    assert md.betti_z2 == (1, 3, 0)  # 2 + 2 obstacles - 1 (closed surface)


def test_product_layout_certifies_kunneth():
    md = (Annulus(12) * Circle(6)).metadata(seed=3)
    assert md.betti_z2 == (1, 2, 1, 0)  # (1,1,0) x (1,1)
    assert md.betti_q == (1, 2, 1, 0)
    assert md.product["kunneth_cross_check"] == "passed"
    assert md.certified["betti_z2"] and md.certified["betti_q"]
    assert md.base_map == "annulus x circle"
    assert md.dim == 3

    md = (Torus(8) * Circle(5)).metadata(seed=1)
    assert md.betti_z2 == (1, 3, 3, 1)  # T^3

    md = (Square(8) * Interval(4)).metadata(seed=1)
    assert md.betti_z2 == (1, 0, 0, 0)  # a box


def test_product_env_runs_and_is_fixed():
    spec = Annulus(10) * Circle(5)
    env = spec.compile(seed=2, max_steps=40)
    obs, info = env.reset(seed=0)
    assert info["topology"]["betti_z2"] == [1, 2, 1, 0]
    start = info["position"]
    for action in range(6):
        obs, reward, terminated, truncated, info = env.step(action)
    # The injected layout is fixed across resets.
    obs2, info2 = env.reset(seed=7)
    assert info2["position"] == start


def test_products_refuse_what_they_cannot_lift():
    with pytest.raises(NotImplementedError, match="doors"):
        (Torus(12).chambers(1) * Circle(5)).layout(seed=1)
    with pytest.raises(NotImplementedError, match="flip-free"):
        (Mobius(8) * Circle(5)).layout(seed=1)
    with pytest.raises(TypeError, match="unsupported product"):
        Product(Torus(8), Torus(8))


def test_unliftable_products_still_have_homology():
    product = Mobius(5) * Circle(4)
    assert product.betti() == product.betti(method="direct")
    klein_x_circle = Klein(4) * Circle(4)
    assert klein_x_circle.betti(2) == (1, 3, 3, 1)  # Kunneth over Z/2
    assert klein_x_circle.betti(3) == (1, 2, 1, 0)  # torsion invisible /Z3
