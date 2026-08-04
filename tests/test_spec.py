"""The compositional spec API: primitives, modifiers, compile."""

from topogym.spec import (
    Annulus,
    Torus,
    XHoles,
)


def test_primitives_are_bare():
    # A primitive is pure topology: no default holes/chambers/decoys.
    md = Torus(8).metadata(seed=1)
    assert md.betti_z2 == (1, 2, 1)
    assert md.n_holes == md.n_chambers == md.n_decoys == 0
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
    from topogym.spec import Square

    env = Square(8).holes(1).compile(seed=3, max_steps=50)
    obs, info = env.reset(seed=0)
    assert info["topology"]["betti_z2"] == [1, 1, 0]
    for action in (2, 0, 2, 1, 2):
        obs, reward, terminated, truncated, info = env.step(action)
    assert info["steps"] == 5
