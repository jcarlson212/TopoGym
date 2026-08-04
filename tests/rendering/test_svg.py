"""The SVG still renderer (used by scripts/new_env.py)."""

from topogym.generation import TopoGenConfig2D, generate_2d
from topogym.rendering.svg import layout_to_svg


def test_layout_to_svg_marks_structure():
    cfg = TopoGenConfig2D(base="square", size=17, n_holes=1, n_chambers=1,
                          n_decoys=1, door_kind="open")
    layout = generate_2d(cfg, seed=1)
    svg = layout_to_svg(layout)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<rect") >= 17 * 17
    assert "#a17438" in svg  # the wood door
    assert "#923f3f" in svg  # revealed decoy walls
    assert "#27ae60" in svg  # the goal


def test_reveal_false_hides_decoys():
    cfg = TopoGenConfig2D(base="square", size=17, n_holes=0, n_chambers=1,
                          n_decoys=1)
    layout = generate_2d(cfg, seed=2)
    hidden = layout_to_svg(layout, reveal=False)
    assert "#923f3f" not in hidden  # decoys look like walls
    assert "#9b59b6" not in hidden  # bump doors look like walls
