"""End-to-end determinism up to seeds — a library guarantee.

(config, seed) fixes the layout byte-for-byte, including the certified
metadata computed through GUDHI; (env, reset seed, action sequence) fixes
the episode, including ``p_slip`` slips. The cross-process test proves the
guarantee does not lean on any per-process state (hash order, caches):
two fresh interpreters must produce identical results.
"""

import json
import subprocess
import sys

_PROBE = r"""
import hashlib
import json

import gymnasium as gym

import topogym  # noqa: F401
from topogym.generation import TopoGenConfig2D, generate_2d

cfg = TopoGenConfig2D(base="torus", size=13, n_holes=1, n_chambers=1,
                      n_decoys=1)
layout = generate_2d(cfg, seed=5)
signature = {
    "cells": sorted(map(repr, layout.cell_types.items())),
    "doors": sorted(map(repr, layout.doors)),
    "start": repr(layout.start),
    "goal": repr(layout.goal),
    "metadata": layout.metadata.to_dict(),
}

env = gym.make("TopoGym/Grid2D-v0", config=cfg, layout_seed=5,
               p_slip=0.3).unwrapped
obs, _ = env.reset(seed=3)
h = hashlib.sha256(obs.tobytes())
for t in range(60):
    obs, r, term, trunc, info = env.step(t % 4)
    h.update(obs.tobytes())
    h.update(repr((r, term, trunc, info["position"])).encode())
signature["rollout_sha"] = h.hexdigest()

print(json.dumps(signature, sort_keys=True, default=repr))
"""


def _run_probe() -> dict:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def test_generation_and_rollout_identical_across_processes():
    assert _run_probe() == _run_probe()
