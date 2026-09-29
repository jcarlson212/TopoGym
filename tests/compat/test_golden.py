"""Every published env id behaves exactly as it did at 0.4.2.

See ``golden_harness.py`` for what a fixture records and COMPATIBILITY.md
for why a mismatch is a bug in the change, never in the fixture.

Recorded ids are compared in full. The cost is dominated by generating
the large worlds, so the rollout comparison is opt-in
(``TOPOGYM_GOLDEN=1``, optionally ``GOLDEN_SHARD=i/n``) and runs in its
own workflow, triggered only by changes that could alter env
behaviour; the cheap checks run everywhere. Ids registered after 0.4.2
have no fixture and are not checked here; they get one when they ship.
"""

from __future__ import annotations

import json
import os
import zlib

import golden_harness as gh
import pytest

RECORDED = sorted(p.stem.replace("__", "/")
                  for p in gh.FIXTURES.glob("*.json"))


def _in_shard(env_id: str) -> bool:
    """``GOLDEN_SHARD=i/n`` keeps the ids whose stable hash lands in
    shard ``i`` of ``n``, so CI can spread the suite over parallel jobs."""
    shard = os.environ.get("GOLDEN_SHARD")
    if not shard:
        return True
    i, n = (int(v) for v in shard.split("/"))
    return zlib.crc32(env_id.encode()) % n == i


SHARD = [i for i in RECORDED if _in_shard(i)]

#: Most of an hour in full, so opt-in: CI's golden job sets the variable.
golden = pytest.mark.skipif(
    not os.environ.get("TOPOGYM_GOLDEN"),
    reason="set TOPOGYM_GOLDEN=1 to run the golden rollouts")


def test_fixtures_exist():
    assert RECORDED, "no golden fixtures found"


def test_every_recorded_id_is_still_registered():
    """Removing an id is a breaking change; it must stay registered."""
    missing = sorted(set(RECORDED) - set(gh.env_ids()))
    assert not missing, f"published ids no longer registered: {missing}"


@pytest.mark.golden
@golden
@pytest.mark.parametrize("env_id", SHARD)
def test_golden(env_id):
    want = json.loads(gh.fixture_path(env_id).read_text())
    got = {"env_id": env_id, **gh.spec_record(env_id), "rollouts": {}}
    assert got["spec"] == want["spec"], (
        f"{env_id}: registered entry point or kwargs changed")
    try:
        for seed, obs_mode, actions in gh.combos():
            key = gh.combo_key(seed, obs_mode, actions)
            rec = gh.rollout(env_id, seed, obs_mode, actions)
            exp = want["rollouts"][key]
            for field in ("spaces", "reset", "info_keys", "resets"):
                assert rec[field] == exp[field], f"{env_id} {key}: {field}"
            for i, (a, b) in enumerate(zip(rec["checkpoints"],
                                           exp["checkpoints"])):
                assert a == b, (
                    f"{env_id} {key}: rollout diverges by step "
                    f"{gh.CHECKPOINTS[i]}")
    finally:
        from topogym.generation import cache

        cache.clear()
