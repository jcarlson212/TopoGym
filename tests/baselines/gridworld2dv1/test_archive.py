"""Layout-keyed archives, and the actor that shares them."""

import pytest

from topogym.baselines.gridworld2dv1.archive import (
    ArchiveService,
    LayoutArchive,
    adjacency_of,
    layout_fingerprint,
    remote_archive,
)
from topogym.baselines.gridworld2dv1.instances import (
    load_split,
    make_instance,
)


def _world(index=0):
    env = make_instance(load_split("test")[index]).unwrapped
    env.reset(seed=0)
    return env


def test_fingerprint_identifies_the_world_not_the_object():
    """Two processes exploring the same world must reach the same
    archive; two different worlds must never share one."""
    first, again, other = _world(0), _world(0), _world(1)
    assert first.layout is not again.layout          # distinct objects
    assert layout_fingerprint(first.layout) == \
        layout_fingerprint(again.layout)
    assert layout_fingerprint(first.layout) != \
        layout_fingerprint(other.layout)


def test_fingerprint_notices_a_changed_world():
    env = _world(0)
    before = layout_fingerprint(env.layout)
    cell = env.layout.free_cells[0]
    env.layout.cell_types[cell] = 1  # wall it off
    assert layout_fingerprint(env.layout) != before


def test_service_keeps_one_archive_per_world():
    first, other = _world(0), _world(1)
    service = ArchiveService(seed=0)
    keys = [layout_fingerprint(w.layout) for w in (first, other)]
    assert service.register(keys[0], adjacency_of(first.layout)) is True
    assert service.register(keys[0], adjacency_of(first.layout)) is False
    service.register(keys[1], adjacency_of(other.layout))

    service.observe(keys[0], set(list(first.layout.free_cells)[:5]))
    assert service.summary(keys[0])["cells"] == 5
    assert service.summary(keys[1])["cells"] == 0  # never mixed


def test_update_and_select_is_one_round_trip():
    env = _world(0)
    key = layout_fingerprint(env.layout)
    service = ArchiveService(seed=0)
    service.register(key, adjacency_of(env.layout))
    visited = set(list(env.layout.free_cells)[:8])
    chosen = service.update_and_select(key, visited)
    assert chosen in visited
    # Observed first, then selected: the order the algorithm requires.
    assert service.summary(key)["cells"] == 8
    assert service.summary(key)["chosen"] == 1


def test_archive_accepts_adjacency_or_a_neighbour_function():
    """A mapping crosses a process boundary; a callable stays cheap in
    one, which matters for a 160,000-cell world."""
    def neighbors(cell):
        x, y = cell
        return [(x + 1, y), (x - 1, y)]

    by_call = LayoutArchive(neighbors=neighbors)
    by_map = LayoutArchive(adjacency={(0, 0): [(1, 0), (-1, 0)]})
    for archive in (by_call, by_map):
        archive.observe({(0, 0)})
        assert archive.score((0, 0)) > 0
    assert by_call.neighbors((0, 0)) == [(1, 0), (-1, 0)]
    assert by_map.neighbors((5, 5)) == ()


def test_service_is_extensible_without_forking_it():
    """Phase 2 needs trajectories in the archive; that is a subclass,
    not a copy of the service."""
    class TrajectoryArchive(LayoutArchive):
        def new_entry(self, cell):
            entry = super().new_entry(cell)
            entry["path"] = []
            return entry

    class TrajectoryService(ArchiveService):
        archive_class = TrajectoryArchive

    service = TrajectoryService(seed=0)
    service.register("key", {(0, 0): []})
    service.observe("key", {(0, 0)})
    assert service.archives["key"].cells[(0, 0)]["path"] == []
    assert isinstance(service.archives["key"], TrajectoryArchive)
    # The base service is untouched by the subclass.
    assert ArchiveService.archive_class is LayoutArchive


@pytest.mark.slow
def test_archive_is_shared_across_processes_as_an_actor():
    """The point of the actor: two callers, one archive per world."""
    ray = pytest.importorskip("ray")

    ray.init(num_cpus=2, log_to_driver=False, include_dashboard=False,
             ignore_reinit_error=True)
    try:
        env = _world(0)
        key = layout_fingerprint(env.layout)
        actor = remote_archive().remote(seed=0)
        ray.get(actor.register.remote(key, adjacency_of(env.layout)))

        cells = list(env.layout.free_cells)
        ray.get(actor.observe.remote(key, set(cells[:4])))
        ray.get(actor.observe.remote(key, set(cells[4:9])))
        summary = ray.get(actor.summary.remote(key))
        assert summary["cells"] == 9  # both callers' cells, one archive

        chosen = ray.get(actor.update_and_select.remote(key,
                                                        set(cells[:2])))
        assert chosen in set(cells[:9])
    finally:
        ray.shutdown()
