"""Control environments: hard to explore, topologically trivial.

These let experiments separate "my agent exploits topology" from "my agent
is just good at generic novelty-seeking": a control maze is size-matched to
the interesting environments but has b1 = 0 (2D) / b1 = b2 = 0 (3D).
"""

from __future__ import annotations


def maze_walls_2d(rng, width, height):
    """Perfect maze (recursive backtracker): a tree, so b1 = 0.

    Rooms live on odd coordinates; returns the WALL cell set. Even-sized
    domains lose their last row/column to walls.
    """
    rooms_x = (width - 1) // 2
    rooms_y = (height - 1) // 2
    if rooms_x < 2 or rooms_y < 2:
        raise ValueError("maze needs size >= 5")
    carved = set()
    room = (0, 0)
    carved.add(room)
    stack = [room]
    visited = {room}
    passages = set()
    while stack:
        x, y = stack[-1]
        options = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < rooms_x and 0 <= ny < rooms_y and (nx, ny) not in visited:
                options.append((nx, ny))
        if not options:
            stack.pop()
            continue
        nxt = options[int(rng.integers(len(options)))]
        visited.add(nxt)
        passages.add(((x, y), nxt))
        stack.append(nxt)

    free = set()
    for x, y in visited:
        free.add((2 * x + 1, 2 * y + 1))
    for (x, y), (nx, ny) in passages:
        free.add((x + nx + 1, y + ny + 1))
    return {
        (x, y) for x in range(width) for y in range(height)
    } - free


def zigzag_walls_2d(width, height):
    """Serpentine corridor: horizontal bars alternately attached to the
    left/right border force one long winding path. Every bar merges with
    the boundary, so b1 = 0."""
    walls = set()
    k = 0
    for y in range(2, height - 1, 2):
        if k % 2 == 0:
            xs = range(0, width - 2)
        else:
            xs = range(2, width)
        walls.update((x, y) for x in xs)
        k += 1
    return walls


def maze_walls_3d(rng, width, height, depth):
    """Perfect 3D maze on odd coordinates: contractible, b1 = b2 = 0."""
    rooms = tuple((s - 1) // 2 for s in (width, height, depth))
    if min(rooms) < 2:
        raise ValueError("3D maze needs size >= 5")
    room = (0, 0, 0)
    visited = {room}
    stack = [room]
    passages = set()
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    while stack:
        cur = stack[-1]
        options = []
        for d in dirs:
            nxt = tuple(c + dd for c, dd in zip(cur, d))
            if all(0 <= n < r for n, r in zip(nxt, rooms)) and nxt not in visited:
                options.append(nxt)
        if not options:
            stack.pop()
            continue
        nxt = options[int(rng.integers(len(options)))]
        visited.add(nxt)
        passages.add((cur, nxt))
        stack.append(nxt)

    free = {tuple(2 * c + 1 for c in room) for room in visited}
    for a, b in passages:
        free.add(tuple(aa + bb + 1 for aa, bb in zip(a, b)))
    return {
        (x, y, z)
        for x in range(width) for y in range(height) for z in range(depth)
    } - free
