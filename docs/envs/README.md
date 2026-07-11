# Environment gallery

Rendered in *reveal* mode: hidden doors (purple), decoy fills
(dark red), start (blue), goal (green), one-way doors (yellow,
arrow = entry side), trapdoors (orange). Agents do not see any
of the revealed information.

## `2d_bench_grid_small`

| env | preview | certified topology |
|---|---|---|
| **square-holes**<br>3 holes on a disc: b1 = 3, nothing hidden | <img src="2d_bench_grid_small/square-holes.svg" width="220"/> | `b = [1, 3, 0]` |
| **square-rooms**<br>holes + hidden chambers + a decoy | <img src="2d_bench_grid_small/square-rooms.svg" width="220"/> | `b = [1, 5, 0]` |
| **square-decoyfield**<br>many identical-looking rooms; most are empty decoys | <img src="2d_bench_grid_small/square-decoyfield.svg" width="220"/> | `b = [1, 5, 0]` |
| **annulus**<br>thick annulus base + one chamber and one decoy | <img src="2d_bench_grid_small/annulus.svg" width="220"/> | `b = [1, 3, 0]` |
| **plane-6holes**<br>plane with 6 large holes: b1 = 6 | <img src="2d_bench_grid_small/plane-6holes.svg" width="220"/> | `b = [1, 6, 0]` |
| **cylinder-rooms**<br>cylinder: one loop is the world itself | <img src="2d_bench_grid_small/cylinder-rooms.svg" width="220"/> | `b = [1, 4, 0]` |
| **mobius-rooms**<br>Mobius band: crossing the seam mirrors you | <img src="2d_bench_grid_small/mobius-rooms.svg" width="220"/> | `b = [1, 4, 0]` |
| **torus-holes**<br>torus with 2 holes: b1 = 3 | <img src="2d_bench_grid_small/torus-holes.svg" width="220"/> | `b = [1, 3, 0]` |
| **torus-rooms**<br>torus with chambers and a decoy | <img src="2d_bench_grid_small/torus-rooms.svg" width="220"/> | `b = [1, 5, 0]` |
| **torus-goal-in-chamber**<br>the goal hides inside a chamber: doors must be found | <img src="2d_bench_grid_small/torus-goal-in-chamber.svg" width="220"/> | `b = [1, 5, 0]` |
| **klein-rooms**<br>Klein bottle: torus-like but non-orientable | <img src="2d_bench_grid_small/klein-rooms.svg" width="220"/> | `b = [1, 4, 0]` |
| **rp2-rooms**<br>RP^2: the antipodal world | <img src="2d_bench_grid_small/rp2-rooms.svg" width="220"/> | `b = [1, 3, 0]` |
| **sphere-holes**<br>cube-sphere with 3 holes | <img src="2d_bench_grid_small/sphere-holes.svg" width="220"/> | `b = [1, 2, 0]` |
| **sphere-rooms**<br>cube-sphere with chambers and a decoy | <img src="2d_bench_grid_small/sphere-rooms.svg" width="220"/> | `b = [1, 2, 0]` |
| **control-maze**<br>control: perfect maze, hard to explore, b1 = 0 | <img src="2d_bench_grid_small/control-maze.svg" width="220"/> | `b = [1, 0, 0]` |
| **control-zigzag**<br>control: serpentine corridor, one long path, b1 = 0 | <img src="2d_bench_grid_small/control-zigzag.svg" width="220"/> | `b = [1, 0, 0]` |

## `2d_bench_grid_small_directed`

| env | preview | certified topology |
|---|---|---|
| **square-traproom**<br>a one-way room: enter and you stay | <img src="2d_bench_grid_small_directed/square-traproom.svg" width="220"/> | `b = [1, 2, 0], SCCs = 2 (1 absorbing)` |
| **square-airlock**<br>a directed circuit: in one door, out the other | <img src="2d_bench_grid_small_directed/square-airlock.svg" width="220"/> | `b = [1, 3, 0], SCCs = 1` |
| **square-trapdoor**<br>a trapdoor room: the way in seals, a hidden hatch leads out | <img src="2d_bench_grid_small_directed/square-trapdoor.svg" width="220"/> | `b = [1, 3, 0], SCCs = 1` |
| **cylinder-trapdoor**<br>trapdoor room on a cylinder | <img src="2d_bench_grid_small_directed/cylinder-trapdoor.svg" width="220"/> | `b = [1, 4, 0], SCCs = 1` |
| **torus-traproom**<br>trap room on a torus | <img src="2d_bench_grid_small_directed/torus-traproom.svg" width="220"/> | `b = [1, 4, 0], SCCs = 2 (1 absorbing)` |
| **torus-airlock-mix**<br>airlock + trapdoor room on a torus | <img src="2d_bench_grid_small_directed/torus-airlock-mix.svg" width="220"/> | `b = [1, 7, 0], SCCs = 1` |
| **sphere-traproom**<br>trap room on the cube-sphere | <img src="2d_bench_grid_small_directed/sphere-traproom.svg" width="220"/> | `b = [1, 1, 0], SCCs = 2 (1 absorbing)` |
| **square-gauntlet**<br>one of everything: trap room, airlock, trapdoor room, decoy | <img src="2d_bench_grid_small_directed/square-gauntlet.svg" width="220"/> | `b = [1, 8, 0], SCCs = 2 (1 absorbing)` |

## `3d_bench_grid_small`

| env | preview | certified topology |
|---|---|---|
| **box-blobs**<br>2 solid obstacles: b2 = 2 enclosing shells | <img src="3d_bench_grid_small/box-blobs.svg" width="220"/> | `b = [1, 0, 2, 0]` |
| **box-ring**<br>a ring obstacle: b1 = 1, b2 = 1 | <img src="3d_bench_grid_small/box-ring.svg" width="220"/> | `b = [1, 1, 1, 0]` |
| **box-rooms**<br>a hollow chamber and a sealed decoy | <img src="3d_bench_grid_small/box-rooms.svg" width="220"/> | `b = [1, 0, 2, 0]` |
| **box-mixed**<br>ring + void + chamber | <img src="3d_bench_grid_small/box-mixed.svg" width="220"/> | `b = [1, 1, 3, 0]` |
| **solid-torus**<br>solid torus: one loop is the world itself | <img src="3d_bench_grid_small/solid-torus.svg" width="220"/> | `b = [1, 2, 2, 0]` |
| **torus3**<br>3-torus: wraps in every direction, b1 = 3 | <img src="3d_bench_grid_small/torus3.svg" width="220"/> | `b = [1, 3, 3, 0]` |
| **shell**<br>spherical shell: a big void you can never enter | <img src="3d_bench_grid_small/shell.svg" width="220"/> | `b = [1, 0, 1, 0]` |
| **control-maze3d**<br>control: perfect 3D maze, b1 = b2 = 0 | <img src="3d_bench_grid_small/control-maze3d.svg" width="220"/> | `b = [1, 0, 0, 0]` |

## `2d_bench_grid_small_bridges`

| env | preview | certified topology |
|---|---|---|
| **square-dumbbell**<br>two rooms, one narrow passage: b1 = 0, all bottleneck | <img src="2d_bench_grid_small_bridges/square-dumbbell.svg" width="220"/> | `b = [1, 0, 0]` |
| **square-twin-passages**<br>two passages through one wall: the loop closure env | <img src="2d_bench_grid_small_bridges/square-twin-passages.svg" width="220"/> | `b = [1, 1, 0]` |
| **square-moat-hidden**<br>a moat you can see across; one open bridge, one hidden | <img src="2d_bench_grid_small_bridges/square-moat-hidden.svg" width="220"/> | `b = [1, 1, 0]` |
| **square-triple-rooms**<br>two dividing walls, three regions | <img src="2d_bench_grid_small_bridges/square-triple-rooms.svg" width="220"/> | `b = [1, 2, 0]` |
| **cylinder-ring-gate**<br>a gate across the cylinder | <img src="2d_bench_grid_small_bridges/cylinder-ring-gate.svg" width="220"/> | `b = [1, 1, 0]` |
| **torus-meridian**<br>a meridian wall on the torus: close the loop through the wrap | <img src="2d_bench_grid_small_bridges/torus-meridian.svg" width="220"/> | `b = [1, 3, 0]` |
| **sphere-belt**<br>an equatorial belt with two passages | <img src="2d_bench_grid_small_bridges/sphere-belt.svg" width="220"/> | `b = [1, 1, 0]` |
| **square-bridge-gauntlet**<br>a moat with a hidden bridge, plus a chamber and a decoy | <img src="2d_bench_grid_small_bridges/square-bridge-gauntlet.svg" width="220"/> | `b = [1, 3, 0]` |

## `3d_bench_grid_small_bridges`

| env | preview | certified topology |
|---|---|---|
| **box-dumbbell**<br>two chambers of space, one tunnel | <img src="3d_bench_grid_small_bridges/box-dumbbell.svg" width="220"/> | `b = [1, 0, 0, 0]` |
| **box-two-tunnels**<br>two tunnels through one wall: b1 = 1 | <img src="3d_bench_grid_small_bridges/box-two-tunnels.svg" width="220"/> | `b = [1, 1, 0, 0]` |
| **box-hidden-tunnel**<br>one tunnel open, one hidden | <img src="3d_bench_grid_small_bridges/box-hidden-tunnel.svg" width="220"/> | `b = [1, 1, 0, 0]` |
| **solid-torus-gate**<br>a gate disc across the solid torus: the wrap loop passes it | <img src="3d_bench_grid_small_bridges/solid-torus-gate.svg" width="220"/> | `b = [1, 1, 1, 0]` |

## `3d_bench_grid_small_directed`

| env | preview | certified topology |
|---|---|---|
| **box-traproom**<br>a one-way room in 3D | <img src="3d_bench_grid_small_directed/box-traproom.svg" width="220"/> | `b = [1, 0, 2, 0], SCCs = 2 (1 absorbing)` |
| **box-airlock**<br>a 3D airlock: directed circuit through a room | <img src="3d_bench_grid_small_directed/box-airlock.svg" width="220"/> | `b = [1, 1, 1, 0], SCCs = 1` |
| **box-trapdoor**<br>3D trapdoor room with a hidden escape hatch | <img src="3d_bench_grid_small_directed/box-trapdoor.svg" width="220"/> | `b = [1, 1, 1, 0], SCCs = 1` |
| **solid-torus-traproom**<br>trap room in a solid torus | <img src="3d_bench_grid_small_directed/solid-torus-traproom.svg" width="220"/> | `b = [1, 1, 1, 0], SCCs = 2 (1 absorbing)` |

