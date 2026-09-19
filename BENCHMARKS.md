# TopoGym-v1 benchmark results

Every reference baseline has run: `random`, `ppo`, `icm-ppo`,
`rnd-ppo`, `go-explore-phase1` and the four `topoexplore-phase1` arms,
each on all 189 `test` worlds under the protocol below.

**Worlds whose goal was reached, of 189.** Archive methods are read on
the training side, where the archive they build is live; the gradient
methods are read on the frozen evaluation, which discards an archive by
construction.

| method | worlds solved | read on |
|---|---|---|
| `topoexplore-phase1-both` | 169 | training |
| `topoexplore-phase1-ricci` | 168 | training |
| `topoexplore-phase1-animation` | 168 | training |
| `topoexplore-phase1-none` | 167 | training |
| `go-explore-phase1` | 152 | training |
| `icm-ppo` | 15 | frozen evaluation |
| `rnd-ppo` | 7 | frozen evaluation |
| `ppo`, `random` | 0 | frozen evaluation |

Paired world by world against `go-explore-phase1`, every TopoExplore
arm wins far more worlds than it loses -- 19 against 2 for `both`, 16
against 1 for `none` -- and a two-sided sign test puts each at
p < 0.001. The per-world table, the per-arm sign tests and the
frozen-evaluation figures are in
[`benchmarks/single_layout/figures/`](benchmarks/single_layout/figures/);
the raw trees they come from are archived outside git, described in
[`benchmarks/single_layout/ARCHIVE.md`](benchmarks/single_layout/ARCHIVE.md).

Regenerate with `scripts/benchmarks/reproduce.sh all`; the chamber-count
and EpicChase studies have their own script,
`scripts/benchmarks/reproduce_gridworld2d_v0.sh`.

The benchmark itself is already pinned. The `gridworld2d-v1` roster in
`topogym/benchmarks.json` is marked frozen, its membership must keep
matching the split CSVs shipped in `docs/splits/` (enforced in both
directions by `tests/test_splits.py`), and every environment id is
stable — so results published against it will remain reproducible
against exactly the worlds they were measured on.

## The protocol

Every world in the `test` split is its own experiment, and every
method faces it on the same terms:

1. **Hyperparameters are chosen on the `tune` split**, never on a
   test world: each candidate configuration spends 100,000 environment
   steps per tuning world, candidates are scored per world and
   averaged, and one configuration is selected per algorithm.
   Candidates are ranked by return when any of them earned one, and by
   coverage otherwise — decided once per search, since ranking one
   candidate by return and another by coverage compares incomparable
   scales.
2. **Each test world grants 1,000,000 environment steps of learning**
   with the selected configuration — gradient updates, an archive,
   both; how a method spends them is its own business.
3. **Evaluation is frozen**: after the budget is spent, the learned
   policy runs fixed evaluation episodes in a fresh copy of the same
   world, and that is the headline number.

One asymmetry in how that budget was spent must be read alongside
every gradient-method number. The RLlib arms (`ppo`, `rnd-ppo`,
`icm-ppo`) train with early stopping -- `patience` validation checks
without improvement end the run -- and on this benchmark every one of
their 570 runs stopped early: the median run spent 20% (ppo), 22%
(rnd-ppo) and 32% (icm-ppo) of its 1,000,000 steps, the least 12% and
the most 98%. The archive methods spend the whole budget by
construction. The gradient methods were therefore not given less than
the archive methods; they declined the rest, because their own
validation signal had flattened. A re-run with early stopping off
(`patience` is a run-config field) is the honest way to close the
question; until then, their numbers are lower bounds on what the full
budget would buy, and the archive-over-gradient gap is an upper bound.

The numbers therefore measure *exploration within a world* — how much
of one world a method uncovers, and whether it reaches the goal, given
a fixed budget in it — not whether a trained policy transfers to
unseen worlds. Every method learns in the world it is scored on, under
the same budget. Budgets are counted in environment steps, never
episodes: horizons across the roster span 130 to 7,680 steps, so a
flat episode count would hand one world fifty times the experience of
another. Every result JSON records the exact budget plan it ran under.

The transfer question — fit on `train`, stop on `val`, evaluate once
on held-out worlds — remains fully posed by the published splits and
the in-repo protocol (`Baseline.run`); we publish the splits so that
benchmark can be run, rather than exercising it ourselves.

## What will be recorded

One result JSON per (algorithm, world) carries the complete native
metric set — coverage milestones, visitation entropy, regret, planning
efficiency, curvature coverage, and (when the run enabled
`--track-topology`) the step at which each hole was first seen —
whether or not a figure plots it. Headline tables will report
per-world and per-slice coverage and goal discovery, alongside
discovery curves and per-world rollout recordings.
