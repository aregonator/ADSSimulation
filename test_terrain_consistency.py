#!/usr/bin/env python3
"""
Regression guard: every cell's terrain-derived values agree with its terrain.

    python3 test_terrain_consistency.py      # exit 0 = pass, 1 = fail

WHAT THIS PROTECTS
------------------
``Cell`` keeps five per-terrain constants as plain instance attributes —
``_terrain_hazard``, ``_terrain_shelter``, ``_food_regen``, ``_water_regen`` and
``_terrain_value`` — because ``hazard``, ``shelter``, ``food_regen``,
``water_regen`` and ``to_dict()`` are read on the engine's and the lookahead's
hottest paths.  A cache is only correct while it agrees with its source,
``TERRAIN_STATS[cell.terrain]``.  ``Cell.terrain`` is a descriptor-typed
dataclass field whose setter re-derives all five on every assignment, so the
two cannot diverge.

The production path is exactly where they could: ``build_simulation_from_plan``
lets ``Simulation`` generate a grid of its own and then relabels every cell
from the plan (``engine/scenario_plan._restore_grid``).  If a relabel ever stops
refreshing the cache, every consumer — health drain, storm shelter, the
flee and mandatory-shelter laws, regeneration, the ADS evidence and rollout,
the parameter estimator — silently runs on the hazard, shelter and regen of
the wrong terrain, while every label-based check (fingerprints, digests,
terrain counts) still passes.  Nothing else in the suite would notice.

Three layers:

  1. **Unit** — construction, every terrain-to-terrain reassignment, rejection
     of a non-``Terrain`` value, the dataclass machinery the codebase and
     the sweep rely on (``replace``, ``asdict``, ``fields``, ``copy``,
     ``deepcopy``, ``pickle``), and the per-cell ``ambient_hazard`` (the
     uniform hazard from the difficulty schedule that is added to the
     terrain hazard).
  2. **Production path** — ``run_full_simulation.build_config()`` driven through
     the real ``benchmark_core._run_one`` up to the point the simulation is
     built, then every cell of the built grid checked, including that its
     hazard is the terrain hazard plus ``engine.difficulty.ambient_hazard(d)``
     for the run's difficulty.  The run itself is not executed: terrain is
     static during a run, so the grid as built is the grid every cycle sees.
  3. **Spawned worker** — the same production-path check inside a ``spawn``
     process, the start method the sweep pins its workers to.  This is what
     proves the ambient hazard reaches a worker through the config rather than
     through anything set in the parent process.

WHY THIS FILE LIVES HERE, AND NOT IN tests/
-------------------------------------------
Same pattern as ``test_seed_derivation.py``: this file sits beside the code
it guards, alongside the project's other top-level regression tests, and
runs under a plain ``python3`` with no test-framework dependency.
"""

from __future__ import annotations

import copy
import dataclasses
import multiprocessing
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from engine.difficulty import ambient_hazard                     # noqa: E402
from engine.grid import TERRAIN_STATS, Cell, Grid, Terrain    # noqa: E402

#: (difficulty, run_idx) coordinates checked on the production path: both ends
#: of the difficulty range, a point on the ambient-hazard ramp (D10) and the
#: middle, at different run indices so the plans are different environments.
_PRODUCTION_COORDS: Tuple[Tuple[int, int], ...] = ((1, 0), (10, 0), (50, 3), (100, 9))

#: Coordinates re-checked inside a spawn worker: one on the ambient ramp, one
#: past the knee.
_SPAWN_COORDS: Tuple[Tuple[int, int], ...] = ((10, 0), (100, 9))

#: Government used to reach ``build_simulation_from_plan`` through ``_run_one``.
#: The grid does not depend on the government (``build_plan`` takes none), so
#: one is enough; ADS is the one whose lookahead reads the cache most.
_PRODUCTION_GOVERNMENT = "ads"


def _check(results: list, name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------

def cell_mismatches(cell: Cell, expected_ambient: Optional[float] = None) -> List[str]:
    """Every terrain-derived value on *cell* that disagrees with its terrain.

    Checks the public surface (the properties and ``to_dict()``) rather than
    the private cache attributes, so it keeps meaning the same thing if the
    cache is ever reorganised.  ``hazard`` is terrain hazard + ambient hazard
    + ``hazard_extra``, capped at 1.0; the expected value is computed with the
    same association.

    *expected_ambient* is the ambient hazard the cell should carry.  ``None``
    takes the cell's own ``ambient_hazard`` (unit checks); the production-path
    checks pass the difficulty schedule's value, so a cell carrying the wrong
    ambient hazard is reported rather than used as its own reference.
    """
    stats = TERRAIN_STATS[cell.terrain]
    ambient = cell.ambient_hazard if expected_ambient is None else expected_ambient
    expected_hazard = min(1.0, stats["hazard"] + ambient + cell.hazard_extra)
    d = cell.to_dict()
    problems = []
    if cell.ambient_hazard != ambient:
        problems.append(f"ambient_hazard {cell.ambient_hazard!r} != {ambient!r}")
    if cell.hazard != expected_hazard:
        problems.append(f"hazard {cell.hazard!r} != {expected_hazard!r}")
    if cell.shelter != stats["shelter"]:
        problems.append(f"shelter {cell.shelter!r} != {stats['shelter']!r}")
    if cell.food_regen != stats["food_regen"]:
        problems.append(f"food_regen {cell.food_regen!r} != {stats['food_regen']!r}")
    if cell.water_regen != stats["water_regen"]:
        problems.append(f"water_regen {cell.water_regen!r} != {stats['water_regen']!r}")
    if d["terrain"] != cell.terrain.value:
        problems.append(f"to_dict terrain {d['terrain']!r} != {cell.terrain.value!r}")
    if d["hazard"] != expected_hazard:
        problems.append(f"to_dict hazard {d['hazard']!r} != {expected_hazard!r}")
    if d["shelter"] != stats["shelter"]:
        problems.append(f"to_dict shelter {d['shelter']!r} != {stats['shelter']!r}")
    return problems


def grid_mismatch_count(grid: Grid, expected_ambient: Optional[float] = None
                        ) -> Tuple[int, int, str]:
    """(mismatched cells, total cells, first mismatch description or '')."""
    bad, total, first = 0, 0, ""
    for cell in grid.all_cells():
        total += 1
        problems = cell_mismatches(cell, expected_ambient)
        if problems:
            bad += 1
            if not first:
                first = f"({cell.row},{cell.col}) {cell.terrain.name}: {'; '.join(problems)}"
    return bad, total, first


# ---------------------------------------------------------------------------
# 1. Unit
# ---------------------------------------------------------------------------

def _check_construction(results: list) -> None:
    print("\nConstruction:")
    default = Cell(row=0, col=0)
    _check(results, "Cell(row=, col=) defaults to PLAINS",
           default.terrain is Terrain.PLAINS, repr(default.terrain))
    _check(results, "the dataclass field default is PLAINS",
           Cell.__dataclass_fields__["terrain"].default is Terrain.PLAINS,
           repr(Cell.__dataclass_fields__["terrain"].default))
    _check(results, "a default cell is consistent",
           not cell_mismatches(default), "; ".join(cell_mismatches(default)))

    for terrain in Terrain:
        cell = Cell(row=1, col=2, terrain=terrain, food=1.0, water=2.0, medicine=3.0)
        problems = cell_mismatches(cell)
        _check(results, f"Cell(terrain={terrain.name}) is consistent",
               cell.terrain is terrain and not problems, "; ".join(problems))

    positional = Cell(3, 4, Terrain.WATER)
    _check(results, "positional construction still sets terrain",
           positional.terrain is Terrain.WATER and not cell_mismatches(positional))


def _check_reassignment(results: list) -> None:
    print("\nReassignment refreshes every derived value (all 25 transitions):")
    failures = []
    for old in Terrain:
        for new in Terrain:
            cell = Cell(row=0, col=0, terrain=old)
            cell.hazard_extra = 0.3
            cell.terrain = new
            problems = cell_mismatches(cell)
            if cell.terrain is not new or problems:
                failures.append(f"{old.name}->{new.name}: {'; '.join(problems)}")
    _check(results, "every old->new terrain reassignment is consistent",
           not failures, "; ".join(failures[:3]))

    # The transitions that change each derived value are the ones that matter:
    # a setter that refreshed nothing would still pass a same-terrain case.
    cell = Cell(row=0, col=0, terrain=Terrain.PLAINS)
    before = (cell.hazard, cell.shelter, cell.food_regen, cell.water_regen)
    cell.terrain = Terrain.MOUNTAIN
    after = (cell.hazard, cell.shelter, cell.food_regen, cell.water_regen)
    _check(results, "PLAINS->MOUNTAIN changes hazard, shelter, food_regen and water_regen",
           all(b != a for b, a in zip(before, after)), f"{before} -> {after}")
    _check(results, "reassignment leaves position, stocks, hazard_extra and contamination untouched",
           _reassigned_keeps_state())

    print("\nRejected values leave the cell unchanged:")
    for bad, label in (("forest", "str"), (None, "None"), (1, "int")):
        cell = Cell(row=0, col=0, terrain=Terrain.FOREST)
        try:
            cell.terrain = bad
            ok, detail = False, "accepted silently"
        except TypeError as exc:
            ok = cell.terrain is Terrain.FOREST and not cell_mismatches(cell)
            detail = f"TypeError: {exc}" if ok else "cell changed despite the error"
        except Exception as exc:                              # noqa: BLE001
            ok, detail = False, f"wrong exception type: {exc!r}"
        _check(results, f"assigning a {label} is rejected", ok, detail)
    try:
        Cell(row=0, col=0, terrain="water")
        ok, detail = False, "accepted silently"
    except TypeError as exc:
        ok, detail = True, f"TypeError: {exc}"
    except Exception as exc:                                  # noqa: BLE001
        ok, detail = False, f"wrong exception type: {exc!r}"
    _check(results, "constructing with a str terrain is rejected", ok, detail)


def _reassigned_keeps_state() -> bool:
    cell = Cell(row=5, col=6, terrain=Terrain.PLAINS, food=11.0, water=12.0,
                medicine=13.0, hazard_extra=0.2, contaminated=True)
    cell.terrain = Terrain.FOREST
    return (cell.row, cell.col, cell.food, cell.water, cell.medicine,
            cell.hazard_extra, cell.contaminated) == (5, 6, 11.0, 12.0, 13.0, 0.2, True)


def _check_dataclass_machinery(results: list) -> None:
    print("\nDataclass machinery:")
    field_names = [f.name for f in dataclasses.fields(Cell)]
    _check(results, "fields() exposes 'terrain' and no private backing field",
           "terrain" in field_names and not any(n.startswith("_") for n in field_names),
           str(field_names))

    src = Cell(row=1, col=1, terrain=Terrain.FOREST, food=9.0)
    replaced = dataclasses.replace(src, terrain=Terrain.WASTELAND)
    _check(results, "dataclasses.replace(terrain=...) re-derives on the new cell",
           replaced.terrain is Terrain.WASTELAND and not cell_mismatches(replaced)
           and src.terrain is Terrain.FOREST and not cell_mismatches(src))

    as_dict = dataclasses.asdict(src)
    _check(results, "asdict() carries the Terrain under 'terrain'",
           as_dict.get("terrain") is Terrain.FOREST, repr(as_dict.get("terrain")))

    _check(results, "equality compares terrain",
           Cell(row=0, col=0, terrain=Terrain.WATER) == Cell(row=0, col=0, terrain=Terrain.WATER)
           and Cell(row=0, col=0, terrain=Terrain.WATER) != Cell(row=0, col=0, terrain=Terrain.FOREST))
    _check(results, "repr shows terrain", "terrain=<Terrain.FOREST" in repr(src), repr(src)[:80])

    for label, clone_fn in (
        ("copy.copy", copy.copy),
        ("copy.deepcopy", copy.deepcopy),
        ("pickle round-trip", lambda c: pickle.loads(pickle.dumps(c))),
    ):
        original = Cell(row=2, col=3, terrain=Terrain.MOUNTAIN, hazard_extra=0.1)
        clone = clone_fn(original)
        consistent = clone.terrain is Terrain.MOUNTAIN and not cell_mismatches(clone)
        clone.terrain = Terrain.WATER
        independent = (not cell_mismatches(clone)
                       and original.terrain is Terrain.MOUNTAIN
                       and not cell_mismatches(original))
        _check(results, f"{label}: clone is consistent and re-derives independently",
               consistent and independent,
               f"consistent={consistent} independent={independent}")


def _check_fresh_grid(results: list) -> None:
    print("\nFreshly generated Grid:")
    grid = Grid(rows=30, cols=30, resource_density=0.8, terrain_variety=1.0, seed=7,
                ambient_hazard=0.0)
    bad, total, first = grid_mismatch_count(grid)
    kinds = {c.terrain for c in grid.all_cells()}
    _check(results, "every terrain type occurs (the check is not vacuous)",
           kinds == set(Terrain), str(sorted(t.name for t in kinds)))
    _check(results, f"0 of {total} cells mismatched", bad == 0, first)


def _check_ambient_hazard(results: list) -> None:
    """``Cell.ambient_hazard``: the uniform hazard added to every terrain hazard.

    Each cell carries its own value (passed down from ``SimulationConfig``
    through ``Grid``), so these checks construct cells and grids with an
    explicit value rather than patching anything process-wide.  Covers fresh
    construction, relabelling (the ``_restore_grid`` path), changing the
    ambient value on an existing cell, the 1.0 cap, validation, the copy
    protocols, and that ``Grid`` refuses to be built without a value.
    """
    print("\nPer-cell ambient hazard:")
    import engine.grid as grid_module

    _check(results, "engine.grid has no process-global ambient-hazard knob",
           not hasattr(grid_module, "AMBIENT_HAZARD"))
    _check(results, "Cell.ambient_hazard is a keyword-only dataclass field defaulting to 0.0",
           Cell.__dataclass_fields__["ambient_hazard"].kw_only
           and Cell.__dataclass_fields__["ambient_hazard"].default == 0.0)

    h0 = 0.01
    failures = []
    for terrain in Terrain:
        cell = Cell(row=0, col=0, terrain=terrain, ambient_hazard=h0)
        expected = min(1.0, TERRAIN_STATS[terrain]["hazard"] + h0)
        got, got_dict = cell.hazard, cell.to_dict()["hazard"]
        if got != expected or got_dict != expected or cell_mismatches(cell):
            failures.append(f"{terrain.name}: hazard={got!r} to_dict={got_dict!r} expected={expected!r}")
    _check(results,
           "construction with ambient_hazard=0.01: every terrain's hazard == "
           "terrain hazard + 0.01 (hazard and to_dict()['hazard'] agree)",
           not failures, "; ".join(failures))

    failures = []
    for terrain in Terrain:
        cell = Cell(row=0, col=0, terrain=Terrain.PLAINS, ambient_hazard=h0)
        cell.terrain = terrain    # reassignment, the _restore_grid path
        expected = min(1.0, TERRAIN_STATS[terrain]["hazard"] + h0)
        if cell.hazard != expected or cell.to_dict()["hazard"] != expected:
            failures.append(f"{terrain.name}: hazard={cell.hazard!r} expected={expected!r}")
    _check(results, "relabelling keeps the cell's ambient hazard in its hazard",
           not failures, "; ".join(failures))

    cell = Cell(row=0, col=0, terrain=Terrain.MOUNTAIN)
    cell.ambient_hazard = h0
    _check(results, "assigning ambient_hazard on an existing cell re-derives its hazard",
           cell.hazard == TERRAIN_STATS[Terrain.MOUNTAIN]["hazard"] + h0
           and not cell_mismatches(cell, h0), repr(cell.hazard))

    # WASTELAND (terrain hazard 0.05) + ambient 0.01 + hazard_extra 0.97 sums
    # to 1.03 raw; both public readers must still clamp to 1.0.
    capped = Cell(row=0, col=0, terrain=Terrain.WASTELAND, hazard_extra=0.97, ambient_hazard=h0)
    _check(results,
           "hazard still caps at 1.0 with ambient + terrain hazard + hazard_extra stacked",
           capped.hazard == 1.0 and capped.to_dict()["hazard"] == 1.0,
           f"hazard={capped.hazard!r} to_dict={capped.to_dict()['hazard']!r}")

    for bad, label, exc_type in ((-0.001, "negative", ValueError), (1.0, "1.0", ValueError),
                                 (float("nan"), "NaN", ValueError), (True, "bool", TypeError),
                                 ("0.01", "str", TypeError), (None, "None", TypeError)):
        cell = Cell(row=0, col=0, terrain=Terrain.FOREST, ambient_hazard=h0)
        try:
            cell.ambient_hazard = bad
            ok, detail = False, "accepted silently"
        except exc_type as exc:
            ok = cell.ambient_hazard == h0 and not cell_mismatches(cell, h0)
            detail = f"{exc_type.__name__}: {exc}" if ok else "cell changed despite the error"
        except Exception as exc:                              # noqa: BLE001
            ok, detail = False, f"wrong exception type: {exc!r}"
        _check(results, f"ambient_hazard {label} is rejected and leaves the cell unchanged", ok, detail)

    for label, clone_fn in (
        ("copy.deepcopy", copy.deepcopy),
        ("pickle round-trip", lambda c: pickle.loads(pickle.dumps(c))),
        ("dataclasses.replace(terrain=...)", lambda c: dataclasses.replace(c, terrain=Terrain.WATER)),
    ):
        clone = clone_fn(Cell(row=2, col=3, terrain=Terrain.FOREST, ambient_hazard=h0))
        _check(results, f"{label} keeps ambient_hazard and a consistent hazard",
               clone.ambient_hazard == h0 and not cell_mismatches(clone, h0),
               "; ".join(cell_mismatches(clone, h0)))

    grid = Grid(rows=10, cols=10, resource_density=0.8, terrain_variety=1.0, seed=11,
                ambient_hazard=h0)
    bad, total, first = grid_mismatch_count(grid, h0)
    _check(results, f"fresh Grid(ambient_hazard=0.01): 0 of {total} cells disagree",
           bad == 0 and grid.ambient_hazard == h0, first)

    try:
        Grid(rows=5, cols=5, resource_density=0.8, terrain_variety=1.0, seed=1)
        ok, detail = False, "Grid built without ambient_hazard"
    except TypeError as exc:
        ok, detail = True, f"TypeError: {exc}"
    _check(results, "Grid refuses to be built without an explicit ambient_hazard", ok, detail)


def _check_hazard_cell_count(results: list) -> None:
    """``RunRecorder._environment_block``'s ``hazard_cell_count`` must not saturate.

    Every cell carries the difficulty's ambient hazard from D2 upward, so a
    cell-count gated on raw ``hazard > 0.0`` would count every cell on the
    grid.  The recorder instead gates on ``hazard > ambient_hazard``, so the
    count tracks cells with terrain or event hazard ON TOP OF the ambient
    floor.  At D50, with a fresh grid and no events yet triggered (cycle 0),
    that is exactly the cells whose terrain hazard is nonzero, i.e. every
    non-plains cell.
    """
    import types

    from engine.events import EventSystem
    from engine.run_recorder import RunRecorder

    print("\nHazard-cell counter (RunRecorder._environment_block):")
    difficulty = 50
    h0 = ambient_hazard(difficulty)
    grid = Grid(rows=50, cols=50, resource_density=0.7, terrain_variety=1.0, seed=17,
                ambient_hazard=h0)
    sim = types.SimpleNamespace(grid=grid, event_system=EventSystem())

    _check(results, "precondition: a fresh sim has no active events at cycle 0",
           sim.event_system.active_events == [], repr(sim.event_system.active_events))

    env = RunRecorder._environment_block(sim)
    cells = grid.all_cells()
    total = len(cells)
    terrain_hazard_cells = sum(
        1 for c in cells if TERRAIN_STATS[c.terrain]["hazard"] > 0.0
    )

    _check(results,
           f"D{difficulty}: hazard_cell_count ({env['hazard_cell_count']}) < total cells ({total})",
           env["hazard_cell_count"] < total,
           f"hazard_cell_count={env['hazard_cell_count']} total={total}")
    _check(results,
           "D{}: hazard_cell_count equals cells with terrain hazard > 0 ({})".format(
               difficulty, terrain_hazard_cells),
           env["hazard_cell_count"] == terrain_hazard_cells,
           f"hazard_cell_count={env['hazard_cell_count']} "
           f"terrain_hazard_cells={terrain_hazard_cells}")
    _check(results,
           "hazard_cell_fraction is hazard_cell_count / total cells",
           env["hazard_cell_fraction"] == round(env["hazard_cell_count"] / total, 4),
           f"hazard_cell_fraction={env['hazard_cell_fraction']!r}")


# ---------------------------------------------------------------------------
# 2. Production path
# ---------------------------------------------------------------------------

class _SimBuilt(Exception):
    """Raised from inside _run_one once the simulation exists, to stop before the run."""

    def __init__(self, sim) -> None:
        super().__init__("simulation built")
        self.sim = sim


def production_grid_report(difficulty: int, run_idx: int) -> Dict[str, object]:
    """Build the production simulation for one coordinate and audit its grid.

    Drives the real ``benchmark_core._run_one`` with the production
    ``run_full_simulation.build_config()``; ``build_simulation_from_plan`` is
    wrapped so the built simulation is captured and ``_run_one`` is stopped
    before the run starts.  ``_restore_grid`` is wrapped too, only to record
    each cell's pre-restore label, so the report can show that the restore
    really relabelled cells (otherwise a 0-mismatch result would say nothing
    about reassignment).  Both wrappers are removed before returning.

    Module-level so a ``spawn`` worker can call it.
    """
    import benchmark_core
    import engine.scenario_plan as scenario_plan
    from run_full_simulation import build_config

    config = build_config()
    pre_restore: Dict[Tuple[int, int], Terrain] = {}
    real_build = benchmark_core.build_simulation_from_plan
    real_restore = scenario_plan._restore_grid

    def capturing_restore(grid, grid_cells):
        for cell in grid.all_cells():
            pre_restore[(cell.row, cell.col)] = cell.terrain
        return real_restore(grid, grid_cells)

    def capturing_build(*args, **kwargs):
        raise _SimBuilt(real_build(*args, **kwargs))

    benchmark_core.build_simulation_from_plan = capturing_build
    scenario_plan._restore_grid = capturing_restore
    try:
        try:
            benchmark_core._run_one(config, _PRODUCTION_GOVERNMENT, difficulty, run_idx,
                                    capture_viz=False, run_dir=None)
        except _SimBuilt as built:
            sim = built.sim
        else:
            raise AssertionError("_run_one returned without calling build_simulation_from_plan")
    finally:
        benchmark_core.build_simulation_from_plan = real_build
        scenario_plan._restore_grid = real_restore

    plan = benchmark_core.build_plan(config, difficulty, run_idx)
    expected_ambient = ambient_hazard(difficulty)
    label_mismatch = sum(
        1 for r, row in enumerate(plan.grid_cells) for c, cell_data in enumerate(row)
        if sim.grid.cell(r, c).terrain.value != cell_data["terrain"]
    )
    relabelled = sum(1 for cell in sim.grid.all_cells()
                     if pre_restore.get((cell.row, cell.col)) is not cell.terrain)
    bad, total, first = grid_mismatch_count(sim.grid, expected_ambient)
    return {
        "expected_ambient": expected_ambient,
        "plan_ambient": plan.config_params.get("ambient_hazard"),
        "config_ambient": sim.config.ambient_hazard,
        "grid_ambient": sim.grid.ambient_hazard,
        "grid": f"{sim.grid.rows}x{sim.grid.cols}",
        "cells": total,
        "mismatched": bad,
        "first_mismatch": first,
        "label_mismatch_vs_plan": label_mismatch,
        "relabelled_by_restore": relabelled,
    }


def _check_production_path(results: list) -> None:
    print("\nProduction path (build_config -> _run_one -> build_plan -> "
          "build_simulation_from_plan):")
    for difficulty, run_idx in _PRODUCTION_COORDS:
        rep = production_grid_report(difficulty, run_idx)
        tag = f"d={difficulty} run={run_idx} ({rep['grid']})"
        _check(results, f"{tag}: grid labels equal the plan's",
               rep["label_mismatch_vs_plan"] == 0,
               f"{rep['label_mismatch_vs_plan']} cells differ")
        _check(results, f"{tag}: the restore relabelled cells (not vacuous)",
               rep["relabelled_by_restore"] > 0,
               f"{rep['relabelled_by_restore']} of {rep['cells']} relabelled")
        _check(results, f"{tag}: 0 of {rep['cells']} cells mismatched "
                        f"(hazard == terrain hazard + ambient_hazard({difficulty}) "
                        f"= {rep['expected_ambient']!r})",
               rep["mismatched"] == 0,
               f"{rep['mismatched']} mismatched; first: {rep['first_mismatch']}"
               if rep["mismatched"] else "")
        _check(results, f"{tag}: plan, config and grid all carry ambient_hazard({difficulty})",
               rep["plan_ambient"] == rep["config_ambient"] == rep["grid_ambient"]
               == rep["expected_ambient"],
               f"plan={rep['plan_ambient']!r} config={rep['config_ambient']!r} "
               f"grid={rep['grid_ambient']!r} expected={rep['expected_ambient']!r}")


# ---------------------------------------------------------------------------
# 3. Spawned worker
# ---------------------------------------------------------------------------

def _worker_report(coord: Tuple[int, int]) -> Dict[str, object]:
    return production_grid_report(*coord)


def _check_spawned_worker(results: list) -> None:
    print("\nSpawned worker (the sweep's start method):")
    try:
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            reports = [(coord, pool.apply(_worker_report, (coord,))) for coord in _SPAWN_COORDS]
    except Exception as exc:                                  # noqa: BLE001
        _check(results, "spawn worker ran the production-path check", False,
               f"worker failed: {exc!r}")
        return
    for coord, rep in reports:
        ok = (rep["mismatched"] == 0 and rep["relabelled_by_restore"] > 0
              and rep["grid_ambient"] == rep["expected_ambient"] == ambient_hazard(coord[0]))
        detail = (f"{rep['mismatched']} of {rep['cells']} mismatched, "
                  f"{rep['relabelled_by_restore']} relabelled, grid ambient "
                  f"{rep['grid_ambient']!r} vs schedule {ambient_hazard(coord[0])!r}")
        _check(results, f"d={coord[0]} run={coord[1]} in a spawn worker: 0 mismatched, "
                        f"every hazard includes ambient_hazard({coord[0]})", ok, detail)


def main() -> int:
    results: list = []
    print("Terrain-derived cell values agree with cell terrain:")
    _check_construction(results)
    _check_reassignment(results)
    _check_dataclass_machinery(results)
    _check_fresh_grid(results)
    _check_ambient_hazard(results)
    _check_hazard_cell_count(results)
    _check_production_path(results)
    _check_spawned_worker(results)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
