#!/usr/bin/env python3
"""
System-level regression tests for the production benchmark pipeline.

Locks in invariants not covered by test_scenarios.py or test_decision_making.py:

  1. run_full_simulation.py reproduces the *published* methodology
     (21 difficulty levels, 100 seeds, 50x50 grid, 500 agents, 150 cycles,
     across all 8 registered governments).
  2. run_quick_test.py stays a smoke test and never drifts to production scale
     (and never writes into benchmark_results/, which would clobber the
     published dataset).
  3. BenchmarkConfig rejects malformed input eagerly rather than hours into
     a sweep.
  4. The real benchmark pipeline runs end to end for a single
     (government, difficulty, seed) cell and emits a well-formed
     final_stats.json.
  5. governments/ads.py exposes no dead code, and its quarantine-bbox helper
     (`_compute_quarantine_bbox`) does not shadow the base class's
     `_compute_quarantine_region` method (which has a different signature
     and different semantics).

Usage (from the Simulation project root):
    python3 tests/system/test_stage1_refactor.py

Runtime: the full suite (all 21 tests) takes on the order of 10-20 minutes,
dominated by the repeated-DEBUG-level fan-out deadlock guard (Test 17: 5
iterations at roughly 90-100s each) and the other real-sweep multiprocessing
tests. See README.md's "tests/ subtree" section for the measured figure and
its filesystem caveat.
"""

import contextlib
import csv
import dataclasses
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root: tests/<subpkg>/<file>.py -> tests/<subpkg> -> tests/ -> root
_PARENT = _HERE  # sys.path target for the flat top-level packages (engine, governments, ...)
sys.path.insert(0, _PARENT)

import benchmark_core as benchmark_core
import governments.ads as ads_module
import run_benchmark as run_benchmark
import run_full_simulation as run_full
import run_quick_test as run_quick
from benchmark_core import (
    SUMMARY_FIELDS,
    BenchmarkConfig,
    ConfigError,
    build_plan,
    configure_multiprocessing,
    run_benchmark as run_benchmark_sweep,
    run_gov_difficulty,
)
from governments import DEFAULT_GOVERNMENTS
from governments.ads import AdsGovernment
from governments.base import Government


# ==========================================================================
# Harness (mirrors test_scenarios.py / test_decision_making.py)
# ==========================================================================

class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []

    def ok(self, name, detail=""):
        self.passed.append((name, detail))
        print(f"  PASS: {name}")

    def fail(self, name, detail=""):
        self.failed.append((name, detail))
        print(f"  FAIL: {name} — {detail}")

    def check(self, condition, name, detail=""):
        """Assert-style helper: record pass or fail based on `condition`."""
        if condition:
            self.ok(name)
        else:
            self.fail(name, detail)

    def expect_config_error(self, name, **kwargs):
        """Assert that BenchmarkConfig(**kwargs) raises ConfigError."""
        try:
            BenchmarkConfig(**kwargs)
        except ConfigError:
            self.ok(name)
        except Exception as exc:
            self.fail(name, f"raised {type(exc).__name__} instead of ConfigError: {exc}")
        else:
            self.fail(name, "no exception raised; invalid config was accepted")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"  RESULTS: {len(self.passed)}/{total} passed, {len(self.failed)} failed")
        if self.failed:
            print(f"\n  FAILURES:")
            for name, detail in self.failed:
                print(f"    - {name}: {detail}")
        print(f"{'='*60}\n")
        return len(self.failed) == 0


results = TestResults()


def _valid_config_kwargs(**overrides):
    """Baseline kwargs known to build a valid BenchmarkConfig, with overrides."""
    kwargs = dict(
        governments=("democracy",),
        difficulties=(1, 50),
        k_runs=1,
        grid_size=10,
        n_agents=20,
        max_cycles=30,
        max_steps=3,
        output_root="/tmp/does-not-need-to-exist",
    )
    kwargs.update(overrides)
    return kwargs


# ==========================================================================
# TEST 1: Production difficulty sweep matches the published methodology
# ==========================================================================

def test_full_simulation_difficulty_sequence():
    print("\n--- Test 1: run_full_simulation.py difficulty sequence ---")

    expected = (1,) + tuple(range(5, 101, 5))
    actual = tuple(run_full.DIFFICULTIES)

    results.check(
        len(actual) == 21,
        "Difficulty sweep has exactly 21 levels",
        f"got {len(actual)}: {actual}",
    )
    results.check(
        actual == expected,
        "Difficulty sweep is [1, 5, 10, ..., 100]",
        f"expected {expected}, got {actual}",
    )
    results.check(
        actual[0] == 1 and actual[-1] == 100,
        "Difficulty sweep spans 1 to 100 inclusive",
        f"first={actual[0]}, last={actual[-1]}",
    )
    results.check(
        list(actual) == sorted(set(actual)),
        "Difficulty sweep is strictly ascending with no duplicates",
        f"got {actual}",
    )
    # The steps-of-5 tail is the part most likely to be "tuned" by accident.
    tail_steps = {b - a for a, b in zip(actual[1:], actual[2:])}
    results.check(
        tail_steps == {5},
        "Difficulty sweep steps by 5 after the initial level 1",
        f"observed step sizes: {sorted(tail_steps)}",
    )


def test_full_simulation_scale_constants():
    print("\n--- Test 2: run_full_simulation.py scale constants ---")

    for name, expected in (
        ("K_RUNS", benchmark_core.PAPER_K_RUNS),
        ("GRID_SIZE", 50),
        ("N_AGENTS", 500),
        ("MAX_CYCLES", 150),
        ("MAX_STEPS", 10),
    ):
        actual = getattr(run_full, name)
        results.check(
            actual == expected,
            f"run_full_simulation.{name} == {expected}",
            f"got {actual} — production scale must match the paper",
        )


def test_full_simulation_config_totals():
    print("\n--- Test 3: run_full_simulation.py config totals ---")

    cfg = run_full.build_config()

    results.check(
        isinstance(cfg, BenchmarkConfig),
        "build_config() returns a BenchmarkConfig",
        f"got {type(cfg).__name__}",
    )
    results.check(
        tuple(cfg.governments) == tuple(DEFAULT_GOVERNMENTS),
        "Production config runs the paper's default government sweep",
        f"config={cfg.governments}, DEFAULT_GOVERNMENTS={tuple(DEFAULT_GOVERNMENTS)}",
    )
    results.check(
        cfg.k_runs == benchmark_core.PAPER_K_RUNS and len(cfg.difficulties) == 21,
        f"Production config carries 21 difficulties x {benchmark_core.PAPER_K_RUNS} seeds",
        f"k_runs={cfg.k_runs}, n_difficulties={len(cfg.difficulties)}",
    )
    # Ground truth: len(DEFAULT_GOVERNMENTS) governments x 21 difficulties x
    # PAPER_K_RUNS seeds, i.e. 21 * PAPER_K_RUNS runs per government. These
    # assertions derive the expected numbers from the same constants the
    # production code uses (run_full_simulation.build_config() passes
    # governments=DEFAULT_GOVERNMENTS, not GOVERNMENT_REGISTRY), so a change
    # to the default sweep roster or the published seed count is reflected
    # automatically instead of going stale.
    expected_total = len(DEFAULT_GOVERNMENTS) * 21 * benchmark_core.PAPER_K_RUNS
    expected_per_gov = 21 * benchmark_core.PAPER_K_RUNS
    results.check(
        cfg.total_runs == expected_total,
        f"Production config implies {expected_total} total runs",
        f"got {cfg.total_runs}",
    )
    results.check(
        cfg.total_runs // len(DEFAULT_GOVERNMENTS) == expected_per_gov,
        f"Production config implies {expected_per_gov} runs per government",
        f"got {cfg.total_runs // len(DEFAULT_GOVERNMENTS)}",
    )
    expected_cells = len(DEFAULT_GOVERNMENTS) * 21
    results.check(
        cfg.total_cells == expected_cells,
        f"Production config has {expected_cells} (government, difficulty) cells",
        f"got {cfg.total_cells}",
    )
    results.check(
        os.path.basename(cfg.output_root) == "benchmark_results",
        "Production config writes to benchmark_results/",
        f"got {cfg.output_root}",
    )


# ==========================================================================
# TEST 4: The quick test must stay quick
# ==========================================================================

def test_quick_test_config_is_small():
    print("\n--- Test 4: run_quick_test.py stays a smoke test ---")

    for name, limit in (
        ("GRID_SIZE", 15),
        ("N_AGENTS", 30),
        ("MAX_CYCLES", 50),
    ):
        actual = getattr(run_quick, name)
        results.check(
            actual <= limit,
            f"run_quick_test.{name} <= {limit}",
            f"got {actual} — quick test has drifted toward production scale",
        )

    results.check(
        run_quick.K_RUNS <= 3,
        "run_quick_test.K_RUNS <= 3",
        f"got {run_quick.K_RUNS}",
    )
    results.check(
        len(run_quick.DIFFICULTIES) <= 5,
        "run_quick_test sweeps at most 5 difficulty levels",
        f"got {len(run_quick.DIFFICULTIES)}: {run_quick.DIFFICULTIES}",
    )

    cfg = run_quick.build_config()
    results.check(
        cfg.total_runs <= 100,
        "Quick test implies at most 100 total runs",
        f"got {cfg.total_runs}",
    )
    # Guard against the smoke test clobbering the published dataset.
    results.check(
        os.path.basename(cfg.output_root) == "quick_test_results",
        "Quick test writes to quick_test_results/, not benchmark_results/",
        f"got {cfg.output_root}",
    )
    results.check(
        os.path.abspath(cfg.output_root)
        != os.path.abspath(run_full.build_config().output_root),
        "Quick test output dir differs from production output dir",
        "quick test would overwrite the production dataset",
    )
    # It should still exercise the paper's default sweep — that is the point of a smoke test.
    results.check(
        tuple(cfg.governments) == tuple(DEFAULT_GOVERNMENTS),
        "Quick test still covers the paper's default government sweep",
        f"got {cfg.governments}",
    )


# ==========================================================================
# TEST 5: BenchmarkConfig input validation
# ==========================================================================

def test_benchmark_config_rejects_invalid_input():
    print("\n--- Test 5: BenchmarkConfig validation ---")

    results.expect_config_error(
        "Rejects empty governments list",
        **_valid_config_kwargs(governments=()),
    )
    results.expect_config_error(
        "Rejects unknown government name",
        **_valid_config_kwargs(governments=("monarchy",)),
    )
    results.expect_config_error(
        "Rejects duplicate governments",
        **_valid_config_kwargs(governments=("democracy", "democracy")),
    )
    results.expect_config_error(
        "Rejects empty difficulties list",
        **_valid_config_kwargs(difficulties=()),
    )
    results.expect_config_error(
        "Rejects difficulty below 1",
        **_valid_config_kwargs(difficulties=(0, 50)),
    )
    results.expect_config_error(
        "Rejects difficulty above 100",
        **_valid_config_kwargs(difficulties=(50, 101)),
    )
    results.expect_config_error(
        "Rejects duplicate difficulty levels",
        **_valid_config_kwargs(difficulties=(50, 50)),
    )
    results.expect_config_error(
        "Rejects k_runs == 0",
        **_valid_config_kwargs(k_runs=0),
    )
    results.expect_config_error(
        "Rejects negative k_runs",
        **_valid_config_kwargs(k_runs=-1),
    )
    results.expect_config_error(
        "Rejects grid_size < 2",
        **_valid_config_kwargs(grid_size=1),
    )
    results.expect_config_error(
        "Rejects n_agents == 0",
        **_valid_config_kwargs(n_agents=0),
    )
    results.expect_config_error(
        "Rejects max_cycles == 0",
        **_valid_config_kwargs(max_cycles=0),
    )
    results.expect_config_error(
        "Rejects max_steps == 0",
        **_valid_config_kwargs(max_steps=0),
    )
    results.expect_config_error(
        "Rejects max_workers == 0",
        **_valid_config_kwargs(max_workers=0),
    )
    results.expect_config_error(
        "Rejects blank output_root",
        **_valid_config_kwargs(output_root="   "),
    )

    # ConfigError must be catchable as ValueError so callers can handle it generically.
    results.check(
        issubclass(ConfigError, ValueError),
        "ConfigError subclasses ValueError",
        f"bases: {ConfigError.__bases__}",
    )


def test_benchmark_config_accepts_valid_input():
    print("\n--- Test 6: BenchmarkConfig valid input and derived values ---")

    try:
        cfg = BenchmarkConfig(**_valid_config_kwargs(
            governments=("democracy", "ads"), difficulties=(1, 50, 100), k_runs=2,
        ))
    except Exception as exc:
        results.fail("Valid config is accepted", f"{type(exc).__name__}: {exc}")
        return

    results.ok("Valid config is accepted")
    results.check(cfg.total_cells == 6, "total_cells == governments x difficulties",
                  f"got {cfg.total_cells}, expected 6")
    results.check(cfg.total_runs == 12, "total_runs == cells x k_runs",
                  f"got {cfg.total_runs}, expected 12")
    results.check(cfg.effective_workers == 2,
                  "effective_workers defaults to one per government",
                  f"got {cfg.effective_workers}, expected 2")

    # Frozen dataclass: mutation must fail so worker processes can share it safely.
    try:
        cfg.k_runs = 99
    except dataclasses.FrozenInstanceError:
        results.ok("BenchmarkConfig is frozen (immutable)")
    except Exception as exc:
        results.fail("BenchmarkConfig is frozen (immutable)",
                     f"raised {type(exc).__name__} instead of FrozenInstanceError")
    else:
        results.fail("BenchmarkConfig is frozen (immutable)",
                     "mutation succeeded; config is not frozen")

    # with_overrides must validate the *result*, not bypass __post_init__.
    try:
        cfg.with_overrides(governments=["not_a_government"])
    except ConfigError:
        results.ok("with_overrides() re-validates the resulting config")
    except Exception as exc:
        results.fail("with_overrides() re-validates the resulting config",
                     f"raised {type(exc).__name__}: {exc}")
    else:
        results.fail("with_overrides() re-validates the resulting config",
                     "invalid override was accepted")


# ==========================================================================
# TEST 7-8: End-to-end pipeline smoke, one cell, one seed
# ==========================================================================

def _run_single_cell(gov_name, difficulty=50):
    """
    Drive the real benchmark pipeline for exactly one (gov, difficulty) cell
    with one seed. Returns (rows, final_stats_path_contents_or_None).
    """
    tmpdir = tempfile.mkdtemp(prefix=f"stage1_smoke_{gov_name}_")
    try:
        cfg = BenchmarkConfig(
            governments=(gov_name,),
            difficulties=(difficulty,),
            k_runs=1,
            grid_size=run_quick.GRID_SIZE,
            n_agents=run_quick.N_AGENTS,
            max_cycles=run_quick.MAX_CYCLES,
            max_steps=run_quick.MAX_STEPS,
            output_root=tmpdir,
            base_seed=run_quick.BASE_SEED,
            label="single-cell smoke",
        )
        # run_gov_difficulty derives its own per-run plans internally; no plan
        # crosses the call boundary.  build_plan is still exercised below as a
        # direct check that it is callable at this config.
        build_plan(cfg, difficulty, 0)
        # run_gov_difficulty logs progress to stdout; silence it to keep test
        # output readable. Exceptions still propagate.
        with contextlib.redirect_stdout(io.StringIO()):
            rows, _frames, _health = run_gov_difficulty(cfg, gov_name, difficulty)

        stats_path = os.path.join(tmpdir, gov_name, str(difficulty),
                                  "run_01", "final_stats.json")
        payload = None
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                payload = json.load(f)
        return rows, payload, stats_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _check_single_cell(gov_name):
    try:
        rows, payload, stats_path = _run_single_cell(gov_name)
    except Exception as exc:
        results.fail(f"Pipeline smoke run completed ({gov_name})",
                     f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return

    # run_gov_difficulty swallows per-run exceptions and continues, so an empty
    # rows list is the signal that the run actually crashed.
    results.check(
        len(rows) == 1,
        f"Pipeline smoke run produced 1 summary row ({gov_name})",
        f"got {len(rows)} rows — a run raised and was swallowed by run_gov_difficulty",
    )
    results.check(
        payload is not None,
        f"final_stats.json was written ({gov_name})",
        f"missing file at {stats_path}",
    )
    if payload is None:
        return

    missing = [k for k in SUMMARY_FIELDS if k not in payload]
    results.check(
        not missing,
        f"final_stats.json contains every SUMMARY_FIELDS key ({gov_name})",
        f"missing: {missing}",
    )

    for key, expected in (("government", gov_name), ("difficulty", 50), ("run", 1)):
        results.check(
            payload.get(key) == expected,
            f"final_stats.json '{key}' == {expected!r} ({gov_name})",
            f"got {payload.get(key)!r}",
        )

    nhs = payload.get("normalized_health_score")
    results.check(
        isinstance(nhs, (int, float)) and 0.0 <= nhs <= 1.0,
        f"normalized_health_score is a fraction in [0, 1] ({gov_name})",
        f"got {nhs!r}",
    )
    survival = payload.get("final_survival_rate")
    results.check(
        isinstance(survival, (int, float)) and 0.0 <= survival <= 1.0,
        f"final_survival_rate is a fraction in [0, 1] ({gov_name})",
        f"got {survival!r}",
    )

    if rows:
        row = rows[0]
        results.check(
            row.get("government") == gov_name and row.get("run") == 1,
            f"Combined row is labelled correctly ({gov_name})",
            f"got {row!r}",
        )


def test_pipeline_smoke_democracy():
    print("\n--- Test 7: single-cell pipeline smoke (democracy) ---")
    _check_single_cell("democracy")


def test_pipeline_smoke_ads():
    # ADS has the most decision-path surface area of any government, so
    # exercise it end to end: this touches the decision round, agent tagging,
    # resource-region scan and the quarantine-bbox helper.
    print("\n--- Test 8: single-cell pipeline smoke (ads) ---")
    _check_single_cell("ads")


# ==========================================================================
# TEST 9-11: governments/ads.py dead-code removal and rename
# ==========================================================================

def test_ads_dead_code_removed():
    print("\n--- Test 9: ads.py dead code is gone ---")

    for name in ("N_GROUPS_MAX", "RESOURCE_REGION_COLS", "_find_resource_rich_region"):
        results.check(
            not hasattr(ads_module, name),
            f"ads.{name} no longer exists",
            "symbol is still defined; dead-code removal was incomplete",
        )

    # The plural function is the live one and must survive.
    results.check(
        hasattr(ads_module, "_find_resource_rich_regions"),
        "ads._find_resource_rich_regions (plural) still exists",
        "the live function was removed by mistake",
    )

    for cls_name, field_name in (
        ("AgentTag", "label"),
        ("FoodEvidence", "resource_rich_region"),
        ("TerrainEvidence", "resource_rich_region"),
        ("AgentGroup", "proposed_law"),
    ):
        cls = getattr(ads_module, cls_name)
        field_names = {f.name for f in dataclasses.fields(cls)}
        results.check(
            field_name not in field_names,
            f"{cls_name}.{field_name} field removed",
            f"still present; fields = {sorted(field_names)}",
        )

    # AgentTag must still be constructible with exactly the two live fields.
    try:
        tag = ads_module.AgentTag("food", 3)
        results.check(
            tag.category == "food" and tag.band == 3,
            "AgentTag(category, band) constructs correctly",
            f"got category={tag.category!r}, band={tag.band!r}",
        )
    except Exception as exc:
        results.fail("AgentTag(category, band) constructs correctly",
                     f"{type(exc).__name__}: {exc}")


def test_ads_quarantine_bbox_rename():
    print("\n--- Test 10: quarantine bbox rename ---")

    results.check(
        "_compute_quarantine_bbox" in AdsGovernment.__dict__,
        "AdsGovernment defines _compute_quarantine_bbox",
        "the renamed method is missing",
    )
    results.check(
        isinstance(AdsGovernment.__dict__.get("_compute_quarantine_bbox"), staticmethod),
        "_compute_quarantine_bbox is a staticmethod",
        f"got {type(AdsGovernment.__dict__.get('_compute_quarantine_bbox')).__name__}",
    )
    # The whole point of the rename: ADS must NOT shadow the base class's
    # instance method, which has a different signature and different semantics.
    results.check(
        "_compute_quarantine_region" not in AdsGovernment.__dict__,
        "AdsGovernment does not shadow Government._compute_quarantine_region",
        "the old name is back in AdsGovernment; base-class shadowing bug has returned",
    )
    results.check(
        "_compute_quarantine_region" in Government.__dict__,
        "Government (base) still provides _compute_quarantine_region",
        "the base-class method other governments rely on was removed",
    )


def test_ads_law_tag_prefix_normalized():
    print("\n--- Test 11: [SCI-*] law tags renamed to [ADS-*] ---")

    with open(os.path.join(_HERE, "governments", "ads.py"), encoding="utf-8") as f:
        source = f.read()

    sci_count = source.count("[SCI-")
    ads_count = source.count("[ADS-")

    results.check(
        sci_count == 0,
        "No [SCI-*] law tags remain in ads.py",
        f"found {sci_count} occurrence(s)",
    )
    results.check(
        ads_count == 10,
        "ads.py carries 10 [ADS-*] law tags",
        f"found {ads_count} — tag set changed; update this test if intentional",
    )

    # Stale-name sweep across the whole package, excluding this test file.
    stale = []
    for root, dirs, files in os.walk(_HERE):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if not fname.endswith(".py") or fname == os.path.basename(__file__):
                continue
            path = os.path.join(root, fname)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for needle in ("[SCI-", "N_GROUPS_MAX", "RESOURCE_REGION_COLS"):
                if needle in text:
                    stale.append(f"{os.path.relpath(path, _HERE)}:{needle}")
    results.check(
        not stale,
        "No removed ads.py debug/dead-code symbols reappear anywhere in the package",
        f"found: {stale}",
    )


def test_ads_grouping_schemes_single_source_of_truth():
    print("\n--- Test 12: grouping-scheme count ---")

    counts = ads_module.GROUP_DISTRIBUTION_COUNTS

    results.check(
        len(counts) == 6,
        "GROUP_DISTRIBUTION_COUNTS has 6 schemes",
        f"got {len(counts)}: {counts} — docstrings claim 6",
    )
    results.check(
        list(counts) == sorted(counts, reverse=True),
        "GROUP_DISTRIBUTION_COUNTS is in descending order",
        f"got {counts}",
    )
    results.check(
        counts[-1] == 1,
        "GROUP_DISTRIBUTION_COUNTS ends at 1 (whole population)",
        f"got {counts[-1]}",
    )
    # No docstring in ads.py may hardcode a scheme count that contradicts the
    # GROUP_DISTRIBUTION_COUNTS constant itself.
    with open(os.path.join(_HERE, "governments", "ads.py"), encoding="utf-8") as f:
        source = f.read()
    wrong = [p for p in ("5 grouping scheme", "five grouping scheme",
                         "5 distributions", "five distributions") if p in source]
    results.check(
        not wrong,
        "No docstring hardcodes a stale grouping-scheme count",
        f"found stale phrases: {wrong}",
    )


# ==========================================================================
# TEST 13: run_benchmark.py deprecation shim
# ==========================================================================

def test_run_benchmark_shim_forwards():
    print("\n--- Test 13: run_benchmark.py deprecation shim ---")

    results.check(
        run_benchmark._full_main is run_full.main,
        "run_benchmark forwards to run_full_simulation.main",
        "shim is wired to something else",
    )
    # The shim must not carry its own (drifted) copy of the sweep constants.
    leftovers = [n for n in ("DIFFICULTY_STEP", "K_RUNS", "DIFFICULTIES", "GRID_SIZE")
                 if hasattr(run_benchmark, n)]
    results.check(
        not leftovers,
        "run_benchmark no longer defines its own sweep constants",
        f"still defines: {leftovers} — the drift bug could recur",
    )
    results.check(
        "DEPRECATED" in (run_benchmark.__doc__ or ""),
        "run_benchmark docstring marks it DEPRECATED",
        "deprecation notice missing",
    )


# ==========================================================================
# TEST 14: the multiprocessing sweep path (worker fan-out + fork-safe plotting)
# ==========================================================================

def test_run_benchmark_multiprocessing_fanout():
    """
    Exercise run_benchmark() itself — the ProcessPoolExecutor fan-out — rather
    than calling run_gov_difficulty directly.

    This is the one path the other smoke tests skip, and the riskiest: the
    workers are forked while matplotlib is already imported, and matplotlib's
    internal locks deadlock if a figure is created inside a forked child.  The
    design depends on frames being collected as plain dicts in the worker and
    rendered only in the parent.  A future change that "helpfully" moves
    rendering into run_gov_difficulty would hang the production sweep with no
    error message.  This test fails fast instead.

    Logging widens the same hazard: it is queue-based across the fork
    boundary, with each worker's `sim.*` records routed through a
    QueueHandler to a listener thread in the parent, and each worker opening
    its own run_detail.jsonl. Both are exactly the kind of inherited handle
    that corrupts a file or deadlocks a child if it crosses a fork wrongly.
    So this test runs several governments at several difficulties *with
    logging on*, and asserts on the resulting log file — worker-context
    records reaching it is the proof that the queue path survives the fork.
    """
    print("\n--- Test 14: run_benchmark() multiprocessing fan-out (with logging) ---")

    tmp = tempfile.mkdtemp(prefix="stage1_mp_")
    try:
        log_path = os.path.join(tmp, "fanout_test.log")
        cfg = BenchmarkConfig(
            governments=("democracy", "ads", "anarchy"),
            difficulties=(20, 80),
            k_runs=1,
            grid_size=8,
            n_agents=12,
            max_cycles=12,
            max_steps=2,
            output_root=tmp,
            label="MP fan-out regression test",
            # Keep the log inside the temp dir rather than polluting the CWD.
            log_file=log_path,
            heartbeat_every=5,
        )
        configure_multiprocessing()

        # NB: `run_benchmark` is bound to the deprecation-shim *module* in this
        # file's namespace; the sweep function is imported as run_benchmark_sweep.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exit_code = run_benchmark_sweep(cfg)
        output = buf.getvalue()

        results.check(
            exit_code == 0,
            "run_benchmark() worker fan-out completes with exit code 0",
            f"got {exit_code}; output tail: {output[-400:]}",
        )

        # Every cell produced its full sample, so the new completeness banner
        # must say so (and must not warn about an incomplete sample).
        results.check(
            "dataset is complete" in output,
            "complete sweep reports a complete dataset",
            f"banner missing; output tail: {output[-400:]}",
        )
        results.check(
            "INCOMPLETE sample" not in output,
            "complete sweep does not warn about an incomplete sample",
            "spurious incomplete-sample warning on a clean run",
        )

        # Results really landed on disk, from every worker process.
        for gov in cfg.governments:
            for diff in cfg.difficulties:
                run_dir = os.path.join(tmp, gov, str(diff), "run_01")
                results.check(
                    os.path.isfile(os.path.join(run_dir, "final_stats.json")),
                    f"worker output written for {gov} D={diff}",
                    f"missing final_stats.json for {gov} D={diff}",
                )
                # The recorder runs inside the worker and closes its file on
                # exit.  A footer proves __exit__ completed, which a
                # deadlocked or killed worker could not have done.
                detail = os.path.join(run_dir, "run_detail.jsonl")
                if not os.path.isfile(detail):
                    results.fail(
                        f"run_detail.jsonl written by worker for {gov} D={diff}",
                        f"missing {detail}",
                    )
                    continue
                with open(detail, encoding="utf-8") as f:
                    records = [json.loads(line) for line in f if line.strip()]
                results.check(
                    bool(records) and records[0]["rec"] == "header"
                    and records[-1]["rec"] == "footer"
                    and records[-1]["status"] == "ok",
                    f"worker-written run_detail.jsonl is complete for {gov} D={diff}",
                    f"records={[r['rec'] for r in records[:2]]}…"
                    f"{[r['rec'] for r in records[-1:]]}",
                )

        combined = os.path.join(tmp, "combined_final_stats.csv")
        if os.path.isfile(combined):
            with open(combined, newline="") as f:
                n_rows = sum(1 for _ in csv.DictReader(f))
            results.check(
                n_rows == cfg.total_runs,
                "combined CSV holds one row per run across all workers",
                f"expected {cfg.total_runs} rows, got {n_rows}",
            )
        else:
            results.fail("combined CSV written", "combined_final_stats.csv missing")

        # Plots are rendered in the parent process after the pool closes.  If
        # matplotlib had been touched in a worker this would typically hang
        # rather than fail, so reaching this assertion at all is the signal.
        plots_dir = os.path.join(tmp, "plots")
        results.check(
            os.path.isdir(plots_dir) and any(
                p.endswith(".png") for p in os.listdir(plots_dir)
            ),
            "summary plots rendered in the main process after fan-out",
            f"no PNGs in {plots_dir}",
        )

        # Snapshot PNGs come from frames pickled back out of a worker.
        viz = os.path.join(tmp, "ads", "20", "visualizations")
        results.check(
            os.path.isdir(viz) and any(p.endswith(".png") for p in os.listdir(viz)),
            "worker-captured frames rendered to snapshot PNGs",
            f"no snapshot PNGs in {viz}",
        )

        # ---- the queue-based logging path across the fork boundary ----------
        if not os.path.isfile(log_path):
            results.fail("sweep log written", f"missing {log_path}")
            return
        with open(log_path, encoding="utf-8") as f:
            log_lines = [line.rstrip("\n") for line in f if line.strip()]

        # Every line must carry the full prefix.  A torn line — the signature of
        # two processes writing the same FileHandler — would not match, so this
        # is the assertion that the workers are NOT writing the file directly.
        line_re = re.compile(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\s+"
            r"(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+\[.{1,40}?\]\s"
        )
        torn = [ln for ln in log_lines if not line_re.match(ln)]
        results.check(
            not torn,
            "every log line is well-formed (no interleaved/torn writes)",
            f"{len(torn)} malformed line(s), first: {torn[0][:120]!r}"
            if torn else "",
        )

        # Records from every worker reached the parent's single file.  This is
        # what proves the QueueHandler -> QueueListener path survived the fork:
        # the worker cannot have written these lines itself.
        for gov in cfg.governments:
            for diff in cfg.difficulties:
                tag = f"[{gov}/D={diff}"
                results.check(
                    any(tag in ln for ln in log_lines),
                    f"worker log records reached the shared file for {gov} D={diff}",
                    f"no line tagged {tag} in {log_path}",
                )
        results.check(
            any("[sweep" in ln for ln in log_lines),
            "parent log records share the file with worker records",
            "no [sweep] context lines in the log",
        )
        results.check(
            any(re.search(r"\[\S+/D=\d+/k=\d+\s*\] cycle ", ln) for ln in log_lines),
            "per-run heartbeat lines carry the narrowed run context",
            "no '<gov>/D=<d>/k=<n>' cycle heartbeat found",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# TEST 15: a failed seed is counted, not silently dropped
# ==========================================================================

def test_failed_run_is_accounted_not_silent():
    """
    run_gov_difficulty deliberately swallows a per-seed exception so one bad
    seed cannot abort a multi-hour sweep.  That is correct, but the loss must be
    *visible*: a cell that yields 2 of 3 seeds would otherwise report success,
    and every downstream mean and confidence interval would be computed over a
    smaller n than the published methodology claims.
    """
    print("\n--- Test 15: failed runs are counted and surfaced ---")

    tmp = tempfile.mkdtemp(prefix="stage1_fail_")
    original_run_one = benchmark_core._run_one
    try:
        cfg = BenchmarkConfig(
            governments=("democracy",),
            difficulties=(50,),
            k_runs=3,
            grid_size=8,
            n_agents=12,
            max_cycles=10,
            max_steps=2,
            output_root=tmp,
            label="failure accounting test",
        )

        # NB: _run_one takes a `run_dir` parameter (it drives the per-run
        # RunRecorder) and derives its own plan internally from run_idx rather
        # than accepting one.  The stand-in mirrors the real signature exactly
        # — including the default — so this test breaks loudly on a genuine
        # contract change rather than on an unrelated one.
        def flaky_run_one(config, gov_name, difficulty, run_idx, capture_viz,
                          run_dir=None):
            if run_idx == 1:          # the middle run fails
                raise RuntimeError("injected failure")
            return original_run_one(
                config, gov_name, difficulty, run_idx, capture_viz, run_dir
            )

        benchmark_core._run_one = flaky_run_one

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rows, _frames, _history = run_gov_difficulty(cfg, "democracy", 50)
        output = buf.getvalue()

        results.check(
            len(rows) == 2,
            "a failed seed drops exactly one row and the cell continues",
            f"expected 2 rows from 3 seeds with 1 injected failure, got {len(rows)}",
        )
        results.check(
            "ONLY 2/3" in output,
            "cell completion line reports the reduced sample size",
            f"expected 'ONLY 2/3' in output; got: {output[-400:]}",
        )
        results.check(
            "FAILED" in output,
            "cell completion line flags the failure",
            f"expected 'FAILED' in output; got: {output[-400:]}",
        )
        results.check(
            "all 3 runs complete" not in output,
            "a short cell never claims all runs completed",
            "cell reported full completion despite a failed seed",
        )
    finally:
        benchmark_core._run_one = original_run_one
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# TEST 16: the fan-out cannot deadlock — enforced with a hard wall-clock limit
# ==========================================================================

#: Generous relative to the ~15s this sweep actually takes, tight relative to
#: the "hangs until someone notices" failure it exists to catch.
FANOUT_TIMEOUT_SECONDS = 120

_DEADLOCK_PROBE = """
import os, sys
sys.path.insert(0, {parent!r})
from benchmark_core import BenchmarkConfig, configure_multiprocessing
from benchmark_core import run_benchmark

cfg = BenchmarkConfig(
    governments={governments!r},
    difficulties=(10, 90),
    k_runs=2,
    grid_size=8,
    n_agents=12,
    max_cycles=12,
    max_steps=2,
    output_root={out!r},
    label="deadlock probe",
    log_file={log!r},
    heartbeat_every=4,
)
configure_multiprocessing()
sys.exit(run_benchmark(cfg))
"""


def test_fanout_cannot_deadlock():
    """
    Run the widest realistic fan-out under a hard wall-clock limit.

    The worst failure mode in this codebase is not a crash: it is a silent
    multi-hour deadlock.  A worker that touches matplotlib after a fork, or that
    inherits a lock in a held state, does not raise — it blocks, and the parent
    blocks with it in `as_completed`.  An in-process test cannot catch that: it
    would hang alongside the code it is testing, taking CI (or an engineer's
    session) down with it.

    So this runs the sweep in a SUBPROCESS with `timeout=`.  A regression
    produces a loud, bounded failure instead of an unbounded hang.  All of the
    paper's default governments (DEFAULT_GOVERNMENTS) at two difficulties with
    two seeds each maximises the number of concurrent workers contending for
    the shared log queue.
    """
    print(f"\n--- Test 16: fan-out completes within "
          f"{FANOUT_TIMEOUT_SECONDS}s (deadlock guard) ---")

    tmp = tempfile.mkdtemp(prefix="stage2_deadlock_")
    try:
        out = os.path.join(tmp, "out")
        log = os.path.join(tmp, "probe.log")
        script = _DEADLOCK_PROBE.format(
            parent=_PARENT, out=out, log=log, governments=tuple(DEFAULT_GOVERNMENTS)
        )

        started = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True,
                timeout=FANOUT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            results.fail(
                "multiprocess fan-out with logging completes without deadlocking",
                f"DEADLOCK: the sweep did not finish within "
                f"{FANOUT_TIMEOUT_SECONDS}s and was killed. This is the "
                f"fork-safety invariant failing — check for matplotlib use, or "
                f"a file/queue handle captured before fork, in the worker path.",
            )
            return
        elapsed = time.monotonic() - started

        results.check(
            proc.returncode == 0,
            "multiprocess fan-out with logging completes without deadlocking",
            f"exit {proc.returncode} after {elapsed:.1f}s; "
            f"stderr tail: {proc.stderr[-400:]}",
        )

        # len(DEFAULT_GOVERNMENTS) governments x 2 difficulties x 2 seeds,
        # every run recorded.
        expected_runs = len(DEFAULT_GOVERNMENTS) * 2 * 2
        detail_files = [
            os.path.join(dirpath, "run_detail.jsonl")
            for dirpath, _dirs, files in os.walk(out)
            if "run_detail.jsonl" in files
        ]
        results.check(
            len(detail_files) == expected_runs,
            "every concurrent worker produced its run_detail.jsonl",
            f"expected {expected_runs} files, found {len(detail_files)}",
        )

        # A footer can only be written by __exit__, so its presence in every
        # file proves no worker was still mid-run when the pool closed.
        missing_footer = []
        for path in detail_files:
            with open(path, encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
            if not lines or json.loads(lines[-1]).get("rec") != "footer":
                missing_footer.append(path)
        results.check(
            not missing_footer,
            "every run closed cleanly (footer present in all detail files)",
            f"{len(missing_footer)} file(s) without a footer, e.g. "
            f"{missing_footer[:1]}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# TEST 17: repeated DEBUG-level fan-out cannot deadlock
# ==========================================================================

#: A deadlock in the fork/logging path only reproduces reliably at
#: --log-level DEBUG, where per-agent debug lines create enough log-queue
#: traffic to widen the fork/emit interleaving window. A single fan-out at
#: the default level (Test 16) samples none of that window. This repeats the
#: fan-out several times at DEBUG, each in its own subprocess under a hard
#: per-iteration timeout, so a deadlock fails the suite in bounded time (a
#: handful of iterations) instead of hanging CI.
#:
#: The timeout is sized with headroom above the sweep's own measured wall
#: time, not just above a nominal "this should be fast" guess:
#: len(DEFAULT_GOVERNMENTS) governments x 2 difficulties x 2 seeds (a couple
#: dozen short DEBUG-logged runs), spawned as fresh interpreters, routinely
#: takes on the order of a minute and a half on a
#: slower or virtualized filesystem (e.g. a Windows-mounted drive under
#: WSL). A timeout close to that healthy wall time would fail the test on
#: ordinary hardware variance rather than on an actual hang, which defeats
#: the point of a deadlock guard. FANOUT_DEBUG_ITER_TIMEOUT_SECONDS is set
#: well above the slowest observed healthy run rather than tuned to the
#: fastest one.
FANOUT_DEBUG_REPEATS = 5
FANOUT_DEBUG_ITER_TIMEOUT_SECONDS = 210

_DEADLOCK_PROBE_DEBUG = """
import os, sys
sys.path.insert(0, {parent!r})
from benchmark_core import BenchmarkConfig, configure_multiprocessing
from benchmark_core import run_benchmark

cfg = BenchmarkConfig(
    governments={governments!r},
    difficulties=(10, 90),
    k_runs=2,
    grid_size=8,
    n_agents=12,
    max_cycles=12,
    max_steps=2,
    output_root={out!r},
    label="deadlock probe (DEBUG)",
    log_file={log!r},
    heartbeat_every=4,
    engine_log_level="DEBUG",
)
configure_multiprocessing()
sys.exit(run_benchmark(cfg))
"""


def test_fanout_cannot_deadlock_repeated_debug():
    """
    Repeat the widest realistic fan-out at --log-level DEBUG, bounded by a
    per-iteration wall-clock timeout.

    Test 16 samples the fork/logging race exactly once, at a level that does
    not generate enough log-queue traffic to widen the interleaving window
    that a deadlock needs to manifest. Reliably surfacing it takes repeated
    DEBUG-level sweeps; this test cannot afford unbounded wall-clock time on
    every CI run, so it trades sample size for a hard per-iteration ceiling —
    any single hang, at any iteration, is caught and reported within
    FANOUT_DEBUG_ITER_TIMEOUT_SECONDS rather than blocking forever.

    The guarantee this test checks is architectural, not probabilistic:
    workers are started with the 'spawn' context
    (benchmark_logging.WORKER_START_METHOD), which means there is no parent
    thread state — and therefore no stream lock — for a child to inherit in
    the first place. A regression here most likely means WORKER_START_METHOD
    (or the `mp_context=` passed to ProcessPoolExecutor) was changed to
    'fork'.
    """
    print(f"\n--- Test 17: {FANOUT_DEBUG_REPEATS}x repeated DEBUG-level fan-out, "
          f"each bounded to {FANOUT_DEBUG_ITER_TIMEOUT_SECONDS}s ---")

    expected_runs = len(DEFAULT_GOVERNMENTS) * 2 * 2  # governments x 2 difficulties x 2 seeds
    hangs = []
    non_zero_exits = []
    incomplete_iters = []

    for i in range(1, FANOUT_DEBUG_REPEATS + 1):
        tmp = tempfile.mkdtemp(prefix=f"stage1_deadlock_debug_{i}_")
        try:
            out = os.path.join(tmp, "out")
            log = os.path.join(tmp, "probe.log")
            script = _DEADLOCK_PROBE_DEBUG.format(
                parent=_PARENT, out=out, log=log, governments=tuple(DEFAULT_GOVERNMENTS)
            )

            started = time.monotonic()
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True, text=True,
                    timeout=FANOUT_DEBUG_ITER_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                hangs.append(i)
                print(f"  iter {i}: HUNG after "
                      f"{FANOUT_DEBUG_ITER_TIMEOUT_SECONDS}s (killed)")
                continue
            elapsed = time.monotonic() - started

            if proc.returncode != 0:
                non_zero_exits.append((i, proc.returncode, proc.stderr[-300:]))

            detail_files = [
                os.path.join(dirpath, "run_detail.jsonl")
                for dirpath, _dirs, files in os.walk(out)
                if "run_detail.jsonl" in files
            ]
            if len(detail_files) != expected_runs:
                incomplete_iters.append((i, len(detail_files)))

            print(f"  iter {i}: OK in {elapsed:.1f}s "
                  f"({len(detail_files)}/{expected_runs} detail files)")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    results.check(
        not hangs,
        f"{FANOUT_DEBUG_REPEATS}x DEBUG-level fan-out never deadlocks",
        f"DEADLOCK on iteration(s) {hangs} of {FANOUT_DEBUG_REPEATS}: the sweep "
        f"did not finish within {FANOUT_DEBUG_ITER_TIMEOUT_SECONDS}s and was "
        f"killed. Check that worker processes are still started via "
        f"benchmark_logging.worker_mp_context() ('spawn'), not 'fork'.",
    )
    results.check(
        not non_zero_exits,
        "every repeated DEBUG fan-out iteration exits 0",
        f"non-zero exit(s): {non_zero_exits}",
    )
    results.check(
        not incomplete_iters,
        "every repeated DEBUG fan-out iteration produces all run_detail.jsonl files",
        f"iteration(s) with a wrong file count (iter, count): {incomplete_iters}; "
        f"expected {expected_runs} per iteration",
    )


# ==========================================================================
# TEST 18: stale-output warning fires on orphan directories, not on a clean
# output_root
# ==========================================================================

def test_stale_output_warning_on_orphans_and_silence_on_clean_dir():
    """
    A re-run into an existing output_root can leave orphan difficulty
    directories behind if nobody notices. ``_warn_on_existing_output`` must
    warn loudly when stale results are already present, and must stay silent
    on a directory that holds nothing (or does not exist yet) — a
    false-positive warning on every fresh run would train people to ignore
    it.
    """
    print("\n--- Test 18: stale-output warning fires on orphans, silent on clean dir ---")

    # -- Case 1: orphan difficulty directory from a discontinued sweep -----
    tmp_stale = tempfile.mkdtemp(prefix="stage1_stale_")
    try:
        # Simulate leftover output from a discontinued difficulty sequence
        # (range(1, 102, 5) -> ...,96,101) colliding with a sweep that only
        # wants {1, 5}.
        orphan_dir = os.path.join(tmp_stale, "democracy", "101")
        os.makedirs(orphan_dir)
        # A difficulty dir that IS in the new sweep should not itself count
        # as an orphan, even though the government directory pre-exists.
        os.makedirs(os.path.join(tmp_stale, "democracy", "1"))

        cfg = BenchmarkConfig(
            governments=("democracy",),
            difficulties=(1, 5),
            k_runs=1, grid_size=8, n_agents=10, max_cycles=5, max_steps=2,
            output_root=tmp_stale,
            label="orphan-detection test",
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            benchmark_core._warn_on_existing_output(cfg)
        output = buf.getvalue()

        results.check(
            "already contains results" in output,
            "warning fires when a pre-existing government directory is found",
            f"expected a warning; got: {output!r}",
        )
        results.check(
            "1 difficulty" in output or "1 director" in output,
            "warning names the orphan count (1: '101' is not in {1, 5})",
            f"expected the orphan count to be reported; got: {output!r}",
        )
        results.check(
            "Nothing is deleted" in output,
            "warning is explicit that it does not delete anything itself",
            f"expected a non-destructive disclaimer; got: {output!r}",
        )
    finally:
        shutil.rmtree(tmp_stale, ignore_errors=True)

    # -- Case 2: clean (empty) output_root — must stay silent ---------------
    tmp_clean = tempfile.mkdtemp(prefix="stage1_clean_")
    try:
        cfg_clean = BenchmarkConfig(
            governments=("democracy",),
            difficulties=(1, 5),
            k_runs=1, grid_size=8, n_agents=10, max_cycles=5, max_steps=2,
            output_root=tmp_clean,
            label="clean-directory test",
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            benchmark_core._warn_on_existing_output(cfg_clean)
        results.check(
            buf.getvalue() == "",
            "no warning on an empty output_root",
            f"expected no output; got: {buf.getvalue()!r}",
        )
    finally:
        shutil.rmtree(tmp_clean, ignore_errors=True)

    # -- Case 3: output_root that does not exist yet — must stay silent -----
    tmp_missing = tempfile.mkdtemp(prefix="stage1_missing_")
    shutil.rmtree(tmp_missing)  # exists on disk nowhere now
    try:
        cfg_missing = BenchmarkConfig(
            governments=("democracy",),
            difficulties=(1, 5),
            k_runs=1, grid_size=8, n_agents=10, max_cycles=5, max_steps=2,
            output_root=tmp_missing,
            label="nonexistent-directory test",
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            benchmark_core._warn_on_existing_output(cfg_missing)
        results.check(
            buf.getvalue() == "",
            "no warning when output_root does not exist yet",
            f"expected no output; got: {buf.getvalue()!r}",
        )
    finally:
        shutil.rmtree(tmp_missing, ignore_errors=True)


# ==========================================================================
# TEST 19: --clean actually removes prior content before the sweep starts
# ==========================================================================

def test_clean_flag_removes_prior_content_before_sweep():
    """
    ``--clean`` (``BenchmarkConfig.clean_output_root``) exists so a deliberate
    fresh run doesn't require a manual ``rm -rf``.  It must remove
    ``output_root`` (everything in it, not just recognised government
    directories) before the sweep writes anything, and the post-clean startup
    warning must therefore NOT fire, since nothing stale survives the clean.
    """
    print("\n--- Test 19: --clean removes prior content before the sweep starts ---")

    tmp = tempfile.mkdtemp(prefix="stage1_cleanflag_")
    try:
        # Leftover from a previous, unrelated run: an orphan difficulty dir
        # AND a stray root-level file that isn't part of the expected layout.
        orphan_dir = os.path.join(tmp, "democracy", "101")
        os.makedirs(orphan_dir)
        with open(os.path.join(orphan_dir, "final_stats.json"), "w") as f:
            f.write("{}")
        stray_root_file = os.path.join(tmp, "leftover.txt")
        with open(stray_root_file, "w") as f:
            f.write("should not survive --clean")

        cfg = BenchmarkConfig(
            governments=("democracy",),
            difficulties=(1,),
            k_runs=1, grid_size=8, n_agents=10, max_cycles=4, max_steps=2,
            output_root=tmp,
            label="clean-flag test",
            clean_output_root=True,
        )
        configure_multiprocessing()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exit_code = run_benchmark_sweep(cfg)
        output = buf.getvalue()

        results.check(
            exit_code == 0,
            "--clean sweep still completes successfully",
            f"got exit code {exit_code}; output tail: {output[-400:]}",
        )
        results.check(
            not os.path.exists(orphan_dir),
            "--clean removes the pre-existing orphan difficulty directory",
            f"{orphan_dir} still exists after a --clean run",
        )
        results.check(
            not os.path.exists(stray_root_file),
            "--clean removes stray files at the root of output_root too",
            f"{stray_root_file} still exists after a --clean run",
        )
        results.check(
            os.path.isdir(os.path.join(tmp, "democracy", "1", "run_01")),
            "the fresh sweep's own output is written after cleaning",
            "expected democracy/1/run_01 to exist post-sweep",
        )
        results.check(
            "already contains results" not in output,
            "the stale-output warning does not fire when --clean left nothing stale",
            f"unexpected warning in output: {output[-400:]}",
        )
        results.check(
            "removing existing output directory" in output,
            "the --clean removal itself is logged, not silent",
            f"expected a '--clean: removing...' log line; got: {output[-400:]}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# TEST 20: --clean is opt-in, not the default
# ==========================================================================

def test_clean_flag_is_not_default():
    """
    Cleaning is destructive, so it must never happen unless explicitly
    requested. Three independent checks: (1) BenchmarkConfig's own default is
    False; (2) both entry points' argument parsers default `clean` to False
    when the flag is omitted; (3) omitting --clean/clean_output_root leaves
    stale content in place and the stale-output warning still fires.
    """
    print("\n--- Test 20: --clean is opt-in, not the default ---")

    results.check(
        BenchmarkConfig(**_valid_config_kwargs()).clean_output_root is False,
        "BenchmarkConfig.clean_output_root defaults to False",
        "a freshly constructed BenchmarkConfig must not clean by default",
    )
    results.check(
        run_full.parse_args([]).clean is False,
        "run_full_simulation.py: --clean is False when omitted",
        f"got {run_full.parse_args([]).clean!r}",
    )
    results.check(
        run_full.parse_args(["--clean"]).clean is True,
        "run_full_simulation.py: --clean parses to True when given",
        f"got {run_full.parse_args(['--clean']).clean!r}",
    )
    results.check(
        run_quick.parse_args([]).clean is False,
        "run_quick_test.py: --clean is False when omitted",
        f"got {run_quick.parse_args([]).clean!r}",
    )

    tmp = tempfile.mkdtemp(prefix="stage1_noclean_")
    try:
        orphan_dir = os.path.join(tmp, "democracy", "101")
        os.makedirs(orphan_dir)
        with open(os.path.join(orphan_dir, "marker.txt"), "w") as f:
            f.write("stale")

        # No clean_output_root override at all — mirrors what happens when a
        # caller simply never passes --clean.
        cfg = BenchmarkConfig(
            governments=("democracy",),
            difficulties=(1,),
            k_runs=1, grid_size=8, n_agents=10, max_cycles=4, max_steps=2,
            output_root=tmp,
            label="no-clean-by-default test",
        )
        results.check(
            cfg.clean_output_root is False,
            "an override-free BenchmarkConfig for these entry points defaults to no cleaning",
            f"expected False, got {cfg.clean_output_root!r}",
        )
        configure_multiprocessing()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exit_code = run_benchmark_sweep(cfg)
        output = buf.getvalue()

        results.check(
            exit_code == 0,
            "sweep without --clean still completes successfully",
            f"got exit code {exit_code}",
        )
        results.check(
            os.path.exists(orphan_dir),
            "omitting --clean leaves prior stale content in place",
            f"{orphan_dir} was unexpectedly removed without --clean being set",
        )
        results.check(
            "already contains results" in output,
            "the stale-output warning still fires when --clean is omitted and stale content exists",
            f"expected the stale-output warning; got: {output[-600:]}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# Main
# ==========================================================================

def run_all():
    print("=" * 60)
    print("  PRODUCTION BENCHMARK PIPELINE — SYSTEM REGRESSION TESTS")
    print("  Entry-point configs, BenchmarkConfig validation, ads.py invariants")
    print("=" * 60)

    # Entry-point configuration
    test_full_simulation_difficulty_sequence()
    test_full_simulation_scale_constants()
    test_full_simulation_config_totals()
    test_quick_test_config_is_small()

    # BenchmarkConfig contract
    test_benchmark_config_rejects_invalid_input()
    test_benchmark_config_accepts_valid_input()

    # End-to-end pipeline
    test_pipeline_smoke_democracy()
    test_pipeline_smoke_ads()

    # ads.py refactor
    test_ads_dead_code_removed()
    test_ads_quarantine_bbox_rename()
    test_ads_law_tag_prefix_normalized()
    test_ads_grouping_schemes_single_source_of_truth()

    # Deprecation shim
    test_run_benchmark_shim_forwards()

    # Multiprocessing sweep path + failure accounting
    test_run_benchmark_multiprocessing_fanout()
    test_failed_run_is_accounted_not_silent()
    test_fanout_cannot_deadlock()
    test_fanout_cannot_deadlock_repeated_debug()

    # Stale-output warning + --clean
    test_stale_output_warning_on_orphans_and_silence_on_clean_dir()
    test_clean_flag_removes_prior_content_before_sweep()
    test_clean_flag_is_not_default()

    return results.summary()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
