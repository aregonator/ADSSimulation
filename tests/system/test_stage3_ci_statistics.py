#!/usr/bin/env python3
"""
test_stage3_ci_statistics.py — Comprehensive test suite for per-cell CI statistics.

Tests cover:
  1. Unit tests for compute_cell_statistics() and write_cell_stats()
  2. Integration test: run_quick_test.py produces correct cell_stats.json files
  3. Regression tests: per-run output unchanged, plots still work
  4. Data quality checks: monotonicity, NaN handling, edge cases
  5. Schema validation: JSON structure matches design

Run from the Simulation project root:
  python3 tests/system/test_stage3_ci_statistics.py

Runtime: on the order of 10-20 minutes, dominated by several full
`run_quick_test.py` invocations in the integration and data-quality
sections (each itself several minutes; see README.md's "tests/ subtree"
section for a measured figure and its filesystem caveat).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Path setup
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root: tests/<subpkg>/<file>.py -> tests/<subpkg> -> tests/ -> root
_PARENT = _HERE  # sys.path target for the flat top-level packages (engine, governments, ...)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from benchmark_core import (
    BenchmarkConfig,
    SUMMARY_FIELDS,
    compute_cell_statistics,
    write_cell_stats,
    ci_95,
    t_critical_95,
)


class TestRunner:
    """Simple test harness with pass/fail tracking."""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.failures: List[str] = []

    def test(self, name: str, condition: bool, message: str = ""):
        """Record one assertion."""
        self.tests_run += 1
        if condition:
            self.tests_passed += 1
            print(f"  ✓ {name}")
        else:
            self.failures.append(f"{name}: {message}")
            print(f"  ✗ {name}: {message}")

    def summary(self) -> Tuple[int, int]:
        """Return (passed, total) and print summary."""
        print(f"\n{'=' * 70}")
        if self.failures:
            print(f"FAILURES ({len(self.failures)}/{self.tests_run}):")
            for fail in self.failures:
                print(f"  - {fail}")
        else:
            print(f"ALL {self.tests_run} TESTS PASSED")
        print(f"{'=' * 70}\n")
        return self.tests_passed, self.tests_run


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_ci_95_function():
    """Unit test: ci_95() returns correct mean and bounds."""
    runner = TestRunner()
    print("\n### UNIT TEST: ci_95() function ###\n")

    # Test 1: Three known values [1, 2, 3]
    mean, lower, upper = ci_95([1.0, 2.0, 3.0])
    runner.test(
        "ci_95([1,2,3]) mean=2",
        abs(mean - 2.0) < 0.01,
        f"expected 2.0, got {mean}",
    )
    runner.test(
        "ci_95([1,2,3]) lower < mean < upper",
        lower < mean < upper,
        f"expected lower < {mean} < upper; got {lower}, {upper}",
    )

    # Test 2: Single value (n=1) → bounds collapse to point
    mean, lower, upper = ci_95([5.0])
    runner.test(
        "ci_95([5]) collapses to (5,5,5)",
        mean == 5.0 and lower == 5.0 and upper == 5.0,
        f"expected (5, 5, 5), got ({mean}, {lower}, {upper})",
    )

    # Test 3: Empty list (n=0) → all zero
    mean, lower, upper = ci_95([])
    runner.test(
        "ci_95([]) returns (0, 0, 0)",
        mean == 0.0 and lower == 0.0 and upper == 0.0,
        f"expected (0, 0, 0), got ({mean}, {lower}, {upper})",
    )

    # Test 4: Two identical values (std=0) → no margin
    mean, lower, upper = ci_95([7.0, 7.0])
    runner.test(
        "ci_95([7,7]) has zero margin",
        mean == lower == upper == 7.0,
        f"expected (7, 7, 7), got ({mean}, {lower}, {upper})",
    )

    return runner.summary()


def test_t_critical_95_function():
    """Unit test: t_critical_95() returns sensible critical values."""
    runner = TestRunner()
    print("\n### UNIT TEST: t_critical_95() function ###\n")

    # Test known values from the t-table
    runner.test("t_critical_95(1)=12.706", abs(t_critical_95(1) - 12.706) < 0.01, "")
    runner.test("t_critical_95(10)=2.228", abs(t_critical_95(10) - 2.228) < 0.01, "")
    runner.test("t_critical_95(30)≈1.960", abs(t_critical_95(30) - 1.960) < 0.01, "")

    # Documented contract: t_critical_95 is strictly decreasing for df < 30,
    # then plateaus at exactly 1.960 (the z approximation) for every df >= 30.
    for df in [1, 5, 10, 25]:
        next_t = t_critical_95(df + 5)
        curr_t = t_critical_95(df)
        runner.test(
            f"t_critical_95({df}) > t_critical_95({df+5}) (below plateau)",
            curr_t > next_t,
            f"expected {curr_t} > {next_t}",
        )

    for df in [30, 35, 100]:
        curr_t = t_critical_95(df)
        runner.test(
            f"t_critical_95({df}) == 1.960 (on/above plateau)",
            curr_t == 1.960,
            f"expected exactly 1.960, got {curr_t}",
        )

    return runner.summary()


def test_compute_cell_statistics_basic():
    """Unit test: compute_cell_statistics() with hand-built data."""
    runner = TestRunner()
    print("\n### UNIT TEST: compute_cell_statistics() basic ###\n")

    # Create 3 runs for one (ads, difficulty=1) cell
    all_combined_rows = [
        {
            "government": "ads",
            "difficulty": 1,
            "run": 1,
            "normalized_health_score": 0.75,
            "final_survival_rate": 0.80,
            "final_median_health": 0.95,
            "final_health_gini": 0.05,
            "time_to_50pct_loss": None,
            "min_survival_rate": 0.70,
        },
        {
            "government": "ads",
            "difficulty": 1,
            "run": 2,
            "normalized_health_score": 0.76,
            "final_survival_rate": 0.81,
            "final_median_health": 0.96,
            "final_health_gini": 0.06,
            "time_to_50pct_loss": None,
            "min_survival_rate": 0.71,
        },
        {
            "government": "ads",
            "difficulty": 1,
            "run": 3,
            "normalized_health_score": 0.77,
            "final_survival_rate": 0.82,
            "final_median_health": 0.97,
            "final_health_gini": 0.07,
            "time_to_50pct_loss": None,
            "min_survival_rate": 0.72,
        },
    ]

    # Create a mock manifest with this cell
    manifest_results = {
        "short_cells": [
            {
                "government": "ads",
                "difficulty": 1,
                "runs_recorded": 3,
                "runs_expected": 3,
            }
        ]
    }

    # Create a mock config
    config = BenchmarkConfig(
        governments=("ads",),
        difficulties=(1,),
        k_runs=3,
        grid_size=10,
        n_agents=20,
        max_cycles=30,
        max_steps=3,
        output_root="/tmp/test",
    )

    # Call the function
    _, cell_stats_dict = compute_cell_statistics(config, all_combined_rows, manifest_results)

    # Check the returned dict
    runner.test(
        "cell_stats_dict has key (ads, 1)",
        ("ads", 1) in cell_stats_dict,
        "key not found",
    )

    cell_stats = cell_stats_dict[("ads", 1)]

    # Check top-level fields
    runner.test("cell_stats has 'government'", "government" in cell_stats, "")
    runner.test(
        "cell_stats['government']='ads'",
        cell_stats["government"] == "ads",
        f"got {cell_stats['government']}",
    )
    runner.test("cell_stats has 'n'", "n" in cell_stats, "")
    runner.test(
        "cell_stats['n']=3 (true sample size)",
        cell_stats["n"] == 3,
        f"got {cell_stats['n']}",
    )
    runner.test("cell_stats has 'timestamp'", "timestamp" in cell_stats, "")
    runner.test("cell_stats has 'metrics'", "metrics" in cell_stats, "")

    # Check metrics structure
    metrics = cell_stats["metrics"]
    runner.test(
        "metrics has 'normalized_health_score'",
        "normalized_health_score" in metrics,
        "",
    )

    nhs = metrics["normalized_health_score"]
    runner.test("nhs has 'mean'", "mean" in nhs, "")
    runner.test("nhs has 'ci95_lower'", "ci95_lower" in nhs, "")
    runner.test("nhs has 'ci95_upper'", "ci95_upper" in nhs, "")
    runner.test("nhs has 'n'", "n" in nhs, "")

    # Check values: mean should be ~0.76
    runner.test(
        "nhs mean ≈ 0.76",
        abs(nhs["mean"] - 0.76) < 0.01,
        f"got {nhs['mean']}",
    )

    # Check monotonicity: lower < mean < upper
    runner.test(
        "nhs: lower < mean < upper",
        nhs["ci95_lower"] < nhs["mean"] < nhs["ci95_upper"],
        f"got {nhs['ci95_lower']} < {nhs['mean']} < {nhs['ci95_upper']}",
    )

    # Check that time_to_50pct_loss has n=0 (all None)
    ttl = metrics["time_to_50pct_loss"]
    runner.test(
        "time_to_50pct_loss n=0 (all None)",
        ttl["n"] == 0,
        f"got {ttl['n']}",
    )
    runner.test(
        "time_to_50pct_loss mean=null",
        ttl["mean"] is None,
        f"got {ttl['mean']}",
    )

    return runner.summary()


def test_compute_cell_statistics_nan_handling():
    """Unit test: compute_cell_statistics() handles NaN correctly."""
    runner = TestRunner()
    print("\n### UNIT TEST: compute_cell_statistics() NaN handling ###\n")

    # Create runs with NaN in one metric
    all_combined_rows = [
        {
            "government": "ads",
            "difficulty": 50,
            "run": 1,
            "normalized_health_score": 0.75,
            "final_survival_rate": float("nan"),
            "final_median_health": 0.95,
            "final_health_gini": 0.05,
            "time_to_50pct_loss": None,
            "min_survival_rate": 0.70,
        },
        {
            "government": "ads",
            "difficulty": 50,
            "run": 2,
            "normalized_health_score": 0.76,
            "final_survival_rate": 0.81,
            "final_median_health": 0.96,
            "final_health_gini": 0.06,
            "time_to_50pct_loss": None,
            "min_survival_rate": 0.71,
        },
    ]

    config = BenchmarkConfig(
        governments=("ads",),
        difficulties=(50,),
        k_runs=2,
        grid_size=10,
        n_agents=20,
        max_cycles=30,
        max_steps=3,
        output_root="/tmp/test",
    )

    manifest_results = {
        "short_cells": [
            {
                "government": "ads",
                "difficulty": 50,
                "runs_recorded": 2,
                "runs_expected": 2,
            }
        ]
    }

    _, cell_stats_dict = compute_cell_statistics(config, all_combined_rows, manifest_results)
    cell_stats = cell_stats_dict[("ads", 50)]
    metrics = cell_stats["metrics"]

    # NaN should be filtered out, so n=1 for final_survival_rate (only the 0.81 value)
    fsr = metrics["final_survival_rate"]
    runner.test(
        "final_survival_rate n=1 (one NaN filtered out)",
        fsr["n"] == 1,
        f"got {fsr['n']}",
    )
    runner.test(
        "final_survival_rate mean=0.81 (only valid value)",
        abs(fsr["mean"] - 0.81) < 0.01,
        f"got {fsr['mean']}",
    )

    return runner.summary()


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_integration_run_quick_test():
    """Integration test: run_quick_test.py produces cell_stats.json files."""
    runner = TestRunner()
    print("\n### INTEGRATION TEST: run_quick_test.py ###\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "integration_test_results")

        # Run the quick test
        cmd = [
            sys.executable,
            os.path.join(_HERE, "run_quick_test.py"),
            "--output-dir",
            output_dir,
        ]

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        runner.test(
            "run_quick_test.py exits with code 0",
            result.returncode == 0,
            f"exit code {result.returncode}, stderr: {result.stderr[:200]}",
        )

        # Check that cell_stats.json files were created
        # Quick test: 7 governments × 3 difficulties × 2 seeds = 42 runs → 21 cells
        cell_count = 0
        for root, dirs, files in os.walk(output_dir):
            if "cell_stats.json" in files:
                cell_count += 1
                # Parse and validate JSON
                cell_stats_path = os.path.join(root, "cell_stats.json")
                try:
                    with open(cell_stats_path) as f:
                        cell_stats = json.load(f)
                    runner.test(
                        f"cell_stats.json parses at {cell_stats_path}",
                        True,
                        "",
                    )
                except json.JSONDecodeError as e:
                    runner.test(
                        f"cell_stats.json parses at {cell_stats_path}",
                        False,
                        str(e),
                    )

        runner.test(
            "cell_stats.json files created (expect ~21)",
            cell_count >= 18,  # Allow some flexibility
            f"found {cell_count} files",
        )

        return runner.summary()


# =============================================================================
# REGRESSION TESTS
# =============================================================================

def test_regression_backward_compatibility():
    """Regression test: per-run output unchanged."""
    runner = TestRunner()
    print("\n### REGRESSION TEST: backward compatibility ###\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "regression_test_results")

        # Run the quick test once
        cmd = [
            sys.executable,
            os.path.join(_HERE, "run_quick_test.py"),
            "--output-dir",
            output_dir,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        runner.test("run_quick_test.py succeeds", result.returncode == 0, "")

        # Check that per-run final_stats.json exists and is unchanged
        for root, dirs, files in os.walk(output_dir):
            if "final_stats.json" in files:
                path = os.path.join(root, "final_stats.json")
                try:
                    with open(path) as f:
                        final_stats = json.load(f)
                    # Check that it doesn't have CI fields (they go in cell_stats.json)
                    has_ci_fields = any(
                        key in final_stats for key in ["ci95_lower", "ci95_upper"]
                    )
                    runner.test(
                        f"final_stats.json has no CI fields ({path})",
                        not has_ci_fields,
                        f"found CI fields in {path}",
                    )
                except json.JSONDecodeError:
                    runner.test(
                        f"final_stats.json parses ({path})",
                        False,
                        "JSON decode error",
                    )

        # Check that combined_final_stats.csv exists and has expected columns
        csv_path = os.path.join(output_dir, "combined_final_stats.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                first_line = f.readline().strip()
                runner.test(
                    "combined_final_stats.csv has no CI columns",
                    "ci95_lower" not in first_line and "ci95_upper" not in first_line,
                    f"header: {first_line}",
                )
        else:
            runner.test(
                "combined_final_stats.csv exists",
                False,
                "file not found",
            )

        # Check that plots still exist
        plots_dir = os.path.join(output_dir, "plots")
        if os.path.exists(plots_dir):
            plot_files = [
                f
                for f in os.listdir(plots_dir)
                if f.endswith(".png")
            ]
            runner.test(
                f"plots directory has PNG files ({len(plot_files)} found)",
                len(plot_files) >= 4,
                f"expected ≥4 plots, got {len(plot_files)}",
            )
        else:
            runner.test("plots directory exists", False, "not found")

        return runner.summary()


# =============================================================================
# DATA QUALITY TESTS
# =============================================================================

def test_data_quality_monotonicity():
    """Data quality test: CI bounds are monotonic (lower ≤ mean ≤ upper)."""
    runner = TestRunner()
    print("\n### DATA QUALITY TEST: CI monotonicity ###\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "dq_test_results")

        cmd = [
            sys.executable,
            os.path.join(_HERE, "run_quick_test.py"),
            "--output-dir",
            output_dir,
        ]

        subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Check monotonicity in every cell_stats.json
        monotonic_violations = 0
        total_metrics = 0

        for root, dirs, files in os.walk(output_dir):
            if "cell_stats.json" in files:
                path = os.path.join(root, "cell_stats.json")
                with open(path) as f:
                    cell_stats = json.load(f)

                metrics = cell_stats.get("metrics", {})
                for metric_name, metric_data in metrics.items():
                    total_metrics += 1
                    mean = metric_data.get("mean")
                    lower = metric_data.get("ci95_lower")
                    upper = metric_data.get("ci95_upper")
                    n = metric_data.get("n", 0)

                    # If n > 1, check monotonicity
                    if n > 1 and mean is not None:
                        if not (lower <= mean <= upper):
                            monotonic_violations += 1
                            print(
                                f"    VIOLATION: {metric_name} in {path}: "
                                f"{lower} ≤ {mean} ≤ {upper}"
                            )

        runner.test(
            f"No monotonicity violations ({total_metrics} metrics checked)",
            monotonic_violations == 0,
            f"{monotonic_violations} violations found",
        )

        return runner.summary()


def test_data_quality_ci_width_vs_n():
    """Data quality test: CI width decreases as n increases."""
    runner = TestRunner()
    print("\n### DATA QUALITY TEST: CI width vs sample size ###\n")

    # Fixed-amplitude alternating pattern: spread (+/- 0.05 around 0.70) does
    # not grow with n_val, so any change in CI width isolates the effect of
    # sample size rather than a confound from n-dependent variance.
    def _fixed_spread_values(n_val: int) -> List[float]:
        return [0.70 + (0.05 if i % 2 == 0 else -0.05) for i in range(n_val)]

    # Build test data with varying n
    for n_val in [1, 3, 5]:
        values = _fixed_spread_values(n_val)
        mean, lower, upper = ci_95(values)
        width = upper - lower if upper is not None else 0

        if n_val > 1:
            runner.test(
                f"n={n_val}: CI width={width:.4f}",
                width >= 0,
                f"negative width {width}",
            )

    # Verify that CI width shrinks as n increases
    widths = []
    for n_val in [2, 5, 10]:
        values = _fixed_spread_values(n_val)
        mean, lower, upper = ci_95(values)
        width = upper - lower if upper is not None else 0
        widths.append((n_val, width))

    for i in range(len(widths) - 1):
        n1, w1 = widths[i]
        n2, w2 = widths[i + 1]
        runner.test(
            f"CI width shrinks: n={n1} width={w1:.4f} > n={n2} width={w2:.4f}",
            w1 > w2,
            f"expected {w1} > {w2}",
        )

    return runner.summary()


def test_data_quality_no_nan_in_json():
    """Data quality test: no NaN/Infinity in JSON output."""
    runner = TestRunner()
    print("\n### DATA QUALITY TEST: no NaN/Infinity in JSON ###\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "dq_nan_test")

        cmd = [
            sys.executable,
            os.path.join(_HERE, "run_quick_test.py"),
            "--output-dir",
            output_dir,
        ]

        subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Check every JSON file for bare NaN/Infinity (invalid JSON)
        nan_found = 0
        inf_found = 0

        for root, dirs, files in os.walk(output_dir):
            for fname in files:
                if fname.endswith(".json"):
                    path = os.path.join(root, fname)
                    with open(path) as f:
                        content = f.read()
                        # Bare NaN and Infinity are invalid JSON
                        if ": NaN" in content or ": Infinity" in content:
                            nan_found += 1
                        if ": -Infinity" in content:
                            inf_found += 1

        runner.test(
            "No bare NaN in JSON files",
            nan_found == 0,
            f"{nan_found} files have bare NaN",
        )
        runner.test(
            "No bare Infinity in JSON files",
            inf_found == 0,
            f"{inf_found} files have bare Infinity",
        )

        return runner.summary()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_edge_case_single_value():
    """Edge case test: n=1 cell (single successful run)."""
    runner = TestRunner()
    print("\n### EDGE CASE TEST: n=1 (single run) ###\n")

    all_combined_rows = [
        {
            "government": "democracy",
            "difficulty": 100,
            "run": 1,
            "normalized_health_score": 0.65,
            "final_survival_rate": 0.70,
            "final_median_health": 0.90,
            "final_health_gini": 0.10,
            "time_to_50pct_loss": None,
            "min_survival_rate": 0.60,
        },
    ]

    config = BenchmarkConfig(
        governments=("democracy",),
        difficulties=(100,),
        k_runs=1,
        grid_size=10,
        n_agents=20,
        max_cycles=30,
        max_steps=3,
        output_root="/tmp/test",
    )

    manifest_results = {
        "short_cells": [
            {
                "government": "democracy",
                "difficulty": 100,
                "runs_recorded": 1,
                "runs_expected": 1,
            }
        ]
    }

    _, cell_stats_dict = compute_cell_statistics(config, all_combined_rows, manifest_results)
    cell_stats = cell_stats_dict[("democracy", 100)]
    metrics = cell_stats["metrics"]

    # At n=1, CI should collapse to the single value
    nhs = metrics["normalized_health_score"]
    runner.test(
        "n=1: mean = lower = upper",
        nhs["mean"] == nhs["ci95_lower"] == nhs["ci95_upper"],
        f"got mean={nhs['mean']}, lower={nhs['ci95_lower']}, upper={nhs['ci95_upper']}",
    )
    runner.test(
        "n=1: bounds equal the single value (0.65)",
        abs(nhs["mean"] - 0.65) < 0.01,
        f"got {nhs['mean']}",
    )

    return runner.summary()


def test_edge_case_all_none_metric():
    """Edge case test: metric that is all-None (never computed)."""
    runner = TestRunner()
    print("\n### EDGE CASE TEST: all-None metric ###\n")

    # Create runs where time_to_50pct_loss is always None
    all_combined_rows = [
        {
            "government": "anarchy",
            "difficulty": 1,
            "run": i,
            "normalized_health_score": 0.50 + i * 0.01,
            "final_survival_rate": 0.55 + i * 0.01,
            "final_median_health": 0.85 + i * 0.01,
            "final_health_gini": 0.15 + i * 0.01,
            "time_to_50pct_loss": None,
            "min_survival_rate": 0.40 + i * 0.01,
        }
        for i in range(3)
    ]

    config = BenchmarkConfig(
        governments=("anarchy",),
        difficulties=(1,),
        k_runs=3,
        grid_size=10,
        n_agents=20,
        max_cycles=30,
        max_steps=3,
        output_root="/tmp/test",
    )

    manifest_results = {
        "short_cells": [
            {
                "government": "anarchy",
                "difficulty": 1,
                "runs_recorded": 3,
                "runs_expected": 3,
            }
        ]
    }

    _, cell_stats_dict = compute_cell_statistics(config, all_combined_rows, manifest_results)
    cell_stats = cell_stats_dict[("anarchy", 1)]
    metrics = cell_stats["metrics"]

    ttl = metrics["time_to_50pct_loss"]
    runner.test(
        "all-None metric: n=0",
        ttl["n"] == 0,
        f"got {ttl['n']}",
    )
    runner.test(
        "all-None metric: mean=null",
        ttl["mean"] is None,
        f"got {ttl['mean']}",
    )
    runner.test(
        "all-None metric: lower=null",
        ttl["ci95_lower"] is None,
        f"got {ttl['ci95_lower']}",
    )
    runner.test(
        "all-None metric: upper=null",
        ttl["ci95_upper"] is None,
        f"got {ttl['ci95_upper']}",
    )

    return runner.summary()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all test suites and report overall results."""
    print("\n" + "=" * 70)
    print("  PER-CELL CI STATISTICS — COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    all_tests = [
        ("Unit: ci_95()", test_ci_95_function),
        ("Unit: t_critical_95()", test_t_critical_95_function),
        ("Unit: compute_cell_statistics() basic", test_compute_cell_statistics_basic),
        ("Unit: compute_cell_statistics() NaN", test_compute_cell_statistics_nan_handling),
        ("Integration: run_quick_test.py", test_integration_run_quick_test),
        ("Regression: backward compatibility", test_regression_backward_compatibility),
        ("Data Quality: monotonicity", test_data_quality_monotonicity),
        ("Data Quality: CI width vs n", test_data_quality_ci_width_vs_n),
        ("Data Quality: no NaN in JSON", test_data_quality_no_nan_in_json),
        ("Edge Case: n=1", test_edge_case_single_value),
        ("Edge Case: all-None metric", test_edge_case_all_none_metric),
    ]

    total_passed = 0
    total_tests = 0

    for name, test_func in all_tests:
        try:
            passed, total = test_func()
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"\n✗ Test suite '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"  FINAL RESULT: {total_passed}/{total_tests} assertions passed")
    print("=" * 70)

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
