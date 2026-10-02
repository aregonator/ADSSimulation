#!/usr/bin/env python3
"""
Regression guard: the median bootstrap confidence interval and its seeding.

    python3 test_median_ci.py          # exit 0 = pass, 1 = fail

WHAT THIS PROTECTS
------------------
``benchmark_core.bootstrap_median_ci`` is the estimator behind every
``*_median_ci.png`` figure: a 95% percentile-bootstrap interval on the
MEDIAN of each cell's runs, with a deliberately separate degenerate-status
vocabulary so a reader can tell "the interval is genuinely narrow" apart
from "there were two runs" apart from "every run returned the same number".
Several of those properties are easy to break silently and hard to notice
by eyeballing a figure:

  1. **The pinned constants drift.**  ``_MEDIAN_CI_N_BOOT`` (10,000),
     ``_MEDIAN_CI_LEVEL`` (0.95) and ``_MEDIAN_CI_MIN_N`` (6) are values baked
     into the sidecar text and the published figures.  A change to any of them
     changes the meaning of every archived interval without changing anything
     that would fail a smoke test.

  2. **The n<6 threshold looks arbitrary but isn't.**  For every n up to 5 the
     percentile bootstrap interval of a median is ALWAYS exactly ``[min, max]``
     — a property of the sample size, not of the data.  The reason is that the
     bootstrap median's distribution is discrete with atoms on the sample's
     order statistics, and while ``P(bootstrap median == min)`` exceeds
     ``alpha/2`` (2.5%) the 2.5th percentile cannot move off the smallest
     observation.  Exact enumeration gives 25.000% / 25.926% / 5.078% / 5.792%
     at n = 2/3/4/5, and only 0.870% at n = 6.

     Setting the threshold below 6 — e.g. only excluding n=2 — would leave
     cells with n of 3, 4 or 5 drawn with error bars that encode nothing but
     their sample size, since the interval is still exactly ``[min, max]`` at
     those n.  Group 3 below reproduces the bootstrap by hand across the
     whole band n = 2..5 (always the range) and at n = 6 (usually narrower), so
     the threshold's justification is verified directly against the mechanism
     rather than against the current constant's value alone.

  3. **It could quietly become a bootstrap on the MEAN.**  This function was
     deliberately NOT built on top of the pre-existing ``ci_95`` (a t-interval
     on the mean); reusing that machinery, or reintroducing it during a
     refactor, would silently change the estimand every figure reports.
     Group 4 differentially reproduces both the median and the mean bootstrap
     on the *same* resample-index draw and asserts this function matches the
     median one exactly and NOT the mean one — a two-sided check, since either
     half alone could pass vacuously (e.g. on a symmetric dataset the two
     estimators can coincide).

  4. **The seed could stop being used, or the wiring to production could
     drift.** Group 6 includes a non-vacuity control: because the bootstrap
     is discrete, many datasets give byte-identical intervals under different
     seeds, so a determinism check alone cannot tell "the seed is honoured"
     apart from "the seed is ignored". Group 9 locks the production seed key
     (``"plot.bootstrap"``) and the estimator separation (``make_median_ci_plots``
     must never call ``ci_95``) at the AST level, so a future edit cannot
     silently re-key every figure or blend the two estimators.

WHY THIS FILE LIVES HERE, AND NOT IN tests/
-------------------------------------------
This file follows the precedent set by ``test_seed_derivation.py`` and
``test_default_governments.py``: it sits beside the code it guards, alongside
the project's other top-level regression tests, imports the way the entry
points do, and runs under a plain ``python3`` with no pytest and no import
gymnastics.

See ``benchmark_core.py`` around ``bootstrap_median_ci``/``MedianCI``/
``make_median_ci_plots`` for the implementation this pins.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no display — before pyplot import

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from engine.scenario_plan import derive_seed                  # noqa: E402
from benchmark_core import (                                  # noqa: E402
    bootstrap_median_ci,
    MedianCI,
    ci_95,
    MEDIAN_CI_STATUS_OK,
    MEDIAN_CI_STATUS_EMPTY,
    MEDIAN_CI_STATUS_INSUFFICIENT_N,
    MEDIAN_CI_STATUS_CONSTANT,
    MEDIAN_CI_STATUS_ZERO_WIDTH,
    _MEDIAN_CI_N_BOOT,
    _MEDIAN_CI_LEVEL,
    _MEDIAN_CI_MIN_N,
)


def _check(results: list, name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Group 1 — constants pinned
# ---------------------------------------------------------------------------

def _check_constants(results: list) -> None:
    print("\nGroup 1 — constants pinned (a silent change re-keys every figure):")
    _check(
        results, "_MEDIAN_CI_N_BOOT == 10000",
        _MEDIAN_CI_N_BOOT == 10000, f"got {_MEDIAN_CI_N_BOOT!r}",
    )
    _check(
        results, "_MEDIAN_CI_LEVEL == 0.95",
        _MEDIAN_CI_LEVEL == 0.95, f"got {_MEDIAN_CI_LEVEL!r}",
    )
    _check(
        results, "_MEDIAN_CI_MIN_N == 6",
        _MEDIAN_CI_MIN_N == 6, f"got {_MEDIAN_CI_MIN_N!r}",
    )


# ---------------------------------------------------------------------------
# Group 2 — degenerate statuses are each reachable and correct
# ---------------------------------------------------------------------------

def _check_degenerate_statuses(results: list) -> None:
    print("\nGroup 2 — the four degenerate statuses, full tuples, drawable, half_width, describe():")

    cases = [
        ("empty", [], 1, (None, None, None, 0, MEDIAN_CI_STATUS_EMPTY)),
        ("insufficient_n (n=1)", [3.0], 1, (3.0, 3.0, 3.0, 1, MEDIAN_CI_STATUS_INSUFFICIENT_N)),
        # n=5 rather than n=2: it is the TOP of the degenerate band and the
        # value that would silently start drawing a false band if the
        # insufficient-n threshold were ever raised to 3. n=2 is still covered
        # exhaustively by Group 3's n=1..5 sweep, so nothing is lost by fixing
        # this fixture at the boundary that matters most.
        ("insufficient_n (n=5)", [1.0, 2.0, 3.0, 4.0, 9.0], 1,
         (3.0, 3.0, 3.0, 5, MEDIAN_CI_STATUS_INSUFFICIENT_N)),
        ("constant", [5.0] * 6, 1, (5.0, 5.0, 5.0, 6, MEDIAN_CI_STATUS_CONSTANT)),
        ("degenerate_zero_width", [1.0] * 9 + [1000.0], 1,
         (1.0, 1.0, 1.0, 10, MEDIAN_CI_STATUS_ZERO_WIDTH)),
        ("ok", list(range(1, 11)), 1, (5.5, 3.0, 8.0, 10, MEDIAN_CI_STATUS_OK)),
    ]

    results_by_label = {}
    for label, vals, seed, expected in cases:
        r = bootstrap_median_ci(vals, seed)
        got = (r.median, r.lower, r.upper, r.n, r.status)
        _check(
            results, f"{label}: full tuple matches",
            got == expected, f"expected {expected}, got {got}",
        )
        results_by_label[label] = r

    # .drawable — True only for 'ok'.
    for label, r in results_by_label.items():
        expected_drawable = (label == "ok")
        _check(
            results, f"{label}: .drawable == {expected_drawable}",
            r.drawable == expected_drawable, f"got {r.drawable}",
        )

    # .half_width — None only for empty; (upper-lower)/2 otherwise; 'ok' == 2.5.
    for label, r in results_by_label.items():
        if label == "empty":
            _check(
                results, "empty: .half_width is None",
                r.half_width is None, f"got {r.half_width!r}",
            )
        else:
            expected_hw = (r.upper - r.lower) / 2.0
            _check(
                results, f"{label}: .half_width == (upper-lower)/2",
                r.half_width == expected_hw, f"expected {expected_hw}, got {r.half_width!r}",
            )
    ok_r = results_by_label["ok"]
    _check(
        results, "ok: .half_width == 2.5",
        ok_r.half_width == 2.5, f"got {ok_r.half_width!r}",
    )

    # .describe() content checks for each non-ok status.
    desc_checks = [
        ("insufficient_n (n=1)", str(_MEDIAN_CI_MIN_N)),
        ("insufficient_n (n=5)", str(_MEDIAN_CI_MIN_N)),
        ("constant", "identical"),
        ("degenerate_zero_width", "order statistic"),
        ("empty", "no runs"),
    ]
    for label, needle in desc_checks:
        desc = results_by_label[label].describe()
        _check(
            results, f"{label}: describe() mentions {needle!r}",
            needle in desc, f"describe() = {desc!r}",
        )


# ---------------------------------------------------------------------------
# Group 3 — the n<6 threshold is justified, not just enforced
#
# The n=2 atom argument generalises.  The percentile endpoints cannot leave the
# extreme order statistics while P(bootstrap median == min) exceeds alpha/2
# (2.5%), and exact enumeration gives 25.000% / 25.926% / 5.078% / 5.792% at
# n = 2/3/4/5, dropping to 0.870% only at n = 6.  So the interval is degenerate
# for the whole band n <= 5, and 6 is the true threshold.
#
# The "always the range" arm spans n = 2..5 (full coverage of the degenerate
# band, and the arm that would fail if the threshold were lowered), and the
# non-vacuity control sits at n = 6 — the first n at which a narrower interval
# is possible at all.  Check COUNT is 3, for a suite total of 56.
# ---------------------------------------------------------------------------

def _check_threshold_justification(results: list) -> None:
    print("\nGroup 3 — n<6 threshold justified against the bootstrap mechanism itself:")

    n_boot = 10_000
    per_n = 100

    # n = 2..5: interval must ALWAYS equal [min, max], for every n in the band.
    gen_rng = np.random.default_rng(20260922)
    degenerate_band = (2, 3, 4, 5)
    deviations = 0
    checked_band = 0
    per_n_deviations = {}
    for n in degenerate_band:
        dev_this_n = 0
        for i in range(per_n):
            v = gen_rng.normal(loc=gen_rng.uniform(-5, 5),
                               scale=gen_rng.uniform(0.1, 10), size=n)
            idx_rng = np.random.default_rng(1_000_000 + 1000 * n + i)
            idx = idx_rng.integers(0, n, size=(n_boot, n))
            bm = np.median(v[idx], axis=1)
            lower = np.percentile(bm, 2.5)
            upper = np.percentile(bm, 97.5)
            if abs(lower - v.min()) > 1e-12 or abs(upper - v.max()) > 1e-12:
                dev_this_n += 1
            checked_band += 1
        per_n_deviations[n] = dev_this_n
        deviations += dev_this_n
    _check(
        results,
        f"n = 2,3,4,5: bootstrap interval == [min, max] for all {per_n} random "
        f"datasets at every n (the whole degenerate band, not just n=2)",
        deviations == 0,
        f"{deviations}/{checked_band} deviated; per-n: {per_n_deviations}",
    )

    # n=6 control: the first n at which a narrower interval is possible, so a
    # strict majority must be NARROWER than [min, max].  Without this arm the
    # check above would pass just as happily if the bootstrap were broken and
    # every interval collapsed to the range at every n.
    gen_rng = np.random.default_rng(20260923)
    n_datasets = 200
    narrower_n6 = 0
    for i in range(n_datasets):
        v = gen_rng.normal(loc=gen_rng.uniform(-5, 5), scale=gen_rng.uniform(0.1, 10), size=6)
        idx_rng = np.random.default_rng(2_000_000 + i)
        idx = idx_rng.integers(0, 6, size=(n_boot, 6))
        bm = np.median(v[idx], axis=1)
        lower = np.percentile(bm, 2.5)
        upper = np.percentile(bm, 97.5)
        if (upper - lower) < (float(v.max()) - float(v.min())) - 1e-12:
            narrower_n6 += 1
    _check(
        results, "n=6 control: strict majority (>100/200) narrower than [min, max]",
        narrower_n6 > n_datasets / 2,
        f"{narrower_n6}/200 narrower than range",
    )

    # bootstrap_median_ci itself must classify every n in 1..5 as
    # insufficient_n, regardless of seed, across several different datasets.
    small_datasets = [
        [1.0], [3.0], [-7.5], [0.0],
        [1.0, 2.0], [5.0, -5.0], [100.0, 100.5], [0.0, 1.0],
        [1.0, 2.0, 3.0], [0.5, -2.0, 7.25], [10.0, 11.0, 9.0],
        [1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 50.0],
        [1.0, 2.0, 3.0, 4.0, 5.0], [9.0, -3.5, 0.25, 4.0, 88.0],
    ]
    assert {len(v) for v in small_datasets} == {1, 2, 3, 4, 5}, (
        "this arm must cover every n in the degenerate band"
    )
    seeds = [1, 2, 3, 42, 999]
    all_insufficient = True
    checked = 0
    offenders = []
    for vals in small_datasets:
        for seed in seeds:
            r = bootstrap_median_ci(vals, seed)
            checked += 1
            if r.status != MEDIAN_CI_STATUS_INSUFFICIENT_N:
                all_insufficient = False
                offenders.append((len(vals), seed, r.status))
    _check(
        results,
        "n = 1..5 all classify as insufficient_n across seeds and datasets",
        all_insufficient,
        f"checked {checked} (dataset, seed) combinations; offenders={offenders[:5]}",
    )


# ---------------------------------------------------------------------------
# Group 4 — bootstrap on the MEDIAN, not the mean (differential, exact)
# ---------------------------------------------------------------------------

def _manual_boot_interval(arr: np.ndarray, seed: int, estimator, n_boot: int = 10_000,
                           ci_level: float = 0.95):
    n = len(arr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = estimator(arr[idx], axis=1)
    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def _check_median_not_mean(results: list) -> None:
    print("\nGroup 4 — bootstrap targets the MEDIAN, differentially verified against the MEAN:")

    skewed_datasets = [
        [1, 1, 1, 1, 1, 2, 3, 4, 50, 200],
        [0.1, 0.2, 0.1, 0.15, 0.12, 5.0, 500.0],
        [10, 11, 9, 10, 12, 11, 9, 1000, 2000, 3000, 10, 11],
    ]
    for i, ds in enumerate(skewed_datasets):
        arr = np.asarray(ds, dtype=float)
        seed = 7
        r = bootstrap_median_ci(ds, seed)
        got_lower, got_upper = r.lower, r.upper

        median_lower, median_upper = _manual_boot_interval(arr, seed, np.median)
        mean_lower, mean_upper = _manual_boot_interval(arr, seed, np.mean)

        matches_median = (got_lower == median_lower) and (got_upper == median_upper)
        matches_mean = (got_lower == mean_lower) and (got_upper == mean_upper)
        _check(
            results,
            f"skewed dataset #{i}: matches median-bootstrap exactly AND NOT mean-bootstrap",
            matches_median and not matches_mean,
            f"median=({median_lower},{median_upper}) mean=({mean_lower},{mean_upper}) "
            f"got=({got_lower},{got_upper})",
        )

    # MedianCI.median must equal np.median(values) exactly, over 50 random
    # datasets of varying n and distribution shape.
    rng = np.random.default_rng(31415)
    mismatches = 0
    for i in range(50):
        n = int(rng.integers(3, 41))
        shape = i % 5
        if shape == 0:
            vals = rng.normal(size=n)
        elif shape == 1:
            vals = rng.uniform(-10, 10, size=n)
        elif shape == 2:
            vals = rng.exponential(scale=3.0, size=n)
        elif shape == 3:
            vals = rng.lognormal(mean=0.0, sigma=1.5, size=n)
        else:
            vals = np.concatenate([
                np.zeros(n // 2), rng.uniform(0, 1, size=n - n // 2)
            ])
        r = bootstrap_median_ci(list(vals), int(rng.integers(0, 1_000_000)))
        if r.median != float(np.median(vals)):
            mismatches += 1
    _check(
        results, "MedianCI.median == np.median(values) exactly over 50 random datasets",
        mismatches == 0, f"{mismatches}/50 mismatched",
    )

    # The median and mean genuinely differ on a skewed dataset — otherwise
    # reusing ci_95 (a mean interval) would have been an accident, not a bug.
    skewed = [1, 1, 1, 1, 1, 2, 3, 4, 50, 200]
    med = float(np.median(skewed))
    mean_val = ci_95(skewed)[0]
    _check(
        results, "median != mean on the skewed dataset (non-vacuity for estimator separation)",
        med != mean_val, f"median={med}, mean={mean_val}",
    )

    # And the intervals genuinely differ too.
    seed = 7
    median_ci = bootstrap_median_ci(skewed, seed)
    mean_ci = ci_95(skewed)
    _check(
        results, "bootstrap_median_ci interval != ci_95 interval on the skewed dataset",
        (median_ci.lower, median_ci.upper) != (mean_ci[1], mean_ci[2]),
        f"median_ci=({median_ci.lower},{median_ci.upper}) ci_95=({mean_ci[1]},{mean_ci[2]})",
    )


# ---------------------------------------------------------------------------
# Group 5 — the CI brackets the median and lies inside the data range
# ---------------------------------------------------------------------------

def _check_bracketing(results: list) -> None:
    print("\nGroup 5 — CI brackets the median and lies within [min(vals), max(vals)]:")

    rng = np.random.default_rng(271828)
    checked = 0
    ok_count = 0
    bracket_failures = 0
    range_failures = 0

    for i in range(200):
        n = int(rng.integers(3, 31))
        shape = i % 6
        if shape == 0:
            vals = rng.normal(size=n)
        elif shape == 1:
            vals = rng.uniform(0, 1, size=n)
        elif shape == 2:
            vals = rng.lognormal(mean=0.0, sigma=1.0, size=n)
        elif shape == 3:
            # bounded [0,1] beta-like
            vals = rng.beta(0.5, 0.5, size=n)
        elif shape == 4:
            # piled at 0.0
            vals = np.where(rng.uniform(size=n) < 0.7, 0.0, rng.uniform(0, 1, size=n))
        else:
            # piled at 1.0
            vals = np.where(rng.uniform(size=n) < 0.7, 1.0, rng.uniform(0, 1, size=n))

        seed = int(rng.integers(0, 1_000_000))
        r = bootstrap_median_ci(list(vals), seed)

        if r.status in (MEDIAN_CI_STATUS_OK, MEDIAN_CI_STATUS_ZERO_WIDTH):
            checked += 1
            if r.status == MEDIAN_CI_STATUS_OK:
                ok_count += 1
            if not (r.lower <= r.median <= r.upper):
                bracket_failures += 1
            if not (float(vals.min()) <= r.lower and r.upper <= float(vals.max())):
                range_failures += 1

    _check(
        results, "lower <= median <= upper for every ok/degenerate_zero_width result",
        bracket_failures == 0, f"{bracket_failures} failures out of {checked} checked",
    )
    _check(
        results, "min(vals) <= lower and upper <= max(vals) for every ok/degenerate_zero_width result",
        range_failures == 0, f"{range_failures} failures out of {checked} checked",
    )
    _check(
        results, "coverage control: at least 20 of 200 cells came back 'ok'",
        ok_count >= 20, f"{ok_count}/200 came back 'ok'",
    )


# ---------------------------------------------------------------------------
# Group 6 — determinism
# ---------------------------------------------------------------------------

def _check_determinism(results: list) -> None:
    print("\nGroup 6 — determinism, in-process, cross-process, and seed non-vacuity:")

    # In-process: same (values, seed) twice gives an identical MedianCI.
    rng = np.random.default_rng(161803)
    identical = True
    for i in range(20):
        n = int(rng.integers(3, 25))
        vals = list(rng.normal(size=n))
        seed = int(rng.integers(0, 1_000_000))
        r1 = bootstrap_median_ci(vals, seed)
        r2 = bootstrap_median_ci(vals, seed)
        if (r1.median, r1.lower, r1.upper, r1.n, r1.status) != \
           (r2.median, r2.lower, r2.upper, r2.n, r2.status):
            identical = False
    _check(
        results, "same (values, seed) called twice in-process gives identical MedianCI (20 datasets)",
        identical, "",
    )

    # Cross-process reproducibility.
    fixed_cases = [
        (list(range(1, 11)), 1),
        ([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0], 42),
        ([0.1, 5.2, 3.3, 9.9, 2.2, 7.7, 1.1, 8.8], 999),
    ]
    expected_reprs = []
    for vals, seed in fixed_cases:
        expected_reprs.append(repr(bootstrap_median_ci(vals, seed)))

    probe_lines = [
        "import sys",
        f"sys.path.insert(0, {_HERE!r})",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "from benchmark_core import bootstrap_median_ci",
    ]
    for vals, seed in fixed_cases:
        probe_lines.append(f"print(repr(bootstrap_median_ci({vals!r}, {seed!r})))")
    probe = "\n".join(probe_lines)

    def _run_probe(env_extra: dict) -> list:
        env = dict(os.environ)
        env.update(env_extra)
        out = subprocess.run(
            [sys.executable, "-c", probe],
            env=env, capture_output=True, text=True, timeout=120, check=True,
        ).stdout.splitlines()
        return out

    try:
        out = _run_probe({})
        ok, detail = out == expected_reprs, f"got {out}"
    except Exception as exc:                                  # noqa: BLE001
        ok, detail = False, f"probe failed: {exc!r}"
    _check(results, "subprocess repr() output matches in-process repr() for 3 fixed cases", ok, detail)

    for hash_seed in ("1", "12345"):
        try:
            out = _run_probe({"PYTHONHASHSEED": hash_seed})
            ok, detail = out == expected_reprs, f"got {out}"
        except Exception as exc:                              # noqa: BLE001
            ok, detail = False, f"probe failed: {exc!r}"
        _check(
            results, f"subprocess repr() matches under PYTHONHASHSEED={hash_seed}", ok, detail,
        )

    # Non-vacuity control: the seed must actually be used. Because the
    # bootstrap is discrete, many datasets alias across seeds, so a bare
    # determinism check would also pass if the seed argument were ignored
    # entirely. Count how many of 60 random n=10 datasets differ between
    # seed 11 and seed 12.
    rng = np.random.default_rng(2718281828)
    differing = 0
    for i in range(60):
        vals = list(rng.normal(size=10))
        r11 = bootstrap_median_ci(vals, 11)
        r12 = bootstrap_median_ci(vals, 12)
        if (r11.lower, r11.upper) != (r12.lower, r12.upper):
            differing += 1
    _check(
        results,
        "non-vacuity: seed 11 vs seed 12 differ on >= 5 of 60 n=10 datasets",
        differing >= 5, f"{differing}/60 differed",
    )


# ---------------------------------------------------------------------------
# Group 7 — input hygiene
# ---------------------------------------------------------------------------

def _check_input_hygiene(results: list) -> None:
    print("\nGroup 7 — non-finite and non-numeric inputs are dropped, n reflects the cleaned count:")

    # Six finite values survive the cleaning, which is the minimum that still
    # reaches `ok` (see Group 3): _MEDIAN_CI_MIN_N is 6, and the property under
    # test is "junk is dropped AND the cleaned sample still produces a real
    # interval" — a smaller fixture would land on insufficient_n and silently
    # stop testing the second half of that.
    r = bootstrap_median_ci(
        [1.0, float("nan"), 2.0, float("inf"), None, "x", 3.0, 4.0,
         float("-inf"), 5.0, 6.0], 1)
    got = (r.median, r.lower, r.upper, r.n, r.status)
    expected = (3.5, 1.5, 5.5, 6, MEDIAN_CI_STATUS_OK)
    _check(
        results, "mixed nan/inf/None/str input: full tuple matches, n reflects cleaned count",
        got == expected, f"expected {expected}, got {got}",
    )

    r_allnan = bootstrap_median_ci([float("nan")] * 5, 1)
    _check(
        results, "all-NaN input -> status empty, n=0",
        r_allnan.status == MEDIAN_CI_STATUS_EMPTY and r_allnan.n == 0,
        f"got status={r_allnan.status!r}, n={r_allnan.n}",
    )

    r_neginf = bootstrap_median_ci([1.0, float("-inf"), 2.0, 3.0], 1)
    _check(
        results, "-inf is dropped too (n excludes it)",
        r_neginf.n == 3, f"got n={r_neginf.n}",
    )

    r_list = bootstrap_median_ci(list(range(1, 11)), 1)
    r_array = bootstrap_median_ci(np.array(list(range(1, 11)), dtype=float), 1)
    got_list = (r_list.median, r_list.lower, r_list.upper, r_list.n, r_list.status)
    got_array = (r_array.median, r_array.lower, r_array.upper, r_array.n, r_array.status)
    _check(
        results, "numpy array input matches list input",
        got_list == got_array, f"list={got_list}, array={got_array}",
    )


# ---------------------------------------------------------------------------
# Group 8 — parameters
# ---------------------------------------------------------------------------

def _check_parameters(results: list) -> None:
    print("\nGroup 8 — ci_level and n_boot parameters:")

    rng = np.random.default_rng(555)
    all_no_wider = True
    widths_detail = []
    for i in range(10):
        n = int(rng.integers(5, 25))
        vals = list(rng.normal(size=n))
        seed = int(rng.integers(0, 1_000_000))
        r95 = bootstrap_median_ci(vals, seed)
        r50 = bootstrap_median_ci(vals, seed, ci_level=0.50)
        if r95.status == MEDIAN_CI_STATUS_OK and r50.status == MEDIAN_CI_STATUS_OK:
            w95 = r95.upper - r95.lower
            w50 = r50.upper - r50.lower
            if w50 > w95 + 1e-12:
                all_no_wider = False
                widths_detail.append((w50, w95))
    _check(
        results, "ci_level=0.50 is no wider than default 0.95 on the same data/seed (10 datasets)",
        all_no_wider, f"violations={widths_detail}" if widths_detail else "",
    )

    rng = np.random.default_rng(777)
    n_boot_small_ok = True
    for i in range(10):
        n = int(rng.integers(5, 25))
        vals = list(rng.normal(size=n))
        seed = int(rng.integers(0, 1_000_000))
        ra = bootstrap_median_ci(vals, seed, n_boot=500)
        rb = bootstrap_median_ci(vals, seed, n_boot=500)
        same = (ra.median, ra.lower, ra.upper, ra.n, ra.status) == \
               (rb.median, rb.lower, rb.upper, rb.n, rb.status)
        brackets = True
        if ra.status in (MEDIAN_CI_STATUS_OK, MEDIAN_CI_STATUS_ZERO_WIDTH):
            brackets = ra.lower <= ra.median <= ra.upper
        if not (same and brackets):
            n_boot_small_ok = False
    _check(
        results, "n_boot=500 is deterministic for a fixed seed and still brackets the median (10 datasets)",
        n_boot_small_ok, "",
    )


# ---------------------------------------------------------------------------
# Group 9 — production seeding wiring
# ---------------------------------------------------------------------------

def _check_production_wiring(results: list) -> None:
    print("\nGroup 9 — derive_seed behaviour and the make_median_ci_plots wiring (source-level):")

    base = 4242
    seeds_by_gov = {
        gov: derive_seed(base, "plot.bootstrap", gov, 50)
        for gov in ("ads", "democracy", "autocracy")
    }
    _check(
        results, "derive_seed produces distinct seeds across governments at the same difficulty",
        len(set(seeds_by_gov.values())) == len(seeds_by_gov), f"{seeds_by_gov}",
    )

    seeds_by_diff = {
        diff: derive_seed(base, "plot.bootstrap", "ads", diff)
        for diff in (10, 50, 95)
    }
    _check(
        results, "derive_seed produces distinct seeds across difficulties for the same government",
        len(set(seeds_by_diff.values())) == len(seeds_by_diff), f"{seeds_by_diff}",
    )

    s1 = derive_seed(base, "plot.bootstrap", "ads", 50)
    s2 = derive_seed(base, "plot.bootstrap", "ads", 50)
    _check(
        results, "derive_seed is stable across repeated calls",
        s1 == s2, f"{s1} vs {s2}",
    )

    # Source-level (AST) wiring checks.
    core_path = os.path.join(_HERE, "benchmark_core.py")
    with open(core_path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=core_path)

    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "make_median_ci_plots":
            func_node = node
            break

    if func_node is None:
        _check(results, "make_median_ci_plots is present in benchmark_core.py (AST)", False,
               "FunctionDef not found")
        return
    _check(results, "make_median_ci_plots is present in benchmark_core.py (AST)", True)

    derive_seed_calls = []
    ci_95_calls = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            fname = None
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            if fname == "derive_seed":
                derive_seed_calls.append(node)
            elif fname == "ci_95":
                ci_95_calls.append(node)

    found_key = False
    for call in derive_seed_calls:
        if len(call.args) >= 2:
            second = call.args[1]
            if isinstance(second, ast.Constant) and second.value == "plot.bootstrap":
                found_key = True
    _check(
        results,
        "make_median_ci_plots calls derive_seed(..., 'plot.bootstrap', ...) (AST)",
        found_key, f"{len(derive_seed_calls)} derive_seed call(s) found in the function body",
    )

    _check(
        results, "make_median_ci_plots never calls ci_95 (AST) — estimators stay separate",
        len(ci_95_calls) == 0,
        f"{len(ci_95_calls)} ci_95 call(s) found" if ci_95_calls else "",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    results: list = []
    print("Median bootstrap confidence interval guards:")
    _check_constants(results)
    _check_degenerate_statuses(results)
    _check_threshold_justification(results)
    _check_median_not_mean(results)
    _check_bracketing(results)
    _check_determinism(results)
    _check_input_hygiene(results)
    _check_parameters(results)
    _check_production_wiring(results)

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
