#!/usr/bin/env python3
"""
Regression guard: the figure layer (Tukey IQR family, median/CI family,
--plots-only, and one caption-correctness defect class).

    python3 test_plot_regression.py          # exit 0 = pass, 1 = fail

WHAT THIS PROTECTS
-------------------
Five independent things, each with its own failure mode that a normal test
run would not catch:

  1. **The Tukey IQR figure path (`make_plots` and everything it calls) is
     content-unchanged.**  This was verified once by hand, two ways: an AST
     identity diff over 9 functions + 9 constants against a reference
     backup of ``benchmark_core.py``, and byte-identical PNGs.  That was a
     one-off audit.  GROUP 1 turns it into a standing guard with golden
     sha256 hashes over docstring-stripped ASTs, so a future refactor of this
     file cannot silently change what these figures draw.

  2. **``--plots-only`` actually regenerates every figure from an archived
     tree, and does nothing else.**  GROUP 2 builds a synthetic archive and
     runs the real replot path end to end: all 8 PNGs + 8 sidecars written,
     the IQR/median-CI caption discipline holds, ``cell_stats.json`` and
     ``analysis/`` are left untouched, and the bootstrap is deterministic
     across two independent replots of the same archive.

  3. **``--plots-only`` refuses a malformed or ambiguous archive instead of
     guessing.**  GROUP 3 exercises the ``ConfigError`` contract in
     ``config_from_manifest`` / ``load_combined_rows`` /
     ``replot_from_output_root``, including the two silent-fallback traps a
     naive implementation could not resist: using the invoking process's
     scale instead of the archive's, and writing into the manifest's
     (foreign-host) output path instead of the directory actually being
     replotted.

  4. **``_half_sample_size_clause`` is genuinely data-derived, not hardcoded
     in disguise.**  A caption that asserts a fixed direction ("the second
     half is systematically thinner") without inspecting the data is wrong
     in the large majority of runs, because the direction and magnitude
     both vary run to run — the clause must be computed from the actual
     counts every time.  This class of defect is easy to reintroduce only
     partially even while guarding against one instance of it, so GROUP 4
     checks both directions, the percentage arithmetic, the near-equal
     branch, the branch boundary, and degenerate zero counts — over a
     300-pair property test, not a couple of examples.

  5. **``_PAPER_K_RUNS`` (figure layer) and ``K_RUNS`` (the entry point that
     actually runs the sweep) do not silently drift apart.**  These are two
     independent hardcoded copies of the same number, and there is no
     ``--k-runs`` CLI override, so the only way to change k is a source edit
     to ``run_full_simulation.py`` — an edit ``_PAPER_K_RUNS`` will not
     follow.  There is no import-time guard connecting the two constants, so
     GROUP 5 is the only thing standing between a k-runs change and a figure
     layer that keeps captioning and projecting to the old k.

  6. **The median/CI figure actually draws the median at the marker and the
     bootstrap interval at the bar — not some other statistic.**  Every other
     check here verifies the numbers upstream of drawing; nothing verified
     that the draw call receives them.  A figure whose marker silently became
     the mean, or whose bar silently became the IQR half-width, would be
     numerically correct at every layer this file otherwise tests and wrong
     on the page.  GROUP 6 monkeypatches ``bootstrap_median_ci`` and
     ``Axes.errorbar`` to record what was computed and what was actually
     handed to matplotlib, and checks them against each other — capturing
     draw calls rather than pixels, since a pixel-golden file is brittle
     across matplotlib patch releases.

WHY THIS FILE LIVES HERE, AND NOT IN tests/
--------------------------------------------
This file follows the precedent set by ``test_seed_derivation.py`` and
``test_default_governments.py``: it sits beside the code it guards, alongside
the project's other top-level regression tests, imports the way the entry
points do, and runs under a plain ``python3`` with no pytest and no import
gymnastics.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import os
import random
import re
import shutil
import struct
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.axes as maxes                                 # noqa: E402
import matplotlib.figure as mfigure                              # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import benchmark_core as bc                                     # noqa: E402
from engine.scenario_plan import derive_seed                    # noqa: E402
from benchmark_core import (                                    # noqa: E402
    BenchmarkConfig,
    ConfigError,
    K_RUNS_SOURCE_CLI,
    K_RUNS_SOURCE_DEFAULT,
    MEDIAN_CI_STATUS_CONSTANT,
    MEDIAN_CI_STATUS_EMPTY,
    MEDIAN_CI_STATUS_INSUFFICIENT_N,
    MEDIAN_CI_STATUS_OK,
    MEDIAN_CI_STATUS_ZERO_WIDTH,
    PLOT_STATS,
    SUMMARY_FIELDS,
    _ADS_GOV_KEY,
    _PAPER_K_RUNS,
    _half_sample_size_clause,
    add_common_arguments,
    config_from_manifest,
    load_combined_rows,
    make_ads_calibration_plots,
    make_scope_matched_calibration_plots,
    parse_k_runs,
    replot_from_output_root,
)

_BENCHMARK_CORE_PATH = os.path.join(_HERE, "benchmark_core.py")
_RUN_FULL_SIM_PATH = os.path.join(_HERE, "run_full_simulation.py")


def _check(results: list, name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# GROUP 1 — IQR non-regression guard
# ---------------------------------------------------------------------------

_IQR_FUNCS = ["_draw_box_series", "_style_difficulty_axes", "make_plots",
              "generate_alt_text", "_clauses_to_alt_text", "load_cell_stats",
              "ci_95", "t_critical_95", "compute_cell_statistics"]
_IQR_CONSTS = ["PLOT_STATS", "SUMMARY_FIELDS", "GOV_COLORS", "GOV_LINESTYLES",
               "GOV_DISPLAY", "GOV_MARKERS", "_HALF_SPLIT_COLORS",
               "_HALF_SPLIT_LINESTYLES", "ADS_CALIBRATION_FIELDS"]

_GOLDEN_IQR_AST = {
    'const ADS_CALIBRATION_FIELDS': "9e9e26a9aad7fc54f18e71b8e6c8078270a1acdf67f52349bf4795446d4216da",
    'const GOV_COLORS': "1eae74c61c466446831c56953849d84eece5de108fa00c7eafc3b976a2bdbb91",
    'const GOV_DISPLAY': "d12641d2e84bbcb57c7208f2b6357a23232ba663c15053017e81a6605a7f6690",
    'const GOV_LINESTYLES': "0b86ba26f9b0efc97035bd7fac7c04a2a79d1706b63d101439350e6fb35f8d5b",
    'const GOV_MARKERS': "76c7c882a48b4e59d9a0ab57b9d89557645d05f7d484fe9329cefdf8104afcd3",
    # Title/axis-label text is bare parameters only — no explanations on figures.
    'const PLOT_STATS': "7ca4d5c2ccfe708e35d70f7a940dc574610b8771592349c81213351aef28f729",
    'const SUMMARY_FIELDS': "8b8678f82d6322fc70b9227428fb99ab396a55d02e3ada7fb9f03c85239b83ed",
    'const _HALF_SPLIT_COLORS': "16b1a27218a0fb119aeb250cfc712067010c3034f62e18e18f97c0181a42339c",
    'const _HALF_SPLIT_LINESTYLES': "95b4b40a60bc71fa6b31fa098d2b215ac981df69e40dc2206cb039e242f4cf53",
    'def _clauses_to_alt_text': "fae759694579ec4eea4927cf85289489e0102266a49de9b6bdf4dcf099bbe227",
    'def _draw_box_series': "cc8182683119a9aa8d6fbecc8de4b5db18cb1f16eac6d2e79b7aea3e72b142bb",
    'def _style_difficulty_axes': "89efd6c7de7dddc8b52034e84df28922aeca48a6e7244002f2f75a20a300d4de",
    'def ci_95': "ef9efa599f61c3da4122afdcb2e95b1b81801d335b3f087dc1836bda1d421616",
    'def compute_cell_statistics': "d6cf5f2b428072de2f8a1a351ec28649c399a9a6ee3b805eb27a81095cd8083b",
    'def generate_alt_text': "f76c11faaf2dc10ba37962fad3a4563413c3704ae85abc9aa7022a8e302825ef",
    'def load_cell_stats': "6371ba23930b151073e6a3b02cd7a7d7409e096fd53000a2c5e3819544e755b4",
    # make_plots emits a "Scale: grid GxG, N agents, C cycles" sidecar line,
    # since the figure title itself carries only the plotted quantity.
    'def make_plots': "0eccf5d38c01a9a6ad065fdc0e4cd378f16945c378ec5140a503fee104e3d461",
    'def t_critical_95': "622b018fdc8a532bc9a97f6b2e180253708c32552dd958446a0fc689b418b709",
}


def _normalised_top_level(path: str) -> dict:
    """Map ``"def " + name`` / ``"const " + name`` -> unparsed source.

    Function bodies have their leading docstring stripped before unparsing —
    docstrings are prose, not behaviour, and a comment/doc fix should not trip
    a guard whose entire point is to catch behavioural drift.
    """
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    out: dict = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _IQR_FUNCS:
            body = list(node.body)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            if not body:
                body = [ast.Pass()]
            new_node = ast.FunctionDef(
                name=node.name,
                args=node.args,
                body=body,
                decorator_list=node.decorator_list,
                returns=node.returns,
                type_comment=None,
                type_params=[],
            )
            ast.fix_missing_locations(new_node)
            out["def " + node.name] = ast.unparse(new_node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in _IQR_CONSTS:
                    out["const " + t.id] = ast.unparse(node)
    return out


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _check_iqr_regression_guard(results: list, skipped: list) -> None:
    print("\nIQR non-regression guard (golden AST hashes):")

    current = _normalised_top_level(_BENCHMARK_CORE_PATH)
    all_keys = set(_IQR_FUNCS) | set(_IQR_CONSTS)
    expected_keys = {("def " + n) for n in _IQR_FUNCS} | {("const " + n) for n in _IQR_CONSTS}
    assert expected_keys == set(_GOLDEN_IQR_AST.keys()), "golden table keys drifted from _IQR_FUNCS/_IQR_CONSTS"

    missing = sorted(k for k in _GOLDEN_IQR_AST if k not in current)
    _check(
        results, "all 18 guarded keys are present in benchmark_core.py",
        not missing,
        f"missing: {missing}" if missing else "",
    )

    mismatched = sorted(
        k for k in _GOLDEN_IQR_AST
        if k in current and _sha256(current[k]) != _GOLDEN_IQR_AST[k]
    )
    _check(
        results, "every present key's hash matches the golden hash",
        not mismatched,
        f"mismatched: {mismatched}" if mismatched else "",
    )

    _check(
        results, "exactly 18 golden keys, no extras",
        len(_GOLDEN_IQR_AST) == 18 and set(_GOLDEN_IQR_AST.keys()) == expected_keys,
        f"{len(_GOLDEN_IQR_AST)} keys, extras={set(_GOLDEN_IQR_AST) - expected_keys}, "
        f"missing_from_table={expected_keys - set(_GOLDEN_IQR_AST)}",
    )

    # CONTROL — the guard must be able to detect a change at all.
    mutated_source = current["def _draw_box_series"] + "\npass"
    mutated_hash = _sha256(mutated_source)
    _check(
        results, "control: a mutated _draw_box_series hashes differently",
        mutated_hash != _GOLDEN_IQR_AST["def _draw_box_series"],
        "mutation produced the SAME hash — the guard cannot detect changes",
    )

    # The golden AST hashes above are the guard: an 18-key, exact-match pin
    # against the shipped source. There is no secondary cross-check against
    # an external reference copy — the golden table itself is the reference.


# ---------------------------------------------------------------------------
# GROUP 2 — --plots-only success path
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG"
_CI_WORD_RE = re.compile(r"\bCI\b")


def _make_archive(root, *, governments, difficulties, k_runs,
                   manifest_overrides=None, write_csv=True, csv_rows=None):
    """Build a synthetic archive: manifest.json + combined_final_stats.csv."""
    config = {
        "governments": list(governments),
        "difficulties": list(difficulties),
        "k_runs": k_runs,
        "grid_size": 20,
        "n_agents": 50,
        "max_cycles": 30,
        "max_steps": 5,
        "base_seed": 12345,
    }
    if manifest_overrides:
        for key, value in manifest_overrides.items():
            if value is None:
                config.pop(key, None)
            else:
                config[key] = value

    manifest = {"config": config}
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    if write_csv:
        header = ["government", "difficulty", "run"] + list(SUMMARY_FIELDS)
        rng = random.Random(0)
        rows = csv_rows
        if rows is None:
            rows = []
            for gov in governments:
                for diff in difficulties:
                    for run in range(k_runs):
                        row = [gov, diff, run]
                        row += [round(rng.uniform(0.0, 1.0), 6) for _ in SUMMARY_FIELDS]
                        rows.append(row)
        with open(os.path.join(root, "combined_final_stats.csv"), "w", encoding="utf-8", newline="") as f:
            import csv as _csv
            writer = _csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)


def _check_plots_only_success(results: list) -> None:
    print("\n--plots-only success path (synthetic archive, end to end):")
    governments = ["ads", "democracy"]
    difficulties = [10, 50]
    k_runs = 3

    root = tempfile.mkdtemp(prefix="plot_regression_success_")
    try:
        _make_archive(root, governments=governments, difficulties=difficulties, k_runs=k_runs)

        rc = replot_from_output_root(root)
        _check(results, "replot_from_output_root returns 0", rc == 0, f"got {rc}")

        plots_dir = os.path.join(root, "plots")
        stat_names = [stat for stat, _, _ in PLOT_STATS]

        iqr_pngs = [os.path.join(plots_dir, f"{stat}.png") for stat in stat_names]
        iqr_sidecars = [os.path.join(plots_dir, f"{stat}_alt_text.txt") for stat in stat_names]
        mci_pngs = [os.path.join(plots_dir, f"{stat}_median_ci.png") for stat in stat_names]
        mci_sidecars = [os.path.join(plots_dir, f"{stat}_median_ci_alt_text.txt") for stat in stat_names]

        missing_iqr_png = [p for p in iqr_pngs if not os.path.isfile(p)]
        _check(results, "all 4 IQR PNGs written", not missing_iqr_png, f"missing: {missing_iqr_png}")
        missing_iqr_sidecar = [p for p in iqr_sidecars if not os.path.isfile(p)]
        _check(results, "all 4 IQR alt-text sidecars written", not missing_iqr_sidecar,
               f"missing: {missing_iqr_sidecar}")

        missing_mci_png = [p for p in mci_pngs if not os.path.isfile(p)]
        _check(results, "all 4 median_ci PNGs written", not missing_mci_png, f"missing: {missing_mci_png}")
        missing_mci_sidecar = [p for p in mci_sidecars if not os.path.isfile(p)]
        _check(results, "all 4 median_ci alt-text sidecars written", not missing_mci_sidecar,
               f"missing: {missing_mci_sidecar}")

        all_pngs = iqr_pngs + mci_pngs
        bad_pngs = []
        for p in all_pngs:
            if not os.path.isfile(p):
                continue
            data = open(p, "rb").read()
            if not data or not data.startswith(_PNG_MAGIC):
                bad_pngs.append(p)
        _check(results, "every written PNG is non-empty and starts with the PNG magic bytes",
               not bad_pngs, f"bad: {bad_pngs}")

        all_sidecars = iqr_sidecars + mci_sidecars
        bad_sidecars = []
        for p in all_sidecars:
            if not os.path.isfile(p):
                continue
            text = open(p, "r", encoding="utf-8").read()
            if not text.strip():
                bad_sidecars.append(p)
        _check(results, "every sidecar is non-empty text", not bad_sidecars, f"bad: {bad_sidecars}")

        # Caption discipline: IQR sidecars must never say CI/confidence interval.
        iqr_violations = []
        for p in iqr_sidecars:
            if not os.path.isfile(p):
                continue
            text = open(p, "r", encoding="utf-8").read()
            if "confidence interval" in text.lower() or _CI_WORD_RE.search(text):
                iqr_violations.append(p)
        _check(results, "IQR sidecars never say 'confidence interval' or a standalone 'CI'",
               not iqr_violations, f"violating: {iqr_violations}")

        mci_missing_ci = []
        for p in mci_sidecars:
            if not os.path.isfile(p):
                continue
            text = open(p, "r", encoding="utf-8").read()
            if "confidence interval" not in text.lower():
                mci_missing_ci.append(p)
        _check(results, "median_ci sidecars DO mention a confidence interval",
               not mci_missing_ci, f"missing mention: {mci_missing_ci}")

        n_missing = []
        for p in mci_sidecars:
            if not os.path.isfile(p):
                continue
            text = open(p, "r", encoding="utf-8").read()
            if "n = " not in text:
                n_missing.append(p)
        _check(results, "median_ci sidecars state the per-cell n ('n = ')",
               not n_missing, f"missing 'n = ': {n_missing}")

        cell_stats_path = os.path.join(root, "cell_stats.json")
        analysis_dir = os.path.join(root, "analysis")
        _check(
            results, "--plots-only does not create cell_stats.json or analysis/",
            not os.path.exists(cell_stats_path) and not os.path.exists(analysis_dir),
            f"cell_stats.json exists={os.path.exists(cell_stats_path)}, "
            f"analysis/ exists={os.path.exists(analysis_dir)}",
        )

        # Determinism: replot twice, PNG bytes must be byte-identical.
        before = {p: open(p, "rb").read() for p in all_pngs if os.path.isfile(p)}
        rc2 = replot_from_output_root(root)
        after = {p: open(p, "rb").read() for p in all_pngs if os.path.isfile(p)}
        diffs = []
        for p in before:
            b, a = before[p], after.get(p, b"")
            if b != a:
                diffs.append((p, abs(len(b) - len(a))))
        _check(
            results, "two replots of the same archive produce byte-identical PNGs",
            rc2 == 0 and not diffs,
            f"rc2={rc2} byte-diffs={diffs}" if (rc2 != 0 or diffs) else "",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# GROUP 3 — --plots-only refusal path
# ---------------------------------------------------------------------------

def _expect_config_error(fn):
    """Run fn(); return (raised_configerror, message)."""
    try:
        fn()
        return False, "did not raise"
    except ConfigError as exc:
        return True, str(exc)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"wrong exception type: {type(exc).__name__}: {exc}"


def _check_plots_only_refusal(results: list) -> None:
    print("\n--plots-only refusal path (must refuse, not guess):")

    # 1. output_root does not exist.
    nonexistent = os.path.join(tempfile.gettempdir(), "plot_regression_does_not_exist_xyz")
    if os.path.exists(nonexistent):
        shutil.rmtree(nonexistent, ignore_errors=True)
    raised, msg = _expect_config_error(lambda: replot_from_output_root(nonexistent))
    _check(results, "output_root that does not exist raises ConfigError", raised, msg)

    # 2. output_root is a file, not a directory.
    tmp_file_fd, tmp_file_path = tempfile.mkstemp(prefix="plot_regression_file_")
    os.close(tmp_file_fd)
    try:
        raised, msg = _expect_config_error(lambda: replot_from_output_root(tmp_file_path))
        _check(results, "output_root that is a file raises ConfigError", raised, msg)
    finally:
        os.remove(tmp_file_path)

    # 3. valid CSV, no manifest.json.
    root = tempfile.mkdtemp(prefix="plot_regression_no_manifest_")
    try:
        _make_archive(root, governments=["ads"], difficulties=[10], k_runs=2)
        os.remove(os.path.join(root, "manifest.json"))
        raised, msg = _expect_config_error(lambda: config_from_manifest(root))
        _check(results, "no manifest.json raises ConfigError mentioning the manifest",
               raised and "manifest" in msg.lower(), msg)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 4. invalid JSON in manifest.json.
    root = tempfile.mkdtemp(prefix="plot_regression_bad_json_")
    try:
        with open(os.path.join(root, "manifest.json"), "w") as f:
            f.write("{not valid json")
        raised, msg = _expect_config_error(lambda: config_from_manifest(root))
        _check(results, "invalid JSON in manifest.json raises ConfigError", raised, msg)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 5. valid JSON, no "config" object.
    root = tempfile.mkdtemp(prefix="plot_regression_no_config_")
    try:
        with open(os.path.join(root, "manifest.json"), "w") as f:
            json.dump({"not_config": True}, f)
        raised, msg = _expect_config_error(lambda: config_from_manifest(root))
        _check(results, "manifest with no 'config' object raises ConfigError", raised, msg)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 6. each of the 7 required fields missing, one at a time.
    required_fields = ("governments", "difficulties", "k_runs", "grid_size",
                        "n_agents", "max_cycles", "max_steps")
    for field in required_fields:
        root = tempfile.mkdtemp(prefix=f"plot_regression_missing_{field}_")
        try:
            _make_archive(root, governments=["ads"], difficulties=[10], k_runs=2,
                          manifest_overrides={field: None})
            raised, msg = _expect_config_error(lambda: config_from_manifest(root))
            _check(results, f"manifest missing '{field}' raises ConfigError naming it",
                   raised and field in msg, msg)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    # 7. k_runs is a non-integer string.
    root = tempfile.mkdtemp(prefix="plot_regression_bad_k_runs_")
    try:
        _make_archive(root, governments=["ads"], difficulties=[10], k_runs=2,
                      manifest_overrides={"k_runs": "ten"})
        raised, msg = _expect_config_error(lambda: config_from_manifest(root))
        _check(results, "k_runs='ten' raises ConfigError (not ValueError)", raised, msg)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 8. valid manifest, no combined_final_stats.csv.
    root = tempfile.mkdtemp(prefix="plot_regression_no_csv_")
    try:
        _make_archive(root, governments=["ads"], difficulties=[10], k_runs=2, write_csv=False)
        raised, msg = _expect_config_error(lambda: replot_from_output_root(root))
        _check(results, "no combined_final_stats.csv raises ConfigError mentioning the CSV",
               raised and "csv" in msg.lower(), msg)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 9. header but zero data rows.
    root = tempfile.mkdtemp(prefix="plot_regression_zero_rows_")
    try:
        _make_archive(root, governments=["ads"], difficulties=[10], k_runs=2, csv_rows=[])
        raised, msg = _expect_config_error(lambda: replot_from_output_root(root))
        _check(results, "zero data rows raises ConfigError mentioning 'no data rows'",
               raised and ("no data row" in msg.lower() or "nothing to plot" in msg.lower()), msg)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 10. archive scale wins over process defaults (no silent fallback).
    root = tempfile.mkdtemp(prefix="plot_regression_archive_scale_")
    try:
        _make_archive(
            root, governments=["ads"], difficulties=[10], k_runs=7,
            manifest_overrides={"k_runs": 7, "grid_size": 33, "n_agents": 77, "max_cycles": 11},
        )
        cfg = config_from_manifest(root)
        _check(
            results, "config_from_manifest uses the ARCHIVE's scale, not process defaults",
            cfg.k_runs == 7 and cfg.grid_size == 33 and cfg.n_agents == 77 and cfg.max_cycles == 11,
            f"got k_runs={cfg.k_runs} grid_size={cfg.grid_size} "
            f"n_agents={cfg.n_agents} max_cycles={cfg.max_cycles}",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 11. control for #10 — the process default (K_RUNS) must genuinely differ from 7.
    k_runs_const = _resolve_run_full_sim_k_runs()
    _check(
        results, "control: run_full_simulation.py's K_RUNS != 7 (check #10 is discriminating)",
        k_runs_const is not None and k_runs_const != 7,
        f"K_RUNS={k_runs_const}",
    )

    # 12. output_root in the config is the caller's path, not the manifest's.
    root = tempfile.mkdtemp(prefix="plot_regression_foreign_path_")
    try:
        _make_archive(root, governments=["ads"], difficulties=[10], k_runs=2,
                      manifest_overrides={"output_root": "/nonexistent/host/path"})
        cfg = config_from_manifest(root)
        _check(
            results, "config_from_manifest.output_root is the abspath of the archive, not the manifest value",
            cfg.output_root == os.path.abspath(root),
            f"got {cfg.output_root!r}, expected {os.path.abspath(root)!r}",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 13. row-count mismatch is a warning, not a failure.
    root = tempfile.mkdtemp(prefix="plot_regression_row_mismatch_")
    try:
        rows = [["ads", 10, r] + [0.5] * len(SUMMARY_FIELDS) for r in range(3)]
        _make_archive(root, governments=["ads"], difficulties=[10], k_runs=10, csv_rows=rows)
        rc = replot_from_output_root(root)
        _check(results, "k_runs=10 manifest with only 3 CSV rows per cell still returns 0",
               rc == 0, f"got {rc}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 14. --plots-only exists on the CLI surface.
    tree = ast.parse(open(_BENCHMARK_CORE_PATH, "r", encoding="utf-8").read())
    found_flag = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == "--plots-only":
                found_flag = True
                break
    _check(results, "a parser.add_argument('--plots-only', ...) call exists", found_flag)

    # 15. the short-circuit exists inside main_from_config.
    main_from_config_src = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main_from_config":
            main_from_config_src = ast.unparse(node)
            break
    _check(
        results, "main_from_config short-circuits on plots_only via replot_from_output_root",
        main_from_config_src is not None
        and "plots_only" in main_from_config_src
        and "replot_from_output_root" in main_from_config_src,
        "main_from_config not found" if main_from_config_src is None else "markers missing",
    )


def _read_module_level_int_constant(path: str, name: str):
    """Read a module-level ``NAME = <int literal>`` assignment via AST."""
    tree = ast.parse(open(path, "r", encoding="utf-8").read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name and isinstance(node.value, ast.Constant):
                    return node.value.value
    return None


def _read_module_level_alias(path: str, name: str):
    """Return the identifier X in a module-level ``NAME = X``, else None."""
    tree = ast.parse(open(path, "r", encoding="utf-8").read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name and isinstance(node.value, ast.Name):
                    return node.value.id
    return None


def _resolve_run_full_sim_k_runs():
    """``run_full_simulation.K_RUNS`` without importing the entry point.

    That module does not define its own literal — it binds
    ``K_RUNS = PAPER_K_RUNS``, importing the single definition from
    ``benchmark_core``.  Resolve the alias here so the checks that depend on the
    value keep working, while still reading the entry point by AST rather than
    importing it (importing would run its import-time design guards and pull in
    the whole engine, which this suite deliberately avoids).
    """
    literal = _read_module_level_int_constant(_RUN_FULL_SIM_PATH, "K_RUNS")
    if literal is not None:
        return literal
    alias = _read_module_level_alias(_RUN_FULL_SIM_PATH, "K_RUNS")
    if alias == "PAPER_K_RUNS":
        return _PAPER_K_RUNS
    return None


# ---------------------------------------------------------------------------
# GROUP 4 — _half_sample_size_clause (data-derived caption clause)
# ---------------------------------------------------------------------------

_THICKER_RE = re.compile(r"The (first|second) half is the THICKER")
_PCT_RE = re.compile(r"(\d+)% LARGER")


def _check_half_sample_size_clause(results: list) -> None:
    print("\n_half_sample_size_clause — direction and arithmetic:")

    # 1. Real production counts.
    real = _half_sample_size_clause(9509, 15745)
    _check(
        results, "production counts (9509, 15745) name the SECOND half as thicker",
        "second" in _THICKER_RE.search(real).group(1).lower() if _THICKER_RE.search(real) else False,
        real,
    )
    _check(results, "production counts contain '66% LARGER'", "66% LARGER" in real, real)
    _check(results, "production counts do NOT contain 'thinner'", "thinner" not in real.lower(), real)

    # 2. Mirrored counts.
    mirrored = _half_sample_size_clause(15745, 9509)
    m = _THICKER_RE.search(mirrored)
    _check(
        results, "mirrored counts (15745, 9509) name the FIRST half as thicker",
        bool(m) and m.group(1).lower() == "first",
        mirrored,
    )
    _check(results, "mirrored counts also contain '66% LARGER'", "66% LARGER" in mirrored, mirrored)

    # 3+4. Property test over 300 random pairs.
    rng = random.Random(20260923)
    pairs = [(rng.randint(1, 50000), rng.randint(1, 50000)) for _ in range(300)]
    branch_5pct = 0
    first_named = 0
    second_named = 0
    direction_violations = []
    pct_violations = []
    for n_first, n_second in pairs:
        text = _half_sample_size_clause(n_first, n_second)
        bigger, smaller = max(n_first, n_second), min(n_first, n_second)
        pct = 100.0 * (bigger - smaller) / smaller
        if pct >= 5.0:
            branch_5pct += 1
            m = _THICKER_RE.search(text)
            if not m:
                direction_violations.append((n_first, n_second, "no THICKER wording found"))
                continue
            named = m.group(1).lower()
            if named == "first":
                first_named += 1
            else:
                second_named += 1
            actually_thicker = "second" if n_second > n_first else "first"
            if named != actually_thicker:
                direction_violations.append((n_first, n_second, f"named {named}, actually {actually_thicker}"))
            pm = _PCT_RE.search(text)
            expected_pct = round(pct)
            if not pm or int(pm.group(1)) != expected_pct:
                pct_violations.append((n_first, n_second, f"got {pm.group(1) if pm else None}, expected {expected_pct}"))

    print(f"    -> {branch_5pct}/300 pairs took the >=5% branch "
          f"(first named {first_named}x, second named {second_named}x)")
    _check(
        results, "property test: the THICKER half named is always the larger one (>=5% branch)",
        not direction_violations, f"{len(direction_violations)} violation(s): {direction_violations[:5]}",
    )
    _check(
        results, "directional coverage: both 'first' and 'second' named >= 20 times each",
        first_named >= 20 and second_named >= 20,
        f"first={first_named} second={second_named}",
    )
    _check(
        results, "property test: quoted percentage == round(100*(max-min)/min) over 300 pairs",
        not pct_violations, f"{len(pct_violations)} violation(s): {pct_violations[:5]}",
    )

    # 5. Near-equal branch.
    near_equal = _half_sample_size_clause(1000, 1020)
    _check(
        results, "(1000, 1020) [2% apart] uses 'comparable samples' wording and claims no thicker half",
        "comparable samples" in near_equal and not _THICKER_RE.search(near_equal),
        near_equal,
    )

    # 6. Boundary: 4.9% -> comparable, 5.1% -> thicker.
    below = _half_sample_size_clause(1000, 1049)
    above = _half_sample_size_clause(1000, 1051)
    _check(
        results, "(1000, 1049) [4.9%] is comparable; (1000, 1051) [5.1%] names a thicker half",
        "comparable samples" in below and bool(_THICKER_RE.search(above)),
        f"below={below!r} above={above!r}",
    )

    # 7. Zero/negative counts.
    for a, b in ((0, 100), (100, 0), (0, 0)):
        text = None
        exc = None
        try:
            text = _half_sample_size_clause(a, b)
        except Exception as e:                                   # noqa: BLE001
            exc = e
        ok = exc is None and text is not None and "not comparable" in text.lower()
        _check(results, f"({a}, {b}) returns 'not comparable' wording without raising",
               ok, f"exc={exc!r} text={text!r}")

    # 8. Symmetry of content across 20 random pairs.
    sym_pairs = [(rng.randint(1, 50000), rng.randint(1, 50000)) for _ in range(20)]
    sym_violations = []
    for a, b in sym_pairs:
        fwd = _half_sample_size_clause(a, b)
        rev = _half_sample_size_clause(b, a)
        pm_fwd, pm_rev = _PCT_RE.search(fwd), _PCT_RE.search(rev)
        m_fwd, m_rev = _THICKER_RE.search(fwd), _THICKER_RE.search(rev)
        if pm_fwd and pm_rev and m_fwd and m_rev:
            if pm_fwd.group(1) != pm_rev.group(1) or m_fwd.group(1) == m_rev.group(1):
                sym_violations.append((a, b))
        elif bool(m_fwd) != bool(m_rev):
            # one side hit the comparable branch and the other didn't — only
            # possible right at the boundary; still a symmetry violation.
            sym_violations.append((a, b))
    _check(
        results, "symmetry: (a,b) and (b,a) quote the same percentage and name opposite halves",
        not sym_violations, f"{len(sym_violations)} violation(s): {sym_violations}",
    )

    # 9. Both raw counts always stated.
    count_pairs = [(rng.randint(1, 50000), rng.randint(1, 50000)) for _ in range(20)]
    count_violations = []
    for a, b in count_pairs:
        text = _half_sample_size_clause(a, b)
        if str(a) not in text or str(b) not in text:
            count_violations.append((a, b))
    _check(
        results, "both raw counts appear in the clause, over 20 random pairs",
        not count_violations, f"{len(count_violations)} violation(s): {count_violations}",
    )


# ---------------------------------------------------------------------------
# GROUP 5 — _PAPER_K_RUNS / K_RUNS coupling guard
# ---------------------------------------------------------------------------

def _check_paper_k_runs_coupling(results: list) -> None:
    """PAPER_K_RUNS / K_RUNS coupling, and the --k-runs plumbing.

    Three properties matter together:

      (a) ``run_full_simulation.K_RUNS`` must resolve to a real value,
          whether it is a literal or an alias of
          ``benchmark_core.PAPER_K_RUNS``;
      (b) there must be only ONE literal defining the published run count.
          Two independent literals that happen to agree today is a guard
          against drift that can erode silently; a single bound name makes
          the drift unspellable;
      (c) if a ``--k-runs`` override exists, it must be correctly plumbed --
          registered as a flag, reaching ``BenchmarkConfig.k_runs``,
          stamping provenance, and not disturbing the projection target.

    The check COUNT is deliberately fixed at 3, so a regression in any one
    of these properties shows up as a change in the suite total, not just an
    assertion failure buried in the log.
    """
    print("\nPAPER_K_RUNS / K_RUNS coupling + --k-runs plumbing:")

    k_runs_value = _resolve_run_full_sim_k_runs()
    _check(
        results, "K_RUNS resolves in run_full_simulation.py (literal or PAPER_K_RUNS alias)",
        k_runs_value is not None,
        f"K_RUNS={k_runs_value}",
    )
    alias = _read_module_level_alias(_RUN_FULL_SIM_PATH, "K_RUNS")
    literal = _read_module_level_int_constant(_RUN_FULL_SIM_PATH, "K_RUNS")
    print(f"    -> run_full_simulation.py K_RUNS = {k_runs_value!r} "
          f"(alias={alias!r}, own_literal={literal!r}), "
          f"benchmark_core.PAPER_K_RUNS = {_PAPER_K_RUNS!r}")

    # (b) Structural: there must be exactly one definition of the published k.
    single_definition = (alias == "PAPER_K_RUNS" and literal is None
                         and k_runs_value == _PAPER_K_RUNS)
    _check(
        results,
        "run_full_simulation.K_RUNS is an ALIAS of benchmark_core.PAPER_K_RUNS, "
        "not a second independent literal",
        single_definition,
        (f"K_RUNS alias={alias!r} own_literal={literal!r}; expected "
         f"`K_RUNS = PAPER_K_RUNS` and no literal. Two independent literals "
         f"for the published run count is exactly the configuration that "
         f"let an archive silently sweep k=10 while the source claimed 100. "
         f"Bind the name instead of copying the number."
         if not single_definition else ""),
    )

    # (c) The override exists AND is plumbed: flag registered, reaches
    #     BenchmarkConfig.k_runs, stamps provenance, and does NOT disturb the
    #     projection target.  Verified behaviourally (a runtime round-trip),
    #     plus an AST check that main_from_config actually wires the flag —
    #     a flag that parses but is never read would pass the round-trip.
    tree = ast.parse(open(_BENCHMARK_CORE_PATH, "r", encoding="utf-8").read())
    has_k_runs_flag = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in ("--k-runs", "--k_runs")
        for node in ast.walk(tree)
    )
    wired_in_main = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main_from_config":
            for call in ast.walk(node):
                if (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "with_overrides"
                        and any(kw.arg == "k_runs" for kw in call.keywords)):
                    wired_in_main = True

    parser = add_common_arguments(
        argparse.ArgumentParser(prog="coupling-probe"), "/tmp/coupling-probe"
    )
    base = BenchmarkConfig(
        governments=("ads",), difficulties=(10,), k_runs=2, grid_size=10,
        n_agents=10, max_cycles=5, max_steps=2, output_root="/tmp/coupling-probe",
    )
    overridden = base.with_overrides(
        k_runs=parse_k_runs(parser.parse_args(["--k-runs", "37"]).k_runs)
    )
    untouched = base.with_overrides(k_runs=parse_k_runs(parser.parse_args([]).k_runs))
    round_trip_ok = (
        overridden.k_runs == 37
        and overridden.k_runs_source == K_RUNS_SOURCE_CLI
        and overridden.paper_k_runs == _PAPER_K_RUNS   # projection target unmoved
        and untouched.k_runs == 2                       # absent flag changes nothing
        and untouched.k_runs_source == K_RUNS_SOURCE_DEFAULT
    )
    _check(
        results,
        "--k-runs exists and is plumbed: reaches config.k_runs, stamps "
        "k_runs_source=cli-override, leaves paper_k_runs alone, no-ops when absent",
        has_k_runs_flag and wired_in_main and round_trip_ok,
        (f"flag_registered={has_k_runs_flag} wired_in_main_from_config={wired_in_main} "
         f"round_trip(k_runs={overridden.k_runs}, source={overridden.k_runs_source}, "
         f"paper_k={overridden.paper_k_runs}; absent -> k_runs={untouched.k_runs}, "
         f"source={untouched.k_runs_source})"),
    )


# ---------------------------------------------------------------------------
# GROUP 6 — the median/CI figure draws what it claims (rendering contract)
# ---------------------------------------------------------------------------

_DEGENERATE_STATUSES = {
    MEDIAN_CI_STATUS_CONSTANT,
    MEDIAN_CI_STATUS_INSUFFICIENT_N,
    MEDIAN_CI_STATUS_ZERO_WIDTH,
    MEDIAN_CI_STATUS_EMPTY,
}


def _group6_rows(governments, difficulties):
    """Build combined_final_stats.csv rows with deliberately planted cells:

    - (anarchy, 10): no rows at all              -> MEDIAN_CI_STATUS_EMPTY
    - (democracy, 50): 8 identical rows          -> MEDIAN_CI_STATUS_CONSTANT
    - (anarchy, 90): only 2 rows                 -> MEDIAN_CI_STATUS_INSUFFICIENT_N (n=2 < min 3)
    - (ads, 10): normalized_health_score skewed  -> non-vacuity control for the median != mean check
    - everything else: 8 unremarkable random rows -> ordinary MEDIAN_CI_STATUS_OK cells
    """
    rng = random.Random(20260924)
    rows = []

    def rand_metrics():
        return [round(rng.uniform(0.05, 0.95), 6) for _ in SUMMARY_FIELDS]

    health_idx = SUMMARY_FIELDS.index("normalized_health_score")

    for gov in governments:
        for diff in difficulties:
            if gov == "anarchy" and diff == 10:
                continue  # empty cell — no rows written at all
            if gov == "democracy" and diff == 50:
                constant_metrics = [0.5] * len(SUMMARY_FIELDS)
                for run in range(8):
                    rows.append([gov, diff, run] + list(constant_metrics))
                continue
            if gov == "anarchy" and diff == 90:
                for run in range(2):
                    rows.append([gov, diff, run] + rand_metrics())
                continue
            if gov == "ads" and diff == 10:
                skewed = [0.1] * 7 + [0.95]
                for run in range(8):
                    metrics = rand_metrics()
                    metrics[health_idx] = skewed[run]
                    rows.append([gov, diff, run] + metrics)
                continue
            for run in range(8):
                rows.append([gov, diff, run] + rand_metrics())
    return rows


def _check_median_ci_render_contract(results: list, notes: list) -> None:
    print("\nmedian/CI rendering contract (draw-call capture, no pixels):")
    governments = ["ads", "democracy", "anarchy"]
    difficulties = [10, 50, 90]

    root = tempfile.mkdtemp(prefix="plot_regression_render_contract_")
    try:
        rows = _group6_rows(governments, difficulties)
        _make_archive(root, governments=governments, difficulties=difficulties,
                      k_runs=8, csv_rows=rows)
        config = config_from_manifest(root)
        loaded_rows = load_combined_rows(os.path.join(root, "combined_final_stats.csv"))
        plots_dir = os.path.join(root, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        bootstrap_calls: list = []
        orig_bootstrap = bc.bootstrap_median_ci

        def recording_bootstrap(values, seed, *args, **kwargs):
            result = orig_bootstrap(values, seed, *args, **kwargs)
            bootstrap_calls.append({"values": tuple(values), "seed": seed, "result": result})
            return result

        errorbar_calls: list = []
        orig_errorbar = maxes.Axes.errorbar

        def recording_errorbar(self, *args, **kwargs):
            y = args[1] if len(args) > 1 else kwargs.get("y")
            errorbar_calls.append({"y": y, "yerr": kwargs.get("yerr")})
            return orig_errorbar(self, *args, **kwargs)

        bc.bootstrap_median_ci = recording_bootstrap
        maxes.Axes.errorbar = recording_errorbar
        try:
            bc.make_median_ci_plots(config, loaded_rows, plots_dir)
        finally:
            bc.bootstrap_median_ci = orig_bootstrap
            maxes.Axes.errorbar = orig_errorbar

        # DISCOVERY, not a test bug: make_median_ci_plots calls bootstrap_median_ci
        # EXACTLY ONCE per (metric, government, difficulty) cell -- the drawing
        # loop caches each cell's MedianCI, and the sidecar's per-cell
        # sample-size table indexes that same cache rather than recomputing
        # it.  This is the STRONGER of two plausible call-count assertions:
        # "called twice, with an identical (values, seed) pair each time"
        # would also be deterministic, and so easy to mistake for correct,
        # but it would hide redundant computation.  "Called exactly once"
        # fails loudly if a duplicate pass is ever introduced, or if a third
        # consumer starts recomputing intervals behind the figure's back.
        # Since every call is a drawing call, the property this protects --
        # errorbar pairs 1:1 and in order with the drawable results -- is
        # asserted below over the whole call list.
        cell_count = len(governments) * len(difficulties)
        expected_calls = cell_count * len(PLOT_STATS)
        _check(
            results,
            "bootstrap_median_ci called exactly governments*difficulties*len(PLOT_STATS) times "
            "(once per cell — the sidecar census reuses the drawing pass's cached MedianCI)",
            len(bootstrap_calls) == expected_calls,
            f"expected {expected_calls}, got {len(bootstrap_calls)}",
        )

        # Every recorded call is a drawing-pass call now.
        draw_calls = list(bootstrap_calls)
        chunk_split_ok = (len(draw_calls) == cell_count * len(PLOT_STATS))

        # b. seed set == expected per-cell seeds, one per (gov, difficulty)
        expected_seeds = {
            (g, d): derive_seed(config.base_seed, "plot.bootstrap", g, d)
            for g in governments for d in difficulties
        }
        seeds_used = {c["seed"] for c in bootstrap_calls}
        _check(
            results,
            "every seed passed equals derive_seed(base_seed,'plot.bootstrap',gov,diff); "
            "distinct seeds == distinct (gov,difficulty) cells (9)",
            seeds_used == set(expected_seeds.values()) and len(seeds_used) == 9,
            f"seeds_used={len(seeds_used)} expected={len(set(expected_seeds.values()))}",
        )

        # c. each per-cell seed reused for all 4 metrics (metric-independent
        # seeding).  There is a single drawing pass, so the expectation is
        # exactly len(PLOT_STATS) calls per seed in total.  The underlying
        # property: the seed depends on (gov, difficulty) and NOT on the
        # metric.
        draw_seed_counts = collections.Counter(c["seed"] for c in draw_calls)
        uniform_four = (
            chunk_split_ok
            and len(draw_seed_counts) == 9
            and all(v == len(PLOT_STATS) for v in draw_seed_counts.values())
        )
        _check(
            results,
            "each of the 9 per-cell seeds is used for exactly len(PLOT_STATS) calls "
            "(one pass, metric-independent seeding)",
            uniform_four,
            f"draw_seed_counts={dict(draw_seed_counts)}",
        )

        # d. geometry: errorbar count == drawable count FROM THE DRAWING PASS ONLY
        # (the sidecar pass never touches matplotlib), and values line up in
        # call order — the drawing pass and errorbar_calls share the same
        # (government, difficulty) iteration order, so a straight zip over the
        # drawable subset lines each errorbar call up with the bootstrap result
        # that produced it.
        drawable = [c for c in draw_calls if c["result"].drawable]
        count_match = chunk_split_ok and len(errorbar_calls) == len(drawable)
        _check(
            results,
            "errorbar call count equals the number of drawable bootstrap results from the drawing pass",
            count_match, f"errorbar_calls={len(errorbar_calls)} drawable_draw_pass_results={len(drawable)}",
        )

        mismatches = []
        if count_match:
            for rec, call in zip(drawable, errorbar_calls):
                res = rec["result"]
                y = call["y"]
                y_val = y[0] if isinstance(y, (list, tuple)) else y
                yerr = call["yerr"]
                try:
                    low = yerr[0][0]
                    high = yerr[1][0]
                except (TypeError, IndexError, KeyError):
                    low = high = None
                lower = res.lower if res.lower is not None else res.median
                upper = res.upper if res.upper is not None else res.median
                expected_low = res.median - lower
                expected_high = upper - res.median
                if (low is None or high is None
                        or abs(y_val - res.median) > 1e-12
                        or abs(low - expected_low) > 1e-12
                        or abs(high - expected_high) > 1e-12):
                    mismatches.append((res.median, y_val, low, expected_low, high, expected_high))
        _check(
            results,
            "for each drawable cell, plotted y == median and yerr == (median-lower, upper-median)",
            count_match and not mismatches,
            f"{len(mismatches)} mismatch(es) (median,y,low,exp_low,high,exp_high): {mismatches[:5]}",
        )

        # e. non-vacuity control: at least one drawable cell has median != mean,
        #    and the plotted y matched the median (not the mean).
        skew_found = False
        skew_detail = "no qualifying (median != mean) drawable cell found"
        if count_match:
            for rec, call in zip(drawable, errorbar_calls):
                res = rec["result"]
                vals = rec["values"]
                if not vals:
                    continue
                mean_v = sum(vals) / len(vals)
                if abs(res.median - mean_v) <= 1e-9:
                    continue
                y = call["y"]
                y_val = y[0] if isinstance(y, (list, tuple)) else y
                if abs(y_val - res.median) < 1e-12 and abs(y_val - mean_v) > 1e-9:
                    skew_found = True
                    skew_detail = f"median={res.median} mean={mean_v} plotted_y={y_val}"
                    break
        _check(
            results,
            "non-vacuity: >=1 drawable cell has median != mean and plots the MEDIAN, not the mean",
            skew_found, skew_detail,
        )

        # f. no errorbar call for a degenerate-status cell.  The drawing pass
        # IS the whole call list, so this compares against every recorded
        # call rather than a 1-of-2 slice of it.
        ok_count = sum(1 for c in draw_calls if c["result"].status == MEDIAN_CI_STATUS_OK)
        produced_statuses = {c["result"].status for c in bootstrap_calls}
        produced_degenerate = sorted(_DEGENERATE_STATUSES & produced_statuses)
        no_degenerate_errorbars = chunk_split_ok and len(errorbar_calls) == ok_count
        _check(
            results,
            f"no errorbar call for a degenerate-status cell (produced: {produced_degenerate})",
            no_degenerate_errorbars,
            f"errorbar_calls={len(errorbar_calls)} ok_status_draw_pass_calls={ok_count} "
            f"all_produced_statuses={sorted(produced_statuses)}",
        )
        missing_degenerate = sorted(_DEGENERATE_STATUSES - produced_statuses)
        if missing_degenerate:
            notes.append(
                f"GROUP 6: could not produce status(es) {missing_degenerate} through the "
                f"archive path — produced {sorted(produced_statuses)} instead; "
                f"treat the un-produced ones as an untested gap, not a pass."
            )
        print(f"    -> produced statuses: {sorted(produced_statuses)}; "
              f"bootstrap_calls={len(bootstrap_calls)} (one pass, no sidecar "
              f"recompute) errorbar_calls={len(errorbar_calls)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# GROUP 7 — calibration-figure rendering defects
#
# Neither defect below is a Tukey-IQR-path regression (GROUP 1 does not cover
# `make_ads_calibration_plots` at all — it was added after the golden-AST
# guard's key list was frozen) and neither is a crash or a skip-path
# condition — nothing else in this suite would otherwise notice a box
# clipped past its own axis or a canvas stretched to an unusable shape.
#
# Both checks below build a REAL, minimal `ads/<difficulty>/run_*/
# final_stats.json` archive and call the actual production function,
# `make_ads_calibration_plots`, rather than reaching into
# `_style_difficulty_axes`/`_series_offsets` directly — a private-function
# unit test could pass while the real call sites still get the geometry
# wrong, since it is the call sites that feed `_series_offsets`'s
# generalized output into the pad computation. Both intercept
# `Figure.savefig` to inspect the actual drawn artists and the actual saved
# PNG, rather than trusting that the function returned without raising.
#
# `gov_keys=(_ADS_GOV_KEY, _ADS_GOV_KEY)` — the same real ADS directory named
# twice — is a deliberate, minimal way to drive both figures through their
# real MULTI-government code paths (`calibration_half_split.png`'s n_series2
# == 4, `calibration_trend_by_cycle.png`'s two-arm headline) without having
# to fabricate `autocracy_lookahead`'s outcome-partitioned
# `calibration_closures_by_outcome_then_scope` / `..._cycle_deltas_by_...`
# schema. The clipping and title-length defects are both purely geometric —
# they reproduce identically regardless of whether the second series' data
# comes from a distinct government or the same one plotted twice — so this
# is a faithful, low-effort repro of the exact n=4 / two-arm cases that can
# break on a real archive, not a synthetic case that happens to also fail.
# ---------------------------------------------------------------------------

def _write_ads_calibration_fixture(
    root, *, difficulties, k_runs, n_cycles, uneven_late_window=False,
):
    """
    Build a minimal, real ``ads/<difficulty>/run_*/final_stats.json`` archive.

    Enough for :func:`make_ads_calibration_plots` to draw all three figures
    (including finding a burn-in breakpoint: ``len(difficulties) * k_runs *
    n_cycles`` must clear ``2 * _BURN_IN_MIN_SEGMENT_OBS`` == 400 pooled
    observations, which the defaults both call sites below use satisfy:
    3 x 3 x 50 == 450) without needing a real simulation run.

    *uneven_late_window*, when true, truncates every other run to roughly
    60% of ``n_cycles`` so it never reaches ``calibration_half_split.png``'s
    post-burn-in "late" window — reproducing, with a single real ADS
    directory (no fabricated ``autocracy_lookahead`` partition schema needed),
    the real archive's ADS-flat-n-vs-A+L-thin-n asymmetry that makes
    ``_n_per_cell_caption`` disagree between the early and late halves. That
    disagreement is what pushes `make_ads_calibration_plots`'s ``distinct_n``
    check past ``len == 1`` and onto the long, one-clause-per-(government,
    half) caption line whose title-length blowup this fixture exists to
    reproduce (see `_check_calibration_aspect_ratios`) — a fixture where
    every run has full, identical cycle coverage never takes that branch, no
    matter how many duplicate ``gov_keys`` are passed in.

    Also writes ``calibration_cycle_deltas_by_outcome_then_scope`` /
    ``calibration_closures_by_outcome_then_scope`` — the partition
    :func:`make_scope_matched_calibration_plots` reads — mirroring the flat
    ``calibration_cycle_deltas`` field under outcome ``"enacted"``, scope
    ``"1"``, the only scope this fixture ever emits. That makes this
    fixture's ADS "n_groups == 1 subset" identical to its own full corpus,
    which is a deliberate simplification for a RENDERING/geometry fixture:
    this file's checks only need the scope-matched figures to have real data
    to draw, not a statistically distinct subset — the scope-FILTERING logic
    itself (that a real two-scope archive gets split correctly) is exercised
    directly, against a genuinely two-scope fixture, in
    ``test_ads_calibration.py``'s scope-matched-filter section. Adding this
    field does not change any existing check in this file:
    ``make_ads_calibration_plots`` never reads the partition for
    ``gov_key == _ADS_GOV_KEY`` (it reads the flat field unfiltered, by
    construction — see that function's docstring), so this is purely
    additive for the pre-existing calibration checks below.
    """
    for diff in difficulties:
        for run in range(k_runs):
            run_dir = os.path.join(root, _ADS_GOV_KEY, str(diff), f"run_{run:02d}")
            os.makedirs(run_dir, exist_ok=True)
            base = 0.05 + 0.001 * diff + 0.001 * run
            run_cycles = n_cycles
            if uneven_late_window and run % 2 == 1:
                run_cycles = max(1, int(n_cycles * 0.6))
            cycle_deltas = [[c, base + 0.0003 * c] for c in range(1, run_cycles + 1)]
            mean_abs_err = sum(v for _c, v in cycle_deltas) / len(cycle_deltas)
            stats = {
                "final_mean_prediction_error": base,
                "calibration_mean_abs_delta_first_half": base + 0.01,
                "calibration_mean_abs_delta_second_half": base - 0.005,
                "calibration_cycle_deltas": cycle_deltas,
                "calibration_cycle_deltas_by_outcome_then_scope": {
                    "enacted": {"1": cycle_deltas},
                },
                "calibration_closures_by_outcome_then_scope": {
                    "enacted": {
                        "1": {"n": len(cycle_deltas), "mean_abs_err": mean_abs_err},
                    },
                },
            }
            with open(os.path.join(run_dir, "final_stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f)


def _png_dimensions(path):
    """``(width, height)`` in pixels, read straight from the PNG IHDR chunk.

    Avoids adding a Pillow dependency to this file for two integers: the IHDR
    chunk is always the first chunk in a valid PNG, immediately after the
    8-byte signature, so bytes 16:24 are big-endian width then height with no
    other parsing needed.
    """
    with open(path, "rb") as f:
        header = f.read(24)
    if not header.startswith(_PNG_MAGIC) or len(header) < 24:
        raise ValueError(f"{path} is not a readable PNG (got {len(header)} header bytes)")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def _check_calibration_half_split_no_clipping(results: list) -> None:
    print("\ncalibration_half_split.png: 4-series geometry does not clip its own axis:")
    root = tempfile.mkdtemp(prefix="calibration_clip_")
    try:
        _write_ads_calibration_fixture(
            root, difficulties=(1, 50, 100), k_runs=3, n_cycles=50,
        )
        plots_dir = os.path.join(root, "plots")

        captured = {}
        orig_savefig = mfigure.Figure.savefig

        def _spy(self, fname, *a, **kw):
            if isinstance(fname, str) and fname.endswith("calibration_half_split.png"):
                ax = self.axes[0]
                captured["xlim"] = ax.get_xlim()
                captured["n_patches"] = len(ax.patches)
                extents = [p.get_path().get_extents() for p in ax.patches]
                if extents:
                    captured["box_lo"] = min(e.x0 for e in extents)
                    captured["box_hi"] = max(e.x1 for e in extents)
            return orig_savefig(self, fname, *a, **kw)

        mfigure.Figure.savefig = _spy
        try:
            # Duplicate gov_keys: see the GROUP 7 header comment for why this
            # is a faithful repro of the real 2-government / n=4-series case.
            written = make_ads_calibration_plots(
                root, plots_dir, gov_keys=(_ADS_GOV_KEY, _ADS_GOV_KEY),
            )
        finally:
            mfigure.Figure.savefig = orig_savefig

        half_split_png = os.path.join(plots_dir, "calibration_half_split.png")
        _check(results, "calibration_half_split.png was written", half_split_png in written,
               f"written={written}")
        _check(results, "the spy actually captured a 4-series draw (test is not vacuous)",
               captured.get("n_patches", 0) >= 4,
               f"captured={captured}")

        xlim = captured.get("xlim")
        box_lo = captured.get("box_lo")
        box_hi = captured.get("box_hi")
        no_clip = (
            xlim is not None and box_lo is not None and box_hi is not None
            and xlim[0] <= box_lo and box_hi <= xlim[1]
        )
        _check(
            results,
            "no drawn box-plot patch extends past ax.get_xlim() on either edge",
            no_clip,
            f"xlim={xlim} box_extent=({box_lo}, {box_hi})" if not no_clip
            else f"xlim={xlim} box_extent=({box_lo:.4f}, {box_hi:.4f}) — fully contained",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


#: Aspect-ratio ceiling for every calibration figure's saved PNG.
#:
#: The regenerated 11-figure set (IQR + median_ci + all three calibration
#: figures, all title-wrapped) sits at 2.30-2.44; 2.7 sits above all of it
#: with real headroom for font/DPI/matplotlib-version variation, while still
#: catching the 3.24-4.87 aspect ratios an unwrapped, multi-clause title can
#: produce on these figures.
_CALIBRATION_ASPECT_RATIO_MAX = 2.7

#: Every PNG `make_ads_calibration_plots` / `make_scope_matched_calibration_plots`
#: writes, checked identically below — not just whichever figure most
#: recently needed a fix. A calibration figure added later is covered by
#: construction, not by remembering to extend this list again. The two
#: `_scope_matched` names share the same multi-arm title-assembly shape and
#: the same aspect-ratio hazard as the other three, so they belong in the
#: same guard rather than a separate one.
#:
#: NOTE: with bare titles, this guard is a regression net against a hazard
#: (a long multi-clause title driving these aspect ratios back into the
#: 3.24-4.87 range) that bare titles make unlikely, rather than a check on
#: active title-wrapping machinery.
_CALIBRATION_PNG_NAMES = (
    "mean_prediction_error.png",
    "calibration_half_split.png",
    "calibration_trend_by_cycle.png",
    "mean_prediction_error_scope_matched.png",
    "calibration_trend_by_cycle_scope_matched.png",
)


def _check_calibration_aspect_ratios(results: list) -> None:
    print("\ncalibration figures: every saved PNG's aspect ratio is bounded:")
    root = tempfile.mkdtemp(prefix="calibration_aspect_")
    try:
        # uneven_late_window=True + more runs/cycles than the clipping fixture
        # (k_runs=6, n_cycles=60, still >= 2*_BURN_IN_MIN_SEGMENT_OBS=400
        # pooled obs after truncation) — needed so early vs late per-cell n
        # actually disagree (see the fixture's own docstring) and
        # calibration_half_split.png's title takes its long, multi-clause
        # branch instead of the short flat-n one a uniform fixture would.
        _write_ads_calibration_fixture(
            root, difficulties=(1, 50, 100), k_runs=6, n_cycles=60,
            uneven_late_window=True,
        )
        # A manifest.json's scale clause (", grid WxH, N agents, M cycles")
        # is part of the real title text this check is guarding against, and
        # omitting it understates the real defect's length: measured this
        # exact fixture at ratio 2.32 without a manifest and 2.66 with one,
        # against the unwrapped title-assembly code, before adding the 3-way
        # gov_keys duplication below closed the remaining gap to a
        # comfortable failing margin.
        with open(os.path.join(root, bc.MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump({"config": {"grid_size": 50, "n_agents": 500, "max_cycles": 150}}, f)
        plots_dir = os.path.join(root, "plots")

        # Triplicate gov_keys (see GROUP 7 header for the pattern; three
        # copies rather than two here specifically): drives every figure's
        # real multi-government code path — the multi-arm trend headline AND
        # calibration_half_split.png's n_series2==6 per-half n disclosure —
        # which is what stretched both figures' titles on the live archive.
        # Measured (this fixture, unwrapped title-assembly code,
        # `_ADS_GOV_KEY` x N): N=2 -> ratio 2.66 (still under a 2.7 bound —
        # not a reliable repro on its own), N=3 -> ratio 3.40, N=4 -> ratio
        # 4.14. N=3 is used so the failure margin is comfortable, not
        # borderline.
        written = make_ads_calibration_plots(
            root, plots_dir, gov_keys=(_ADS_GOV_KEY,) * 3,
        )
        # Same triplicate-gov_keys stress on the scope-matched pair: they share
        # the same multi-arm title-assembly shape as the figures above (see
        # _CALIBRATION_PNG_NAMES's docstring for the now-removed wrapping
        # machinery this guard predates), so the same geometry hazard
        # applies. The
        # fixture's ADS "n_groups == 1" partition is written by
        # `_write_ads_calibration_fixture` alongside the flat field (see that
        # function's docstring) so this has real data to draw from.
        written += make_scope_matched_calibration_plots(
            root, plots_dir, gov_keys=(_ADS_GOV_KEY,) * 3,
        )
        for name in _CALIBRATION_PNG_NAMES:
            png = os.path.join(plots_dir, name)
            _check(results, f"{name} was written", png in written, f"written={written}")

            width, height = _png_dimensions(png)
            ratio = width / height
            _check(
                results,
                f"{name}: saved PNG width/height ratio is bounded "
                f"(<= {_CALIBRATION_ASPECT_RATIO_MAX}; declared figsize ratio is {16/7:.3f})",
                ratio <= _CALIBRATION_ASPECT_RATIO_MAX,
                f"{width}x{height} px, ratio={ratio:.3f}",
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _check_bare_titles_and_half_split_ci(results: list) -> None:
    """Titles/legends carry only parameters; half-split CI sibling."""
    print("\ncalibration figures: bare one-line titles, no explanatory legends, "
          "and calibration_half_split_median_ci.png:")
    root = tempfile.mkdtemp(prefix="calibration_titles_")
    try:
        _write_ads_calibration_fixture(
            root, difficulties=(1, 50, 100), k_runs=3, n_cycles=50,
        )
        plots_dir = os.path.join(root, "plots")
        seen = {}
        orig_savefig = mfigure.Figure.savefig

        def _spy(self, fname, *a, **kw):
            if isinstance(fname, str) and fname.endswith(".png"):
                ax = self.axes[0]
                leg = ax.get_legend()
                seen[os.path.basename(fname)] = (
                    ax.get_title(), ax.get_ylabel(),
                    [t.get_text() for t in leg.get_texts()] if leg else [],
                    len(ax.lines), len(ax.collections),
                )
            return orig_savefig(self, fname, *a, **kw)

        mfigure.Figure.savefig = _spy
        try:
            written = make_ads_calibration_plots(
                root, plots_dir, gov_keys=(_ADS_GOV_KEY, _ADS_GOV_KEY),
            )
        finally:
            mfigure.Figure.savefig = orig_savefig

        ci_png = os.path.join(plots_dir, "calibration_half_split_median_ci.png")
        _check(results, "calibration_half_split_median_ci.png and its sidecar "
                        "were written",
               ci_png in written
               and ci_png.replace(".png", "_alt_text.txt") in written,
               f"written={[os.path.basename(w) for w in written]}")
        ci = seen.get("calibration_half_split_median_ci.png")
        _check(results, "the CI figure draws markers/bars, not boxes, and has "
                        "the same title as the box figure",
               ci is not None and ci[3] > 0
               and ci[0] == seen.get("calibration_half_split.png", ("",))[0],
               f"{ci}")

        banned = ("(", "—", "lower = better", "n =", "n=", "not ", "see ",
                  "artifact", "interpretable")
        bad = []
        for name, (title, ylabel, legend, _l, _c) in seen.items():
            if "\n" in title:
                bad.append(f"{name}: multi-line title")
            for text in [title, ylabel]:
                bad += [f"{name}: {b!r} in {text!r}" for b in banned
                        if b in text and not (b == "(" and "(cycles" in text)]
            for text in legend:
                bad += [f"{name}: {b!r} in legend {text!r}" for b in banned
                        if b in text and not (b == "(" and text.startswith("Burn-in (cycles"))]
        _check(results, "no explanatory text in any calibration title, y-label "
                        "or legend entry", not bad, "; ".join(bad[:5]))
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# GROUP 8 — manifest schedule_digest golden
# ---------------------------------------------------------------------------

#: `benchmark_core._schedule_block()["schedule_digest"]`, pinned per slope.
#:
#: FORMAT v3: the config table plus the event/engine channel
#: (_auto_n_waves, _auto_base_severity, grid depletion diff_scale, epidemic
#: infection_pct, health recovery_cycles) with `difficulty_multiplier(d)`
#: (the drain multiplier) APPENDED as the last column of the event/engine
#: channel.  Earlier format versions never covered drain -- there is no
#: separate drain field on `SimulationConfig` -- so a drain-only regression
#: could pass an earlier golden unnoticed.  v1 and v2 digests are
#: earlier-format references and are NOT comparable to v3; see
#: _schedule_block's docstring.
#:
#: Verified: v3 at slope 1.0 equals the v3 digest computed from the
#: pre-relaxation config/wave functions (floors: warnings>=5, regen>=0.25,
#: n_waves<=10) byte-for-byte -- relaxing those floors alone does not move
#: this digest at the reference slope.
#:
#: The config table includes `SimulationConfig.ambient_hazard`
#: (`engine.difficulty.ambient_hazard`: 0 at D1 -> 0.008 at D20, constant
#: above).  All three goldens were re-pinned when that column entered the
#: config: with the `ambient_hazard` key removed from each config row, each
#: digest reproduces the previous golden exactly (slope 1.0 bd4dcde1...,
#: slope 0.0 5ac3ea2f..., slope 1.75 0751099a...), so the column is the
#: only change.  The ambient hazard does not depend on the slope (its ramp
#: ends at D20, below the knee), so it moves all three digests alike.
_GOLDEN_SCHEDULE_DIGEST_SLOPE1 = (
    "9055e22815bf5fbbd7f5c1e62f73828b0ad125dee7fa8f9d79e73c93b62777e5"
)
_GOLDEN_SCHEDULE_DIGEST_SLOPE0 = (
    "949ca59c3046cbc64a8b1518e8cc6d3bf34a49482e2f0c110a037deffa64584c"
)

#: The shipped schedule (slope 1.75).  This is the LIVE/AMBIENT-slope golden.
#: Archived manifests written before the ambient-hazard column existed carry
#: the previous digest (0751099a...) and will not match this value.
_GOLDEN_SCHEDULE_DIGEST_SLOPE_STAR = (
    "de1793138fa3f2ba7dba6aff5d4859109a6a75b8becf9923e3c2b1ae72ac2f83"
)


def _check_schedule_digest_golden(results: list) -> None:
    """Pin `_schedule_block()["schedule_digest"]` at both slopes, and prove
    it is sensitive to the event, drain and config channels.

    A digest that stays unchanged when
    `scenarios.scenario_base._auto_event_schedule` is stubbed out is a blind
    spot: it would let a severity-ceiling regression go undetected by a
    "digest matches the pilot manifest" provenance gate.  The controls below
    assert the opposite is true: stubbing `_auto_base_severity`,
    `_auto_n_waves`, or `difficulty_multiplier` (the two functions
    `_schedule_block` reads for the event channel, plus the drain column
    added in v3) MUST move the digest.

    The slope-1.0 assertion is pinned to an EXPLICIT `DIFFICULTY_TAIL_SLOPE =
    1.0` patch, not the ambient module slope, because the ambient slope is
    whatever `DIFFICULTY_TAIL_SLOPE` currently ships with --
    `_GOLDEN_SCHEDULE_DIGEST_SLOPE_STAR` is the live-slope golden, pinned
    below via its own explicit `DIFFICULTY_TAIL_SLOPE = 1.75` patch, and the
    slope-1.0 assertion keeps proving the reference schedule is still
    reproducible by flipping the constant back.  Every patch also calls
    `validate_schedule()`, matching every other slope-mutating site in this
    codebase.
    """
    import engine.difficulty as difficulty_mod

    print("\nschedule_digest golden (v3 format, event/drain/config channels):")

    original_slope = difficulty_mod.DIFFICULTY_TAIL_SLOPE
    try:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = 1.0
        difficulty_mod.validate_schedule()
        real_slope1 = bc._schedule_block()
    finally:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = original_slope
    _check(results, "reference: schedule_digest at an EXPLICIT slope 1.0 "
                    "patch matches the v3 golden",
           real_slope1["schedule_digest"] == _GOLDEN_SCHEDULE_DIGEST_SLOPE1,
           f"got {real_slope1['schedule_digest']}")

    try:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = 0.0
        difficulty_mod.validate_schedule()
        slope0 = bc._schedule_block()
    finally:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = original_slope
    _check(results, "schedule_digest at slope 0.0 matches the v3 golden",
           slope0["schedule_digest"] == _GOLDEN_SCHEDULE_DIGEST_SLOPE0,
           f"got {slope0['schedule_digest']}")

    # The shipped slope is 1.75.  Pinned via its own EXPLICIT patch (same
    # pattern as slope 1.0 and 0.0 above), so this assertion is
    # self-contained rather than trusting whatever the ambient module
    # constant happens to be.
    try:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = 1.75
        difficulty_mod.validate_schedule()
        real_slope_star = bc._schedule_block()
    finally:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = original_slope
    _check(results, "shipped: schedule_digest at an EXPLICIT slope 1.75 "
                    "patch matches the v3 golden",
           real_slope_star["schedule_digest"] == _GOLDEN_SCHEDULE_DIGEST_SLOPE_STAR,
           f"got {real_slope_star['schedule_digest']}")
    _check(results, "shipped: the AMBIENT module schedule_digest (no patch) "
                    "matches the same slope-1.75 golden -- proves 1.75 is "
                    "really what ships, not just what an explicit patch "
                    "can reach",
           bc._schedule_block()["schedule_digest"] == _GOLDEN_SCHEDULE_DIGEST_SLOPE_STAR,
           f"got {bc._schedule_block()['schedule_digest']}")

    _check(results, "CONTROL: slope 0.0, slope 1.0 and slope 1.75 digests are "
                    "all pairwise distinct (the pins above cannot pass by "
                    "any two slopes collapsing to one constant)",
           len({_GOLDEN_SCHEDULE_DIGEST_SLOPE0, _GOLDEN_SCHEDULE_DIGEST_SLOPE1,
                _GOLDEN_SCHEDULE_DIGEST_SLOPE_STAR}) == 3)

    # CONTROL: stub the event-channel functions _schedule_block reads.  If
    # the digest does not move, it is not actually covering that channel --
    # run as a standing regression guard against exactly that blind spot.
    # Compared against the shipped (1.75) reference, since that is what the
    # ambient `bc._schedule_block()` calls below actually compute against.
    orig_severity = bc._auto_base_severity
    try:
        bc._auto_base_severity = lambda d: 0.0
        stubbed = bc._schedule_block()
        _check(results, "CONTROL: stubbing _auto_base_severity moves "
                        "schedule_digest",
               stubbed["schedule_digest"] != real_slope_star["schedule_digest"],
               "digest is insensitive to _auto_base_severity -- the event "
               "channel is not covered")
    finally:
        bc._auto_base_severity = orig_severity

    orig_waves = bc._auto_n_waves
    try:
        bc._auto_n_waves = lambda d: 0
        stubbed = bc._schedule_block()
        _check(results, "CONTROL: stubbing _auto_n_waves moves "
                        "schedule_digest",
               stubbed["schedule_digest"] != real_slope_star["schedule_digest"],
               "digest is insensitive to _auto_n_waves -- the event channel "
               "is not covered")
    finally:
        bc._auto_n_waves = orig_waves

    # CONTROL (v3): stub the drain multiplier the v3 digest added.  A
    # drain-only regression must move the digest, or v3's whole reason for
    # existing -- v2 never covered drain, because SimulationConfig has no
    # drain field -- has regressed.
    import engine.difficulty as _difficulty_for_stub
    orig_mult = _difficulty_for_stub.difficulty_multiplier
    try:
        _difficulty_for_stub.difficulty_multiplier = lambda d: 0.0
        stubbed = bc._schedule_block()
        _check(results, "CONTROL (v3): stubbing difficulty_multiplier (the "
                        "drain column) moves schedule_digest",
               stubbed["schedule_digest"] != real_slope_star["schedule_digest"],
               "digest is insensitive to difficulty_multiplier -- the v3 "
               "drain column has regressed")
    finally:
        _difficulty_for_stub.difficulty_multiplier = orig_mult

    # CONTROL: stub the ambient-hazard schedule where from_difficulty looks it
    # up.  The config channel must cover it, or an ambient-hazard regression
    # would pass a "digest matches the pilot manifest" gate.
    import engine.simulation as _simulation_for_stub
    orig_ambient = _simulation_for_stub.ambient_hazard
    try:
        _simulation_for_stub.ambient_hazard = lambda d: 0.0
        stubbed = bc._schedule_block()
        _check(results, "CONTROL: stubbing ambient_hazard (config channel) "
                        "moves schedule_digest",
               stubbed["schedule_digest"] != real_slope_star["schedule_digest"],
               "digest is insensitive to the ambient-hazard schedule")
    finally:
        _simulation_for_stub.ambient_hazard = orig_ambient

    # Sanity: stubs fully unwound, real (ambient, shipped) digest reproduces
    # exactly.
    _check(results, "schedule_digest is unchanged after the stub controls "
                    "are unwound (no leakage into other tests)",
           bc._schedule_block()["schedule_digest"] == real_slope_star["schedule_digest"])


def main() -> int:
    results: list = []
    skipped: list = []
    notes: list = []
    print("Plot-regression guards (IQR parity, --plots-only, half-split captions):")
    _check_iqr_regression_guard(results, skipped)
    _check_plots_only_success(results)
    _check_plots_only_refusal(results)
    _check_half_sample_size_clause(results)
    _check_paper_k_runs_coupling(results)
    _check_median_ci_render_contract(results, notes)
    _check_calibration_half_split_no_clipping(results)
    _check_calibration_aspect_ratios(results)
    _check_bare_titles_and_half_split_ci(results)
    _check_schedule_digest_golden(results)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if skipped:
        print(f"{len(skipped)} skipped")
        for s in skipped:
            print(f"  - {s}")
    if notes:
        print("\nNOTES:")
        for n in notes:
            print(f"  - {n}")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
