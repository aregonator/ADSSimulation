#!/usr/bin/env python3
"""
Regression lock for the difficulty schedule in `engine/difficulty.py`.

    .venv-local/bin/python test_difficulty_schedule.py     # exit 0 = pass

WHAT THIS PROTECTS
------------------
``engine/difficulty.py`` is the single implementation of the difficulty
"knee"; every call site that scales behaviour with difficulty level must
delegate to it rather than re-deriving the formula.  Tuning
``DIFFICULTY_TAIL_SLOPE`` is a deliberate behavioural change to D91-D100 and
must be a NON-change to D1-D90.  Nothing in the code enforces that.  This
file does.

Three distinct failure modes are covered, and they need different tests:

  1. **The baseline drifts.**  Somebody edits the lerp endpoints, the
     knee level, or the rounding, and every archived run in
     ``benchmark_results/`` now describes a schedule the code cannot
     reproduce.  ``_GOLDEN_SWEEP`` / ``_GOLDEN_ALL_LEVELS_SHA256`` pin exact
     ``repr()`` values -- exact float equality, not a tolerance.

  2. **Tuning the slope leaks below the knee.**  The whole safety argument for
     letting ``DIFFICULTY_TAIL_SLOPE`` be raised is that doing so cannot touch
     D1-D89.  ``_check_slope_tuning_safety`` re-runs the golden table at slopes
     0.25/0.5/0.75/1.0 and requires D1-D89 to be *identical* while D91-D100
     *must* differ.  The second half is the control: without it the test would
     also pass if the knob were disconnected entirely.

  3. **A call site falls off the schedule.**  Site 6 -- ``recovery_cycles`` in
     ``_apply_health_dynamics`` -- is a fourth copy of the knee logic inside
     ``simulation.py``.  It is verified here at SOURCE level (the real
     expression is extracted from ``engine/simulation.py`` by AST and
     evaluated) rather than by re-implementing it, and it is verified to
     *extend* at a nonzero slope, which is the property that matters for safe
     tuning.

NON-VACUITY
-----------
A test that only asserts "A == B" passes just as happily when A and B are both
broken, or when the lever it pulls does nothing.  Every equality claim here has
a paired control that must FAIL to be equal:

  * the reference transcription is shown to disagree out-of-domain (the two
    known ``difficulty <= 0`` differences), so it is not a copy of the code
    under test;
  * the slope monkeypatch is shown to move D91-D100, so it is wired;
  * the AST scan is shown to detect a synthetic open-coded knee.

WHY THIS FILE LIVES HERE, AND NOT IN tests/
-------------------------------------------
This file follows the same pattern as ``test_seed_derivation.py`` and
``test_default_governments.py``: it sits beside the code it guards and runs
under a plain interpreter with no pytest, alongside the project's other
top-level regression tests.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import os
import subprocess
import sys
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import engine.difficulty as difficulty_mod          # noqa: E402
from engine.difficulty import (                     # noqa: E402
    DIFFICULTY_KNEE_LEVEL,
    DIFFICULTY_T_MAX,
    DIFFICULTY_TAIL_SLOPE,
    difficulty_multiplier,
    difficulty_t,
    effective_difficulty,
    past_knee,
    validate_schedule,
)
from engine.simulation import SimulationConfig, _lerp  # noqa: E402

# ---------------------------------------------------------------------------
# Golden tables
#
# PROVENANCE.  Captured from `engine/difficulty.py` at
# DIFFICULTY_TAIL_SLOPE = 0.0, and independently cross-checked in
# `_check_matches_pre_tuning_reference` against a hand-transcribed reference
# implementation of the same arithmetic (see that function's docstring).
# They are therefore not merely a snapshot of the code that produces them.
#
# Three slope settings are pinned as separate golden tables, each anchoring a
# different archived dataset: slope 0.0 (`_GOLDEN_SWEEP` /
# `_GOLDEN_ALL_LEVELS_SHA256`), slope 1.0 (`_GOLDEN_TAIL_SLOPE1` /
# `_GOLDEN_ALL_LEVELS_SHA256_SLOPE1`), and slope 1.75
# (`_GOLDEN_TAIL_SLOPE_STAR` / `_GOLDEN_ALL_LEVELS_SHA256_SLOPE_STAR`), which
# is the value `DIFFICULTY_TAIL_SLOPE` ships with.  Each table is evaluated
# under an explicit `_slope(...)` override so the knee machinery is checked
# against every schedule it must still be able to reproduce.
#
# IF THESE FAIL.  Do not update the constants to make the test green.  A change
# here re-keys every figure, every cell statistic and every archived run in
# benchmark_results/ against a schedule the archives no longer match.
# ---------------------------------------------------------------------------

#: The 21 levels the production sweep actually visits.
SWEEP_LEVELS = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                55, 60, 65, 70, 75, 80, 85, 90, 95, 100)

#: difficulty -> (effective_difficulty, difficulty_t, difficulty_multiplier)
_GOLDEN_SWEEP = {
      1: (1.0, 0.0, 0.5),
      5: (5.0, 0.04040404040404041, 0.5606),
     10: (10.0, 0.09090909090909091, 0.6364),
     15: (15.0, 0.1414141414141414, 0.7121),
     20: (20.0, 0.1919191919191919, 0.7879),
     25: (25.0, 0.24242424242424243, 0.8636),
     30: (30.0, 0.29292929292929293, 0.9394),
     35: (35.0, 0.3434343434343434, 1.0152),
     40: (40.0, 0.3939393939393939, 1.0909),
     45: (45.0, 0.4444444444444444, 1.1667),
     50: (50.0, 0.494949494949495, 1.2424),
     55: (55.0, 0.5454545454545454, 1.3182),
     60: (60.0, 0.5959595959595959, 1.3939),
     65: (65.0, 0.6464646464646465, 1.4697),
     70: (70.0, 0.696969696969697, 1.5455),
     75: (75.0, 0.7474747474747475, 1.6212),
     80: (80.0, 0.797979797979798, 1.697),
     85: (85.0, 0.8484848484848485, 1.7727),
     90: (90.0, 0.898989898989899, 1.8485),
     95: (90.0, 0.898989898989899, 1.8485),
    100: (90.0, 0.898989898989899, 1.8485),
}

#: The knee itself.  D89/D90/D91 is the boundary that a slope change moves.
_GOLDEN_KNEE_NEIGHBOURHOOD = {
     85: (85.0, 0.8484848484848485, 1.7727),
     86: (86.0, 0.8585858585858586, 1.7879),
     87: (87.0, 0.8686868686868687, 1.803),
     88: (88.0, 0.8787878787878788, 1.8182),
     89: (89.0, 0.8888888888888888, 1.8333),
     90: (90.0, 0.898989898989899, 1.8485),
     91: (90.0, 0.898989898989899, 1.8485),
     92: (90.0, 0.898989898989899, 1.8485),
     93: (90.0, 0.898989898989899, 1.8485),
}

#: sha256 over "d|repr(eff)|repr(t)|repr(mult)" for d in 1..100, newline-joined.
#: Complete coverage; the tables above exist so a failure is diagnosable.
_GOLDEN_ALL_LEVELS_SHA256 = (
    "d236c6474038ae1963822f26156a48191dcce81e14be15968abef57a9bb2aed5"
)

#: The reference schedule (slope 1.0) at the swept levels that differ from
#: slope 0.0.  Every other swept level must equal `_GOLDEN_SWEEP` exactly.
_GOLDEN_TAIL_SLOPE1 = {
     91: (91.0, 0.9090909090909091, 1.8636),
     92: (92.0, 0.9191919191919192, 1.8788),
     93: (93.0, 0.9292929292929293, 1.8939),
     95: (95.0, 0.9494949494949495, 1.9242),
    100: (100.0, 1.0, 2.0),
}

#: sha256 of the level table at the reference slope 1.0 (same format above).
_GOLDEN_ALL_LEVELS_SHA256_SLOPE1 = (
    "41e6c0d9062dfb4d318e1a96065a8e8e52f31b541dc45e65d1c39c24a88c12c7"
)

#: The currently shipped schedule (slope 1.75) at the swept levels that
#: differ from slope 0.0.  Every other swept level must equal
#: `_GOLDEN_SWEEP` exactly (D1-D90 are bit-identical at every slope).
#: Computed against the live tree with `./.venv-local/bin/python`.
_GOLDEN_TAIL_SLOPE_STAR = {
     91: (91.75, 0.9166666666666666, 1.875),
     92: (93.5, 0.9343434343434344, 1.9015),
     93: (95.25, 0.952020202020202, 1.928),
     95: (98.75, 0.9873737373737373, 1.9811),
    100: (107.5, 1.0757575757575757, 2.1136),
}

#: sha256 of the level table at the shipped slope 1.75 (same format above).
_GOLDEN_ALL_LEVELS_SHA256_SLOPE_STAR = (
    "8611b75f7949e2c926d12c31dbec3906e04f6a880070a11f88731ae8c874c319"
)

#: Site 6 -- `recovery_cycles` for difficulty 1..100 at slope 0.0.
_GOLDEN_RECOVERY_CYCLES = (
    15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 21,
    21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 27, 27,
    27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 33, 33, 33,
    34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 39, 39, 39, 40,
    40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 45, 45, 45, 46, 46,
    46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
)

#: Candidate slope values covered by the "D1-D89 cannot move" guarantee, so
#: any future tuning value in this range is already exercised by these tests.
#:
#: The admissible domain is capped at DIFFICULTY_T_MAX = 1.25, i.e.
#: slope <= 3.475: 3.5 gives difficulty_t(100) = 1.2525... > 1.25, which
#: validate_schedule() rejects (see _check_t_max_guard), so 3.5 itself is not
#: a valid candidate.  1.5/1.75/2.0/2.5/3.0/3.475 span the candidate grid
#: (1.75 is the shipped interpolation point) plus the exact upper bound.
_CANDIDATE_SLOPES = (0.25, 0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 3.0, 3.475)

#: Packages whose executable code must not contain an open-coded knee.
_SOURCE_DIRS = ("engine", "scenarios", "governments", "agents")


# ---------------------------------------------------------------------------
# Harness (same shape as test_seed_derivation.py / test_default_governments.py)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _slope(value):
    """Temporarily set the module-level tail slope; always restored.

    Calls ``validate_schedule()`` immediately after setting the constant, so
    a caller that hands this an out-of-domain slope fails loudly at entry
    with the real guard's ``ValueError``, instead of silently computing
    extrapolated values past ``DIFFICULTY_T_MAX``.  Every slope used with
    this helper in this file is therefore proved admissible by the guard
    itself, not just by convention.
    """
    original = difficulty_mod.DIFFICULTY_TAIL_SLOPE
    difficulty_mod.DIFFICULTY_TAIL_SLOPE = value
    try:
        difficulty_mod.validate_schedule()
        yield
    finally:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = original


def _check(results, name, ok, detail=""):
    results.append((name, bool(ok), str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail and not ok else ""))


def _all_levels_digest():
    blob = "\n".join(
        f"{d}|{effective_difficulty(d)!r}|{difficulty_t(d)!r}"
        f"|{difficulty_multiplier(d)!r}"
        for d in range(1, 101)
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _recovery_cycles_via_helper(d):
    """The canonical form of site 6, expressed through `difficulty_t`."""
    return int(15 + difficulty_t(d) * 35)


# ---------------------------------------------------------------------------
# 1. The slope-0 baseline is pinned
# ---------------------------------------------------------------------------

def _check_golden_values(results):
    _check(results, "DIFFICULTY_KNEE_LEVEL is 90",
           DIFFICULTY_KNEE_LEVEL == 90, f"got {DIFFICULTY_KNEE_LEVEL!r}")
    _check(results, "DIFFICULTY_TAIL_SLOPE ships as 1.75",
           repr(DIFFICULTY_TAIL_SLOPE) == "1.75",
           f"got {DIFFICULTY_TAIL_SLOPE!r}")
    with _slope(0.0):
        _check_golden_values_slope0(results)


def _check_golden_values_slope0(results):

    bad = []
    for d, (ge, gt, gm) in _GOLDEN_SWEEP.items():
        got = (effective_difficulty(d), difficulty_t(d), difficulty_multiplier(d))
        if (repr(got[0]), repr(got[1]), repr(got[2])) != (repr(ge), repr(gt), repr(gm)):
            bad.append(f"D{d}: {got!r} != {(ge, gt, gm)!r}")
    _check(results, f"slope 0.0: the {len(SWEEP_LEVELS)} swept levels match golden (exact repr)",
           not bad, "; ".join(bad[:4]))

    bad = []
    for d, (ge, gt, gm) in _GOLDEN_KNEE_NEIGHBOURHOOD.items():
        got = (effective_difficulty(d), difficulty_t(d), difficulty_multiplier(d))
        if (repr(got[0]), repr(got[1]), repr(got[2])) != (repr(ge), repr(gt), repr(gm)):
            bad.append(f"D{d}: {got!r} != {(ge, gt, gm)!r}")
    _check(results, "slope 0.0: the knee neighbourhood D85-D93 matches golden (exact repr)",
           not bad, "; ".join(bad[:4]))

    got_digest = _all_levels_digest()
    _check(results, "slope 0.0: all 100 levels hash to the golden digest",
           got_digest == _GOLDEN_ALL_LEVELS_SHA256,
           f"got {got_digest}, want {_GOLDEN_ALL_LEVELS_SHA256}")

    # The digest above would also pass if the golden constant were recomputed
    # from live code, so prove the digest is sensitive to a one-ULP change.
    perturbed = hashlib.sha256(
        ("\n".join(f"{d}|{effective_difficulty(d)!r}|{difficulty_t(d)!r}"
                   f"|{difficulty_multiplier(d)!r}" for d in range(1, 100))
         + f"\n100|{90.0!r}|{difficulty_t(100)!r}"
           f"|{difficulty_multiplier(100) + 1e-12!r}").encode()
    ).hexdigest()
    _check(results, "CONTROL: the level digest changes under a 1e-12 perturbation",
           perturbed != _GOLDEN_ALL_LEVELS_SHA256,
           "digest is insensitive -- the hash is not covering the values")


def _check_shipped_schedule(results):
    """The shipped slope-1.75 schedule: D1-D90 unchanged at every slope,
    D91-D100 extrapolate past the reference D100 endpoint.

    The slope-1.0 schedule is kept as an exact oracle, evaluated under an
    explicit `_slope(1.0)` override -- the same pattern `_check_golden_values`
    already uses for the slope-0.0 oracle.  Its identity claim
    ("effective_difficulty(d) == float(d) for every d in 1..100") is TRUE
    ONLY at slope 1.0 -- at any slope > 1.0 the tail extrapolates past
    float(d), so that assertion must stay scoped to the explicit patch and
    must not be run at the ambient (shipped) slope.
    """
    with _slope(1.0):
        _check_shipped_schedule_slope1_historical(results)
    _check_shipped_schedule_live(results)


def _check_shipped_schedule_slope1_historical(results):
    """Reference slope-1.0 oracle."""
    identity = [d for d in range(1, 101)
                if repr(effective_difficulty(d)) != repr(float(d))]
    _check(results, "reference slope 1.0: effective_difficulty(d) == "
                    "float(d) for every d in 1..100 (no difficulty plateau "
                    "at slope 1.0 -- identity holds ONLY at this slope, not "
                    "at the shipped slope 1.75)",
           not identity, f"not identity at {identity[:6]}")

    bad = []
    for d, g in _GOLDEN_SWEEP.items():
        want = _GOLDEN_TAIL_SLOPE1.get(d, g)
        got = (effective_difficulty(d), difficulty_t(d), difficulty_multiplier(d))
        if tuple(map(repr, got)) != tuple(map(repr, want)):
            bad.append(f"D{d}: {got!r} != {want!r}")
    for d, want in _GOLDEN_TAIL_SLOPE1.items():
        got = (effective_difficulty(d), difficulty_t(d), difficulty_multiplier(d))
        if tuple(map(repr, got)) != tuple(map(repr, want)):
            bad.append(f"D{d}: {got!r} != {want!r}")
    _check(results, "reference slope 1.0: swept levels + D91-D93 match the "
                    "slope-1.0 golden",
           not bad, "; ".join(bad[:4]))

    got_digest = _all_levels_digest()
    _check(results, "reference slope 1.0: all 100 levels hash to the "
                    "slope-1.0 golden digest",
           got_digest == _GOLDEN_ALL_LEVELS_SHA256_SLOPE1,
           f"got {got_digest}")


def _check_shipped_schedule_live(results):
    """The shipped slope 1.75, evaluated at the AMBIENT module slope -- this
    is what production runs."""
    below_knee = [d for d in range(1, DIFFICULTY_KNEE_LEVEL + 1)
                  if repr(effective_difficulty(d)) != repr(float(d))]
    _check(results, "shipped: effective_difficulty(d) == float(d) for every "
                    "d in 1..90 (D1-D90 have no plateau or extrapolation at "
                    "ANY slope)",
           not below_knee, f"not identity at {below_knee[:6]}")

    extrapolated = [
        d for d in range(DIFFICULTY_KNEE_LEVEL + 1, 101)
        if repr(effective_difficulty(d))
           != repr(DIFFICULTY_KNEE_LEVEL + (d - DIFFICULTY_KNEE_LEVEL)
                   * DIFFICULTY_TAIL_SLOPE)
    ]
    _check(results, "shipped: effective_difficulty(d) == 90 + (d-90)*1.75 "
                    "for every d in 91..100 -- the tail EXTRAPOLATES past "
                    "the reference D100 endpoint at the shipped slope "
                    "(no plateau, and not capped at float(d) either)",
           not extrapolated, f"wrong at {extrapolated[:6]}")

    bad = []
    for d, g in _GOLDEN_SWEEP.items():
        want = _GOLDEN_TAIL_SLOPE_STAR.get(d, g)
        got = (effective_difficulty(d), difficulty_t(d), difficulty_multiplier(d))
        if tuple(map(repr, got)) != tuple(map(repr, want)):
            bad.append(f"D{d}: {got!r} != {want!r}")
    for d, want in _GOLDEN_TAIL_SLOPE_STAR.items():
        got = (effective_difficulty(d), difficulty_t(d), difficulty_multiplier(d))
        if tuple(map(repr, got)) != tuple(map(repr, want)):
            bad.append(f"D{d}: {got!r} != {want!r}")
    _check(results, "shipped: swept levels + D91-D93 match the slope-1.75 "
                    "golden",
           not bad, "; ".join(bad[:4]))

    got_digest = _all_levels_digest()
    _check(results, "shipped: all 100 levels hash to the slope-1.75 golden "
                    "digest",
           got_digest == _GOLDEN_ALL_LEVELS_SHA256_SLOPE_STAR,
           f"got {got_digest}")

    _check(results, "shipped: D90 < D95 < D100 are three distinct, increasingly "
                    "harsh conditions",
           difficulty_multiplier(90) < difficulty_multiplier(95)
           < difficulty_multiplier(100),
           f"{difficulty_multiplier(90)} / {difficulty_multiplier(95)} / "
           f"{difficulty_multiplier(100)}")

    # past_knee(): False everywhere at slope 0.0, == (d > knee) at any slope > 0
    # (including the shipped 1.75 -- the property does not depend on which
    # positive slope is live).
    with _slope(0.0):
        pk0 = [d for d in range(1, 101) if past_knee(d)]
    _check(results, "past_knee is False for every level at slope 0.0",
           not pk0, f"True at {pk0[:6]}")
    wrong = [d for d in range(1, 101)
             if past_knee(d) != (d > DIFFICULTY_KNEE_LEVEL)]
    _check(results, "past_knee(d) == (d > knee) for every level at the "
                    "shipped slope (1.75)",
           not wrong, f"wrong at {wrong[:6]}")


def _check_event_schedule(results):
    """scenario_base._auto_event_schedule: severity ceiling and wave count.

    Every severity assertion below calls the real
    ``scenario_base._auto_base_severity`` / ``_auto_n_waves`` helpers rather
    than re-implementing the severity clamp locally: a local
    re-implementation can silently drift from the real formula (for example
    if the D2 severity ceiling is edited) while the wave-count comparison
    still passes, which would make the check pass against a broken ceiling.
    The D100 block additionally exercises the real ``_auto_event_schedule``
    output over many seeds so the wiring from helper to schedule is checked
    too, not just the helper in isolation.
    """
    import scenarios.scenario_base as sb

    ceiling = sb._raw_base_severity(difficulty_t(DIFFICULTY_KNEE_LEVEL - 1))
    with _slope(0.0):
        ceiling0 = sb._raw_base_severity(difficulty_t(DIFFICULTY_KNEE_LEVEL - 1))
    _check(results, "the derived severity ceiling is exactly 1.4 at slope 0.0 "
                    "and at the shipped slope (D89 is below the knee and "
                    "unaffected by the tail slope at every value)",
           repr(ceiling) == "1.4" and repr(ceiling0) == "1.4",
           f"{ceiling!r} / {ceiling0!r}")

    def base_and_waves(d):
        """The real functions, not a local re-implementation."""
        return sb._auto_base_severity(d), sb._auto_n_waves(d)

    # Reference slope 1.0 is kept as an exact oracle under an explicit
    # `_slope(1.0)` override, the same pattern
    # `_check_shipped_schedule_slope1_historical` uses.
    with _slope(1.0):
        want_hist = {85: (1.36, 8), 90: (1.40, 9), 95: (1.45, 9), 100: (1.50, 10)}
        got_hist = {d: base_and_waves(d) for d in want_hist}
        waves_hist = {d: len(sb._auto_event_schedule(150, d, 7))
                      for d in (85, 90, 95, 100)}
    _check(results, "reference slope 1.0: real _auto_base_severity/"
                    "_auto_n_waves give D85 1.36/8, D90 1.40/9, D95 1.45/9, "
                    "D100 1.50/10",
           got_hist == want_hist, f"{got_hist}")
    _check(results, "reference slope 1.0: _auto_event_schedule emits "
                    "8/9/9/10 waves at D85/D90/D95/D100",
           waves_hist == {85: 8, 90: 9, 95: 9, 100: 10}, f"{waves_hist}")

    # Shipped slope 1.75, at the ambient module slope.
    want = {85: (1.36, 8), 90: (1.40, 9), 95: (1.49, 10), 100: (1.57, 11)}
    got = {d: base_and_waves(d) for d in want}
    _check(results, "shipped: real _auto_base_severity/_auto_n_waves give "
                    "D85 1.36/8, D90 1.40/9, D95 1.49/10, D100 1.57/11",
           got == want, f"{got}")

    # The real function: wave count per level, seed-independent.
    waves = {d: len(sb._auto_event_schedule(150, d, 7)) for d in (85, 90, 95, 100)}
    _check(results, "shipped: _auto_event_schedule emits 8/9/10/11 waves at "
                    "D85/D90/D95/D100 (D100's 11th wave is a scheduled "
                    "epidemic)",
           waves == {85: 8, 90: 9, 95: 10, 100: 11}, f"{waves}")

    # Real _auto_event_schedule output, not just the helper: over many seeds
    # at D100 every emitted per-event severity must fall in
    # [base*0.85, base*1.15] (the rng.uniform jitter band around the real
    # base_severity), and at least one seed must clear the unconditional
    # ceiling 1.40 * 1.15 = 1.61 that would apply without the D2
    # severity-ceiling formula -- so a broken ceiling could not produce a
    # schedule whose severities ever reach this band.  This is a
    # non-vacuous alternative check, run IN ADDITION to the direct helper
    # assertions above, not instead of them.  At the shipped slope 1.75,
    # base100 is 1.57 -- comfortably past the 1.61 bound even before jitter
    # is applied on the high side.
    base100 = sb._auto_base_severity(100)
    sevs_d100 = [
        sev
        for seed100 in range(200)
        for _, _, _, sev in sb._auto_event_schedule(150, 100, seed100)
    ]
    lo100, hi100 = base100 * 0.85, base100 * 1.15
    _check(results, "shipped: D100 real _auto_event_schedule severities "
                    "(200 seeds) lie in [base*0.85, base*1.15] and some "
                    "exceed the unconditional ceiling 1.40*1.15=1.61",
           bool(sevs_d100)
           and all(lo100 - 1e-9 <= s <= hi100 + 1e-9 for s in sevs_d100)
           and any(s > 1.40 * 1.15 for s in sevs_d100),
           f"n={len(sevs_d100)} range=({min(sevs_d100)}, {max(sevs_d100)}) "
           f"bound=({lo100:.4f}, {hi100:.4f})")

    with _slope(0.0):
        waves0 = {d: len(sb._auto_event_schedule(150, d, 7)) for d in (90, 95, 100)}
        sev0 = {d: sb._auto_base_severity(d) for d in (90, 95, 100)}
    _check(results, "slope 0.0: D90-D100 still collapse to 9 waves at base "
                    "1.40 (reference schedule reproducible)",
           set(waves0.values()) == {9} and set(sev0.values()) == {1.40},
           f"{waves0} {sev0}")

    mono = all(base_and_waves(d) <= base_and_waves(d + 1) for d in range(1, 100))
    _check(results, "shipped: base severity and wave count are monotone over "
                    "1..100", mono)


# ---------------------------------------------------------------------------
# 2. The knee boundary
# ---------------------------------------------------------------------------

def _check_knee_boundary(results):
    with _slope(0.0):
        _check_knee_boundary_slope0(results)


def _check_knee_boundary_slope0(results):
    _check(results, "D89 is below the knee and folds to itself",
           repr(effective_difficulty(89)) == repr(89.0),
           repr(effective_difficulty(89)))
    _check(results, "D90 IS the knee and folds to itself (boundary is inclusive)",
           repr(effective_difficulty(90)) == repr(90.0),
           repr(effective_difficulty(90)))
    _check(results, "slope 0.0: D91 is above the knee and folds back to 90.0",
           repr(effective_difficulty(91)) == repr(90.0),
           repr(effective_difficulty(91)))

    _check(results, "D90 == D95 == D100 are one condition, not three (slope 0.0)",
           (repr(effective_difficulty(90)) == repr(effective_difficulty(95))
            == repr(effective_difficulty(100))),
           f"{effective_difficulty(90)!r} / {effective_difficulty(95)!r} / "
           f"{effective_difficulty(100)!r}")

    # D89 -> D90 must still be a real step, or the knee has swallowed the level
    # below it too.
    _check(results, "CONTROL: D89 and D90 are genuinely distinct conditions",
           effective_difficulty(89) != effective_difficulty(90),
           "D89 and D90 collapsed -- the knee moved down a level")

    nondecreasing = all(
        effective_difficulty(d) <= effective_difficulty(d + 1)
        for d in range(1, 100)
    )
    _check(results, "effective_difficulty is monotone non-decreasing over 1..100",
           nondecreasing)

    in_range = all(0.0 <= difficulty_t(d) <= 1.0 for d in range(1, 101))
    _check(results, "difficulty_t stays in [0, 1] over 1..100", in_range)

    exact = all(
        repr(difficulty_t(d)) == repr((effective_difficulty(d) - 1.0) / 99.0)
        for d in range(1, 101)
    )
    _check(results, "difficulty_t is exactly (effective_difficulty - 1)/99", exact)

    # The documented reason level space is primitive: the reverse round trip is
    # inexact.  If this ever becomes exact the docstring's rationale is stale.
    inexact = [d for d in range(1, 101)
               if repr(1 + ((d - 1) / 99) * 99) != repr(float(d))]
    _check(results, "the 1 + t*99 round trip is still inexact at 7 levels "
                    "(the reason level space is primitive)",
           len(inexact) == 7 and 55 in inexact and 60 in inexact,
           f"inexact at {inexact}")


# ---------------------------------------------------------------------------
# 3. Out-of-domain behaviour
# ---------------------------------------------------------------------------

def _check_out_of_domain(results):
    for d in (-1000, -5, 0):
        _check(results, f"difficulty {d} clamps to the D1 floor",
               (repr(effective_difficulty(d)) == repr(1.0)
                and repr(difficulty_t(d)) == repr(0.0)
                and repr(difficulty_multiplier(d)) == repr(0.5)),
               f"{effective_difficulty(d)!r} / {difficulty_t(d)!r} / "
               f"{difficulty_multiplier(d)!r}")

    # The INPUT clamps to the D100 level, but the OUTPUT still extrapolates
    # through the knee at whatever slope is live: effective_difficulty(>=100)
    # == 90 + 10*slope, not a hardcoded 100.0.  At the shipped slope 1.75
    # that is 107.5, not 100.0 -- this assertion is general over whatever
    # slope is live, so it holds unchanged if the shipped slope is retuned
    # without editing this check.
    for d in (101, 200, 10_000):
        _check(results, f"difficulty {d}: INPUT clamps to the D100 level; "
                        f"OUTPUT is 90 + 10*slope, not a hardcoded ceiling "
                        f"(shipped slope 1.75 -> 107.5)",
               repr(effective_difficulty(d))
               == repr(90.0 + 10.0 * difficulty_mod.DIFFICULTY_TAIL_SLOPE),
               repr(effective_difficulty(d)))
        with _slope(0.0):
            _check(results, f"slope 0.0: difficulty {d} clamps to 100 then "
                            f"folds to the knee",
                   repr(effective_difficulty(d)) == repr(90.0),
                   repr(effective_difficulty(d)))
        with _slope(2.0):
            _check(results, f"slope 2.0: difficulty {d} clamps to 100 then "
                            f"extrapolates past the knee to 110.0",
                   repr(effective_difficulty(d)) == repr(110.0),
                   repr(effective_difficulty(d)))

    # The clamp is what makes the events.py negative-infection path impossible.
    # Assert the consequence, not just the clamp.
    infection_pct = 0.01 + (effective_difficulty(-5) - 1) / 99 * 0.09
    _check(results, "the events.py infection_pct cannot go negative "
                    "at difficulty <= 0",
           infection_pct >= 0.01, f"got {infection_pct!r}")

    _check(results, "a float difficulty is truncated by int(), not rounded",
           (repr(effective_difficulty(50.9)) == repr(50.0)
            and repr(effective_difficulty(-0.5)) == repr(1.0)),
           f"{effective_difficulty(50.9)!r} / {effective_difficulty(-0.5)!r}")

    # difficulty_t's range is [0, DIFFICULTY_T_MAX], not [0, 1] -- at any
    # above-knee candidate slope, difficulty_t(100) must exceed 1.0, or the
    # extrapolation this range is meant to allow is not actually happening.
    with _slope(2.0):
        _check(results, "INVERT THE GUARD: at slope 2.0 (above the knee), "
                        "difficulty_t(100) EXCEEDS 1.0 -- the [0, 1] bound "
                        "only holds at slope 0.0",
               difficulty_t(100) > 1.0, f"got {difficulty_t(100)!r}")
        _check(results, "slope 2.0: difficulty_t(100) still fits inside "
                        "DIFFICULTY_T_MAX",
               difficulty_t(100) <= DIFFICULTY_T_MAX, f"got {difficulty_t(100)!r}")


# ---------------------------------------------------------------------------
# 4. The slope-0 no-op claim, re-derived rather than trusted
# ---------------------------------------------------------------------------

def _check_matches_pre_tuning_reference(results):
    with _slope(0.0):
        _check_matches_pre_tuning_reference_slope0(results)


def _check_matches_pre_tuning_reference_slope0(results):
    """Hand-transcribed reference arithmetic for the plateau-only (slope 0.0)
    schedule, independent of ``engine/difficulty.py``.

    This is the independent arm of the no-op claim: a transcription, not an
    import, so it cannot merely reflect a shared bug in the module under
    test.  The control below matters: the transcription MUST disagree with
    the new code somewhere (out of domain), or it is a copy rather than a
    cross-check.
    """
    plateau = 90
    plateau_t = (plateau - 1) / 99

    def old_multiplier(d):
        dd = max(1, min(plateau, int(d)))
        return round(0.50 + (dd - 1) / 99 * 1.50, 4)

    def old_t_cap(d):
        t = (max(1, min(100, int(d))) - 1) / 99
        return min(t, plateau_t)

    def old_level_cap(d):
        return min(d, 90)

    def old_recovery(d):
        return int(15 + (min(d, 90) - 1) / 99 * 35)

    bad = []
    for d in range(1, 101):
        if repr(difficulty_multiplier(d)) != repr(old_multiplier(d)):
            bad.append(f"mult D{d}")
        if repr(difficulty_t(d)) != repr(old_t_cap(d)):
            bad.append(f"t D{d}")
        if repr(effective_difficulty(d)) != repr(float(old_level_cap(d))):
            bad.append(f"eff D{d}")
        if _recovery_cycles_via_helper(d) != old_recovery(d):
            bad.append(f"recovery D{d}")
    _check(results, "slope 0.0 reproduces the reference (pre-tuning) arithmetic "
                    "at every in-domain level (400 exact-repr comparisons)",
           not bad, f"{len(bad)} mismatch(es): {bad[:6]}")

    # CONTROL.  The transcription is only evidence if it can disagree.  The two
    # documented differences are at difficulty <= 0, where events.py did not
    # clamp.  If these become equal, the transcription has been "fixed" into a
    # copy of the new code and the arm above is vacuous.
    differs = [d for d in (-5, 0)
               if repr(float(old_level_cap(d))) != repr(effective_difficulty(d))]
    _check(results, "CONTROL: the transcription still disagrees out-of-domain "
                    "at difficulty -5 and 0 (the 2 known differences)",
           differs == [-5, 0],
           f"disagreements found at {differs}, expected [-5, 0]")


# ---------------------------------------------------------------------------
# 5. The headline: tuning the slope cannot move D1-D89
# ---------------------------------------------------------------------------

def _check_slope_tuning_safety(results):
    original = difficulty_mod.DIFFICULTY_TAIL_SLOPE
    below_knee = list(range(1, DIFFICULTY_KNEE_LEVEL))          # 1..89
    above_knee = list(range(DIFFICULTY_KNEE_LEVEL + 1, 101))    # 91..100

    with _slope(0.0):
        baseline = {d: (repr(effective_difficulty(d)), repr(difficulty_t(d)),
                        repr(difficulty_multiplier(d)),
                        _recovery_cycles_via_helper(d))
                    for d in range(1, 101)}

    try:
        for slope in _CANDIDATE_SLOPES:
            difficulty_mod.DIFFICULTY_TAIL_SLOPE = slope
            # Re-validate after every constant mutation.  Every slope in
            # _CANDIDATE_SLOPES is <= 3.475 (DIFFICULTY_T_MAX's exact bound),
            # so this must never raise here; _check_t_max_guard is what
            # proves it DOES raise just above that.
            difficulty_mod.validate_schedule()

            leaked = [d for d in below_knee
                      if (repr(effective_difficulty(d)), repr(difficulty_t(d)),
                          repr(difficulty_multiplier(d)),
                          _recovery_cycles_via_helper(d)) != baseline[d]]
            _check(results,
                   f"slope {slope}: D1-D89 are bit-identical to the slope-0 "
                   f"baseline",
                   not leaked,
                   f"{len(leaked)} level(s) moved below the knee: {leaked[:6]}")

            at_knee_ok = repr(effective_difficulty(DIFFICULTY_KNEE_LEVEL)) == \
                baseline[DIFFICULTY_KNEE_LEVEL][0]
            _check(results, f"slope {slope}: D90 (the knee itself) does not move",
                   at_knee_ok, repr(effective_difficulty(DIFFICULTY_KNEE_LEVEL)))

            # CONTROL.  Without this the whole block passes if the knob is dead.
            moved = [d for d in above_knee
                     if repr(effective_difficulty(d)) != baseline[d][0]]
            _check(results,
                   f"CONTROL slope {slope}: D91-D100 DO move (the knob is wired)",
                   len(moved) == len(above_knee),
                   f"only {len(moved)}/{len(above_knee)} levels responded")

            _check(results, f"slope {slope}: difficulty_t stays in "
                            f"[0, DIFFICULTY_T_MAX] (not the narrower [0, 1] "
                            f"that holds only while slope cannot exceed 1.0)",
                   all(0.0 <= difficulty_t(d) <= DIFFICULTY_T_MAX
                       for d in range(1, 101)))

            _check(results, f"slope {slope}: effective_difficulty stays monotone "
                            f"non-decreasing",
                   all(effective_difficulty(d) <= effective_difficulty(d + 1)
                       for d in range(1, 100)))

            _check(results, f"slope {slope}: difficulty_t is still exactly "
                            f"(effective_difficulty - 1)/99",
                   all(repr(difficulty_t(d))
                       == repr((effective_difficulty(d) - 1.0) / 99.0)
                       for d in range(1, 101)))

        # At slope 1.0 the schedule must reach the true endpoint, or the knee
        # is still clipping something.
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = 1.0
        _check(results, "slope 1.0: D100 reaches the uncapped endpoint "
                        "(eff 100.0, t 1.0, mult 2.0)",
               (repr(effective_difficulty(100)) == repr(100.0)
                and repr(difficulty_t(100)) == repr(1.0)
                and repr(difficulty_multiplier(100)) == repr(2.0)),
               f"{effective_difficulty(100)!r} / {difficulty_t(100)!r} / "
               f"{difficulty_multiplier(100)!r}")
    finally:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = original

    _check(results, "the slope global was restored after the tuning sweep",
           repr(difficulty_mod.DIFFICULTY_TAIL_SLOPE) == repr(original),
           repr(difficulty_mod.DIFFICULTY_TAIL_SLOPE))


# ---------------------------------------------------------------------------
# 5b. The T_MAX guard
# ---------------------------------------------------------------------------

def _check_t_max_guard(results):
    """`validate_schedule()` raises exactly at the documented boundary.

    Exact maximum slope: difficulty_t(100) = (89 + 10*s)/99 <= 1.25 iff
    s <= 3.475.  3.475 must pass (t(100) == 1.25 exactly); 3.5 must raise
    (t(100) == 1.2525...).  Both are asserted in float, not by construction,
    so a rounding regression in DIFFICULTY_T_MAX or difficulty_t would be
    caught here.
    """
    original = difficulty_mod.DIFFICULTY_TAIL_SLOPE
    try:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = 3.475
        try:
            difficulty_mod.validate_schedule()
            raised = False
        except ValueError:
            raised = True
        _check(results, "slope 3.475: validate_schedule() does NOT raise "
                        "(exactly at DIFFICULTY_T_MAX)",
               not raised and repr(difficulty_t(100)) == repr(1.25),
               f"raised={raised} difficulty_t(100)={difficulty_t(100)!r}")

        difficulty_mod.DIFFICULTY_TAIL_SLOPE = 3.5
        try:
            difficulty_mod.validate_schedule()
            raised = False
        except ValueError as exc:
            raised = True
            message = str(exc)
        _check(results, "slope 3.5: validate_schedule() DOES raise "
                        "(just past DIFFICULTY_T_MAX)",
               raised and difficulty_t(100) > DIFFICULTY_T_MAX,
               f"raised={raised} difficulty_t(100)={difficulty_t(100)!r}")
        if raised:
            _check(results, "slope 3.5: the ValueError names the slope and "
                            "the offending difficulty_t(100) value",
                   "3.5" in message and "1.25" in message,
                   f"message: {message!r}")

        # CONTROL: an invalid slope (negative) must also raise, independent of
        # T_MAX -- otherwise the "DIFFICULTY_TAIL_SLOPE < 0" branch is dead.
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = -0.5
        try:
            difficulty_mod.validate_schedule()
            raised_neg = False
        except ValueError:
            raised_neg = True
        _check(results, "CONTROL: a negative slope raises for a different "
                        "reason (monotonicity), proving that branch is live",
               raised_neg)
    finally:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = original
        difficulty_mod.validate_schedule()

    _check(results, "the slope global was restored after the T_MAX guard "
                    "probe and re-validates clean",
           repr(difficulty_mod.DIFFICULTY_TAIL_SLOPE) == repr(original))


# ---------------------------------------------------------------------------
# 5c. No output bound binds
#
# This is the real double-knee guard.  `_check_unification`'s AST scan only
# detects a literal `min(..., 90)`; it is blind to `min(t, 1.0)`,
# `min(10, ...)` or `max(5, ...)` -- the shape a per-channel clamp takes.
# This check asserts, at the VALUE level and at every candidate slope, that
# every difficulty-scaled channel equals its un-clamped formula, so a
# re-introduced per-channel t<=1 clamp shows up as a mismatch the moment any
# candidate slope pushes t past 1.0.
# ---------------------------------------------------------------------------

#: Slopes swept by the value check.  1.0 is the no-op point for the physical
#: floors (they are inert there); 1.75 is the shipped slope; the rest are the
#: candidate-grid slopes plus the exact T_MAX bound.
_BOUND_CHECK_SLOPES = (1.0, 1.5, 1.75, 2.0, 2.5, 3.0, 3.475)


def _check_no_output_bound_binds(results):
    import scenarios.scenario_base as sb

    def real_recovery(d):
        _, recovery_src = _extract_recovery_site()
        return eval(recovery_src, {"__builtins__": {"int": int}},
                    {"d_eff": effective_difficulty(d)})

    mismatches = {n: [] for n in range(1, 9)}
    for slope in _BOUND_CHECK_SLOPES:
        with _slope(slope):
            for d in range(1, 101):
                t = difficulty_t(d)
                c = SimulationConfig.from_difficulty(d, seed=0)

                # 1. warning cycles: physical floor 1, not the old t<=1 clamp of 5.
                want = round(_lerp(20, 5, t))
                if c.event_warning_cycles != want or c.event_warning_cycles < 1:
                    mismatches[1].append((slope, d, c.event_warning_cycles, want))

                # 2. regen_mult: physical floor 0.0, not the old t<=1 clamp of 0.25.
                want = round(_lerp(1.0, 0.25, t), 4)
                if c.regen_mult != want or c.regen_mult < 0.0:
                    mismatches[2].append((slope, d, c.regen_mult, want))

                # 3. resource_density / stocks: uncapped extrapolation, still positive.
                want_density = round(_lerp(0.90, 0.40, t), 3)
                want_stock = round(_lerp(15, 5, t), 2)
                if (c.resource_density != want_density or c.resource_density <= 0
                        or c.initial_food_stock != want_stock
                        or c.initial_food_stock <= 0
                        or c.initial_water_stock != want_stock
                        or c.initial_water_stock <= 0):
                    mismatches[3].append((slope, d, c.resource_density,
                                          want_density, c.initial_food_stock,
                                          want_stock))

                # 4. event_frequency / event_severity / metabolic_rate.
                want_freq = round(_lerp(0.01, 0.06, t), 4)
                want_sev = round(_lerp(0.40, 1.20, t), 3)
                want_metab = round(_lerp(0.05, 0.25, t), 4)
                if (c.event_frequency != want_freq
                        or not (0 < c.event_frequency <= 1)
                        or c.event_severity != want_sev
                        or c.metabolic_rate != want_metab):
                    mismatches[4].append((slope, d, c.event_frequency, want_freq,
                                          c.event_severity, want_sev,
                                          c.metabolic_rate, want_metab))

                # 5. difficulty_multiplier: the real function, called, not copied.
                want_mult = round(0.50 + 1.50 * t, 4)
                if difficulty_multiplier(d) != want_mult:
                    mismatches[5].append((slope, d, difficulty_multiplier(d),
                                          want_mult))

                # 6. n_waves: THE check that goes RED if min(10, ...) comes back.
                want_waves = max(0, round(10 * t))
                if sb._auto_n_waves(d) != want_waves:
                    mismatches[6].append((slope, d, sb._auto_n_waves(d), want_waves))

                # 7. base_severity: real helper == round(raw lerp, 2), except the
                #    one documented pre-knee-ceiling exception at d == 90 exactly.
                if d == 90:
                    if sb._auto_base_severity(d) != 1.40:
                        mismatches[7].append((slope, d, sb._auto_base_severity(d),
                                              "1.40 (documented D90 exception)"))
                else:
                    want_sev2 = round(sb._raw_base_severity(t), 2)
                    if sb._auto_base_severity(d) != want_sev2:
                        mismatches[7].append((slope, d, sb._auto_base_severity(d),
                                              want_sev2))

                # 8. recovery_cycles, via the AST-extracted real expression.
                want_recovery = int(15 + 35 * t)
                if real_recovery(d) != want_recovery:
                    mismatches[8].append((slope, d, real_recovery(d), want_recovery))

    labels = {
        1: "event_warning_cycles == round(lerp(20,5,t)) and >= 1 (physical floor)",
        2: "regen_mult == round(lerp(1,.25,t),4) and >= 0.0 (physical floor)",
        3: "resource_density/stocks == uncapped lerp and > 0",
        4: "event_frequency/severity/metabolic_rate == uncapped lerp",
        5: "difficulty_multiplier(d) == round(0.50+1.50t,4) (real function)",
        6: "_auto_n_waves(d) == max(0, round(10t)) -- no ceiling clamp",
        7: "_auto_base_severity(d) == round(_raw_base_severity(t),2), D90 exception",
        8: "recovery_cycles == int(15+35t) via AST-extracted source",
    }
    for n, label in labels.items():
        bad = mismatches[n]
        _check(results, f"no-output-bound-binds [{n}]: {label} "
                        f"(slopes {_BOUND_CHECK_SLOPES}, d=1..100)",
               not bad, f"{len(bad)} mismatch(es): {bad[:4]}")

    # SENSITIVITY CONTROL: prove the sweep above actually reaches territory
    # where a per-channel clamp would bind.  Without this, checks 1/2/6
    # could pass vacuously if every slope here stayed below such a clamp's
    # thresholds.
    with _slope(2.0):
        old_floor_would_bind_1 = any(
            round(_lerp(20, 5, difficulty_t(d))) < 5 for d in range(1, 101))
        old_floor_would_bind_2 = any(
            _lerp(1.0, 0.25, difficulty_t(d)) < 0.25 for d in range(1, 101))
        old_ceiling_would_bind_6 = any(
            round(10 * difficulty_t(d)) > 10 for d in range(1, 101))
    _check(results, "SENSITIVITY CONTROL: at slope 2.0 an "
                    "event_warning_cycles floor of 5 WOULD bind "
                    "somewhere in 1..100 (proves check [1] is not vacuous)",
           old_floor_would_bind_1)
    _check(results, "SENSITIVITY CONTROL: at slope 2.0 a regen_mult "
                    "floor of 0.25 WOULD bind somewhere in 1..100 "
                    "(proves check [2] is not vacuous)",
           old_floor_would_bind_2)
    _check(results, "SENSITIVITY CONTROL: at slope 2.0 an n_waves "
                    "ceiling of 10 WOULD bind somewhere in 1..100 "
                    "(proves check [6] is not vacuous)",
           old_ceiling_would_bind_6)


# ---------------------------------------------------------------------------
# 6. Site 6 -- recovery_cycles, verified at source level
# ---------------------------------------------------------------------------

def _extract_recovery_site():
    """Pull the two real statements out of `_apply_health_dynamics` by AST.

    Reading the source rather than re-implementing it is the point: a
    re-implementation would keep passing after the real site was edited.
    Returns ``(d_eff_src, recovery_src)`` as unparsed strings.
    """
    path = os.path.join(_HERE, "engine", "simulation.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_health_dynamics":
            d_eff_src = recovery_src = None
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                        and isinstance(stmt.targets[0], ast.Name):
                    if stmt.targets[0].id == "d_eff":
                        d_eff_src = ast.unparse(stmt.value)
                    elif stmt.targets[0].id == "recovery_cycles":
                        recovery_src = ast.unparse(stmt.value)
            return d_eff_src, recovery_src
    return None, None


def _check_site6_recovery_cycles(results):
    d_eff_src, recovery_src = _extract_recovery_site()

    _check(results, "site 6: _apply_health_dynamics still assigns d_eff and "
                    "recovery_cycles",
           d_eff_src is not None and recovery_src is not None,
           f"d_eff={d_eff_src!r} recovery={recovery_src!r}")

    if d_eff_src is None or recovery_src is None:
        return

    _check(results, "site 6: d_eff comes from effective_difficulty(), not an "
                    "open-coded min(..., 90)",
           d_eff_src == "effective_difficulty(self.config.difficulty)",
           f"source reads: {d_eff_src}")

    _check(results, "site 6: the recovery_cycles expression is unchanged",
           recovery_src == "int(15 + (d_eff - 1) / 99 * 35)",
           f"source reads: {recovery_src}")

    # Evaluate the REAL extracted expression, not a copy of it, at every level.
    def real_recovery(d):
        return eval(recovery_src, {"__builtins__": {"int": int}},
                    {"d_eff": effective_difficulty(d)})

    with _slope(0.0):
        bad = [d for d in range(1, 101)
               if real_recovery(d) != _GOLDEN_RECOVERY_CYCLES[d - 1]]
    _check(results, "site 6: slope 0.0: recovery_cycles matches golden at all "
                    "100 levels",
           not bad, f"{len(bad)} mismatch(es): {bad[:6]}")

    # Site 6 open-codes `(d_eff - 1) / 99` instead of calling difficulty_t().
    # That is equivalent today; lock the equivalence so a future edit to the
    # normalisation in difficulty.py cannot leave this site behind.
    drifted = [d for d in range(1, 101)
               if real_recovery(d) != _recovery_cycles_via_helper(d)]
    _check(results, "site 6: the open-coded (d_eff-1)/99 still equals "
                    "difficulty_t() at every level",
           not drifted, f"drifted at {drifted[:6]}")

    # THE STEP-2 PROPERTY.  At slope 0 recovery is frozen at 46 from D89 up.
    # It must extend when the slope rises, or the health-recovery window stays
    # pinned at the D90 value while every other parameter moves -- a silent,
    # plausible-looking miscalibration.
    original = difficulty_mod.DIFFICULTY_TAIL_SLOPE
    try:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = 0.0
        frozen = {d: real_recovery(d) for d in (90, 95, 100)}
        _check(results, "site 6: at slope 0.0 recovery is flat across D90-D100",
               len(set(frozen.values())) == 1 and frozen[100] == 46,
               f"{frozen}")

        difficulty_mod.DIFFICULTY_TAIL_SLOPE = 1.0
        extended = {d: real_recovery(d) for d in (90, 95, 100)}
        _check(results, "site 6: at slope 1.0 recovery EXTENDS past the knee "
                        "(D90 46 -> D100 50)",
               (extended[90] == 46 and extended[100] == 50
                and extended[90] < extended[95] < extended[100]),
               f"{extended}")

        below = [d for d in range(1, 90)
                 if real_recovery(d) != _GOLDEN_RECOVERY_CYCLES[d - 1]]
        _check(results, "site 6: at slope 1.0 recovery below the knee is "
                        "unchanged",
               not below, f"moved at {below[:6]}")
    finally:
        difficulty_mod.DIFFICULTY_TAIL_SLOPE = original


# ---------------------------------------------------------------------------
# 7. Unification: imports, re-exports, and no residual open-coded knees
# ---------------------------------------------------------------------------

_EXPECTED_IMPORTERS = {
    os.path.join("engine", "simulation.py"): {"effective_difficulty", "difficulty_t",
                                              "difficulty_multiplier"},
    os.path.join("engine", "grid.py"): {"effective_difficulty"},
    os.path.join("engine", "events.py"): {"effective_difficulty"},
    os.path.join("scenarios", "scenario_base.py"): {"difficulty_t", "past_knee",
                                                    "DIFFICULTY_KNEE_LEVEL"},
}


def _check_unification(results):
    for rel, wanted in _EXPECTED_IMPORTERS.items():
        path = os.path.join(_HERE, rel)
        tree = ast.parse(open(path, encoding="utf-8").read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and \
                    node.module.endswith("difficulty"):
                imported |= {a.name for a in node.names}
        _check(results, f"{rel} imports the schedule from engine.difficulty",
               wanted <= imported, f"imports {sorted(imported)}, need {sorted(wanted)}")

    # engine/difficulty.py must stay a leaf, or the circular-import argument
    # that justifies the module's existence stops holding.
    tree = ast.parse(open(os.path.join(_HERE, "engine", "difficulty.py"),
                          encoding="utf-8").read())
    intra = [ast.unparse(n) for n in ast.walk(tree)
             if isinstance(n, ast.ImportFrom) and (n.level or 0) > 0]
    intra += [ast.unparse(n) for n in ast.walk(tree)
              if isinstance(n, ast.Import)
              and any(a.name.startswith("engine") for a in n.names)]
    _check(results, "engine/difficulty.py has no intra-package imports (leaf)",
           not intra, f"found {intra}")

    # No residual open-coded knee anywhere in executable code.
    offenders = []
    for d in _SOURCE_DIRS:
        base = os.path.join(_HERE, d)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for fn in files:
                if not fn.endswith(".py") or ".bak" in fn:
                    continue
                path = os.path.join(dirpath, fn)
                tree = ast.parse(open(path, encoding="utf-8").read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and \
                            isinstance(node.func, ast.Name) and node.func.id == "min":
                        for a in node.args:
                            if isinstance(a, ast.Constant) and a.value == 90:
                                offenders.append(
                                    f"{os.path.relpath(path, _HERE)}: "
                                    f"{ast.unparse(node)}")
    _check(results, "no open-coded min(..., 90) knee survives in engine/, "
                    "scenarios/, governments/, agents/",
           not offenders, f"{offenders}")

    # CONTROL: the scanner must actually detect one.
    probe = ast.parse("d_eff = min(self.difficulty, 90)")
    found = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "min"
                and any(isinstance(a, ast.Constant) and a.value == 90
                        for a in n.args)
                for n in ast.walk(probe))
    _check(results, "CONTROL: the open-coded-knee scanner detects a synthetic "
                    "min(self.difficulty, 90)", found)


def _check_reexports(results):
    import engine.simulation as sim
    for name in ("effective_difficulty", "difficulty_t", "difficulty_multiplier"):
        a = getattr(sim, name, None)
        b = getattr(difficulty_mod, name)
        _check(results, f"engine.simulation.{name} IS engine.difficulty.{name} "
                        f"(identical object, not a copy)",
               a is b, f"simulation has {a!r}")

    for name in ("DIFFICULTY_KNEE_LEVEL", "DIFFICULTY_TAIL_SLOPE"):
        _check(results, f"engine.simulation re-exports {name} with the same value",
               repr(getattr(sim, name, None)) == repr(getattr(difficulty_mod, name)),
               f"{getattr(sim, name, None)!r} vs "
               f"{getattr(difficulty_mod, name)!r}")

    # The import path via engine.simulation's re-export, resolved in a FRESH
    # process so this cannot pass off the already-populated module cache of
    # this one.
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {_HERE!r})
        from engine.simulation import difficulty_multiplier
        import engine.difficulty as d
        print(difficulty_multiplier is d.difficulty_multiplier,
              difficulty_multiplier.__module__,
              repr(difficulty_multiplier(90)))
    """)
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True)
    out = proc.stdout.strip()
    _check(results, "in a fresh process, `from engine.simulation import "
                    "difficulty_multiplier` resolves to engine.difficulty's object",
           out == "True engine.difficulty 1.8485",
           f"rc={proc.returncode} stdout={out!r} stderr={proc.stderr.strip()[:200]!r}")

    _check(results, "engine.difficulty.__all__ names exactly the public surface",
           set(difficulty_mod.__all__) == {
               "DIFFICULTY_KNEE_LEVEL", "DIFFICULTY_TAIL_SLOPE", "DIFFICULTY_T_MAX",
               "effective_difficulty", "past_knee", "difficulty_t",
               "difficulty_multiplier", "validate_schedule",
               "AMBIENT_HAZARD_FULL", "AMBIENT_HAZARD_RAMP_END_LEVEL",
               "ambient_hazard"},
           f"{sorted(difficulty_mod.__all__)}")


#: `ambient_hazard(d)` pinned by exact repr.  D1/D5/D10/D15 sit on the ramp;
#: D20 and above are the constant full-hazard value 0.008.
_GOLDEN_AMBIENT_HAZARD = {
    1: "0.0",
    5: "0.0016842105263157893",
    10: "0.003789473684210526",
    15: "0.005894736842105263",
    20: "0.008",
    50: "0.008",
    100: "0.008",
}

#: The ramp's full-value threshold, expressed as difficulty_t.  One ULP above
#: 19/99 by construction, so the ramp formula below is reproduced only at the
#: levels where the ramp is actually engaged (D1-D15); at and above D20,
#: ambient_hazard is the flat 0.008 constant already pinned above.
_RAMP_T_FULL = float("0.19191919191919193")
_RAMP_LEVELS = (1, 5, 10, 15)


def _check_ambient_hazard_schedule(results):
    from engine.difficulty import (
        AMBIENT_HAZARD_FULL, AMBIENT_HAZARD_RAMP_END_LEVEL, ambient_hazard,
    )
    import engine.simulation as sim_mod

    for d, want in _GOLDEN_AMBIENT_HAZARD.items():
        got = repr(ambient_hazard(d))
        _check(results, f"ambient_hazard({d}) == {want} (exact repr)", got == want,
               f"got {got}")

    ramp_bad = [d for d in _RAMP_LEVELS
                if ambient_hazard(d) != 0.008 * min(1.0, difficulty_t(d) / _RAMP_T_FULL)]
    _check(results, "bit-identical to the ramp formula at every level it covers "
                    f"{_RAMP_LEVELS}", not ramp_bad, f"differs at {ramp_bad}")

    # CONTROL: the t-space expression of the same ramp differs from the
    # level-space form by one ULP at D5, so the exact-repr pins above can
    # tell the two forms apart.
    t_form = 0.008 * min(1.0, difficulty_t(5) / difficulty_t(AMBIENT_HAZARD_RAMP_END_LEVEL))
    _check(results, "CONTROL: the t-space form differs from the level-space form at D5 "
                    "(the pins distinguish them)",
           t_form != ambient_hazard(5), f"t-form {t_form!r} == level-form")

    _check(results, "named constants: full value 0.008, ramp end D20",
           AMBIENT_HAZARD_FULL == 0.008 and AMBIENT_HAZARD_RAMP_END_LEVEL == 20,
           f"{AMBIENT_HAZARD_FULL!r}, {AMBIENT_HAZARD_RAMP_END_LEVEL!r}")

    values = [ambient_hazard(d) for d in range(1, 101)]
    _check(results, "ambient_hazard is monotone non-decreasing over D1-D100",
           all(a <= b for a, b in zip(values, values[1:])))
    _check(results, "ambient_hazard is strictly increasing over D1-D20",
           all(a < b for a, b in zip(values[:20], values[1:20])))
    not_const = [d for d in range(20, 101) if ambient_hazard(d) != AMBIENT_HAZARD_FULL]
    _check(results, "ambient_hazard == AMBIENT_HAZARD_FULL at every D >= 20",
           not not_const, f"differs at {not_const[:5]}")
    _check(results, "ambient_hazard clamps out-of-range input like every other channel",
           ambient_hazard(0) == ambient_hazard(1) == 0.0
           and ambient_hazard(150) == ambient_hazard(100))

    per_slope = {}
    for slope in (0.0, 1.0, 1.75):
        with _slope(slope):
            per_slope[slope] = [ambient_hazard(d) for d in range(1, 101)]
    _check(results, "ambient_hazard does not depend on DIFFICULTY_TAIL_SLOPE (0.0/1.0/1.75)",
           per_slope[0.0] == per_slope[1.0] == per_slope[1.75])

    cfg_bad = [d for d in range(1, 101)
               if SimulationConfig.from_difficulty(d, seed=0).ambient_hazard != ambient_hazard(d)
               or SimulationConfig(difficulty=d).ambient_hazard != ambient_hazard(d)]
    _check(results, "SimulationConfig carries ambient_hazard(d) at every level, via "
                    "from_difficulty AND via direct construction",
           not cfg_bad, f"differs at {cfg_bad[:5]}")
    _check(results, "an explicit ambient_hazard override is kept as given",
           SimulationConfig.from_difficulty(50, ambient_hazard=0.0).ambient_hazard == 0.0
           and SimulationConfig(difficulty=50, ambient_hazard=0.002).ambient_hazard == 0.002)
    _check(results, "engine.simulation.ambient_hazard IS engine.difficulty.ambient_hazard",
           getattr(sim_mod, "ambient_hazard", None) is ambient_hazard)

    for name, bad in (("AMBIENT_HAZARD_RAMP_END_LEVEL", 95),
                      ("AMBIENT_HAZARD_RAMP_END_LEVEL", 1),
                      ("AMBIENT_HAZARD_FULL", -0.001),
                      ("AMBIENT_HAZARD_FULL", 1.0)):
        original = getattr(difficulty_mod, name)
        setattr(difficulty_mod, name, bad)
        try:
            difficulty_mod.validate_schedule()
            ok, detail = False, "accepted"
        except ValueError as exc:
            ok, detail = True, str(exc)
        finally:
            setattr(difficulty_mod, name, original)
        _check(results, f"validate_schedule rejects {name}={bad!r}", ok, detail)
    difficulty_mod.validate_schedule()


def main():
    results = []
    print("Difficulty-schedule regression lock (engine/difficulty.py):")
    print("\n-- 1. golden values (shipped slope 1.75; reference slopes 1.0, 0.0) --")
    _check_golden_values(results)
    _check_shipped_schedule(results)
    print("\n-- 1b. auto event schedule ceilings --")
    _check_event_schedule(results)
    print("\n-- 2. knee boundary --")
    _check_knee_boundary(results)
    print("\n-- 3. out-of-domain --")
    _check_out_of_domain(results)
    print("\n-- 4. no-op vs the untuned (slope 0.0) baseline --")
    _check_matches_pre_tuning_reference(results)
    print("\n-- 5. slope-tuning safety (D1-D89 lock) --")
    _check_slope_tuning_safety(results)
    print("\n-- 5b. T_MAX guard --")
    _check_t_max_guard(results)
    print("\n-- 5c. no output bound binds --")
    _check_no_output_bound_binds(results)
    print("\n-- 6. site 6: recovery_cycles --")
    _check_site6_recovery_cycles(results)
    print("\n-- 7. unification and re-exports --")
    _check_unification(results)
    _check_reexports(results)
    print("\n-- 8. ambient hazard schedule --")
    _check_ambient_hazard_schedule(results)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("\nFAILURES:")
        for name, _ok, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
