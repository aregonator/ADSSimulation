#!/usr/bin/env python3
"""
Regression guard: the seed lattice and the environment fingerprint.

    python3 test_seed_derivation.py          # exit 0 = pass, 1 = fail

WHAT THIS PROTECTS
------------------
Every random stream in the benchmark is derived from one root integer by
``engine.scenario_plan.derive_seed``.  Three separate things can go wrong there,
and all three are silent:

  1. **The derivation changes.**  Swap blake2b for sha256, reorder the payload,
     stop masking to 63 bits — the code still runs, every sweep still completes,
     and every archived run in ``benchmark_results/`` now describes an
     environment that cannot be reproduced.  The GOLDEN_SEEDS table
     below pins specific inputs to specific outputs so that change cannot
     happen without a test failing.  If you are here because these failed and
     you MEANT to re-key the corpus: that decision invalidates the archive and
     must be made deliberately and documented, not absorbed silently by
     updating the constants.

  2. **The derivation stops being process-stable.**  ``hash()`` is the obvious
     shortcut and is randomised per process by ``PYTHONHASHSEED``; the sweep
     starts its workers with ``spawn``, so two workers would derive different
     "identical" environments and the fairness property would fail without any
     visible symptom.  The subprocess checks re-derive under several explicit
     hash seeds and compare.

  3. **The lattice keys drift.**  Which keys each stream depends on IS the
     experimental design: the environment must NOT depend on the government
     (or governments are not comparable) and MUST depend on the run index (or
     n counts replicates).  Those are asserted directly, on
     ``ScenarioPlan.fingerprint()`` — the same value the sweep audits — rather
     than on the seeds alone, so a plan that ignores its seed would also be
     caught.

WHY THIS FILE LIVES HERE, AND NOT IN tests/
-------------------------------------------
This file follows the precedent set by ``test_default_governments.py``: it
sits beside the code it guards, alongside the project's other top-level
regression tests, imports the way the entry points do, and runs under a
plain ``python3`` with no pytest and no import gymnastics.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from engine.scenario_plan import derive_seed                  # noqa: E402
from benchmark_core import (                                  # noqa: E402
    BenchmarkConfig,
    FingerprintAuditError,
    build_plan,
    verify_env_fingerprints,
)


# ---------------------------------------------------------------------------
# Golden values
#
# (base_seed, domain, parts) -> expected derived seed.
#
# Hard-coded on purpose.  A derived expectation ("compute it the same way and
# compare") would move in lockstep with the bug it is meant to catch, which is
# the entire failure mode here: a re-keying refactor is self-consistent by
# construction and only detectable against values captured BEFORE it.
#
# Generated 2026-09-18 from the reference implementation, on CPython 3.12.
# ---------------------------------------------------------------------------
GOLDEN_SEEDS = {
    # Minimal case: no parts at all.  Pins the trailing separator in the
    # payload format, which is otherwise invisible and easy to "tidy away".
    (0, "env", ()): 7304608822140095016,

    # The environment stream, at two difficulties and the same run index.
    # These two differing is requirement 2 (environments vary) expressed at the
    # seed level; their equality would mean d=50/run=3 and d=95/run=3 share a
    # world — a population-seed collision.
    (42, "env", (50, 3)): 1979897192167990243,
    (42, "env", (95, 3)): 7798145959078841231,

    # Population, same lattice keys as the environment, different domain.
    # MUST differ from the "env" value at the same keys — that difference is
    # what domain separation buys.
    (42, "pop", (50, 3)): 2167880937326965897,

    # Government streams DO take the government name as a key.
    (42, "gov", ("ads", 50, 3)): 5159075378115916770,
    (42, "gov", ("democracy", 50, 3)): 3023815998204188478,
    (42, "gov.eval", ("ads", 50, 3)): 87982768449197522,

    # Plan sub-streams, derived off the plan's own seed with no extra keys.
    (42, "env.grid", ()): 7552513344929634149,
    (42, "env.placement", ()): 1454483229624110184,
    (42, "env.events.random", ()): 235996514964546313,

    # Scheduled events are indexed by position in the schedule, so each one
    # draws an independent seed; these two values differing confirms events at
    # different positions do not seed the same cell.
    (42, "env.events.sched", (50, 3, 0)): 8290614668007568592,
    (42, "env.events.sched", (50, 3, 1)): 5336227717844068763,

    # Boundary inputs: a negative root and a root above 2**63.  Both are legal
    # Python ints and both must produce an in-range seed rather than raising or
    # returning something a consumer would narrow.
    (-1, "env", (1, 0)): 2293749317502731473,
    (2 ** 63, "env", (1, 0)): 8340897662251217416,
}


def _check(results: list, name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _check_golden_values(results: list) -> None:
    print("\nGolden values (pin the derivation against silent re-keying):")
    for (base, domain, parts), expected in GOLDEN_SEEDS.items():
        got = derive_seed(base, domain, *parts)
        _check(
            results,
            f"derive_seed({base}, {domain!r}, *{parts})",
            got == expected,
            f"expected {expected}, got {got}",
        )


def _check_range_and_types(results: list) -> None:
    print("\nRange and input validation:")
    mask = (1 << 63) - 1
    values = [
        derive_seed(s, "env", d, k)
        for s in (0, 42, -7)
        for d in (1, 50, 100)
        for k in range(10)
    ]
    _check(
        results, "every derived seed is in [0, 2**63)",
        all(0 <= v <= mask for v in values),
        f"min={min(values)} max={max(values)}",
    )
    # Not a cryptographic claim — just that the low bits are not being thrown
    # away by a masking mistake, which would make whole families of runs alias.
    _check(
        results, "90 lattice points give 90 distinct seeds",
        len(set(values)) == len(values),
        f"{len(set(values))} distinct of {len(values)}",
    )

    for bad_base, label in ((4.0, "float"), ("42", "str"), (True, "bool")):
        try:
            derive_seed(bad_base, "env")
            ok, detail = False, "accepted silently"
        except TypeError as exc:
            ok, detail = True, f"TypeError: {exc}"
        except Exception as exc:                              # noqa: BLE001
            ok, detail = False, f"wrong exception type: {exc!r}"
        _check(results, f"a {label} base_seed is rejected", ok, detail)

    for bad_domain, label in (("", "empty"), (None, "None")):
        try:
            derive_seed(42, bad_domain)
            ok, detail = False, "accepted silently"
        except ValueError as exc:
            ok, detail = True, f"ValueError: {exc}"
        except Exception as exc:                              # noqa: BLE001
            ok, detail = False, f"wrong exception type: {exc!r}"
        _check(results, f"a {label} domain is rejected", ok, detail)


def _check_domain_separation(results: list) -> None:
    print("\nDomain separation and key sensitivity:")
    _check(
        results, "env and pop differ at the same lattice point",
        derive_seed(42, "env", 50, 3) != derive_seed(42, "pop", 50, 3),
    )
    _check(
        results, "gov differs per government",
        derive_seed(42, "gov", "ads", 50, 3)
        != derive_seed(42, "gov", "democracy", 50, 3),
    )
    _check(
        results, "gov and gov.eval are independent streams",
        derive_seed(42, "gov", "ads", 50, 3)
        != derive_seed(42, "gov.eval", "ads", 50, 3),
    )
    _check(
        results, "part ORDER is significant",
        derive_seed(42, "gov", "ads", 50, 3)
        != derive_seed(42, "gov", 50, 3, "ads"),
    )
    _check(
        results, "the root seed is significant",
        derive_seed(42, "env", 50, 3) != derive_seed(43, "env", 50, 3),
    )
    # The specific aliasing that arithmetic keying (base_seed + run_idx) cannot
    # avoid and that this function exists to eliminate.
    _check(
        results, "no aliasing across the (difficulty, run) lattice",
        len({
            derive_seed(42, "env", d, k)
            for d in range(1, 101, 5) for k in range(100)
        }) == 20 * 100,
    )


def _check_cross_process_stability(results: list) -> None:
    print("\nCross-process stability (PYTHONHASHSEED must not matter):")
    probe = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %r)
        from engine.scenario_plan import derive_seed
        print(derive_seed(42, "env", 50, 3))
        print(derive_seed(42, "gov", "ads", 50, 3))
        """
    ) % _HERE

    expected = [
        str(derive_seed(42, "env", 50, 3)),
        str(derive_seed(42, "gov", "ads", 50, 3)),
    ]
    for hash_seed in ("0", "1", "12345", "random"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        try:
            out = subprocess.run(
                [sys.executable, "-c", probe],
                env=env, capture_output=True, text=True, timeout=120, check=True,
            ).stdout.split()
            ok, detail = out == expected, f"got {out}"
        except Exception as exc:                              # noqa: BLE001
            ok, detail = False, f"probe failed: {exc!r}"
        _check(
            results, f"identical under PYTHONHASHSEED={hash_seed}", ok, detail
        )


def _tiny_config(**overrides) -> BenchmarkConfig:
    """A config small enough that build_plan is milliseconds, not seconds."""
    kwargs = dict(
        governments=("ads",),
        difficulties=(50,),
        k_runs=3,
        grid_size=8,
        n_agents=10,
        max_cycles=20,
        max_steps=2,
        output_root=os.path.join(_HERE, "_seed_derivation_scratch"),
        base_seed=42,
        label="seed derivation guard",
    )
    kwargs.update(overrides)
    return BenchmarkConfig(**kwargs)


def _check_plan_fingerprint(results: list) -> None:
    print("\nScenarioPlan.fingerprint — the fairness/variation witness:")
    cfg = _tiny_config()

    plan = build_plan(cfg, 50, 0)
    _check(
        results, "fingerprint is stable across repeated calls",
        plan.fingerprint() == plan.fingerprint(),
    )
    _check(
        results, "fingerprint is 16 hex characters",
        len(plan.fingerprint()) == 16
        and all(ch in "0123456789abcdef" for ch in plan.fingerprint()),
        plan.fingerprint(),
    )

    # FAIRNESS at the source: build_plan does not take a government, so two
    # builds at the same coordinate must agree byte for byte.  This is the
    # property the whole per-worker-derivation design rests on.
    _check(
        results, "two independent builds at (d, k) are identical",
        build_plan(cfg, 50, 0).fingerprint()
        == build_plan(cfg, 50, 0).fingerprint(),
    )

    # VARIATION: distinct run indices, distinct environments.
    per_run = {k: build_plan(cfg, 50, k).fingerprint() for k in range(6)}
    _check(
        results, "6 run indices give 6 distinct environments",
        len(set(per_run.values())) == 6,
        f"{per_run}",
    )
    _check(
        results, "the same run index at another difficulty differs",
        build_plan(cfg, 50, 0).fingerprint()
        != build_plan(cfg, 95, 0).fingerprint(),
    )
    _check(
        results, "a different base_seed gives a different environment",
        build_plan(cfg, 50, 0).fingerprint()
        != build_plan(_tiny_config(base_seed=43), 50, 0).fingerprint(),
    )

    # A fingerprint must survive the archive round-trip, or the standalone
    # re-audit of a published dataset is checking a different object.
    from engine.scenario_plan import ScenarioPlan
    _check(
        results, "fingerprint survives a to_dict/from_dict round-trip",
        ScenarioPlan.from_dict(plan.to_dict()).fingerprint() == plan.fingerprint(),
    )

    # And across processes, for the same reason the seeds must.
    probe = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %r)
        from benchmark_core import BenchmarkConfig, build_plan
        cfg = BenchmarkConfig(
            governments=("ads",), difficulties=(50,), k_runs=3, grid_size=8,
            n_agents=10, max_cycles=20, max_steps=2,
            output_root=%r, base_seed=42, label="probe",
        )
        print(build_plan(cfg, 50, 0).fingerprint())
        """
    ) % (_HERE, cfg.output_root)
    for hash_seed in ("0", "999"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        try:
            out = subprocess.run(
                [sys.executable, "-c", probe],
                env=env, capture_output=True, text=True, timeout=300, check=True,
            ).stdout.strip()
            ok, detail = out == plan.fingerprint(), f"got {out!r}"
        except Exception as exc:                              # noqa: BLE001
            ok, detail = False, f"probe failed: {exc!r}"
        _check(
            results,
            f"fingerprint identical in a subprocess (PYTHONHASHSEED={hash_seed})",
            ok, detail,
        )


def _check_audit(results: list) -> None:
    print("\nverify_env_fingerprints — the audit must actually fail:")

    def rows(spec):
        return [
            {"government": g, "difficulty": d, "run": r, "env_fingerprint": f}
            for g, d, r, f in spec
        ]

    good = rows([
        ("ads", 50, 1, "aaaa"), ("democracy", 50, 1, "aaaa"),
        ("ads", 50, 2, "bbbb"), ("democracy", 50, 2, "bbbb"),
        ("ads", 95, 1, "cccc"), ("democracy", 95, 1, "cccc"),
        ("ads", 95, 2, "dddd"), ("democracy", 95, 2, "dddd"),
    ])
    try:
        summary = verify_env_fingerprints(good, k_runs=2)
        ok, detail = summary["status"] == "pass", str(summary["status"])
    except Exception as exc:                                  # noqa: BLE001
        ok, detail = False, f"raised on a valid dataset: {exc!r}"
    _check(results, "a fair, varying dataset passes", ok, detail)

    # One government saw a different world at (50, 1).
    unfair = rows([
        ("ads", 50, 1, "aaaa"), ("democracy", 50, 1, "XXXX"),
        ("ads", 50, 2, "bbbb"), ("democracy", 50, 2, "bbbb"),
    ])
    try:
        verify_env_fingerprints(unfair, k_runs=2)
        ok, detail = False, "returned instead of raising"
    except FingerprintAuditError as exc:
        ok, detail = True, str(exc)[:60]
    except Exception as exc:                                  # noqa: BLE001
        ok, detail = False, f"wrong exception type: {exc!r}"
    _check(results, "a FAIRNESS violation raises", ok, detail)

    # Both runs saw the same world — the VARIATION invariant this check
    # exists to catch.
    non_varying = rows([
        ("ads", 50, 1, "aaaa"), ("democracy", 50, 1, "aaaa"),
        ("ads", 50, 2, "aaaa"), ("democracy", 50, 2, "aaaa"),
    ])
    try:
        verify_env_fingerprints(non_varying, k_runs=2)
        ok, detail = False, "returned instead of raising"
    except FingerprintAuditError as exc:
        ok, detail = True, str(exc)[:60]
    except Exception as exc:                                  # noqa: BLE001
        ok, detail = False, f"wrong exception type: {exc!r}"
    _check(results, "a VARIATION violation raises", ok, detail)

    # A short cell must NOT be reported as a variation failure: dropped runs
    # are the short_cells mechanism's business, and conflating the two would
    # turn a reported partial sweep into a spurious fairness abort.
    short = rows([
        ("ads", 50, 1, "aaaa"), ("democracy", 50, 1, "aaaa"),
        ("ads", 50, 2, "bbbb"),
    ])
    try:
        summary = verify_env_fingerprints(short, k_runs=5)
        ok, detail = summary["status"] == "pass", str(summary["status"])
    except Exception as exc:                                  # noqa: BLE001
        ok, detail = False, f"raised on a legitimately short cell: {exc!r}"
    _check(results, "a short cell passes and is reported, not failed", ok, detail)

    # An archive with no fingerprints recorded at all; the audit must say so
    # rather than pass vacuously.
    legacy = [{"government": "ads", "difficulty": 50, "run": 1}]
    try:
        summary = verify_env_fingerprints(legacy, k_runs=1)
        ok = summary["status"] == "not_applicable"
        detail = str(summary["status"])
    except Exception as exc:                                  # noqa: BLE001
        ok, detail = False, f"raised: {exc!r}"
    _check(results, "a fingerprint-less dataset reports not_applicable", ok, detail)


def main() -> int:
    results: list = []
    print("Seed-derivation and environment-fingerprint guards:")
    _check_golden_values(results)
    _check_range_and_types(results)
    _check_domain_separation(results)
    _check_cross_process_stability(results)
    _check_plan_fingerprint(results)
    _check_audit(results)

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
