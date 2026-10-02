#!/usr/bin/env python3
"""
Regression guard: the default sweep is exactly the paper's eight primary
regimes, and the registry/default-set seam still exists.

    python3 test_default_governments.py          # exit 0 = pass, 1 = fail

WHAT THIS GUARDS
-------------------------
``autocracy_lookahead`` is a full primary regime, not an ablation control, so
the default sweep is 8 governments / 16,800 runs with no excluded condition.

This file protects two things:

  1. The run-count arithmetic.  A silent drift in either direction — dropping
     to 7 governments, or registering a ninth — changes the published
     methodology, and 16,800 is the number the paper states.
  2. The registry/default seam.  ``ABLATION_GOVERNMENTS`` is empty today, so
     ``DEFAULT_GOVERNMENTS == tuple(GOVERNMENT_REGISTRY)``.  That equality is a
     coincidence of the current membership, not an invariant, and the tempting
     "simplification" is to delete ``DEFAULT_GOVERNMENTS`` and write
     ``tuple(GOVERNMENT_REGISTRY)`` at the call sites.  Doing that would mean
     merely *registering* a government enrols it in the published sweep. The
     assertions below therefore check the seam itself, not just the current
     membership.

WHY THE SEAM STILL MATTERS
-------------------------------------------
If ``run_full_simulation.py`` and ``run_quick_test.py`` read
``governments=tuple(GOVERNMENT_REGISTRY)`` directly, *merely registering* a new
government would enrol it in the published sweep. The registry is the
validation set and ``DEFAULT_GOVERNMENTS`` the default-sweep set — but the
registry-reading idiom is the obvious-looking one, and a future edit that
reintroduces it would silently change the published methodology while every
test still passed.  That risk is real even though the two sets currently have
equal membership and the difference looks like dead abstraction.

``tests/system/test_stage1_refactor.py`` carries the equivalent assertions
(``tuple(cfg.governments) == tuple(DEFAULT_GOVERNMENTS)`` and the total-run
arithmetic), derived directly from ``DEFAULT_GOVERNMENTS`` and
``benchmark_core.PAPER_K_RUNS`` so they track the default-sweep roster and
seed count automatically — not from ``GOVERNMENT_REGISTRY``, precisely
because of the seam this file guards. This file is a lighter, standalone
counterpart alongside its sibling top-level regression tests
(``test_ads_calibration.py``, ``test_difficulty_schedule.py``, ...) that
needs no test framework to run.
"""

from __future__ import annotations

import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from governments import (                                  # noqa: E402
    ABLATION_GOVERNMENTS,
    DEFAULT_GOVERNMENTS,
    GOVERNMENT_REGISTRY,
)
import run_full_simulation                                 # noqa: E402
import run_quick_test                                      # noqa: E402

#: The paper's design. Hard-coded on purpose: a derived expectation would move
#: in lockstep with the bug it is meant to catch.
EXPECTED_PRIMARY_REGIMES = 8
EXPECTED_TOTAL_RUNS = 16_800       # 8 governments x 21 difficulties x 100 seeds
EXPECTED_QUICK_RUNS = 48           # 8 governments x 3 difficulties x 2 seeds

#: The regime that must stay a primary condition rather than an ablation
#: control.  Named explicitly rather than inferred, so a regression that drops
#: it from the default sweep fails here by name instead of only as an
#: off-by-one count.
PROMOTED_REGIME = "autocracy_lookahead"


def _check(results: list, name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    results: list = []
    print("Default-sweep membership and run-count guards:")

    # --- membership ------------------------------------------------------
    _check(results, f"{PROMOTED_REGIME} is registered",
           PROMOTED_REGIME in GOVERNMENT_REGISTRY,
           f"registry={tuple(GOVERNMENT_REGISTRY)}")

    _check(results, f"{PROMOTED_REGIME} IS in DEFAULT_GOVERNMENTS (promoted)",
           PROMOTED_REGIME in DEFAULT_GOVERNMENTS,
           f"default={DEFAULT_GOVERNMENTS}")

    _check(results, f"DEFAULT_GOVERNMENTS has {EXPECTED_PRIMARY_REGIMES} regimes",
           len(DEFAULT_GOVERNMENTS) == EXPECTED_PRIMARY_REGIMES,
           f"got {len(DEFAULT_GOVERNMENTS)}")

    _check(results, "DEFAULT_GOVERNMENTS covers the whole registry today",
           set(DEFAULT_GOVERNMENTS) == set(GOVERNMENT_REGISTRY),
           f"excluded={sorted(set(GOVERNMENT_REGISTRY) - set(DEFAULT_GOVERNMENTS))}")

    _check(results, "no duplicate names in DEFAULT_GOVERNMENTS",
           len(set(DEFAULT_GOVERNMENTS)) == len(DEFAULT_GOVERNMENTS),
           f"{DEFAULT_GOVERNMENTS}")

    # --- the seam --------------------------------------------------------
    # This value check confirms the derivation currently *agrees* with the
    # registry.  It does NOT prove DEFAULT_GOVERNMENTS is still derived: a
    # hand-written literal holding today's eight names in today's order passes
    # it, and is only caught on the next registry change.  The behavioural
    # probe below is what actually catches the literal, at the moment it is
    # written.
    _check(results, "DEFAULT_GOVERNMENTS agrees with registry - ABLATION",
           tuple(DEFAULT_GOVERNMENTS) == tuple(
               n for n in GOVERNMENT_REGISTRY if n not in ABLATION_GOVERNMENTS),
           "value check only — see the re-derivation probe below")

    # Structural check on the SOURCE: is the value assigned to
    # DEFAULT_GOVERNMENTS actually computed from GOVERNMENT_REGISTRY, or is it a
    # hand-written literal?
    #
    # A substring test (`"name for name in GOVERNMENT_REGISTRY…" in source`)
    # has a false-PASS path: the derivation text satisfies it from anywhere in
    # the file, including a comment, so a literal tuple with the derivation
    # expression merely preserved in a comment would still pass. Parsing the
    # AST asks the question the name promises: it inspects the assigned
    # *node*, so comments are invisible to it and reformatting does not break
    # it.
    #
    # What it accepts: any expression that is not a bare literal and that
    # mentions GOVERNMENT_REGISTRY — the tuple(genexp) form used today, or a
    # comprehension, or a call to a helper that takes the registry.
    # What it rejects: a Tuple/List/Set literal, and any expression that never
    # references GOVERNMENT_REGISTRY at all.
    # What it still cannot catch: an expression that references the registry but
    # ignores it (e.g. `tuple(GOVERNMENT_REGISTRY) and ("anarchy", …)`). That is
    # sabotage, not drift, and the value check above plus the count checks would
    # catch any version of it that changes the result.
    _init_path = os.path.join(_HERE, "governments", "__init__.py")
    try:
        _tree = ast.parse(open(_init_path, encoding="utf-8").read())
        _rhs = None
        for _node in ast.walk(_tree):
            _targets = ([_node.target] if isinstance(_node, ast.AnnAssign)
                        else getattr(_node, "targets", []))
            for _t in _targets:
                if isinstance(_t, ast.Name) and _t.id == "DEFAULT_GOVERNMENTS":
                    _rhs = _node.value
        if _rhs is None:
            _derived, _why = False, "no DEFAULT_GOVERNMENTS assignment found"
        elif isinstance(_rhs, (ast.Tuple, ast.List, ast.Set, ast.Constant)):
            _derived, _why = False, f"assigned a bare {type(_rhs).__name__} literal"
        else:
            _names = {n.id for n in ast.walk(_rhs) if isinstance(n, ast.Name)}
            _derived = "GOVERNMENT_REGISTRY" in _names
            _why = (f"{type(_rhs).__name__} referencing {sorted(_names)}"
                    if _derived else
                    f"{type(_rhs).__name__} never references GOVERNMENT_REGISTRY")
    except Exception as exc:                                  # noqa: BLE001
        _derived, _why = False, f"could not parse {_init_path}: {exc!r}"

    _check(results, "DEFAULT_GOVERNMENTS is DERIVED from the registry in source (AST)",
           _derived,
           _why + " — a hand-written tuple stops tracking the registry silently")

    _check(results, "every excluded name (if any) is still registered",
           all(a in GOVERNMENT_REGISTRY for a in ABLATION_GOVERNMENTS),
           f"excluded={sorted(ABLATION_GOVERNMENTS)}")

    # --- entry points ----------------------------------------------------
    for label, module in (("run_full_simulation", run_full_simulation),
                          ("run_quick_test", run_quick_test)):
        cfg = module.build_config()
        _check(results, f"{label}.build_config() defaults to DEFAULT_GOVERNMENTS",
               tuple(cfg.governments) == tuple(DEFAULT_GOVERNMENTS),
               f"got {tuple(cfg.governments)}")
        _check(results,
               f"{label} default sweep includes {PROMOTED_REGIME}",
               PROMOTED_REGIME in cfg.governments,
               f"got {tuple(cfg.governments)}")
        _check(results,
               f"{label} sweeps {EXPECTED_PRIMARY_REGIMES} governments",
               len(cfg.governments) == EXPECTED_PRIMARY_REGIMES,
               f"got {len(cfg.governments)}")

    full_cfg = run_full_simulation.build_config()
    _check(results, f"full sweep is {EXPECTED_TOTAL_RUNS:,} runs",
           full_cfg.total_runs == EXPECTED_TOTAL_RUNS,
           f"got {full_cfg.total_runs:,}")

    quick_cfg = run_quick_test.build_config()
    _check(results, f"quick test is {EXPECTED_QUICK_RUNS} runs",
           quick_cfg.total_runs == EXPECTED_QUICK_RUNS,
           f"got {quick_cfg.total_runs}")

    # --- the seam, probed rather than assumed ----------------------------
    # The checks above cannot tell `governments=DEFAULT_GOVERNMENTS` apart from
    # `governments=tuple(GOVERNMENT_REGISTRY)`, because with an empty exclusion
    # set the two are equal.  That is precisely the regression this file exists
    # to catch, so probe it directly: temporarily rebind each entry point's
    # DEFAULT_GOVERNMENTS to a short list and confirm build_config() follows it.
    # A call site reading the registry instead would ignore the rebind and
    # return all eight.
    probe = ("anarchy", "ads")
    for label, module in (("run_full_simulation", run_full_simulation),
                          ("run_quick_test", run_quick_test)):
        saved = module.DEFAULT_GOVERNMENTS
        try:
            module.DEFAULT_GOVERNMENTS = probe
            got = tuple(module.build_config().governments)
        finally:
            module.DEFAULT_GOVERNMENTS = saved
        _check(results,
               f"{label}.build_config() reads DEFAULT_GOVERNMENTS, not the registry",
               got == probe,
               f"rebound to {probe}, got {got}")

    # --- selectability ---------------------------------------------------
    # Every registered name must be reachable via --governments, whether or not
    # it is in the default sweep.
    from benchmark_core import parse_government_list
    for name in GOVERNMENT_REGISTRY:
        try:
            parsed = parse_government_list(name)
            ok = parsed == (name,)
        except Exception as exc:                      # noqa: BLE001
            ok, parsed = False, repr(exc)
        _check(results, f"--governments {name} is accepted", ok, f"got {parsed}")

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
