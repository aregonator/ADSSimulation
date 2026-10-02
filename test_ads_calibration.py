#!/usr/bin/env python3
"""
Regression guards for the ADS government's parameter-inference machinery and
the shared lookahead code it drives.

    python3 test_ads_calibration.py          # exit 0 = pass, 1 = fail

WHAT THIS GUARDS, AND WHY EACH ONE IS HERE
------------------------------------------
Every assertion below corresponds to a property that is either (a) invisible
from the output of a run that "worked", or (b) the kind of thing a plausible
future simplification would break silently.  None of them is a smoke test —
``run_quick_test.py`` already covers "does it run".

  1. THE ESTIMATOR GENERALISES THE ONE ALREADY IN THE TREE.  ``ShrunkRatioEstimator``
     at ``prior_mean=0, prior_cycles=20`` on unit exposure must reproduce
     ``EventSystem.observed_event_stats``' rate formula exactly.  This is what
     licenses the claim that the shrinkage estimator generalises the existing
     pattern rather than replacing it with something merely similar — and it
     is why ``observed_event_stats`` itself is left un-refactored.

  2. RECOVERY ON A SYNTHETIC WORLD.  Driven from a known multiplier, the
     estimator must converge to it and must approach from the prior side
     without overshoot.

  3. ONCE PER CYCLE, EXACTLY.  ``observe()`` must raise on a repeated cycle and
     on a skipped one.  A silently double-counted or missed cycle biases every
     estimate with no symptom, which makes this the most important defensive
     check in the design.

  4. DEGENERATE CLOSURES ARE GRADED, NOT SKIPPED.  A forecast of exactly 0.0 —
     a group projected extinct — cannot simply be excluded from the multiplier
     update, because ``realized / 0`` crashes.  With no ratio left to compute
     there is nothing to skip, so it must be graded like any other closure and
     still counted.

  5. ``autocracy_lookahead`` RECEIVES THE CRUDE ENVIRONMENT.  THE HEADLINE
     REGRESSION TEST IN THIS FILE.  If this arm silently received the full
     evidence-based event statistics AND the real ``drain_mult`` /
     ``regen_mult``, "foresight without inference" would be false and the
     middle arm of the three-way comparison would be invalidated.  The failure
     mode is invisible — it looks like unusually good control results.

  6. COMMON RANDOM NUMBERS.  Every candidate scored in one decision round must
     face the bit-identical phantom future, or a candidate can win on a luckier
     draw.  Also load-bearing for the crude model: setting every rate to 0.0
     must leave the injection stream's CONSUMPTION unchanged.

  7. THE T+20 REVIEW DATE, and that closure is decoupled from law lifecycle.

  8. THE SCIENTIFIC-VALIDITY FIREWALL, STATIC AND DYNAMIC.  ``observed_event_stats``
     must never see ``scheduled_events``; and a live ADS run must complete
     without touching ``sim.drain_mult``, ``sim.config.regen_mult``,
     ``sim.config.max_steps_per_cycle`` or ``sim.config.difficulty``.  The
     dynamic half is strictly stronger than the static one.

  9. REAL EVENTS ARE TIME-BOUNDED IN THE ROLLOUT.  A storm with two cycles left
     must stop hurting after two fake cycles, not persist for the full horizon.
     This is especially load-bearing in the crude rollout, where real events
     are the ONLY events.

 10. THE MOVEMENT ESTIMATOR IS SAFE.  It must never exceed the true cap (the
     one-sided error that would matter, since over-estimating mobility credits
     relocation laws with travel that cannot happen), must be monotone, and must
     stay pinned at 1 when the true cap is 1.

 11. THE CANARY.  A completed run's summary must carry
     ``ads_forecast_schema_version``.  Its absence proves the ``getattr``
     duck-typed reporting chain is broken — a failure that otherwise vanishes
     silently rather than raising.

TESTS THIS FILE DELIBERATELY DOES NOT CONTAIN
----------------------------------------------
These are omitted on purpose rather than overlooked:

  * **No assertion that the estimates match the config.**  ``regen_mult_hat``
    and ``drain_mult_hat`` are expected to DIFFER from ``sim.config.regen_mult``
    and ``sim.drain_mult``, for stated structural reasons (unmodelled global
    depletion; a coefficient mismatch between the evaluator and the engine).
    Such a test would look reasonable, fail, and invite someone to "repair" a
    correct estimator into a broken one.
  * **No convergence deadline for the movement estimator.**  Whether it reaches
    the true cap depends on some agent actually exhausting its move budget,
    which is a coverage property of the scenario, not of the estimator.
  * **No "ADS's first round is bit-identical to autocracy_lookahead's".**  That
    holds only before any event has been observed; the narrow three-field form
    is asserted in section 5 instead.

WHY THIS FILE LIVES HERE, AND NOT IN tests/
-------------------------------------------
Same pattern as ``test_default_governments.py`` and ``test_seed_derivation.py``:
this file sits beside the code it guards, alongside the project's other
top-level regression tests, and imports the way the entry points do — no
pytest or test-framework dependency required to run it.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: Imported as a MODULE, not just for its names: the partition-assignment
#: check's positive controls patch `_scope_key` / `_outcome_key` on the module
#: object, and rebinding a name imported `from ... import` would patch this
#: test's copy and leave the emitter's own lookup untouched — a control that
#: silently never fires.
import governments.ads as _ads_mod                             # noqa: E402
import governments.forecast_ledger as _fl                      # noqa: E402

from engine.events import (                                   # noqa: E402
    EVENT_RATE_CAP,
    EVENT_RATE_PRIOR_CYCLES,
    ActiveEvent,
    EventType,
)
from engine.scenario_plan import (                            # noqa: E402
    build_simulation_from_plan,
    derive_seed,
    generate_scenario_plan,
)
from engine.grid import Terrain                                # noqa: E402
from engine.simulation import SimulationConfig                # noqa: E402
from governments.ads import (                                 # noqa: E402
    ADS_FORECAST_SCHEMA_VERSION,
    LOOKAHEAD_CYCLES,
    PHANTOM_CATEGORIES,
    AdsGovernment,
    EvaluatorNode,
    ProposedLaw,
)
from governments.autocracy_lookahead import (                 # noqa: E402
    AutocracyLookaheadGovernment,
)
from governments.parameter_inference import (                 # noqa: E402
    CRUDE_ENVIRONMENT,
    MOVEMENT_PRIOR_CEILING,
    EnvironmentModel,
    ParameterEstimator,
    ShrunkRatioEstimator,
)
from governments.forecast_ledger import (                     # noqa: E402
    _CROSS_ROUNDING_TOL,
    ForecastLedger,
)
from benchmark_core import (                                  # noqa: E402
    _ADS_GOV_KEY,
    _CALIBRATION_GOV_KEYS,
    _filtered_cycle_deltas,
    _run_calibration_half_split,
    _run_calibration_mean,
    CROSS_ARM_OUTCOMES,
    filtered_calibration_mean,
    filtered_half_split,
    load_scope_matched_calibration_deltas,
)
from scenarios.scenario_base import (                         # noqa: E402
    _make_citizen_population as make_citizen_population,
)

#: The ledger logs; a bare logger with a NullHandler keeps the partition-
#: assignment and filtered-mean-enforcement checks' synthetic fixtures silent
#: without suppressing anything the real runs emit.
_NULL_LOGGER = logging.getLogger("test_ads_calibration.null")
_NULL_LOGGER.addHandler(logging.NullHandler())
_NULL_LOGGER.propagate = False


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _check(results: list, name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))


def _raises(exc_type, fn) -> bool:
    """True iff *fn* raises *exc_type*.

    Deliberately does NOT swallow other exception types: a guard that was
    supposed to raise ValueError and instead raised AttributeError has a
    different bug, and silently reporting "it raised, good" would hide it.
    """
    try:
        fn()
    except exc_type:
        return True
    return False


#: Deliberately larger than ``run_quick_test.py``'s 30 cycles.  A prediction
#: opened at the first decision round (cycle 5) closes at cycle 25, so a
#: 30-cycle run exercises only the earliest closures and would let a broken
#: T+20 trigger pass unnoticed.  60 cycles gives every early bucket several
#: closures and is still under two seconds.
_TEST_CYCLES = 60


def _build_sim(government, *, difficulty: int = 50, seed: int = 4242,
               n_agents: int = 24, grid: int = 12, cycles: int = _TEST_CYCLES,
               steps: int = 3):
    """A small but structurally real simulation, built the way the sweep does.

    *steps* is ``max_steps_per_cycle``.  The default of 3 matches
    ``run_quick_test.py``; the movement-estimator section overrides it, because
    that estimator's whole behaviour is a function of this value and testing it
    at one setting would prove nothing.  Note that ``from_difficulty`` emits 1
    and every real entry point overrides it — 10 in production — so 1 is NOT
    the production path despite being the dataclass default.
    """
    sim_config = SimulationConfig.from_difficulty(
        difficulty,
        seed=derive_seed(seed, "env", difficulty, 0),
        max_steps_per_cycle=steps,
        grid_rows=grid,
        grid_cols=grid,
        num_agents=n_agents,
        max_cycles=cycles,
    )
    plan = generate_scenario_plan(sim_config)
    agents = make_citizen_population(
        n_agents, seed=derive_seed(seed, "pop", difficulty, 0)
    )
    return build_simulation_from_plan(
        plan, government, agents, verbose=False, record_audit_trail=False,
    )


def _run(sim, cycles: int = _TEST_CYCLES) -> None:
    for cycle in range(cycles):
        sim.cycle = cycle
        sim._step(cycle)


# ---------------------------------------------------------------------------
# 1. The shrinkage estimator generalises the one already in the tree
# ---------------------------------------------------------------------------

def _check_estimator_reduction(results: list) -> None:
    print("\n1. ShrunkRatioEstimator reduces to observed_event_stats' rate formula:")

    # THE LICENCE FOR NOT REFACTORING observed_event_stats.
    #
    # engine/events.py computes `rate = n / (cycle + EVENT_RATE_PRIOR_CYCLES)`.
    # ShrunkRatioEstimator computes
    # `(prior_mean*K + sum_obs) / (K + sum_exp)` with
    # `K = prior_cycles * (sum_exp / n_cycles)`.  Substituting prior_mean=0 and
    # one unit of exposure per cycle gives K = prior_cycles and hence
    # `n / (prior_cycles + n_cycles)` — the same formula.
    #
    # This is asserted rather than argued because the alternative was to
    # re-point observed_event_stats at the new class, and its exact arithmetic
    # is load-bearing for the published corpus.  Proving the correspondence buys
    # the confidence without taking the risk.
    #
    # NOTE FOR ANYONE EXTENDING THIS: `cycle` here is the NUMBER OF OBSERVATIONS
    # FOLDED IN, not the highest cycle index.  Observing cycles 0..c inclusive is
    # c+1 observations and reduces to n/(c+1+20), not n/(c+20).  An earlier draft
    # of this test looped `range(cycle+1)` and appeared to show a broken
    # estimator; the estimator was correct and the harness was off by one.
    exact = True
    rows = []
    for n_events, n_obs in ((0, 5), (1, 3), (3, 10), (7, 40), (11, 99), (1, 1)):
        est = ShrunkRatioEstimator("probe", 0.0, EVENT_RATE_PRIOR_CYCLES, 0.0, 1.0)
        fired = 0
        for c in range(n_obs):
            observed = 1.0 if fired < n_events else 0.0
            fired += int(observed)
            est.observe(c, observed, 1.0)
        expected = fired / (n_obs + EVENT_RATE_PRIOR_CYCLES)
        if est.value != expected:
            exact = False
            rows.append(f"n={fired} obs={n_obs}: {est.value!r} != {expected!r}")
    _check(results, "matches n/(cycle+prior) exactly on a table of (n, cycle)",
           exact, "; ".join(rows) if rows else "6 cases, exact equality")

    # No evidence -> the prior, exactly.  This is the structural elimination of
    # the divide-by-zero case: there is no guard anywhere because there is no
    # path to guard.
    fresh = ShrunkRatioEstimator("probe", 1.0, 20.0, 0.0, 5.0)
    _check(results, "an unobserved estimator returns exactly its prior",
           fresh.value == 1.0, f"got {fresh.value!r}")
    zero_exposure = ShrunkRatioEstimator("probe", 1.0, 20.0, 0.0, 5.0)
    for c in range(5):
        zero_exposure.observe(c, 0.0, 0.0)
    _check(results, "five zero-exposure cycles still return exactly the prior",
           zero_exposure.value == 1.0 and zero_exposure.n_cycles == 5,
           f"value={zero_exposure.value!r}, n_cycles={zero_exposure.n_cycles}")


# ---------------------------------------------------------------------------
# 2. Recovery of a known multiplier on a synthetic world
# ---------------------------------------------------------------------------

def _check_estimator_recovery(results: list) -> None:
    print("\n2. The estimator recovers a known multiplier:")

    # A synthetic world with a KNOWN answer, which the live simulation cannot
    # provide: there, the estimand is deliberately not any config constant (see
    # parameter_inference's module docstring), so "did it converge to the right
    # number" is only askable here.
    #
    # READ THIS BEFORE TIGHTENING THE TOLERANCE.  A shrinkage estimator does NOT
    # reach the truth in finite samples and is not supposed to.  It retains
    # prior weight `prior_cycles / (prior_cycles + n)` forever, so at n = 400
    # with prior_cycles = 20 it still sits 4.8% of the way from the truth toward
    # the prior.  That is the bias-variance trade the shrinkage exists to make:
    # the same property that stops one freak early cycle from dominating the
    # estimate also stops the estimate from ever fully forgetting the prior.
    #
    # So this asserts the CLOSED FORM exactly, and consistency separately.  An
    # earlier draft asserted `abs(value - truth) < 1e-9` and failed for three of
    # four multipliers — the estimator was correct and the assertion encoded a
    # false expectation.  Asserting the closed form is also strictly stronger
    # than a loose tolerance would be: it pins the exact arithmetic, so a change
    # to the shrinkage rule cannot hide inside a tolerance band.
    def _closed_form(prior_mean, prior_cycles, truth, n):
        return (prior_mean * prior_cycles + truth * n) / (prior_cycles + n)

    for truth in (0.5, 1.0, 1.85, 2.4):
        est = ShrunkRatioEstimator("probe", 1.0, 20.0, 0.0, 5.0)
        trajectory = []
        for c in range(400):
            exposure = 0.05 + (c % 7) * 0.01      # varying, always positive
            est.observe(c, truth * exposure, exposure)
            trajectory.append(est.value)

        expected = _closed_form(1.0, 20.0, truth, 400)
        _check(results, f"truth={truth}: matches the shrinkage closed form exactly",
               math.isclose(est.value, expected, rel_tol=0.0, abs_tol=1e-12),
               f"got {est.value:.12f}, closed form {expected:.12f}")

        # CONSISTENCY: the residual bias must shrink toward zero as evidence
        # accumulates.  This is the property that actually matters for the
        # design's claim that "inference gets more accurate as cycles
        # accumulate", and it is what a tolerance-based test was groping at.
        errors = []
        for n in (50, 200, 1000, 5000):
            e = ShrunkRatioEstimator("probe", 1.0, 20.0, 0.0, 5.0)
            for c in range(n):
                exposure = 0.05 + (c % 7) * 0.01
                e.observe(c, truth * exposure, exposure)
            errors.append(abs(e.value - truth))
        shrinking = all(a > b for a, b in zip(errors, errors[1:])) if truth != 1.0 else True
        _check(results, f"truth={truth}: the residual bias shrinks as n grows",
               shrinking,
               "n=50,200,1000,5000 -> " + ", ".join(f"{e:.6f}" for e in errors))

        # Monotone approach from the prior, no overshoot.  A shrinkage estimator
        # is a weighted average of prior and evidence, so every intermediate
        # value must lie between them; an overshoot would mean the weights went
        # negative, the signature of an arithmetic sign error.
        lo, hi = min(1.0, truth), max(1.0, truth)
        outside = [v for v in trajectory if not (lo - 1e-12 <= v <= hi + 1e-12)]
        _check(results, f"truth={truth}: never leaves [prior, truth] while converging",
               not outside,
               f"{len(outside)} excursion(s), e.g. {outside[:1]}" if outside
               else f"all 400 values within [{lo}, {hi}]")


# ---------------------------------------------------------------------------
# 3. observe() must run exactly once per cycle
# ---------------------------------------------------------------------------

def _check_once_per_cycle(results: list) -> None:
    print("\n3. A repeated or skipped cycle raises rather than biasing silently:")

    # THE SINGLE MOST IMPORTANT DEFENSIVE CHECK IN THE DESIGN.  A
    # double-counted cycle inflates n and biases every later estimate; a skipped
    # cycle loses per-cycle evidence that cannot be reconstructed, and for the
    # regen diff specifically would attribute two cycles of regrowth to one
    # cycle of exposure.  Neither has any symptom at the output.
    est = ShrunkRatioEstimator("probe", 1.0, 20.0, 0.0, 5.0)
    est.observe(0, 1.0, 1.0)
    est.observe(1, 1.0, 1.0)
    for label, cycle in (("repeated", 1), ("backwards", 0)):
        try:
            est.observe(cycle, 1.0, 1.0)
            ok, detail = False, "no exception raised"
        except ValueError as exc:
            ok, detail = True, type(exc).__name__
        _check(results, f"ShrunkRatioEstimator rejects a {label} cycle", ok, detail)

    _check(results, "a negative exposure raises",
           _raises(ValueError, lambda: ShrunkRatioEstimator(
               "p", 1.0, 20.0, 0.0, 5.0).observe(0, 1.0, -1.0)),
           "ValueError")

    # A negative OBSERVATION is clamped and counted rather than raising: unlike
    # a negative exposure (an arithmetic error in our own model) it can arise
    # from a broken channel, and losing the run over it is the worse trade.
    clamping = ShrunkRatioEstimator("probe", 1.0, 20.0, 0.0, 5.0)
    clamping.observe(0, -0.5, 1.0)
    _check(results, "a negative observation is clamped to 0 and counted",
           clamping.n_clamped == 1 and clamping.sum_observed == 0.0,
           f"n_clamped={clamping.n_clamped}, sum_observed={clamping.sum_observed}")

    # And at the ParameterEstimator level, which is where the contiguity rule
    # actually lives: a GAP must raise too, not merely a repeat.
    gov = AdsGovernment(seed=71, eval_seed=71)
    sim = _build_sim(gov, cycles=6)
    for cycle in range(4):
        sim.cycle = cycle
        sim._step(cycle)
    _check(results, "ParameterEstimator rejects a repeated cycle",
           _raises(ValueError, lambda: gov._estimator.observe(3, sim)), "ValueError")
    _check(results, "ParameterEstimator rejects a skipped cycle",
           _raises(ValueError, lambda: gov._estimator.observe(9, sim)), "ValueError")


# ---------------------------------------------------------------------------
# 4. Degenerate closures are graded, not skipped
# ---------------------------------------------------------------------------

def _check_degenerate(results: list) -> None:
    print("\n4. A predicted == 0.0 closure is GRADED, not skipped:")

    # Under an additive-error scheme, a zero forecast has no ratio to compute,
    # so there is nothing to skip: such a closure is graded exactly like any
    # other, contributes its error to the series, and the counter survives only
    # as a diagnostic.  A test asserting "a zero forecast leaves the bucket
    # untouched" would assert the wrong invariant for this scheme.
    gov = AdsGovernment(seed=5, eval_seed=5)
    sim = _build_sim(gov, cycles=30)
    gov._sim = sim

    # Drive a single closure through `gov._ledger`, with a forecast of exactly
    # 0.0 — the value reachable whenever a whole evaluated group is projected
    # dead.  The entry is opened through the real `open()` rather than
    # assembled as a literal dict, so a hand-built entry cannot drift out of
    # step with the shape the emitter actually produces.  Opening at
    # `7 - LOOKAHEAD_CYCLES` puts the review date on cycle 7 by the horizon
    # arithmetic rather than by a hard-coded `close_cycle`.
    measured = [a.agent_id for a in sim.agents[:4]]
    gov._ledger.open(
        cycle=7 - LOOKAHEAD_CYCLES,
        predicted=0.0,
        evaluated_agent_ids=list(measured),
        tags={
            "law_id": "D1",
            "outcome": "enacted",
            "n_groups": 1,
            "category": "health",
            "law_type": "EPIDEMIC_RESPONSE",
            "applies_to": list(measured),
            "adjusted_at_enactment": 0.0,
        },
    )
    before_closed = gov._ledger.closed_total
    try:
        gov._ledger.close(7, sim, {law.law_id for law in gov.active_laws})
        crashed = None
    except Exception as exc:                                  # noqa: BLE001
        crashed = repr(exc)

    _check(results, "a zero forecast does not raise", crashed is None,
           crashed or "no exception")
    _check(results, "a zero forecast is COUNTED as closed (not skipped)",
           gov._ledger.closed_total == before_closed + 1,
           f"closed_total {before_closed} -> {gov._ledger.closed_total}")
    _check(results, "a zero forecast contributes to the error series",
           len(gov._ledger.observations) == 1
           and gov._ledger.observations[0][0] == 7,
           f"observations={gov._ledger.observations}")
    _check(results, "a zero forecast increments the degenerate counter",
           gov._ledger.n_degenerate == 1, f"n={gov._ledger.n_degenerate}")

    # The recorded error must be the real gap, not zero: these agents are alive
    # and healthy, so a forecast of 0.0 was badly wrong and must be scored as
    # badly wrong.  Under the old scheme this observation was discarded, which
    # flattered the reported accuracy.
    realized = sum(a.health for a in sim.agents[:4] if a.alive) / len(measured)
    recorded = gov._ledger.observations[0][1]
    _check(results, "the recorded error is abs(0 - realized), not 0",
           math.isclose(recorded, abs(0.0 - realized), rel_tol=0, abs_tol=1e-12),
           f"recorded={recorded:.6f}, expected={abs(0.0 - realized):.6f}")

    # And end to end at the hardest difficulty, where projected extinction is
    # most likely to actually occur.
    gov6 = AdsGovernment(seed=11, eval_seed=11)
    sim6 = _build_sim(gov6, difficulty=100)
    try:
        _run(sim6)
        ok, detail = True, (f"closed={gov6._ledger.closed_total}, "
                            f"degenerate={gov6._ledger.n_degenerate}")
    except Exception as exc:                                  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    _check(results, "a full hard-difficulty run completes cleanly", ok, detail)


# ---------------------------------------------------------------------------
# 5. autocracy_lookahead has ADS's foresight and none of its organisation.
#
# `autocracy_lookahead` is DELIBERATELY GRANTED the same evidence-based
# foresight as ADS — its own estimator, its own ledger, real event statistics
# rather than the crude environment with rates pinned at 0.0.  This is what
# lets the three-arm chain
#     autocracy (no foresight)
#       -> autocracy_lookahead (good foresight, no organisation)
#       -> ads (good foresight + organisation)
# isolate COORDINATION instead of confounding coordination with information
# quality: the middle arm must differ from ADS on organisation alone, not on
# what it knows.
#
# Same spy harness, same coverage-independent instrument as the rest of this
# suite, so the tree stays guarded on this axis.  What this check now
# primarily protects is the two dimensions where the arm remains deliberately
# impoverished relative to ADS: the flat target group and the hand-authored
# menu.
# ---------------------------------------------------------------------------

def _check_lookahead_control(results: list) -> None:
    print("\n5. autocracy_lookahead has ADS's foresight and none of its "
          "organisation:")

    gov = AutocracyLookaheadGovernment(seed=21)

    # EPISTEMIC PARITY.  autocracy_lookahead owns the same evidence-based
    # machinery ADS does:
    _check(results, "it constructs its own ParameterEstimator",
           isinstance(getattr(gov, "_estimator", None), ParameterEstimator),
           f"{type(getattr(gov, '_estimator', None)).__name__}")
    _check(results, "it constructs its own ForecastLedger",
           isinstance(getattr(gov, "_ledger", None), ForecastLedger),
           f"{type(getattr(gov, '_ledger', None)).__name__}")

    # INSTANCE OWNERSHIP.  "By construction" is the kind of guarantee that
    # survives until someone adds a module-level cache; sharing either object
    # would destroy both the ablation and the reproducibility story.
    _ads_probe = AdsGovernment(seed=21, eval_seed=21)
    _check(results, "its estimator is NOT the ADS's instance",
           gov._estimator is not _ads_probe._estimator, "distinct instances")
    _check(results, "its ledger is NOT the ADS's instance",
           gov._ledger is not _ads_probe._ledger, "distinct instances")

    # STRUCTURAL: ADS ORGANISATION does not exist here, so there is nothing to
    # drift and nothing a later "make this configurable" commit could switch
    # on.  This is the primary invariant this section guards.
    #
    # `_predictions_due` / `_calib_observations` / `_record_prediction` /
    # `_close_predictions` are not asserted absent here: those names exist on
    # NEITHER government (both delegate to `self._ledger`), so asserting their
    # absence would be vacuous — it would pass against a regime that had grown
    # the whole ledger.
    for attr in ("_calibration", "_hg_node"):
        _check(results, f"AutocracyLookaheadGovernment has no {attr}",
               not hasattr(gov, attr), f"hasattr -> {hasattr(gov, attr)}")
    for meth in ("_update_calibration", "propose_laws"):
        _check(results, f"AutocracyLookaheadGovernment has no {meth}",
               not hasattr(gov, meth), f"hasattr -> {hasattr(gov, meth)}")

    # It must NOT join the `get_ads_metrics` hook.  That hook feeds
    # `final_mean_prediction_error`, which is by construction an UNFILTERED
    # all-time mean — and this arm grades `retained` / `repealed` / `no_action`
    # closures ADS never grades, so the number would be a population mismatch
    # dressed as a comparison.  The absence also keeps this arm's
    # health_stats.csv byte-identical, which is what makes its no-op gate
    # assertable at all.  This is a deliberate asymmetry with a stated reason,
    # not an oversight.
    _check(results, "it does NOT implement get_ads_metrics",
           not hasattr(gov, "get_ads_metrics"),
           f"hasattr -> {hasattr(gov, 'get_ads_metrics')}")

    # The two reporting hooks move from "must be absent" to "must be present and
    # SHAPE-IDENTICAL to ADS's" — strictly stronger than a hasattr check.
    _check(results, "its calibration record is shape-identical to ADS's",
           set(gov.get_calibration_record(0))
           == set(_ads_probe.get_calibration_record(0)),
           f"al={sorted(gov.get_calibration_record(0))}")
    _check(results, "its calibration summary is shape-identical to ADS's",
           set(gov.get_calibration_summary(50))
           == set(_ads_probe.get_calibration_summary(50)),
           f"symmetric_difference="
           f"{sorted(set(gov.get_calibration_summary(50)) ^ set(_ads_probe.get_calibration_summary(50)))}")
    _check(results, "neither arm's summary emits final_mean_prediction_error",
           "final_mean_prediction_error" not in gov.get_calibration_summary(50)
           and "final_mean_prediction_error"
           not in _ads_probe.get_calibration_summary(50),
           "absent from both (it belongs to MetricsCollector.summary())")

    params = gov.get_params()
    ads_params = _ads_probe.get_params()
    _check(results, "it reports environment_model='inferred' in its params",
           params.get("environment_model") == "inferred",
           str(params.get("environment_model")))
    _check(results, "it reports parameter_inference_enabled=True",
           params.get("parameter_inference_enabled") is True,
           str(params.get("parameter_inference_enabled")))
    _check(results, "it reports organization='none'",
           params.get("organization") == "none", str(params.get("organization")))
    # THE SYMMETRY REGRESSION GUARD.  The failure mode this guards against is an
    # ASYMMETRY — this arm describing its own epistemics while ADS does not —
    # so asserting only this arm's side would re-admit exactly that failure.
    # Both arms, one check.
    _check(results, "ADS reports the same three header fields (symmetry guard)",
           ads_params.get("environment_model") == "inferred"
           and ads_params.get("parameter_inference_enabled") is True
           and ads_params.get("organization") == "multinode",
           f"ads: env={ads_params.get('environment_model')}, "
           f"inference={ads_params.get('parameter_inference_enabled')}, "
           f"org={ads_params.get('organization')}")
    _check(results, "it reports the shared evaluator's horizon",
           params.get("lookahead_cycles") == LOOKAHEAD_CYCLES,
           str(params.get("lookahead_cycles")))

    # THE DIRECT ASSERTION.  Capture the EnvironmentModel each regime actually
    # hands the evaluator, on a run long enough that several events have been
    # observed: the two regimes must receive materially different
    # environments, since that difference is exactly what the ablation rests
    # on.
    def _captured_envs(government, cycles=_TEST_CYCLES):
        seen = []
        rounds = []
        real_evaluate = government._evaluator.evaluate

        def _spy(proposal, group_agents, sim, *, cycle=None, env=CRUDE_ENVIRONMENT):
            seen.append(env)
            return real_evaluate(proposal, group_agents, sim, cycle=cycle, env=env)

        government._evaluator.evaluate = _spy       # type: ignore[assignment]
        s = _build_sim(government, cycles=cycles)
        for c in range(cycles):
            s.cycle = c
            s._step(c)
            # Snapshot each round as it happens: _last_decision_details is
            # overwritten on the next round, so reading it only at the end
            # would check one round instead of all of them.
            details = government._last_decision_details
            if details and details.get("cycle") == c:
                rounds.append(details)
        return seen, s, rounds

    ctrl_envs, ctrl_sim, ctrl_rounds = _captured_envs(AutocracyLookaheadGovernment(seed=22))
    ads_gov = AdsGovernment(seed=23, eval_seed=23)
    ads_envs, ads_sim, _ = _captured_envs(ads_gov)

    _check(results, "both regimes actually scored candidates (probe is not vacuous)",
           bool(ctrl_envs) and bool(ads_envs),
           f"control={len(ctrl_envs)} evaluations, ads={len(ads_envs)}")

    _check(results, "autocracy_lookahead's evaluator only ever sees "
                    "source='inferred'",
           all(e.source == "inferred" for e in ctrl_envs),
           f"sources={sorted({e.source for e in ctrl_envs})}")
    _check(results, "ADS's evaluator only ever sees source='inferred'",
           all(e.source == "inferred" for e in ads_envs),
           f"sources={sorted({e.source for e in ads_envs})}")

    # THE SUBSTANTIVE HALF.  `source` is a diagnostic tag that nothing branches
    # on, so a test checking only the tag would pass even if the numbers had
    # not actually moved.  These assertions check the numbers directly: rates
    # are not all 0.0, and the multiplier triple is not pinned at crude
    # (1.0, 1.0, 1).
    ctrl_rates = {
        float(stats.get("rate") or 0.0)
        for env in ctrl_envs for stats in env.event_stats.values()
    }
    _check(results, "the control's event rates are NOT all 0.0 "
                    "(it now injects phantom events)",
           ctrl_rates != {0.0}, f"rates seen: {sorted(ctrl_rates)[:6]}")
    ctrl_triples = {
        (e.drain_mult, e.regen_mult, e.max_steps_per_cycle) for e in ctrl_envs
    }
    _check(results, "the control's multipliers are NOT pinned at crude "
                    "(1.0, 1.0, 1) — the estimator moved",
           ctrl_triples != {(1.0, 1.0, 1)},
           f"{len(ctrl_triples)} distinct triples, e.g. "
           f"{sorted(ctrl_triples)[:2]}")

    # And the control that proves the comparison is not vacuous: ADS must
    # actually have learned something nonzero over the same run, or "the two
    # regimes differ" would be satisfied by an estimator that never moved.
    ads_nonzero_rate = any(
        float(stats.get("rate") or 0.0) > 0.0
        for env in ads_envs for stats in env.event_stats.values()
    )
    _check(results, "ADS did observe a nonzero event rate (test is not vacuous)",
           ads_nonzero_rate,
           "at least one category had rate > 0" if ads_nonzero_rate
           else "no events observed in this run — lengthen it")

    # CRUDE/INFERRED IDENTITY AT ZERO EVIDENCE.
    #
    # Comparing only the three scalars against CRUDE_* would be nearly
    # tautological, since `ParameterEstimator.__init__` sets its priors FROM
    # those very constants and could only fail if someone edited one constant
    # and not the other.
    #
    # The form below asserts the property that actually matters: at zero
    # evidence the two models are numerically interchangeable END TO END —
    # equal on the event-statistics TABLE as well as the scalars, and producing
    # a BIT-IDENTICAL rollout score.  `event_stats` legitimately diverges the
    # moment an event is observed, which is why this is pinned at cycle 0 and
    # why the control below is mandatory rather than decorative.
    fresh_gov = AdsGovernment(seed=24, eval_seed=24)
    fresh_sim = _build_sim(fresh_gov, cycles=4)
    at_zero = fresh_gov._estimator.environment_model(fresh_sim, 0)
    _check(results, "before any cycle, ADS's model equals CRUDE on all three scalars",
           (at_zero.drain_mult == CRUDE_ENVIRONMENT.drain_mult
            and at_zero.regen_mult == CRUDE_ENVIRONMENT.regen_mult
            and at_zero.max_steps_per_cycle == CRUDE_ENVIRONMENT.max_steps_per_cycle),
           f"drain={at_zero.drain_mult}, regen={at_zero.regen_mult}, "
           f"steps={at_zero.max_steps_per_cycle}")

    # The half the narrow version missed: the event-statistics table.
    _zero_cats = sorted(set(at_zero.event_stats) | set(CRUDE_ENVIRONMENT.event_stats))
    _zero_diffs = []
    for _cat in _zero_cats:
        _a = dict(at_zero.event_stats.get(_cat) or {})
        _b = dict(CRUDE_ENVIRONMENT.event_stats.get(_cat) or {})
        for _key in sorted(set(_a) | set(_b)):
            if _a.get(_key) != _b.get(_key):
                _zero_diffs.append(f"{_cat}.{_key}: {_a.get(_key)!r} vs {_b.get(_key)!r}")
    _check(results, "at zero evidence the event-statistics tables are "
                    "field-for-field identical",
           not _zero_diffs,
           "; ".join(_zero_diffs) if _zero_diffs
           else f"{len(_zero_cats)} categories, all fields equal")

    # The behavioural form: a full rollout must score bit-identically, which is
    # the property the deleted invariant actually guaranteed.
    _z_living = [a for a in fresh_sim.agents if a.alive][:8]
    _z_probe = ProposedLaw(law_type="MANDATORY_SHELTER", params={}, description="p",
                           source_category="terrain", duration=10,
                           applies_to=[a.agent_id for a in _z_living])
    _z_inferred = EvaluatorNode(eval_seed=24).evaluate(
        _z_probe, _z_living, fresh_sim, cycle=0, env=at_zero)
    _z_crude = EvaluatorNode(eval_seed=24).evaluate(
        _z_probe, _z_living, fresh_sim, cycle=0, env=CRUDE_ENVIRONMENT)
    _check(results, "at zero evidence a full rollout scores bit-identically under "
                    "inferred vs crude",
           _z_inferred == _z_crude, f"{_z_inferred!r} vs {_z_crude!r}")

    # NON-VACUITY CONTROL — without this the equalities above are worthless,
    # since they would also hold if the estimator never moved or if `evaluate`
    # ignored `env` entirely.  After evidence accumulates the two MUST diverge.
    # `ads_envs` is the tail of the 30-cycle ADS run captured above, so this
    # reuses evidence already gathered rather than paying for another run.
    _late_env = ads_envs[-1]
    _l_living = [a for a in ads_sim.agents if a.alive][:8]
    _late_differs = None
    if _l_living:
        _l_probe = ProposedLaw(law_type="MANDATORY_SHELTER", params={}, description="p",
                               source_category="terrain", duration=10,
                               applies_to=[a.agent_id for a in _l_living])
        _l_inf = EvaluatorNode(eval_seed=25).evaluate(
            _l_probe, _l_living, ads_sim, cycle=ads_sim.cycle, env=_late_env)
        _l_cru = EvaluatorNode(eval_seed=25).evaluate(
            _l_probe, _l_living, ads_sim, cycle=ads_sim.cycle, env=CRUDE_ENVIRONMENT)
        _late_differs = _l_inf != _l_cru
    _check(results, "CONTROL: once evidence exists the two models score DIFFERENTLY "
                    "(so the zero-evidence equality is a real invariant)",
           _late_differs is True,
           f"inferred={_l_inf!r} crude={_l_cru!r} (est drain={_late_env.drain_mult:.4f} "
           f"regen={_late_env.regen_mult:.4f} steps={_late_env.max_steps_per_cycle})"
           if _late_differs is not None else "no living agents to score")

    # The environment model is shared by reference across a round, so it must be
    # impossible for one candidate to perturb it for the next.
    _check(results, "EnvironmentModel is frozen against per-candidate mutation",
           _raises(Exception, lambda: setattr(ads_envs[0], "drain_mult", 99.0)),
           "assignment raises")

    # The evaluator itself holds no regime state, which is what makes sharing it
    # between the two arms safe in the first place.
    ev_attrs = [a for a in vars(gov._evaluator)
                if "calib" in a.lower() or "estimat" in a.lower()]
    _check(results, "the shared EvaluatorNode holds no calibration/estimator state",
           not ev_attrs, f"found {ev_attrs}" if ev_attrs else "none")

    # Behavioural: every recorded ranking score equals raw_score / population
    # exactly — the property that would break if anyone reintroduced a
    # correction term.
    mismatches = []
    n_scored = 0
    for details in ctrl_rounds:
        k = max(1, int(details.get("population", 1)))
        for decision in details.get("decisions", []):
            for row in decision.get("candidates_scored", decision.get("scored", [])):
                raw, norm = row.get("raw_score"), row.get("norm_score")
                if raw is None or norm is None:
                    continue
                n_scored += 1
                if not math.isclose(norm, raw / k, rel_tol=0, abs_tol=5e-4):
                    mismatches.append((raw, norm, raw / k))
    _check(results, "every recorded norm_score equals raw_score / population",
           not mismatches and n_scored > 0,
           f"{n_scored} scores checked, {len(mismatches)} mismatched"
           + (f", e.g. {mismatches[0]}" if mismatches else ""))


# ---------------------------------------------------------------------------
# 6. Common random numbers within a decision round
# ---------------------------------------------------------------------------

def _check_common_random_numbers(results: list) -> None:
    print("\n6. Every candidate in one decision round faces the same phantom future:")

    gov = AdsGovernment(seed=31, eval_seed=31)
    sim = _build_sim(gov, cycles=24)
    _run(sim, cycles=24)

    ev = gov._evaluator
    # Hand-built environment models are the only way to hand the evaluator a
    # stats table directly.  `source` is diagnostic and nothing
    # branches on it, so "inferred" here just labels these as evidence-based.
    stats_env = EnvironmentModel(
        event_stats=sim.event_system.observed_event_stats(sim.cycle),
        drain_mult=1.0, regen_mult=1.0, max_steps_per_cycle=1, source="inferred",
    )
    living = [a for a in sim.agents if a.alive][:6]
    if not living:
        _check(results, "the CRN probe has living agents to score", False, "none alive")
        return

    # Two structurally different proposals, scored at the same cycle.  Their
    # scores will differ; their phantom schedules must not.
    p1 = ProposedLaw(law_type="FOOD_RATION", params={"max_per_cycle": 2.0},
                     description="probe A", source_category="food",
                     applies_to=[a.agent_id for a in living], duration=5)
    p2 = ProposedLaw(law_type="MANDATORY_SHELTER", params={},
                     description="probe B", source_category="terrain",
                     applies_to=[a.agent_id for a in living], duration=12)

    ev.evaluate(p1, living, sim, cycle=sim.cycle, env=stats_env)
    sched_a = list(ev.last_phantom_summary["injected"])
    ev.evaluate(p2, living, sim, cycle=sim.cycle, env=stats_env)
    sched_b = list(ev.last_phantom_summary["injected"])
    # Same proposal, different group size — the case that would break if the
    # injection draws shared the epidemic-spread stream, whose consumption
    # count varies with group size.
    ev.evaluate(p1, living[:2], sim, cycle=sim.cycle, env=stats_env)
    sched_c = list(ev.last_phantom_summary["injected"])

    _check(results, "two different proposals at one cycle share a phantom schedule",
           sched_a == sched_b, f"A={sched_a} B={sched_b}")
    _check(results, "two different group sizes at one cycle share a phantom schedule",
           sched_a == sched_c, f"A={sched_a} C={sched_c}")

    # Different rounds must NOT share, or the "stochastic" injection is a fixed
    # schedule in disguise.  Compared over several cycles because any two
    # individual cycles can legitimately coincide when the rate is low.
    schedules = []
    for c in range(5, 25):
        ev.evaluate(p1, living, sim, cycle=c, env=stats_env)
        schedules.append(tuple(ev.last_phantom_summary["injected"]))
    _check(results, "different decision rounds draw different phantom futures",
           len(set(schedules)) > 1,
           f"{len(set(schedules))} distinct schedules over {len(schedules)} cycles")

    # Replay: the same seed and cycle must reproduce exactly, or "stochastic but
    # reproducible" is only the first half.
    ev2 = EvaluatorNode(eval_seed=31)
    ev2.evaluate(p1, living, sim, cycle=12, env=stats_env)
    replay_a = list(ev2.last_phantom_summary["injected"])
    ev3 = EvaluatorNode(eval_seed=31)
    ev3.evaluate(p1, living, sim, cycle=12, env=stats_env)
    replay_b = list(ev3.last_phantom_summary["injected"])
    _check(results, "the same (eval_seed, cycle) replays bit-for-bit",
           replay_a == replay_b, f"{replay_a} vs {replay_b}")
    ev4 = EvaluatorNode(eval_seed=32)
    ev4.evaluate(p1, living, sim, cycle=12, env=stats_env)
    _check(results, "a different eval_seed gives a different stream",
           ev4._event_rng.random() != ev3._event_rng.random(), "streams differ")

    # Draw count must be candidate-independent: exactly one uniform per category
    # per fake cycle, drawn even when the rate is zero.  If a zero-rate category
    # skipped its draw, stream position would depend on the statistics and the
    # identity above would hold only by luck.
    # Uses the REAL CRUDE_ENVIRONMENT rather than a hand-written zero-rate dict,
    # which makes this simultaneously the CRN guard it always was and a proof
    # that the crude regime consumes the injection stream identically to the
    # inferred one.  That equality is what lets autocracy_lookahead's numbers
    # differ from ADS's without re-keying its RNG stream.
    probe = EvaluatorNode(eval_seed=77)
    probe.evaluate(p1, living, sim, cycle=3, env=CRUDE_ENVIRONMENT)
    expected_draws = len(PHANTOM_CATEGORIES) * LOOKAHEAD_CYCLES
    reference = EvaluatorNode(eval_seed=77)
    reference._event_rng.seed(derive_seed(77, "eval.round", 3))
    for _ in range(expected_draws):
        reference._event_rng.random()
    _check(results, "exactly one draw per category per fake cycle, even at rate 0",
           probe._event_rng.random() == reference._event_rng.random(),
           f"expected {expected_draws} draws consumed")
    _check(results, "a zero-rate world injects nothing",
           probe.last_phantom_summary["n_injected"] == 0,
           str(probe.last_phantom_summary["n_injected"]))

    # The diagnostic must never report a stale cycle.  `evaluate` returns early
    # on an empty group; if the summary kept the PREVIOUS call's schedule in
    # that case, the once-per-round EVAL_PHANTOM log would show two different
    # cycles apparently sharing a phantom future — exactly the signature of a
    # seeding bug, even though the underlying injection is correct and only the
    # log would be lying.  Pinned because a misleading diagnostic costs more
    # than a missing one.
    stale = EvaluatorNode(eval_seed=88)
    stale.evaluate(p1, living, sim, cycle=5, env=stats_env)
    at_five = dict(stale.last_phantom_summary)
    stale.evaluate(p1, [], sim, cycle=6, env=stats_env)
    after_empty = stale.last_phantom_summary
    _check(results, "an empty-group evaluate re-stamps the diagnostic cycle",
           after_empty.get("cycle") == 6 and at_five.get("cycle") == 5,
           f"empty-group summary cycle={after_empty.get('cycle')}")
    _check(results, "an empty-group evaluate reports no phantom schedule",
           after_empty.get("injected") == []
           and after_empty.get("skipped") == "empty_group",
           f"injected={after_empty.get('injected')}, "
           f"skipped={after_empty.get('skipped')}")


# ---------------------------------------------------------------------------
# 7. The T+20 review date
# ---------------------------------------------------------------------------

def _check_review_date(results: list) -> None:
    print(f"\n7. Predictions close exactly {LOOKAHEAD_CYCLES} cycles after enactment:")

    gov = AdsGovernment(seed=41, eval_seed=41)
    sim = _build_sim(gov)

    opened: dict = {}
    closed: list = []
    # This is the only test pinning that closure is decoupled from the law
    # lifecycle, and that arithmetic lives in `gov._ledger` — a module boundary
    # worth spying on directly.  The spies wrap the ledger's two mutators,
    # `open` and `close`.
    real_open = gov._ledger.open

    def _spy_open(**kwargs):
        real_open(**kwargs)
        opened[kwargs["tags"]["law_id"]] = kwargs["cycle"]
    gov._ledger.open = _spy_open                              # type: ignore[assignment]

    # Closures are observed by wrapping `close` and reading the bucket it is
    # about to drain, through the ledger's own `due_law_ids` accessor rather
    # than its private dict.  This is a stronger test of the REVIEW DATE than
    # spying on an update rule would be, because it observes every prediction
    # that comes due rather than only those an update rule chooses to accept
    # (a rule that skips degenerate closures would leave those windows
    # unchecked).
    real_close = gov._ledger.close

    def _spy_close(cycle, sim_, active_ids):
        for law_id in gov._ledger.due_law_ids(cycle):
            closed.append((law_id, cycle))
        real_close(cycle, sim_, active_ids)
    gov._ledger.close = _spy_close                            # type: ignore[assignment]

    _run(sim)

    _check(results, "predictions were opened and closed during the run",
           bool(opened) and bool(closed),
           f"{len(opened)} opened, {len(closed)} closed")
    bad = [(lid, c, opened.get(lid)) for lid, c in closed
           if opened.get(lid) is None or c - opened[lid] != LOOKAHEAD_CYCLES]
    _check(results, f"every closure lands exactly T+{LOOKAHEAD_CYCLES}",
           not bad, f"{len(bad)} off-schedule" + (f", e.g. {bad[0]}" if bad else ""))

    # Closure is decoupled from the law's fate.  Laws run 5-20 cycles and some
    # are repealed or superseded, so if closure were still keyed on expiry the
    # windows above could not all be exactly 20.  The direct evidence is the
    # active/lifted split recorded at closure time: most reviews must find the
    # law already gone, and grading them anyway is the explicit design choice.
    #
    # NOTE this counts the law's state AT THE REVIEW DATE, not
    # `_law_expiry_reasons`, which `_expire_laws` replaces wholesale every cycle
    # and is therefore always empty of 20-cycle-old expiries — asserting against
    # that dict instead would silently fail.  The correct state lives in
    # `_calib_n_closed_law_lifted`.
    n_lifted = gov._ledger.n_closed_law_lifted
    n_active = gov._ledger.n_closed_law_active
    # This split is THREE-way: `n_closed_no_law` is the third bucket, for a
    # decision that enacted nothing.  ADS never opens such a forecast, so its
    # count is a true zero here and the two-way sum still accounts for every
    # closure — asserting that is itself a guard on ADS's behaviour not
    # drifting.
    _check(results, "the active/lifted split accounts for every closure",
           n_lifted + n_active == gov._ledger.closed_total
           and gov._ledger.n_closed_no_law == 0,
           f"lifted={n_lifted}, active={n_active}, "
           f"no_law={gov._ledger.n_closed_no_law}, "
           f"total={gov._ledger.closed_total}")
    _check(results, "predictions do close for laws that had already lifted",
           n_lifted > 0, f"{n_lifted} of {gov._ledger.closed_total} closures")

    # The ledger drains: everything except the last 20 cycles' worth must have
    # closed, and the open count must match the surviving buckets exactly.
    summary = gov.get_calibration_summary(_TEST_CYCLES)
    bucketed = sum(len(gov._ledger.due_law_ids(c))
                   for c in gov._ledger.due_cycles())
    _check(results, "the open-prediction counter matches the ledger contents",
           gov._ledger.open_predictions == bucketed,
           f"counter={gov._ledger.open_predictions}, ledger={bucketed}")
    _check(results, "every still-open prediction is due after the run ends",
           all(k >= _TEST_CYCLES for k in gov._ledger.due_cycles()),
           f"due cycles {gov._ledger.due_cycles()[:5]}")
    _check(results, "the summary's error series covers every closure",
           len(summary["calibration_cycle_deltas"]) == gov._ledger.closed_total,
           f"n={gov._ledger.closed_total}")

    # The paired-null fields are absent from the summary, and their absence is
    # asserted rather than assumed: a consumer that still reads them would
    # silently get None from a .get() and plot an empty series.
    for dead in ("calibration_cycle_deltas_raw", "calibration_mean_abs_err_raw",
                 "calibration_mean_abs_err_cal", "calibration_table",
                 "calibration_n_infeasible", "calibration_schema_version"):
        _check(results, f"the summary does not carry {dead}",
               dead not in summary, "absent" if dead not in summary else "STILL PRESENT")


# ---------------------------------------------------------------------------
# 8. Event statistics: the firewall, static and dynamic
# ---------------------------------------------------------------------------

def _check_event_stats(results: list) -> None:
    print("\n8. Observed-event statistics see the past only:")

    # RUN LENGTH IS LOAD-BEARING, not incidental.  The "projected score ignores
    # the scheduled catastrophe" assertion below only exercises the
    # statistics -> phantom-injection path if the inferred model carries a
    # NONZERO event rate, and this seed observes its first event at cycle 30 —
    # a 20-cycle run leaves every rate at 0.0 and injects nothing in either arm.
    # `_TEST_CYCLES` gives comfortable headroom over that threshold, and the
    # explicit rate check below turns any future regression (a seed change, a
    # quieter event generator) into a FAILURE rather than a silent return to
    # comparing two empty rollouts.
    gov = AdsGovernment(seed=51, eval_seed=51)
    sim = _build_sim(gov, cycles=_TEST_CYCLES)
    _run(sim, cycles=_TEST_CYCLES)
    es = sim.event_system

    before = es.observed_event_stats(sim.cycle)

    # A catastrophe scheduled for the future must be invisible.  This is the
    # single most important assertion in this file: a leak here would make the
    # foresight comparison meaningless while improving the results.
    doom = ActiveEvent(event_type=EventType.STORM, start_cycle=sim.cycle + 3,
                       duration=99, severity=9.9, storm_damage=0.4)
    es.schedule(sim.cycle + 3, doom)
    after = es.observed_event_stats(sim.cycle)
    _check(results, "a scheduled future event does not change the statistics",
           before == after, "unchanged" if before == after else f"{before} -> {after}")

    living = [a for a in sim.agents if a.alive][:6]
    if living:
        probe = ProposedLaw(law_type="FOOD_RATION", params={"max_per_cycle": 2.0},
                            description="p", source_category="food",
                            applies_to=[a.agent_id for a in living], duration=5)

        # THE `env=` HERE IS LOAD-BEARING — do not drop it back to the default.
        #
        # This assertion exists to prove that the statistics -> phantom-injection
        # path cannot see the future.  If these two calls omitted `env=`, they
        # would silently default to CRUDE_ENVIRONMENT: every category's rate is
        # 0.0 there, so ZERO phantom events would be injected in either arm and
        # the two scores would match trivially.  The test would still pass and
        # would still catch a direct `scheduled_events` read by `evaluate`, but
        # it would stop exercising the mechanism it exists to test.
        #
        # Built from the 20-cycle run above so the rates are nonzero, and built
        # ONCE and shared by both arms: a per-arm model would reintroduce a
        # second difference and confound the comparison.  It is also built
        # BEFORE the doom is scheduled, though `environment_model` reads only
        # `observed_event_stats` — which the `before == after` check immediately
        # above has just proven is past-only.
        env = gov._estimator.environment_model(sim, sim.cycle)
        _rates = [float(s.get("rate") or 0.0) for s in env.event_stats.values()]
        _check(results, "the inferred env carries a nonzero event rate "
                        "(so phantom injection is actually exercised)",
               any(r > 0.0 for r in _rates),
               f"rates={[round(r, 4) for r in _rates]}"
               + ("" if any(r > 0.0 for r in _rates)
                  else " — ALL ZERO, this test is back to comparing two empty rollouts"))

        ev = EvaluatorNode(eval_seed=51)
        score_with_doom = ev.evaluate(probe, living, sim, cycle=sim.cycle, env=env)
        es.scheduled_events.remove((sim.cycle + 3, doom))
        ev2 = EvaluatorNode(eval_seed=51)
        score_without = ev2.evaluate(probe, living, sim, cycle=sim.cycle, env=env)
        _check(results, "the projected score ignores the scheduled catastrophe",
               score_with_doom == score_without,
               f"{score_with_doom!r} vs {score_without!r}")

    # Estimator shape.
    for cat, s in sorted(after.items()):
        _check(results, f"{cat}: rate is a probability within the cap",
               0.0 <= s["rate"] <= EVENT_RATE_CAP, f"rate={s['rate']:.4f}")
    _check(results, "the rate prior damps the early-run estimate",
           all(s["rate"] <= s["n"] / (sim.cycle + EVENT_RATE_PRIOR_CYCLES) + 1e-12
               for s in after.values()),
           "n/(cycle+prior) respected")

    # An empty history yields rate 0, hence no injection at all — the property
    # that makes the "no evidence" case safe rather than merely defaulted.
    fresh = _build_sim(AdsGovernment(seed=52), cycles=2)
    empty = fresh.event_system.observed_event_stats(0)
    _check(results, "with no history every rate is exactly 0.0",
           all(s["rate"] == 0.0 for s in empty.values()),
           str({k: v["rate"] for k, v in empty.items()}))

    # -----------------------------------------------------------------------
    # THE STATIC FIREWALL — A COVERAGE-INDEPENDENT CHECK ON GROUND-TRUTH LEAKAGE
    # -----------------------------------------------------------------------
    #
    # WHY THIS EXISTS, IN ONE SENTENCE: the dynamic check above (the projected
    # score ignoring a scheduled catastrophe) is a COVERAGE instrument, so a
    # ground-truth leak sitting on a line the test suite never executes would
    # slip past it silently.
    #
    # THE CONCRETE RISK: `ads.py`'s ecological-ration heuristic reconstructs
    # `sim.drain_mult` via `difficulty_multiplier(sim.config.difficulty)`,
    # gated on `food_depletion_fraction >= _DEPLETION_THRESHOLD * 0.72 = 0.2016`.
    # The largest configuration in this suite reaches a depletion fraction of
    # only 0.0489, so a run at this suite's scale walks past that branch every
    # cycle and would report a clean firewall — while on the published sweep's
    # shape (500 agents, 50x50, 150 cycles) the gate opens at cycle 128 at
    # difficulty D100 and the line executes, leaking ground truth exactly where
    # the dynamic check cannot see it.
    #
    # A scan over the SOURCE has no coverage to be gated by.  It cannot replace
    # the dynamic check (it cannot see a read performed through getattr, a
    # computed attribute name, or a helper in engine/ called with the config as
    # an argument), so both run.  They fail in different directions, which is
    # the point.
    #
    # WHY AST AND NOT grep.  `parameter_inference.py`'s module docstring
    # ENUMERATES these very names ("MUST NOT read: sim.drain_mult,
    # sim.config.regen_mult, ...") as documentation.  A text grep flags that
    # prose, someone silences it, and the guard is decorative within a week.
    # `ast` never sees string or comment content, so the documentation is
    # invisible to it for free — no allowlist, no pragma, nothing to keep in
    # sync as the docs change.
    #
    # THE RULE IS NOT "no government mentions these names".  `env.drain_mult`
    # and `self._estimator.drain_mult` are the legitimate inferred substitutes
    # these governments are meant to read, and a guard that flagged them would
    # be reverted immediately.  The rule is: no module under governments/ may
    # obtain a difficulty-derived quantity FROM THE SIMULATION OR ITS CONFIG.

    #: Quantities that are functions of the hidden difficulty knob.  Mirrors the
    #: "MUST NOT read" list in `parameter_inference`'s module docstring.
    _HIDDEN_ATTRS = frozenset({
        "drain_mult", "regen_mult", "max_steps_per_cycle", "difficulty",
        "metabolic_rate", "hunger_drain", "thirst_drain", "disease_drain",
        "hunger_build_rate", "thirst_build_rate",
    })
    #: Base expressions denoting the Simulation or its config.  `env` and
    #: `_estimator` are deliberately ABSENT — reading `drain_mult` off those is
    #: exactly what governments are meant to do.
    _GROUND_TRUTH_BASES = frozenset({
        "sim", "_sim", "simulation", "config", "_config", "cfg",
    })
    _FORBIDDEN_FUNC = "difficulty_multiplier"

    def _base_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _scan_source(src: str, filename: str) -> list:
        """Firewall violations in one module's source, as readable strings."""
        out = []
        base = os.path.basename(filename)
        for node in ast.walk(ast.parse(src, filename=filename)):
            # Pattern 1 — any reference to difficulty_multiplier, under any
            # alias.  There is no legitimate use: it is a pure function of
            # config.difficulty, so reaching it AT ALL reconstructs ground
            # truth.  This is the pattern that catches ads.py:770.
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == _FORBIDDEN_FUNC:
                        out.append(f"{base}:{node.lineno} imports {_FORBIDDEN_FUNC}"
                                   + (f" as {alias.asname}" if alias.asname else ""))
            elif isinstance(node, ast.Name) and node.id == _FORBIDDEN_FUNC:
                out.append(f"{base}:{node.lineno} references {_FORBIDDEN_FUNC}")
            elif isinstance(node, ast.Attribute) and node.attr == _FORBIDDEN_FUNC:
                out.append(f"{base}:{node.lineno} references .{_FORBIDDEN_FUNC}")
            # Pattern 2 — a hidden attribute read off the simulation or config.
            if isinstance(node, ast.Attribute) and node.attr in _HIDDEN_ATTRS:
                if _base_name(node.value) in _GROUND_TRUTH_BASES:
                    out.append(f"{base}:{node.lineno} reads "
                               f"{_base_name(node.value)}.{node.attr}")
        return out

    gov_dir = os.path.join(_HERE, "governments")
    scanned, static_hits = [], []
    for entry in sorted(os.listdir(gov_dir)):
        if not entry.endswith(".py"):
            continue
        scanned.append(entry)
        with open(os.path.join(gov_dir, entry), "r", encoding="utf-8") as fh:
            static_hits.extend(_scan_source(fh.read(), os.path.join(gov_dir, entry)))

    _check(results, "the static scan covers every module under governments/",
           len(scanned) >= 4 and "ads.py" in scanned
           and "parameter_inference.py" in scanned,
           f"{len(scanned)} modules: {scanned}")
    _check(results, "no module under governments/ reads a hidden parameter off "
                    "sim/config (coverage-independent)",
           not static_hits, "; ".join(static_hits) if static_hits else "clean")

    # POSITIVE CONTROLS.  A scanner that cannot fire certifies the tree forever.
    # Each case below is the exact shape of a violation that would defeat the
    # intended protection.
    for _label, _src in (
        ("a ground-truth-reconstruction pattern's general form",
         "def f(self):\n    return difficulty_multiplier(self._sim.config.difficulty)\n"),
        ("the FOOD_RATION heuristic's exact form",
         "from engine.simulation import difficulty_multiplier as _dm\n"
         "def f(sim):\n    return _dm(sim.config.difficulty)\n"),
        ("a bare sim.drain_mult read", "def f(sim):\n    return sim.drain_mult\n"),
        ("a config.regen_mult read", "def f(cfg):\n    return cfg.regen_mult\n"),
        ("a move-budget read",
         "def f(sim):\n    return sim.config.max_steps_per_cycle\n"),
    ):
        _hits = _scan_source(_src, "synthetic.py")
        _check(results, f"static scan flags {_label} (positive control)",
               bool(_hits),
               "; ".join(_hits) if _hits else "NOT FLAGGED — scanner is blind here")

    # NEGATIVE CONTROLS.  Equally load-bearing: a scanner that flags the
    # legitimate inferred reads, or the docstring that documents the rule, gets
    # deleted — which is the same as never having written it.
    for _label, _src in (
        ("env.drain_mult (the inferred substitute)",
         "def f(env):\n    return env.drain_mult\n"),
        ("self._estimator.drain_mult (the correct inferred substitute)",
         "def f(self):\n    return self._estimator.drain_mult\n"),
        ("env.max_steps_per_cycle inside the evaluator",
         "def f(env):\n    return env.max_steps_per_cycle\n"),
        ("the docstring enumerating the forbidden names",
         '"""MUST NOT read: sim.drain_mult, sim.config.difficulty."""\n'),
        ("a comment naming difficulty_multiplier",
         "# see difficulty_multiplier(sim.config.difficulty) for context\nx = 1\n"),
    ):
        _hits = _scan_source(_src, "synthetic.py")
        _check(results, f"static scan does NOT flag {_label} (negative control)",
               not _hits, "; ".join(_hits) if _hits else "correctly ignored")

    # -----------------------------------------------------------------------
    # THE DYNAMIC FIREWALL.  Catches what the static scan cannot: a read via
    # getattr, a computed attribute name, or an engine helper handed the
    # config.
    # -----------------------------------------------------------------------
    #
    # The static test proves observed_event_stats does not read the schedule.
    # This one proves the GOVERNMENT does not read the four difficulty-derived
    # quantities it is not entitled to, anywhere, over a whole run — including
    # from code paths a manual survey would not think to check, such as a
    # FOOD_RATION heuristic reconstructing sim.drain_mult via
    # difficulty_multiplier from nowhere near the evaluator.
    #
    # Implemented by poisoning the attributes rather than by proxying the whole
    # object: a property that raises on __get__ is the narrowest possible
    # instrument, catches reads from any call site, and cannot itself perturb
    # the simulation.
    #
    # CRITICAL SUBTLETY, and the reason this is a stack-walking tripwire rather
    # than a plain raise.  THE ENGINE LEGITIMATELY READS ALL FOUR OF THESE, every
    # cycle: `_step` passes `config.regen_mult` to `grid.regenerate`,
    # `_execute_actions` reads `config.max_steps_per_cycle` for the move budget,
    # and `_apply_health_dynamics` reads `self.drain_mult` and
    # `config.difficulty`.  They are the engine's OWN parameters — it is
    # supposed to read them.  The firewall being tested is not "nobody may read
    # these", it is "no GOVERNMENT may read these".  A tripwire that raised on
    # any read would fail on the engine's first regeneration call and would be
    # testing nothing about ADS at all.
    #
    # So the getter walks the call stack and fires only when some frame belongs
    # to a module under governments/.  That is the precise encoding of the
    # property, and it catches a read from ANY government call path —
    # including paths a manual survey would not think to check, such as the
    # FOOD_RATION heuristic above.
    #
    # RUN AGAINST BOTH GOVERNMENTS.  autocracy_lookahead owns its own
    # estimator, its own ledger and an out-of-band decision round, which makes
    # it the more permissive of the two arms on exactly this axis — pointing
    # the strongest guard in the file only at ADS would leave the more
    # permissive arm unguarded.  This is the instrument that catches a
    # violation a manual code survey would miss.
    _GOV_DIR = os.path.join(_HERE, "governments") + os.sep

    class _Tripwire:
        def __init__(self) -> None:
            self.touched = []

    def _run_tripwire(government, cycles=30):
        """
        Poison the four hidden parameters on a fresh sim, run it, and report
        whether any frame under governments/ read one.

        Returns ``(completed, detail, tripwire, poisoned_sim)``.

        EACH CALL GETS ITS OWN ``_Tripwire``.  Sharing one across two runs would
        cross-contaminate ``touched`` and make the positive control's
        ``"_probe"`` filter ambiguous — a violation by the second government
        could be filtered away as the first run's control, or vice versa.
        """
        tripwire = _Tripwire()

        def _poison(obj, attr):
            cls = type(obj)
            if not cls.__dict__.get("_poisoned_subclass"):
                cls = type(f"Poisoned{cls.__name__}", (cls,),
                           {"_poisoned_subclass": True})
                obj.__class__ = cls
            # Move the real value out of the instance dict, where it would
            # otherwise shadow nothing (a data descriptor on the class wins) but
            # would leak if the property were ever removed.
            real = obj.__dict__.pop(attr)

            def _guarded(self, _attr=attr, _real=real):
                frame = sys._getframe(1)
                while frame is not None:
                    filename = frame.f_code.co_filename
                    if filename.startswith(_GOV_DIR):
                        tripwire.touched.append(
                            f"{os.path.basename(filename)}:"
                            f"{frame.f_code.co_name} read {_attr}"
                        )
                        raise AssertionError(tripwire.touched[-1])
                    frame = frame.f_back
                return _real                  # engine read — legitimate

            setattr(cls, attr, property(_guarded))

        sim = _build_sim(government, cycles=cycles)
        _poison(sim, "drain_mult")
        _poison(sim.config, "regen_mult")
        _poison(sim.config, "max_steps_per_cycle")
        _poison(sim.config, "difficulty")

        try:
            for cycle in range(cycles):
                sim.cycle = cycle
                sim._step(cycle)
            return (True,
                    f"{cycles} cycles, no government read any of the four",
                    tripwire, sim)
        except AssertionError as exc:
            return False, str(exc), tripwire, sim
        except Exception as exc:                              # noqa: BLE001
            return (False, f"unexpected {type(exc).__name__}: {exc}",
                    tripwire, sim)

    # 30 cycles matches the foresight-parity check's spy harness and is long
    # enough to exercise the decision round, the warning round and the
    # estimator on both arms.
    ads_completed, ads_detail, ads_tw, ads_fw_sim = _run_tripwire(
        AdsGovernment(seed=53, eval_seed=53)
    )
    al_completed, al_detail, al_tw, al_fw_sim = _run_tripwire(
        AutocracyLookaheadGovernment(seed=53)
    )

    # Positive control: the instrument must be capable of firing.  Without this,
    # a tripwire broken in the "never fires" direction would report a clean
    # firewall forever — the single most dangerous way for this test to fail.
    #
    # RUN ONCE PER ARM, NOT ONCE IN TOTAL.  The mechanism is NOT "the same
    # object in both calls, so proving it fires once proves it for both":
    # `_poison` derives a fresh `Poisoned*` subclass and installs a fresh
    # `property` per sim instance, and `_run_tripwire` gives each call its own
    # `_Tripwire`. ADS's instrument and A+L's are two separate instruments.
    # Proving one fires says nothing about the other — and A+L is the arm with
    # the more permissive epistemics, so it is exactly the arm where an
    # unproven instrument reporting "clean" would be most costly.
    def _deliberate_read(poisoned_sim):
        import governments.parameter_inference as _probe_module
        src = ("def _probe(sim):\n"
               "    return sim.config.regen_mult\n")
        namespace = {}
        exec(compile(src, _probe_module.__file__, "exec"), namespace)
        return namespace["_probe"](poisoned_sim)

    for _arm, _sim in (("ADS", ads_fw_sim), ("autocracy_lookahead", al_fw_sim)):
        _works = _raises(AssertionError, lambda s=_sim: _deliberate_read(s))
        _check(results,
               f"the {_arm} firewall tripwire is capable of firing "
               f"(positive control)",
               _works,
               f"a read from a governments/ frame raises on the {_arm} "
               f"poisoned sim" if _works
               else f"THE {_arm} TRIPWIRE IS BROKEN — a clean result below "
                    f"means nothing for this arm")

    # `touched` is filtered to exclude the positive control's own entry.
    ads_touches = [t for t in ads_tw.touched if "_probe" not in t]
    _check(results, "a live ADS run completes without any government reading a "
                    "hidden parameter",
           ads_completed and not ads_touches,
           ads_detail + (f" touched={ads_touches}" if ads_touches else ""))
    al_touches = [t for t in al_tw.touched if "_probe" not in t]
    _check(results, "a live autocracy_lookahead run completes without any "
                    "government reading a hidden parameter",
           al_completed and not al_touches,
           al_detail + (f" touched={al_touches}" if al_touches else ""))


# ---------------------------------------------------------------------------
# 9. Real events are time-bounded inside the rollout
# ---------------------------------------------------------------------------

def _check_event_time_bounding(results: list) -> None:
    print("\n9. A pre-existing event stops when it would really stop:")

    gov = AdsGovernment(seed=61, eval_seed=61)
    sim = _build_sim(gov, cycles=6)
    _run(sim, cycles=6)
    es = sim.event_system
    living = [a for a in sim.agents if a.alive][:8]
    if not living:
        _check(results, "the time-bounding probe has living agents", False, "none alive")
        return

    # Neutralize shelter for any probe agent the scenario RNG happened to seat
    # on FOREST/MOUNTAIN terrain (leave everyone else's cell, and hence their
    # resource dynamics, untouched). The storm-drain term below is gated on
    # `on_shelter` and is exactly zero for a sheltered agent (governments/
    # ads.py, ``EvaluatorNode._step``'s storm-damage branch); if enough of
    # the 8-agent probe group sits on shelter, the graded, duration-dependent
    # response this check depends on collapses into a flat step function.
    # Retagging only the sheltered agents' own cells to PLAINS (never
    # shelter, per TERRAIN_STATS in engine/grid.py) makes the storm drain
    # apply to the whole probe group
    # without depending on which cells the seed happened to land agents on.
    for a in living:
        if a.position and sim.grid.in_bounds(*a.position):
            cell = sim.grid.cell(*a.position)
            if cell.shelter:
                cell.terrain = Terrain.PLAINS

    # The crude environment injects nothing, so the only difference between the
    # two scores below is how long the pre-existing storm is held on for.  This
    # is exactly the model autocracy_lookahead now runs on, which makes this
    # section a direct guard on that regime rather than a synthetic probe.
    quiet = CRUDE_ENVIRONMENT
    probe = ProposedLaw(law_type="NO_ACTION", params={}, description="p",
                        source_category="terrain",
                        applies_to=[a.agent_id for a in living], duration=5)

    def _score_with_storm(duration):
        es.active_events.clear()
        if duration is not None:
            es.active_events.append(ActiveEvent(
                event_type=EventType.STORM, start_cycle=sim.cycle,
                duration=duration, severity=1.0, storm_damage=0.04))
        s = EvaluatorNode(eval_seed=61).evaluate(
            probe, living, sim, cycle=sim.cycle, env=quiet)
        es.active_events.clear()
        return s

    ladder = [(d, _score_with_storm(d))
              for d in (None, 2, 5, 10, 15, 18, LOOKAHEAD_CYCLES,
                        LOOKAHEAD_CYCLES * 2)]
    baseline = ladder[0][1]
    scores = [s for _, s in ladder]

    # Monotone: a longer residual window can only hurt more.
    _check(results, "projected health is non-increasing in the storm's remaining window",
           all(a >= b - 1e-12 for a, b in zip(scores, scores[1:])),
           ", ".join(f"{d}:{s:.4f}" for d, s in ladder))

    # THE sharp assertion: a storm that outlasts the horizon must score exactly
    # the same as one that exactly fills it.  This is what proves the window is
    # clamped with a `min` rather than allowed to run past the rollout, and it
    # is an exact-equality check because a clamp either fires or it does not.
    _check(results, "a storm longer than the horizon scores exactly like a horizon-long one",
           scores[-1] == scores[-2],
           f"{LOOKAHEAD_CYCLES}:{scores[-2]:.9f} vs "
           f"{LOOKAHEAD_CYCLES * 2}:{scores[-1]:.9f}")

    # Bounding is graded, not all-or-nothing: some finite window shorter than
    # the horizon must land strictly between "no storm" and "storm throughout".
    strictly_between = [d for d, s in ladder[1:-1] if baseline > s > scores[-1]]
    _check(results, "an intermediate window scores strictly between none and horizon-long",
           bool(strictly_between),
           f"durations {strictly_between} lie strictly between "
           f"{baseline:.4f} and {scores[-1]:.4f}")

    # RECOVERY DOMINANCE — pinned deliberately, because it is counter-intuitive,
    # it is a property of the OBJECTIVE rather than of the time-bounding code,
    # and the obvious "any storm must hurt" assertion would encode the opposite
    # belief.  A storm ending several cycles before the horizon leaves NO trace
    # in the score: projected agents regain roughly 0.18 health per cycle from
    # eating and drinking (2 x 0.05 + 2 x 0.04), clipped at 1.0, against a storm
    # drain of 0.04 x drain_mult, so a transient drain on a healthy group is
    # fully repaid long before cycle 20 and the terminal sum is unchanged.
    #
    # Two consequences a reader of the projection needs:
    #   * the terminal-value objective isnearly blind to short residual events, so
    #     the time-bounding fix matters mainly for events still live near the
    #     END of the horizon, and for mortality, which is irreversible;
    #   * it therefore BOUNDS the over-pessimism the old freeze-for-the-whole-
    #     horizon behaviour caused.  That defect was real, but its magnitude was
    #     smaller than a naive reading of "20 cycles of phantom storm" suggests.
    #
    # If this assertion ever fails, the objective or the recovery rates changed,
    # and the calibration error levels are not comparable across that change.
    washed_out = [d for d, s in ladder[1:] if d is not None and d <= 10
                  and s == baseline]
    _check(results, "a storm ending well before the horizon leaves the score unchanged",
           len(washed_out) >= 2,
           f"durations {washed_out} score exactly {baseline:.6f}")

    # Expired-but-uncleaned events must contribute nothing, which is what the
    # `cycles_remaining == 0` branch is for.
    es.active_events.append(
        ActiveEvent(event_type=EventType.STORM, start_cycle=sim.cycle - 10,
                    duration=2, severity=1.0, storm_damage=0.04))
    score_stale = EvaluatorNode(eval_seed=61).evaluate(
        probe, living, sim, cycle=sim.cycle, env=quiet)
    es.active_events.clear()
    _check(results, "an already-expired event contributes nothing to the rollout",
           score_stale == baseline, f"{score_stale:.6f} vs {baseline:.6f}")


# ---------------------------------------------------------------------------
# 10. The movement estimator is safe and monotone
# ---------------------------------------------------------------------------

def _check_movement_estimator(results: list) -> None:
    print("\n10. The movement estimator never over-states mobility:")

    # THE ONE-SIDED ERROR THAT MATTERS.  Over-estimating mobility credits
    # relocation laws (MANDATORY_SHELTER, storm-triggered MOVE_TO_REGION) with
    # travel the engine cannot actually perform, which inflates their projected
    # benefit.  Under-estimating merely reproduces the crude, uninferred
    # behaviour.  So the assertion is one-sided by design.
    #
    # NOT asserted, deliberately: that it REACHES the true cap by any
    # particular cycle.  Convergence needs some agent to actually exhaust its
    # move budget, which is a property of the scenario, not of the estimator.
    for true_cap in (1, 3, 10):
        gov = AdsGovernment(seed=61 + true_cap, eval_seed=61)
        sim = _build_sim(gov, cycles=40, steps=true_cap)

        # Sampled BEFORE any cycle runs.  The trajectory below is sampled after
        # each _step, by which point cycle 0 has already been observed and the
        # estimate may legitimately have risen — so reading trajectory[0] would
        # not tell us where the estimator STARTED.
        _check(results, f"cap={true_cap}: the estimate starts at exactly the prior ceiling",
               gov._estimator.max_steps_per_cycle == MOVEMENT_PRIOR_CEILING == 1,
               f"pre-run value {gov._estimator.max_steps_per_cycle}")

        trajectory = []
        for cycle in range(40):
            sim.cycle = cycle
            sim._step(cycle)
            trajectory.append(gov._estimator.max_steps_per_cycle)

        over = [v for v in trajectory if v > true_cap]
        _check(results, f"cap={true_cap}: the estimate never exceeds the true cap",
               not over, f"max seen {max(trajectory)} vs cap {true_cap}")

        monotone = all(a <= b for a, b in zip(trajectory, trajectory[1:]))
        _check(results, f"cap={true_cap}: the estimate is monotone non-decreasing",
               monotone, f"trajectory {trajectory[0]}..{trajectory[-1]}")

        _check(results, f"cap={true_cap}: every sampled value is within [1, cap]",
               all(MOVEMENT_PRIOR_CEILING <= v <= true_cap for v in trajectory),
               f"range [{min(trajectory)}, {max(trajectory)}] vs cap {true_cap}")

        if true_cap == 1:
            # THE PROPERTY THAT KEEPS autocracy_lookahead's PROJECTIONS
            # UNCHANGED.  With a true cap of 1 the estimate can never move, so
            # ADS's rollout arithmetic on this axis is identical to the crude
            # one — which is what makes the Chebyshev generalisation safe to
            # land in shared code.
            _check(results, "cap=1: the estimate stays pinned at 1 for the whole run",
                   set(trajectory) == {1}, f"values seen: {sorted(set(trajectory))}")


# ---------------------------------------------------------------------------
# 11. The reporting canary
# ---------------------------------------------------------------------------

def _check_schema_canary(results: list) -> None:
    print("\n11. A completed run reports its forecast schema version:")

    # WHY A CANARY RATHER THAN A RENAME.  `get_calibration_summary` and
    # `get_calibration_record` are reached by getattr duck-typing
    # (benchmark_core._merge_calibration_summary,
    # run_recorder._calibration_payload).  A missed rename there does not raise:
    # the getattr returns None and the whole block silently vanishes from
    # final_stats.json.  That is the worst available failure mode, so the method
    # names were deliberately left imperfect and this field proves the chain is
    # intact instead.
    gov = AdsGovernment(seed=81, eval_seed=81)
    sim = _build_sim(gov, cycles=30)
    _run(sim, cycles=30)

    summary = gov.get_calibration_summary(30)
    _check(results, "the summary carries ads_forecast_schema_version",
           summary.get("ads_forecast_schema_version") == ADS_FORECAST_SCHEMA_VERSION,
           f"got {summary.get('ads_forecast_schema_version')!r}, "
           f"expected {ADS_FORECAST_SCHEMA_VERSION}")
    _check(results, "the summary reports the final parameter estimates",
           set(summary.get("parameter_estimates_final", {}))
           == {"drain_mult", "regen_mult", "max_steps"},
           str(sorted(summary.get("parameter_estimates_final", {}))))

    # The per-cycle record must carry the estimates too, since that is the
    # series a reader needs to tell "the estimator never converged" apart from
    # "the estimator converged and the model is still wrong".
    record = gov.get_calibration_record(29)
    _check(results, "the per-cycle record carries parameter_estimates",
           set(record.get("parameter_estimates", {}))
           == {"drain_mult", "regen_mult", "max_steps"},
           str(sorted(record.get("parameter_estimates", {}))))
    _check(results, "the per-cycle record does not carry calibration_table",
           "calibration_table" not in record,
           "absent" if "calibration_table" not in record else "STILL PRESENT")

    # No clamp should ever fire in a healthy run.  A clamp means the exposure
    # model has drifted from the engine — an assertion failure, not a mechanism
    # — so this is a direct health check on the estimator's arithmetic.
    clamps = {
        name: block.get("n_bound_clamps", 0)
        for name, block in summary["parameter_estimates_final"].items()
    }
    _check(results, "no estimator hit its assertion bounds during the run",
           all(v == 0 for v in clamps.values()), str(clamps))


# ---------------------------------------------------------------------------
# 12. The partition's cell ASSIGNMENT
# ---------------------------------------------------------------------------
#
# WHY THIS NEEDS ITS OWN CHECK.  A transposition that files a whole-population
# closure under "10" and a ten-band closure under "1" slips past every other
# instrument in this suite:
#
#   * The projection identity SORTS BOTH SIDES before comparing, so it proves
#     the multiset of (cycle, delta) pairs survives re-indexing and nothing
#     whatever about which cell each pair landed in;
#   * The key-domain check bounds the DOMAIN, not the assignment;
#   * The per-cycle count check joins per (close_cycle, n_groups) COUNT, which
#     a permutation confined to a single cycle preserves by definition.
#
# That is not a hypothetical exposure: on the gate fixture (h=90) 14 of 50 ADS
# decision cycles — 28.0% — co-enact laws of differing `n_groups`, matching the
# archive's 27%. A mis-assignment there would corrupt the cross-arm comparison
# silently, because every other instrument stays green.
#
# This check lives in the standing runner rather than as a one-off script, so
# that someone editing `_partition()` cannot get 386 green assertions while
# this property silently breaks.
#
# THE CONSTRUCTION.  Two forecasts opened on the SAME cycle with DIFFERENT
# n_groups AND DIFFERENT predicted values, so each delta is uniquely
# attributable. Equal predicted values would make the transposition undetectable
# even here, which is exactly why they are not equal.

class _FakeAgent:
    """Minimal agent: the ledger reads `agent_id`, `health` and `alive`."""

    def __init__(self, agent_id: str, health: float, alive: bool = True) -> None:
        self.agent_id = agent_id
        self.health = health
        self.alive = alive


class _FakeSim:
    def __init__(self, agents: list) -> None:
        self.agents = agents


def _t5j_ledger():
    """
    Two co-opened forecasts at cycle 0, closing together at cycle 2.

    scope "1"  (whole population): 2 agents at health 0.5, predicted 0.90 -> 0.40
    scope "10" (ten bands):        2 agents at health 0.2, predicted 0.25 -> 0.05

    An order of magnitude apart, so a transposition is unmistakable in the cell
    MEANS and not merely in a count.
    """
    sim = _FakeSim([
        _FakeAgent("w1", 0.5), _FakeAgent("w2", 0.5),
        _FakeAgent("t1", 0.2), _FakeAgent("t2", 0.2),
    ])
    led = ForecastLedger(horizon=2, logger=_NULL_LOGGER)
    led.open(cycle=0, predicted=0.90, evaluated_agent_ids=["w1", "w2"],
             tags={"law_id": "LW", "outcome": "enacted", "n_groups": 1,
                   "category": "food", "law_type": "RATION",
                   "applies_to": [], "adjusted_at_enactment": 0.90})
    led.open(cycle=0, predicted=0.25, evaluated_agent_ids=["t1", "t2"],
             tags={"law_id": "LT", "outcome": "enacted", "n_groups": 10,
                   "category": "food", "law_type": "RATION",
                   "applies_to": [], "adjusted_at_enactment": 0.25})
    led.close(2, sim, {"LW", "LT"})
    return led, {"1": 0.40, "10": 0.05}


def _t5j_assertions(results: list, tag: str, record: bool) -> list:
    """
    Evaluate the partition-assignment property.  *record* False means
    "evaluate silently" — the positive controls need the OUTCOME of these
    assertions without filling the result table with expected failures.
    """
    led, expected = _t5j_ledger()
    summary = led.as_summary(total_cycles=4)
    deltas = summary["calibration_cycle_deltas_by_outcome_then_scope"]
    closures = summary["calibration_closures_by_outcome_then_scope"]
    outs: list = []

    def _c(name: str, ok: bool, detail: str = "") -> None:
        outs.append(bool(ok))
        if record:
            _check(results, name, ok, detail)

    cell = deltas.get("enacted", {})
    _c(f"{tag}both co-closed forecasts are in the partition",
       sorted(cell) == ["1", "10"], f"scope keys={sorted(cell)}")

    for scope, want in expected.items():
        pairs = cell.get(scope) or []
        got = [round(d, 6) for _cy, d in pairs]
        _c(f"{tag}scope {scope!r} holds EXACTLY its own delta ({want:.2f}), "
           f"not the other cell's",
           got == [round(want, 6)], f"expected [{want:.6f}], got {got}")
        agg = (closures.get("enacted") or {}).get(scope) or {}
        _c(f"{tag}scope {scope!r} aggregate mean matches its own delta",
           agg.get("n") == 1
           and abs(float(agg.get("mean_abs_err", -1)) - want) < 1e-9,
           f"agg={agg}")

    # Outcome axis, same construction: two co-closing forecasts differing only
    # in `outcome`.  A+L is the arm where this axis is informative.
    led2 = ForecastLedger(horizon=2, logger=_NULL_LOGGER)
    led2.open(cycle=0, predicted=0.9, evaluated_agent_ids=["a"],
              tags={"law_id": "L1", "outcome": "enacted", "n_groups": 1})
    led2.open(cycle=0, predicted=0.2, evaluated_agent_ids=["b"],
              tags={"law_id": None, "outcome": "no_action", "n_groups": 1})
    led2.close(2, _FakeSim([_FakeAgent("a", 0.5), _FakeAgent("b", 0.1)]), {"L1"})
    d2 = led2.as_summary(4)["calibration_cycle_deltas_by_outcome_then_scope"]
    _c(f"{tag}'enacted' holds 0.40, not the no_action delta",
       [round(x, 6) for _cy, x in (d2.get("enacted", {}).get("1") or [])] == [0.4],
       f"got {d2.get('enacted')}")
    _c(f"{tag}'no_action' holds 0.10, not the enacted delta",
       [round(x, 6) for _cy, x in (d2.get("no_action", {}).get("1") or [])] == [0.1],
       f"got {d2.get('no_action')}")
    return outs


def _check_partition_assignment(results: list) -> None:
    _t5j_assertions(results, "", record=True)

    # ---- POSITIVE CONTROLS.  A test that cannot go red proves nothing. ------
    # These are the load-bearing part of this section, not decoration: this
    # check and the scope-domain-derivation check are the ONLY instruments in
    # the tree that can see a cell mis-assignment, so a green result here is
    # worth precisely as much as the demonstration that it is capable of going
    # red.
    real_scope, real_outcome = _fl._scope_key, _fl._outcome_key
    try:
        def _transposing_scope(tag):
            k = real_scope(tag)
            return {"1": "10", "10": "1"}.get(k, k)
        _fl._scope_key = _transposing_scope
        outs = _t5j_assertions(results, "[ctrl] ", record=False)
        _check(results,
               "POSITIVE CONTROL: a transposing _scope_key makes this check RED",
               not all(outs),
               f"{sum(1 for o in outs if not o)} of {len(outs)} assertion(s) "
               f"failed under the transposition")
    finally:
        _fl._scope_key = real_scope

    try:
        def _transposing_outcome(tag):
            k = real_outcome(tag)
            return {"enacted": "no_action", "no_action": "enacted"}.get(k, k)
        _fl._outcome_key = _transposing_outcome
        outs = _t5j_assertions(results, "[ctrl] ", record=False)
        _check(results,
               "POSITIVE CONTROL: a transposing _outcome_key makes this check RED",
               not all(outs),
               f"{sum(1 for o in outs if not o)} of {len(outs)} assertion(s) "
               f"failed under the transposition")
    finally:
        _fl._outcome_key = real_outcome

    # NEGATIVE control: with both restored this check is green again, so the
    # two results above are attributable to the patch and not to leakage.
    _check(results,
           "NEGATIVE CONTROL: the check is green again once the patches are reverted",
           all(_t5j_assertions(results, "[ctrl] ", record=False)))


# ---------------------------------------------------------------------------
# 13. The scope domain is DERIVED, never written out
# ---------------------------------------------------------------------------

#: DISJOINT from the real [10, 5, 4, 3, 2, 1].  Disjointness is the entire point:
#: a hard-coded domain list inside the emitter would keep producing the real keys
#: under this patch, and the disjointness is what turns that into a failure
#: rather than a coincidence.  7 and 6 are also feasible group counts for the
#: population size below, so the run does not degenerate to zero closures.
_SENTINEL_GROUP_COUNTS = [7, 6]


def _ads_scope_keys(cycles: int = 60, seed: int = 909):
    gov = AdsGovernment(seed=seed, eval_seed=seed)
    sim = _build_sim(gov, n_agents=30, grid=14, cycles=cycles)
    _run(sim, cycles)
    summary = gov.get_calibration_summary(cycles)
    part = summary["calibration_cycle_deltas_by_outcome_then_scope"]
    keys = sorted({g for by in part.values() for g in by})
    return keys, len(summary["calibration_cycle_deltas"])


def _check_scope_domain_derivation(results: list) -> None:
    real_keys, real_n = _ads_scope_keys()
    _check(results, "baseline ADS run closes predictions (non-vacuity)",
           real_n > 0, f"{real_n} closures, scope keys {real_keys}")
    _check(results, "baseline scope keys all come from GROUP_DISTRIBUTION_COUNTS",
           set(real_keys) <= {str(n) for n in _ads_mod.GROUP_DISTRIBUTION_COUNTS},
           f"keys={real_keys}")

    saved = list(_ads_mod.GROUP_DISTRIBUTION_COUNTS)
    try:
        _ads_mod.GROUP_DISTRIBUTION_COUNTS = list(_SENTINEL_GROUP_COUNTS)
        sent_keys, sent_n = _ads_scope_keys()
    finally:
        _ads_mod.GROUP_DISTRIBUTION_COUNTS = saved

    _check(results, "sentinel run closes predictions (non-vacuity)",
           sent_n > 0,
           f"{sent_n} closures under "
           f"GROUP_DISTRIBUTION_COUNTS={_SENTINEL_GROUP_COUNTS}")
    _check(results,
           "under a DISJOINT sentinel domain the emitted scope keys FOLLOW it",
           bool(sent_keys)
           and set(sent_keys) <= {str(n) for n in _SENTINEL_GROUP_COUNTS},
           f"keys={sent_keys}, sentinel={_SENTINEL_GROUP_COUNTS}")
    _check(results,
           "the sentinel keys are disjoint from the real domain "
           "(so a hard-coded list would have been caught)",
           not (set(sent_keys) & {str(n) for n in saved}),
           f"sentinel keys={sent_keys}, real domain={saved}")
    _check(results, "GROUP_DISTRIBUTION_COUNTS was restored",
           _ads_mod.GROUP_DISTRIBUTION_COUNTS == saved,
           f"{_ads_mod.GROUP_DISTRIBUTION_COUNTS}")


# ---------------------------------------------------------------------------
# 14. The filtered-mean enforcement
# ---------------------------------------------------------------------------
#
# `filtered_calibration_mean` is documented as "THE ONLY FUNCTION A FIGURE MAY
# USE TO OBTAIN AN ARM'S CALIBRATION MEAN" and `ForecastLedger.as_summary` calls
# the pair "the enforcement" against plotting A+L's unfiltered scalar beside
# ADS's.  Without this section, that function has zero callers and zero tests,
# so nothing in the tree would catch a regression in the single guard standing
# between a figure and a plausible-looking wrong number.
#
# WHAT THIS SECTION DELIBERATELY DOES NOT DO: hard-code any of the magnitudes.
# The gate fixture's exact numbers shift with the run horizon without changing
# anything about the underlying mechanism.  A test pinned to those numbers
# would go red for a reason that is not a defect — which is how a suite earns
# the right to be ignored.  The invariants below are horizon-stable:
#
#   * on a MIXED-outcome ledger the filtered and unfiltered means DISAGREE, and
#     the filtered mean is the n-weighted mean of the named cells exactly;
#   * on a DEGENERATE (ADS-shaped) ledger they AGREE to within the ledger's own
#     derived cross-rounding bound, `_CROSS_ROUNDING_TOL`;
#   * widening the outcome filter to everything collapses filtered onto
#     unfiltered — the positive control that proves the filtering does work.

def _mixed_outcome_stats() -> dict:
    """
    A `final_stats`-shaped summary from a real ledger with mixed outcomes.

    Two batches, straddling the half-split boundary (`total_cycles=8` puts it at
    cycle 4), each carrying one `enacted` and one `retained` closure:

        batch 1, closes cycle 2 (first half):  enacted 0.40, retained 0.05
        batch 2, closes cycle 6 (second half): enacted 0.30, retained 0.10

    Every one of the six derived numbers is therefore distinct, so no assertion
    below can pass by coincidence:

        filtered   mean   = (0.40 + 0.30) / 2               = 0.35
        unfiltered mean   = (0.40 + 0.05 + 0.30 + 0.10) / 4 = 0.2125
        filtered   halves = (0.40, 0.30)
        unfiltered halves = (0.225, 0.20)
    """
    agents = [_FakeAgent(f"a{i}", 0.5) for i in range(4)]
    sim = _FakeSim(agents)
    led = ForecastLedger(horizon=2, logger=_NULL_LOGGER)
    led.open(cycle=0, predicted=0.90, evaluated_agent_ids=["a0"],
             tags={"law_id": "E1", "outcome": "enacted", "n_groups": 1})
    led.open(cycle=0, predicted=0.55, evaluated_agent_ids=["a1"],
             tags={"law_id": "R1", "outcome": "retained", "n_groups": 1})
    led.close(2, sim, {"E1", "R1"})
    led.open(cycle=4, predicted=0.80, evaluated_agent_ids=["a2"],
             tags={"law_id": "E2", "outcome": "enacted", "n_groups": 1})
    led.open(cycle=4, predicted=0.60, evaluated_agent_ids=["a3"],
             tags={"law_id": "R2", "outcome": "retained", "n_groups": 1})
    led.close(6, sim, {"E1", "R1", "E2", "R2"})
    return led.as_summary(total_cycles=8)


def _close(a, b, tol: float = 1e-9) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def _check_filtered_enforcement(results: list) -> None:
    stats = _mixed_outcome_stats()

    # -- non-vacuity.  Every assertion below is worthless on an empty ledger: a
    #    check that only compares {} against {} on freshly constructed
    #    governments would pass without exercising anything.
    n_closed = len(stats.get("calibration_cycle_deltas") or ())
    outcomes = sorted(
        stats.get("calibration_closures_by_outcome_then_scope") or {})
    _check(results, "filtered-mean fixture is a NON-EMPTY, MIXED-outcome ledger",
           n_closed == 4 and outcomes == ["enacted", "retained"],
           f"{n_closed} closures, outcomes={outcomes}")

    unfiltered = stats.get("calibration_mean_abs_err")
    filtered = filtered_calibration_mean(stats)
    _check(results, "filtered_calibration_mean returns the enacted-only mean",
           _close(filtered, 0.35), f"expected 0.35, got {filtered!r}")
    _check(results, "the unfiltered scalar is the all-outcome mean "
                    "(so the two genuinely differ)",
           _close(unfiltered, 0.2125), f"expected 0.2125, got {unfiltered!r}")
    _check(results, "filtered and unfiltered DISAGREE on a mixed-outcome ledger",
           not _close(filtered, unfiltered, _CROSS_ROUNDING_TOL),
           f"filtered={filtered!r} unfiltered={unfiltered!r}")

    unf_halves = (stats.get("calibration_mean_abs_delta_first_half"),
                  stats.get("calibration_mean_abs_delta_second_half"))
    fil_halves = filtered_half_split(stats)
    _check(results, "filtered_half_split returns the enacted-only halves",
           _close(fil_halves[0], 0.40) and _close(fil_halves[1], 0.30),
           f"expected (0.40, 0.30), got {fil_halves!r}")
    _check(results, "the unfiltered halves are the all-outcome halves",
           _close(unf_halves[0], 0.225) and _close(unf_halves[1], 0.20),
           f"expected (0.225, 0.20), got {unf_halves!r}")
    _check(results, "filtered_half_split splits at the run's OWN emitted "
                    "boundary, not a recomputed one",
           stats.get("calibration_half_split_cycle") == 4,
           f"calibration_half_split_cycle={stats.get('calibration_half_split_cycle')!r}")

    # -- POSITIVE CONTROL.  Widen the filter to every outcome the ledger holds:
    #    the filtered mean must collapse onto the unfiltered one.  Without this,
    #    a `filtered_calibration_mean` that had quietly stopped filtering — the
    #    failure mode that reintroduces the whole hazard — would still satisfy
    #    every assertion above that does not name 0.35.
    widened = frozenset({"enacted", "retained", "repealed", "no_action",
                         "replaced"})
    collapsed = filtered_calibration_mean(stats, outcomes=widened)
    _check(results, "POSITIVE CONTROL: widening the outcome filter to ALL "
                    "outcomes collapses filtered onto unfiltered",
           _close(collapsed, unfiltered, _CROSS_ROUNDING_TOL),
           f"widened={collapsed!r} unfiltered={unfiltered!r}")
    collapsed_halves = filtered_half_split(stats, outcomes=widened)
    _check(results, "POSITIVE CONTROL: the same widening collapses the halves",
           _close(collapsed_halves[0], unf_halves[0], _CROSS_ROUNDING_TOL)
           and _close(collapsed_halves[1], unf_halves[1], _CROSS_ROUNDING_TOL),
           f"widened={collapsed_halves!r} unfiltered={unf_halves!r}")

    # -- the documented None boundary.  "No named outcome cell present" is a
    #    real state (an arm that enacted nothing), not an error, and the
    #    distinction matters: a 0.0 here would plot as a perfect forecaster.
    _check(results, "filtered_calibration_mean returns None (not 0.0) when no "
                    "named outcome cell is present",
           filtered_calibration_mean(stats, outcomes=frozenset({"repealed"}))
           is None)
    _check(results, "filtered_half_split returns (None, None) on a summary with "
                    "no partition",
           filtered_half_split({}) == (None, None))
    _check(results, "filtered_calibration_mean returns None on a summary with "
                    "no partition",
           filtered_calibration_mean({}) is None)

    # -- the two arms, on ledgers built by real runs rather than by hand -------
    ads_gov = AdsGovernment(seed=77, eval_seed=77)
    _run(_build_sim(ads_gov, n_agents=30, grid=14))
    ads_stats = ads_gov.get_calibration_summary(_TEST_CYCLES)
    al_gov = AutocracyLookaheadGovernment(seed=77)
    _run(_build_sim(al_gov, n_agents=30, grid=14))
    al_stats = al_gov.get_calibration_summary(_TEST_CYCLES)

    ads_n = len(ads_stats.get("calibration_cycle_deltas") or ())
    al_n = len(al_stats.get("calibration_cycle_deltas") or ())
    _check(results, "both arms' live ledgers are non-empty (non-vacuity)",
           ads_n > 0 and al_n > 0, f"ADS {ads_n} closures, A+L {al_n}")

    ads_f = filtered_calibration_mean(ads_stats)
    ads_u = ads_stats.get("calibration_mean_abs_err")
    _check(results,
           "on ADS filtered == unfiltered to within the derived cross-rounding "
           "bound (its outcome tag is constant, so they coincide BY ACCIDENT)",
           _close(ads_f, ads_u, _CROSS_ROUNDING_TOL),
           f"filtered={ads_f!r} unfiltered={ads_u!r} "
           f"gap={abs(float(ads_f) - float(ads_u)):.3e} "
           f"bound={_CROSS_ROUNDING_TOL:.3e}")

    al_outcomes = sorted(
        al_stats.get("calibration_closures_by_outcome_then_scope") or {})
    _check(results, "A+L's live ledger really is mixed-outcome "
                    "(otherwise the check below is vacuous)",
           len(al_outcomes) > 1, f"outcomes={al_outcomes}")
    al_f = filtered_calibration_mean(al_stats)
    al_u = al_stats.get("calibration_mean_abs_err")
    _check(results,
           "on A+L filtered != unfiltered by far more than the rounding bound "
           "— this is the 'plausible-looking wrong figure' the pair prevents",
           not _close(al_f, al_u, _CROSS_ROUNDING_TOL),
           f"filtered={al_f!r} unfiltered={al_u!r} "
           f"gap={abs(float(al_f) - float(al_u)):.3e} "
           f"bound={_CROSS_ROUNDING_TOL:.3e}")
    _check(results, "A+L's OWN summary (get_calibration_summary) still does not "
                    "emit final_mean_prediction_error -- deliberate: it would be "
                    "an unfiltered all-time mean. (NOTE: the persisted "
                    "final_stats.json DOES carry that key, set to None by the "
                    "metrics layer, not by this government -- this assertion is "
                    "about the government's own summary object, not the "
                    "on-disk artifact.)",
           "final_mean_prediction_error" not in al_stats
           and not hasattr(al_gov, "get_ads_metrics"))

    # -- THE WIRED CALL SITE ---------------------------------------------------
    # `make_ads_calibration_plots` takes a `gov_keys` parameter, and
    # `_CALIBRATION_GOV_KEYS` (the tuple the two production call sites actually
    # pass) names a non-ADS government.  This is a POSITIVE guard: it proves
    # every non-ADS government's number is ACTUALLY routed through the filtered
    # engine, not merely that a caller exists syntactically.
    # `_run_calibration_mean` / `_run_calibration_half_split` are the wiring
    # `load_ads_calibration_series` and `load_ads_calibration_cycle_deltas_by_run`
    # call for every non-ADS `gov_key` — see their docstrings — so exercising
    # them directly on the mixed-outcome fixture already built above (filtered
    # mean 0.35, unfiltered 0.2125) is a direct test of the hazard this section
    # is named for, not an indirect one via a rendered figure.
    _check(results,
           "_CALIBRATION_GOV_KEYS names ADS plus at least one non-ADS "
           "government (the calibration figures have a real multi-arm "
           "caller, not just a multi-arm-capable signature)",
           (_CALIBRATION_GOV_KEYS[:1] == (_ADS_GOV_KEY,)
            and len(_CALIBRATION_GOV_KEYS) > 1),
           f"_CALIBRATION_GOV_KEYS={_CALIBRATION_GOV_KEYS!r}")

    non_ads_gov = next(g for g in _CALIBRATION_GOV_KEYS if g != _ADS_GOV_KEY)

    routed_mean = _run_calibration_mean(stats, non_ads_gov)
    _check(results,
           f"_run_calibration_mean routes a non-ADS gov_key ({non_ads_gov!r}) "
           f"through filtered_calibration_mean rather than an unfiltered "
           f"scalar (a positive routing guard)",
           _close(routed_mean, 0.35),
           f"expected 0.35 (the filtered mean computed earlier in this "
           f"section), got {routed_mean!r} — 0.2125 would mean it silently "
           f"fell back to the unfiltered mean, and None would mean it stopped "
           f"routing altogether; either is exactly the 'present, plausible "
           f"and wrong' hazard filtered_calibration_mean exists to prevent.")

    routed_first, routed_second = _run_calibration_half_split(stats, non_ads_gov)
    _check(results,
           f"_run_calibration_half_split routes a non-ADS gov_key "
           f"({non_ads_gov!r}) through filtered_half_split rather than the "
           f"unfiltered halves (a positive routing guard)",
           _close(routed_first, 0.40) and _close(routed_second, 0.30),
           f"expected (0.40, 0.30) (the filtered halves computed earlier in "
           f"this section), got ({routed_first!r}, {routed_second!r}) — "
           f"(0.225, 0.20) would mean it silently fell back to the unfiltered "
           f"halves.")

    ads_routed_mean = _run_calibration_mean(stats, _ADS_GOV_KEY)
    _check(results,
           "_run_calibration_mean does NOT filter ADS's own scalar — it "
           "reads final_mean_prediction_error verbatim (correct for ADS by "
           "construction), which this ForecastLedger-built fixture never "
           "sets, so the routed value must be None here rather than silently "
           "falling back to the filtered engine for ADS too",
           ads_routed_mean is None,
           f"got {ads_routed_mean!r}")


# ---------------------------------------------------------------------------
# 15. The scope-matched loader actually selects n_groups == 1, not the full
#     corpus -- the scope-matched figures' own version of the filtered-mean
#     enforcement check's unfiltered-mean hazard and the partition-assignment
#     /scope-domain-derivation checks' scope-transposition hazard
# ---------------------------------------------------------------------------
#
# make_scope_matched_calibration_plots exists because ADS's FULL corpus is not
# a like-for-like comparison with A+L: 81% of it is n_groups == 10, a
# different, easier prediction problem than A+L's whole-population grading
# (see load_scope_matched_calibration_deltas's own docstring for the corpus
# breakdown). The silent failure mode this section guards against is a
# scope-matched loader that quietly stops filtering by scope -- reads the
# whole partition regardless of the scopes argument, or is later rewired to
# read ADS's raw (unfiltered) calibration_cycle_deltas field the way
# _run_calibration_cycle_deltas does for the FULL-corpus figures -- and
# produces a series that LOOKS like a legitimate scope-matched figure while
# actually reproducing the full-corpus number. That failure mode is exactly
# the one the filtered-mean enforcement check (section 14) and the
# partition-assignment check (section 12) guard at the outcome and
# partition-assignment layers respectively; this section is the same
# discipline applied to the scope axis specifically.
#
# Reuses _t5j_ledger() (the partition-assignment and scope-domain-derivation
# checks' own fixture: scope "1" delta 0.40, scope "10" delta 0.05, both under
# outcome "enacted") rather than building a fresh one -- one canonical
# two-scope ledger for every scope-keyed assertion in this file, per that
# fixture's own docstring.

def _write_final_stats(root: str, gov_key: str, difficulty: int, run: int,
                        stats: dict) -> None:
    """Write *stats* as a real ``<gov_key>/<difficulty>/run_NN/final_stats.json``.

    Mirrors ``test_plot_regression.py``'s ``_write_ads_calibration_fixture``,
    scaled down to the single file this section's assertions need -- the
    production entry point under test,
    :func:`load_scope_matched_calibration_deltas`, reads from an on-disk tree,
    not from an in-memory ``final_stats`` mapping, so the test must build one.
    """
    run_dir = os.path.join(root, gov_key, str(difficulty), f"run_{run:02d}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "final_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f)


def _check_scope_matched_filter(results: list) -> None:
    led, _expected = _t5j_ledger()
    stats = led.as_summary(total_cycles=4)

    # -- non-vacuity: the fixture really does carry two DIFFERENT scope cells
    #    under the same outcome, an order of magnitude apart (0.40 vs 0.05),
    #    so no assertion below can pass by the two cells coinciding.
    cell = stats["calibration_cycle_deltas_by_outcome_then_scope"].get("enacted", {})
    _check(results,
           "fixture carries BOTH scope '1' (delta 0.40) and scope '10' "
           "(delta 0.05) under the same outcome",
           sorted(cell) == ["1", "10"], f"scope keys={sorted(cell)}")

    # -- the engine primitive, one level down from the loader under test:
    #    _filtered_cycle_deltas(scopes={"1"}) must return ONLY the scope-1
    #    delta, not the pooled pair.
    scope1_pairs = _filtered_cycle_deltas(
        stats, CROSS_ARM_OUTCOMES, frozenset({"1"}))
    _check(results,
           "_filtered_cycle_deltas(scopes={'1'}) returns exactly the scope-1 "
           "delta (0.40), not scope-10's (0.05)",
           [round(v, 6) for _c, v in scope1_pairs] == [0.40],
           f"got {scope1_pairs}")

    unfiltered_pairs = _filtered_cycle_deltas(stats, CROSS_ARM_OUTCOMES)
    _check(results,
           "POSITIVE CONTROL: omitting the scopes argument pools BOTH cells "
           "-- proves 0.40-vs-0.05 are genuinely distinguishable, so the "
           "assertion above is not a coincidence of the fixture",
           sorted(round(v, 6) for _c, v in unfiltered_pairs) == [0.05, 0.40],
           f"got {unfiltered_pairs}")

    # -- filtered_calibration_mean: the same guard one level up, at the
    #    n-weighted scalar a figure actually plots.
    scope1_mean = filtered_calibration_mean(
        stats, CROSS_ARM_OUTCOMES, frozenset({"1"}))
    _check(results,
           "filtered_calibration_mean(scopes={'1'}) returns 0.40, not the "
           "pooled 0.225",
           _close(scope1_mean, 0.40), f"expected 0.40, got {scope1_mean!r}")
    pooled_mean = filtered_calibration_mean(stats, CROSS_ARM_OUTCOMES)
    _check(results,
           "POSITIVE CONTROL: without the scopes argument the mean IS the "
           "pooled 0.225 -- confirms 0.40 above is attributable to the "
           "filter, not to the fixture only ever having one cell",
           _close(pooled_mean, 0.225), f"expected 0.225, got {pooled_mean!r}")

    # -- THE PRODUCTION ENTRY POINT: load_scope_matched_calibration_deltas
    #    walking a REAL on-disk tree, exactly what
    #    make_scope_matched_calibration_plots calls. This is the level a
    #    silently-unfiltered regression would actually ship at -- everything
    #    above establishes that the engine primitives work; this establishes
    #    that the loader actually calls them with the scope restriction in
    #    place, rather than, say, reading the same unfiltered field ADS's
    #    full-corpus figures use.
    tmp = tempfile.mkdtemp(prefix="scope_matched_filter_")
    try:
        _write_final_stats(tmp, _ADS_GOV_KEY, 50, 0, stats)
        loaded = load_scope_matched_calibration_deltas(tmp, gov_key=_ADS_GOV_KEY)
        got = sorted(round(v, 6) for pairs in loaded.values() for _c, v in pairs)
        _check(results,
               "load_scope_matched_calibration_deltas(gov_key=ADS) returns "
               "ONLY the scope-1 delta (0.40) from a real on-disk "
               "final_stats.json -- NOT scope-10's 0.05, and NOT both",
               got == [0.40], f"got {got}")

        # -- THE REGRESSION THIS TEST EXISTS TO CATCH, DEMONSTRATED DIRECTLY.
        #    A loader that silently stopped restricting by scope -- the exact
        #    "silently unfiltered series that looks plausible" hazard named in
        #    the implementation brief -- would return BOTH deltas here and
        #    still look like a legitimate, finite series. Widening this
        #    function's OWN *scopes* argument is the most direct way to prove
        #    the assertion above is a real guard on the scope restriction and
        #    not an artifact of the fixture or the temp tree: if this control
        #    did NOT go red, the assertion above would not be testing scope
        #    filtering at all.
        unfiltered_loaded = load_scope_matched_calibration_deltas(
            tmp, gov_key=_ADS_GOV_KEY, scopes=frozenset({"1", "10"}))
        got_unfiltered = sorted(
            round(v, 6) for pairs in unfiltered_loaded.values() for _c, v in pairs)
        _check(results,
               "POSITIVE CONTROL: widening load_scope_matched_calibration_deltas's "
               "own scopes argument to {'1','10'} makes the ADS series NOT "
               "n_groups==1-only (0.05 leaks back in) -- proves the assertion "
               "above actually exercises scope filtering rather than passing "
               "vacuously",
               got_unfiltered == [0.05, 0.40], f"got {got_unfiltered}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # -- Symmetric with A+L, which is scope=="1" already: the scope-matched
    #    filter must be a NO-OP on an arm whose own n_groups is always 1, not
    #    a filter that only happens to work on ADS's synthetic fixture.
    al_gov = AutocracyLookaheadGovernment(seed=91)
    _run(_build_sim(al_gov, n_agents=30, grid=14))
    al_stats = al_gov.get_calibration_summary(_TEST_CYCLES)
    al_scope_keys = sorted({
        scope
        for by_scope in (al_stats.get(
            "calibration_cycle_deltas_by_outcome_then_scope") or {}).values()
        for scope in by_scope
    })
    _check(results,
           "A+L's live ledger really is scope=='1' only (n_groups hardcoded "
           "to 1) -- otherwise the no-op check below is vacuous",
           al_scope_keys == ["1"], f"scope keys={al_scope_keys}")
    al_scope_matched = _filtered_cycle_deltas(
        al_stats, CROSS_ARM_OUTCOMES, frozenset({"1"}))
    al_full = _filtered_cycle_deltas(al_stats, CROSS_ARM_OUTCOMES)
    _check(results,
           "scope-matching A+L to {'1'} is a NO-OP -- it already only ever "
           "carries scope '1', so its scope-matched series equals its "
           "full-corpus (outcome-filtered) series exactly",
           sorted(round(v, 6) for _c, v in al_scope_matched)
           == sorted(round(v, 6) for _c, v in al_full),
           f"scope-matched={len(al_scope_matched)} obs, "
           f"full={len(al_full)} obs")


# ---------------------------------------------------------------------------

def main() -> int:
    results: list = []
    print("ADS lookahead + parameter-inference guards:")
    _check_estimator_reduction(results)
    _check_estimator_recovery(results)
    _check_once_per_cycle(results)
    _check_degenerate(results)
    _check_lookahead_control(results)
    _check_common_random_numbers(results)
    _check_review_date(results)
    _check_event_stats(results)
    _check_event_time_bounding(results)
    _check_movement_estimator(results)
    _check_schema_canary(results)
    print("\nThe shared ForecastLedger's partition and its readers:")
    _check_partition_assignment(results)
    _check_scope_domain_derivation(results)
    _check_filtered_enforcement(results)
    _check_scope_matched_filter(results)

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
