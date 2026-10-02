"""
Evidence-based inference of the environment's hidden dynamics parameters.

WHAT THIS MODULE IS FOR
-----------------------
``EvaluatorNode`` (``governments/ads.py``) rolls a candidate law forward 20
cycles and reports the projected surviving health.  That rollout consumes a
handful of *environment-dynamics parameters* — how fast health drains, how fast
resources regrow, how far an agent can travel in a cycle, how often events
arrive.  Reading those straight off the ``Simulation`` (``sim.drain_mult``,
``sim.config.regen_mult``, a hardcoded step size) would be ground truth a
government is not supposed to have.

This module supplies estimates built from what a government has legitimately
observed instead, and packages every such parameter into one frozen
:class:`EnvironmentModel` that the evaluator takes as a single argument.

THE ESTIMAND — READ THIS BEFORE JUDGING ANY NUMBER HERE
-------------------------------------------------------
    The estimand is the multiplier that would have made THE EVALUATOR'S OWN
    MODEL reproduce the observed outcome.  Agreement with ``sim.drain_mult`` /
    ``sim.config.regen_mult`` is a diagnostic coincidence, NOT a correctness
    criterion.

This is the single most consequential decision here, and it is deliberately not
the obvious one.  Two concrete consequences, both correct behaviour rather than
bias:

* ``regen_mult`` REPORTS **above** the config value at every difficulty and
  scale measured, while the raw evidence underneath it sits **below**.  Both
  halves of that sentence matter: expecting only a downward-biased figure and
  treating a high one as suspicious is the wrong read — the opposite is what a
  reader will actually see.

  The engine applies a *global* depletion haircut to every cell each cycle
  (``Grid.regenerate``) that the evaluator does not model at all, so the raw
  evidence ratio ``sum_observed / sum_exposure`` is genuinely lower than the
  config constant, and increasingly so as difficulty rises.  That much is real.
  But the reported figure is not that ratio: :class:`ShrunkRatioEstimator`
  returns a convex combination of the ratio and a prior whose mean is **1.0**,
  and 1.0 lies ABOVE ``config.regen_mult`` at every difficulty > 0.  Shrinkage
  therefore always pulls *up* here, and at realistic run lengths it more than
  cancels the haircut.

  Substituting ``K = prior_cycles * (sum_exposure / n_cycles)`` and dividing
  through by ``sum_exposure`` collapses the estimate to a weight that depends
  only on the CYCLE COUNT, which is what makes the claim falsifiable::

      value = w*1.0 + (1 - w)*r,    w = prior_cycles / (prior_cycles + n_cycles)

  Measured (reconstruction exact to 2.2e-16, so this is arithmetic, not a fit):

  =========================  =============  ========  ==========  ======
  configuration              raw ratio ``r``  config    reported    ``w``
  =========================  =============  ========  ==========  ======
  D50   500ag/50x50/150c     0.5920 (-5.8%)   0.6288   0.6400      0.1176
  D100  500ag/50x50/150c     0.2790 (-14.4%)  0.3258   0.3638      0.1176
  D50   24ag/12x12/60c       0.6207 (-1.3%)   0.6288   0.7156      0.2500
  D100  24ag/12x12/60c       0.2978 (-8.6%)   0.3258   0.4734      0.2500
  =========================  =============  ========  ==========  ======

  So the overshoot is a FINITE-SAMPLE ARTEFACT decaying like
  ``20/(20 + n_cycles)``, not a bias: +45.3% at 60 cycles and +11.7% at 150 at
  D100, tending toward the (negative) raw ratio as a run lengthens.  Two
  practical consequences.  A reported value above the config constant is the
  EXPECTED reading and not evidence of a fault.  And the gap narrows with run
  length, so figures from runs of different lengths are not comparable on this
  axis — which matters because the quick-test corpus is 30-60 cycles and the
  published sweep is 150.

  None of this is a defect to repair.  Under this module's estimand the target
  was never ``config.regen_mult``, and the prior is doing exactly its job:
  refusing to over-trust a handful of noisy early cycles.
* ``drain_mult`` converges to ``sim.drain_mult * w``, not to ``sim.drain_mult``,
  where ``w`` is the channel-mix-weighted ratio of the engine's drain
  coefficients to the evaluator's.  The two sets differ (the engine uses 0.025 /
  0.020 / 0.015 for hunger / thirst / disease where the evaluator uses 0.02 /
  0.015 / 0.03), so ``w`` runs from 0.5 for a disease-dominated regime to 1.33
  for a thirst-dominated one.  Reconciling those coefficient sets is a model
  change needing its own design pass; until then the mismatch is deliberately
  ABSORBED here rather than papered over.

**Do not "fix" this by asserting the estimates match the config.**  Such a test
looks reasonable, fails, and invites someone to repair a correct estimator into
a broken one.

THE EPISTEMIC BOUNDARY
-----------------------
Difficulty is the hidden environmental variable of the experiment.  Everything
downstream of it is hidden; everything independent of it is public.

MAY read: any agent's public state (``health``, ``hunger``, ``thirst``,
``epidemic_ids``, ``position``, ``alive``, ``*_stock``); any cell's public state
and static terrain properties; grid geometry; the active-event channel
(``active_events``, ``storm_damage_per_cycle``, ``observed_event_stats``); and
the three realized-outcome channels
(``Simulation.last_health_drain``, ``Simulation.last_move_counts``,
``Grid.collected_cells_this_cycle``).

MUST NOT read: ``sim.drain_mult``, ``sim.config.regen_mult``,
``sim.config.max_steps_per_cycle``, ``sim.config.difficulty``,
``sim.config.metabolic_rate``, ``sim.hunger_build_rate`` /
``thirst_build_rate`` / ``hunger_drain`` / ``thirst_drain`` /
``disease_drain``, ``event_system.scheduled_events``,
``event_system.pending_warnings``, or anything on ``scenario_plan``.

Note which side of the line the base drain constants fall on: this module does
NOT read ``sim.hunger_drain``.  It uses the evaluator's own coefficients
(``DRAIN_COEF_*`` below), which are part of the model being calibrated, not
knowledge about the world.  The boundary stays clean — the estimator reads
observations and its own model, never the engine's parameters.

DEPENDENCY DIRECTION
--------------------
``engine`` <- ``parameter_inference`` <- ``ads`` <- ``autocracy_lookahead``.

This module imports from ``engine.grid`` and ``engine.events`` only.  It must
NEVER import ``governments.ads``.

Both arms want the SAME thing from this module: each constructs its own
:class:`ParameterEstimator` and scores against an inferred environment, so that
the ablation between them isolates organisation instead of information quality.

Keeping this module free of ``governments.ads`` is what makes that possible.  It
is the neutral ground both arms can reach without either importing the other —
so the two share an estimator CLASS while never sharing an INSTANCE, and neither
arm's inference can drift relative to the other's.

Draws no randomness and mutates no simulation state.  A run stays reproducible
from its seed with this module in the loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from engine.events import (
    DISEASE_SPREAD_PER_SEVERITY,
    DROUGHT_FACTOR_PER_SEVERITY,
    FALLBACK_EVENT_DURATION,
    OBSERVED_EVENT_CATEGORIES,
    STORM_DAMAGE_PER_SEVERITY,
)
from engine.grid import FOOD_CAP, WATER_CAP, Terrain

if TYPE_CHECKING:
    from engine.simulation import Simulation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Weight of the prior, expressed in cycles.  See :class:`ShrunkRatioEstimator`
#: for why "cycles" rather than "exposure units" is the unit that makes one
#: class serve three parameters whose exposure scales differ by four orders of
#: magnitude.
#:
#: Numerically equal to ``engine.events.EVENT_RATE_PRIOR_CYCLES`` and
#: DELIBERATELY NOT IMPORTED FROM IT.  The match is a judgement that the two
#: should decay on the same timescale, not a shared definition; tying them
#: together would mean a future tuning pass on the event-rate prior silently
#: re-tuned the drain and regen estimators too.
ESTIMATOR_PRIOR_CYCLES = 20

#: Assertion bounds on ``drain_mult``.  The engine maximum is
#: ``engine.difficulty.difficulty_multiplier(100)``: 2.1136 at the shipped tail
#: slope of 1.75, and at most 2.375 if the tail slope is ever tuned up to
#: ``DIFFICULTY_T_MAX`` = 1.25.  The channel-mix factor described in the module
#: docstring lifts the estimand correspondingly above that per-channel maximum,
#: so 5.0 remains a sanity ceiling rather than a mechanism — see
#: :class:`ShrunkRatioEstimator` on why a clamp that fires is a bug report.
#: ``DRAIN_MULT_MAX`` stays inert at every level in 1..100 at every slope up to
#: ``DIFFICULTY_T_MAX``.
DRAIN_MULT_MIN = 0.0
DRAIN_MULT_MAX = 5.0

#: Assertion bounds on ``regen_mult``.  **0.0 is reachable and LEGITIMATE** — a
#: heavily depleted late run genuinely has no net regeneration once the global
#: haircut is netted off, and that is what the evaluator needs to be told.  Do
#: not raise this floor to "protect" the projection.  The difficulty schedule's
#: own ``regen_mult`` floor is 0.0 but never reaches exactly 0.0 inside
#: ``DIFFICULTY_T_MAX`` (minimum ~0.0625 at t=1.25), so this estimator bound
#: stays strictly below the schedule's range at every slope up to T_MAX.
REGEN_MULT_MIN = 0.0
REGEN_MULT_MAX = 2.0

#: Floor for the movement estimate: the evaluator's pre-existing hardcoded step.
#: Makes the crude regime match the evaluator's original hardcoded-step
#: behaviour on this axis by construction rather than by coincidence.
MOVEMENT_PRIOR_CEILING = 1

#: Assertion bound on the movement estimate; larger than any grid this project
#: runs, so it can only fire on a genuine accounting error.
MOVEMENT_ESTIMATE_CAP = 50

#: Cells at or above this fraction of their terrain cap are excluded from the
#: regen observation.  ONE constant, TWO justifications, which is precisely why
#: it is one threshold and not two:
#:
#:   * **Cap censoring.**  A cell at its cap shows zero realized regen no matter
#:     what ``regen_mult`` is, so including it would drag every estimate toward
#:     zero for a reason that has nothing to do with the parameter.
#:   * **Global-depletion confound.**  The unmodelled haircut costs a cell
#:     ``f * (1 - m)``, proportional to its stock, while the regen signal is
#:     independent of stock.  Restricting to low-stock cells therefore maximises
#:     signal-to-confound.
#:
#: It also disposes of ``FOOD_CAP[WATER] == 0`` and the WASTELAND row for free:
#: ``f >= 0.5 * 0`` is always true, so those are excluded automatically.
REGEN_OBS_MAX_FILL = 0.5

# --- The evaluator's own drain coefficients --------------------------------
#
# Moved here out of ``EvaluatorNode._step`` so that ``_step`` and
# :meth:`ParameterEstimator.observe` read the SAME five literals.  That shared
# reading is what makes ``drain_mult_hat`` mean "the multiplier that fixes *this*
# model" rather than "some number near the engine's constant".
#
# EACH LINE BELOW PAIRS WITH A ``delta -=`` LINE IN
# ``Simulation._apply_health_dynamics``, AND THE PAIRS DO NOT ALL MATCH:
#
#   here                      engine (_apply_health_dynamics)     ratio
#   DRAIN_COEF_HUNGER  0.02   self.hunger_drain    0.025          1.25
#   DRAIN_COEF_THIRST  0.015  self.thirst_drain    0.020          1.33
#   DRAIN_COEF_DISEASE 0.03   self.disease_drain   0.015          0.50
#   DRAIN_COEF_HAZARD  1.0    cell.hazard * dm     (unit)         1.00
#   DRAIN_COEF_STORM   1.0    storm_damage * dm    (unit)         1.00
#
# The three mismatches are deliberately out of scope to fix; the estimator
# absorbs them, which is why ``drain_mult_hat`` does not converge to
# ``sim.drain_mult``.  The table is here so that a future change to either
# side is a visible divergence rather than a silent one.
#
# If the engine gains a SIXTH drain channel with no matching term added below,
# it will be quietly absorbed into ``drain_mult_hat``.  Under this module's
# estimand that is arguably the correct degradation, but it is worth knowing
# about.
DRAIN_COEF_HUNGER = 0.02
DRAIN_COEF_THIRST = 0.015
DRAIN_COEF_DISEASE = 0.03
#: Structural terms: the evaluator charges terrain hazard and storm damage at
#: face value.  Named anyway, so the block above is a complete statement of the
#: drain model rather than a partial one.
DRAIN_COEF_HAZARD = 1.0
DRAIN_COEF_STORM = 1.0

#: Threshold above which hunger/thirst begin to cost health.  Shared with
#: ``EvaluatorNode._step`` for the same anti-drift reason as the coefficients.
DRAIN_NEED_THRESHOLD = 0.4

# --- The crude baseline's values ---------------------------------------------
#
# "Zero-effect defaults" is read as THE NEUTRAL ELEMENT OF EACH PARAMETER'S OWN
# ALGEBRA, not as literal zero everywhere.  The four values below are the whole
# of that decision; flipping to a literal-zero reading is an edit to these
# constants and nothing else.
#
# THESE ARE PRIOR MEANS, NOT AN ARM'S ENVIRONMENT.  `CRUDE_DRAIN_MULT` and
# `CRUDE_REGEN_MULT` are the two `ShrunkRatioEstimator` PRIOR MEANS (see
# `ParameterEstimator.__init__`), which is what makes "zero evidence =>
# inferred == crude" a theorem rather than a coincidence, and that identity is
# pinned by a test.  Both arms' headers also record all four as the reference
# point their inferred values are read against: "drain 1.42" only means
# something next to "crude would have been 1.0".
#
# Why not literal zeros, in one line each (stated against the prior-mean role):
#   * ``drain_mult = 0.0`` as a prior mean would shrink every early estimate
#     toward "no projected agent ever loses health", so every candidate would
#     score its group size, every comparison would tie, and BOTH arms' rankings
#     would collapse into their tie-break RNG until enough evidence accumulated
#     to overcome the prior.
#   * ``regen_mult = 0.0`` is the mirror image: unconditional resource collapse
#     predicted under every law, which again discriminates between none of them.
#   * an event rate of 0.0 IS the neutral element for that parameter, because
#     rates compose additively — it means "inject no SPECULATIVE future events",
#     while currently-active real events still apply and still expire at their
#     true ``cycles_remaining``.
#
# Same trap as a bucket defaulting to 0.0 and multiplying the score by zero:
# a 1.0 default here is not cosmetic, it is a contract that keeps an
# un-observed pairing from being silently vetoed.
CRUDE_DRAIN_MULT = 1.0
CRUDE_REGEN_MULT = 1.0
CRUDE_MAX_STEPS_PER_CYCLE = 1
CRUDE_EVENT_RATE = 0.0


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

class ShrunkRatioEstimator:
    """
    Conjugate-shrinkage ratio estimator with a self-scaling prior weight.

        estimate(t) = (prior_mean * K_t + sum observed) / (K_t + sum exposure)
        K_t         = prior_cycles * (sum exposure / n_cycles)

    K is expressed as "``prior_cycles`` worth of average per-cycle exposure"
    rather than as a fixed constant in exposure units.  That is what lets one
    class serve three parameters whose exposure units differ by four orders of
    magnitude (health-units, resource-units, dimensionless) from a single shared
    ``prior_cycles``, and it is *exactly* faithful to the estimator it
    generalises: substitute ``prior_mean=0``, ``prior_cycles=20`` and one unit of
    exposure per cycle and you get ``K_t = 20`` and ``estimate = n / (cycle + 20)``,
    which is ``EventSystem.observed_event_stats``' rate formula
    (``engine/events.py:615``) character for character.

    Expressing K absolutely rather than relatively would put it in the wrong
    units; expressing it relatively, as above, avoids that.  A unit test pins
    the reduction.

    Numerically total: ``value`` returns ``prior_mean`` exactly when no cycle has
    been observed or when total exposure is zero, and otherwise the denominator
    is ``(prior_cycles/n + 1) * sum_exposure > 0``.  There is no
    division-by-zero path to guard at any call site — cycle 0 is the ordinary
    case, not a special one.
    """

    def __init__(
        self,
        name: str,
        prior_mean: float,
        prior_cycles: float,
        lo: float,
        hi: float,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if prior_cycles < 0.0:
            raise ValueError(f"{name}: prior_cycles must be >= 0, got {prior_cycles!r}")
        if lo > hi:
            raise ValueError(f"{name}: lo ({lo}) exceeds hi ({hi})")
        self.name = name
        self.prior_mean = float(prior_mean)
        self.prior_cycles = float(prior_cycles)
        self.lo = float(lo)
        self.hi = float(hi)
        self._logger = logger or logging.getLogger("sim.parameter_inference")

        self._n_cycles = 0
        self._sum_observed = 0.0
        self._sum_exposure = 0.0
        self._last_cycle: Optional[int] = None
        #: Observations whose realized value arrived negative and was clamped to
        #: 0.0.  Nonzero means the observation channel is broken, not that the
        #: world was unusual.
        #:
        #: KNOWN LIMITATION — ``n_clamped == 0`` DOES NOT MEAN "no faults
        #: occurred".  It means no NEGATIVE observation reached this estimator,
        #: and how much that is worth differs sharply by channel:
        #:
        #: * ``regen_mult`` — INERT BY CONSTRUCTION, unconditionally.
        #:   ``ParameterEstimator._observe_regen`` floors the pooled per-cycle
        #:   total at 0 *before* calling :meth:`observe`, so a negative value
        #:   can never arrive here no matter what the engine does.  For this
        #:   channel the counter is not a fault indicator at all; the channel is
        #:   simply uninstrumented.  A broken ``collected_cells_this_cycle`` or
        #:   a bad stock diff would be silently netted to zero with no counter
        #:   and no log.  Accepted deliberately, not fixed.
        #: * ``drain_mult`` — unreachable given the engine's CURRENT dynamics,
        #:   but a live guard.  ``_apply_health_dynamics`` composes health
        #:   change from five ``delta -=`` terms and no additive one, and the
        #:   clip ``max(0.0, min(1.0, health + delta))`` can only truncate the
        #:   magnitude of a loss, so ``prev_health - health >= 0`` identically.
        #:   Add any health-INCREASING term (a medicine effect, a recovery
        #:   mechanic) and this counter starts earning its keep immediately.
        #:
        #: The difference is worth preserving in wording: the first can never
        #: fire, the second does not fire *yet*.
        self._n_clamped_observations = 0
        #: Times the running estimate landed outside ``[lo, hi]`` and was
        #: clamped.  Nonzero means the EXPOSURE MODEL has drifted from the
        #: engine — see the class note on clamps being assertions.
        self._n_bound_clamps = 0
        self._warned_bound = False

    def observe(self, cycle: int, observed: float, exposure: float) -> None:
        """
        Fold in ONE cycle's aggregate observation.

        Raises ``ValueError`` if called twice for the same *cycle* or with a
        cycle below the last one seen.  Both are silent-corruption bugs
        otherwise: double-counting inflates ``n`` and biases every later
        estimate with no symptom, and an out-of-order call means the caller has
        been moved out of the unconditional per-cycle hook.  Raising is the
        point — this is the single most important defensive check in the design.

        A negative *exposure* raises: exposure is a sum of non-negative model
        terms, so a negative one is an arithmetic error, not data.  A negative
        *observed* is clamped to 0.0 and counted, because that one CAN arise
        from a broken channel and losing the whole run over it would be a worse
        trade than recording it.

        A cycle with ``exposure == 0.0`` is recorded (``n`` increments) but
        contributes nothing to either sum.  That is deliberate and correct: it
        is a cycle that carried no information, not a cycle that did not happen,
        and the distinction matters because ``n`` is the divisor that turns
        total exposure into the per-cycle average K is denominated in.
        """
        cycle = int(cycle)
        if self._last_cycle is not None and cycle <= self._last_cycle:
            raise ValueError(
                f"{self.name}: observe() called for cycle {cycle} after cycle "
                f"{self._last_cycle}. Each cycle must be folded in exactly once, "
                f"in order; a repeat silently double-counts and a backwards call "
                f"means the caller is no longer on the unconditional per-cycle hook."
            )
        exposure = float(exposure)
        if exposure < 0.0:
            raise ValueError(
                f"{self.name}: negative exposure {exposure!r} at cycle {cycle}. "
                f"Exposure is a sum of non-negative model terms; a negative value "
                f"is an arithmetic error in the exposure expression."
            )
        observed = float(observed)
        if observed < 0.0:
            self._n_clamped_observations += 1
            self._logger.warning(
                "cycle=%d PARAM_EST_NEGATIVE_OBSERVATION estimator=%s "
                "observed=%.6g clamped_to=0.0 n_clamped=%d",
                cycle, self.name, observed, self._n_clamped_observations,
            )
            observed = 0.0

        self._last_cycle = cycle
        self._n_cycles += 1
        self._sum_observed += observed
        self._sum_exposure += exposure

        # Bound check runs HERE rather than inside `value`, so that `value`
        # stays a pure function that can be read as often as a caller likes.
        # Incrementing a diagnostic counter from a property getter would make
        # the counter a measure of how often the estimate was READ.
        raw = self._raw_estimate()
        if raw < self.lo or raw > self.hi:
            self._n_bound_clamps += 1
            if not self._warned_bound:
                self._warned_bound = True
                self._logger.warning(
                    "cycle=%d PARAM_EST_CLAMP estimator=%s raw=%.6g bounds=[%.3g, %.3g] "
                    "n_cycles=%d sum_observed=%.6g sum_exposure=%.6g — a clamp here is "
                    "an ASSERTION FAILURE, not a mechanism: it means the exposure model "
                    "has drifted from the engine. Logged once per estimator per run.",
                    cycle, self.name, raw, self.lo, self.hi,
                    self._n_cycles, self._sum_observed, self._sum_exposure,
                )

    def _raw_estimate(self) -> float:
        """The unclamped estimate.  ``prior_mean`` when there is no evidence."""
        if self._n_cycles == 0 or self._sum_exposure <= 0.0:
            return self.prior_mean
        k = self.prior_cycles * (self._sum_exposure / self._n_cycles)
        denominator = k + self._sum_exposure
        if denominator <= 0.0:
            # Reachable only if prior_cycles == 0 AND sum_exposure == 0, which
            # the branch above already caught.  Kept because an unreachable
            # division is cheaper to exclude than to debug.
            return self.prior_mean
        return (self.prior_mean * k + self._sum_observed) / denominator

    @property
    def value(self) -> float:
        """
        The current estimate, clamped to ``[lo, hi]``.

        Returns ``prior_mean`` exactly when no cycle has been observed or when
        total exposure is zero, so there is no division-by-zero path.  Pure: no
        state is mutated, so this is safe to read as often as wanted.
        """
        return min(max(self._raw_estimate(), self.lo), self.hi)

    @property
    def n_cycles(self) -> int:
        return self._n_cycles

    @property
    def sum_observed(self) -> float:
        return self._sum_observed

    @property
    def sum_exposure(self) -> float:
        return self._sum_exposure

    @property
    def n_clamped(self) -> int:
        return self._n_clamped_observations

    @property
    def n_bound_clamps(self) -> int:
        return self._n_bound_clamps

    def as_dict(self) -> Dict[str, Any]:
        """
        JSON-ready snapshot for the per-cycle run record.

        Surfaces as ``parameter_estimates`` in ``run_detail.jsonl`` and
        ``parameter_estimates_final`` in ``final_stats.json``.

        READING ``n_clamped`` FROM THESE FILES: a 0 is not a clean bill of
        health.  On the ``regen_mult`` channel the counter cannot increment at
        all — ``_observe_regen`` nets negatives away before this estimator sees
        them — so 0 there means "not instrumented", not "no faults".  On
        ``drain_mult`` a 0 is meaningful but weak: it is currently unreachable
        given the engine's monotone health decay, and only becomes informative
        if a health-increasing term is ever added.  See the field's definition
        in ``__init__`` for the full statement.  ``n_bound_clamps`` is the
        counter that carries real signal today; a nonzero value there is an
        assertion failure about the exposure model.
        """
        return {
            "value": self.value,
            "n_cycles": self._n_cycles,
            "sum_observed": self._sum_observed,
            "sum_exposure": self._sum_exposure,
            "n_clamped": self._n_clamped_observations,
            "n_bound_clamps": self._n_bound_clamps,
        }


class FlooredMaxEstimator:
    """
    Running max of noise-free lower bounds, floored at a prior ceiling.

        estimate(t) = max(prior_ceiling, max over observed samples)

    **A shrinkage blend is WRONG for this quantity and must not be substituted.**
    Each observation is an exact lower bound on a hard physical cap, not a noisy
    draw around a mean.  Averaging an observed 4-cell cycle against a prior of 1
    to report 2.5 discards proof: the world has demonstrated that 4 is possible,
    and no amount of prior belief makes it less possible.

    Known limitation, accepted and documented rather than corrected: the
    estimate under-reports until a sufficiently mobile agent-cycle is actually
    sampled.  That is a COVERAGE gap, not a noise gap, and no estimator shape
    fixes it — which is why the test plan asserts the one-sided bound (never
    above the true cap) and monotonicity, and explicitly does NOT assert that
    convergence happens by any particular cycle.
    """

    def __init__(
        self,
        name: str,
        prior_ceiling: int,
        hi: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.name = name
        self.prior_ceiling = int(prior_ceiling)
        self.hi = int(hi)
        self._logger = logger or logging.getLogger("sim.parameter_inference")
        self._running_max = int(prior_ceiling)
        self._n_cycles = 0
        self._last_cycle: Optional[int] = None
        self._n_bound_clamps = 0
        self._warned_bound = False

    def observe(self, cycle: int, sample: int) -> None:
        """Fold in one cycle's sample.  Same duplicate/out-of-order guard as
        :meth:`ShrunkRatioEstimator.observe`, and for the same reason."""
        cycle = int(cycle)
        if self._last_cycle is not None and cycle <= self._last_cycle:
            raise ValueError(
                f"{self.name}: observe() called for cycle {cycle} after cycle "
                f"{self._last_cycle}. Each cycle must be folded in exactly once, "
                f"in order."
            )
        sample = int(sample)
        if sample < 0:
            raise ValueError(
                f"{self.name}: negative sample {sample!r} at cycle {cycle}; a move "
                f"count cannot be negative."
            )
        self._last_cycle = cycle
        self._n_cycles += 1
        if sample > self._running_max:
            self._running_max = sample
        if self._running_max > self.hi and not self._warned_bound:
            self._n_bound_clamps += 1
            self._warned_bound = True
            self._logger.warning(
                "cycle=%d PARAM_EST_CLAMP estimator=%s running_max=%d cap=%d — a clamp "
                "here is an ASSERTION FAILURE: no grid this project runs permits that "
                "many steps, so the move counter is being incremented by something "
                "other than a completed relocation.",
                cycle, self.name, self._running_max, self.hi,
            )

    @property
    def value(self) -> int:
        """The estimate, floored at ``prior_ceiling`` and capped at ``hi``."""
        return min(max(self._running_max, self.prior_ceiling), self.hi)

    @property
    def n_cycles(self) -> int:
        return self._n_cycles

    @property
    def running_max(self) -> int:
        return self._running_max

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready snapshot for the per-cycle run record."""
        return {
            "value": self.value,
            "n_cycles": self._n_cycles,
            "running_max": self._running_max,
            "prior_ceiling": self.prior_ceiling,
            "n_bound_clamps": self._n_bound_clamps,
        }


# ---------------------------------------------------------------------------
# EnvironmentModel and the crude baseline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvironmentModel:
    """
    Every environment-dynamics parameter ``EvaluatorNode``'s rollout consumes,
    in one immutable object.

    One object rather than four keyword arguments, so that adding a fifth
    parameter later is a one-field change rather than a fourth argument threaded
    through three call sites.

    FROZEN ON PURPOSE.  It is shared by reference across every candidate scored
    in a decision round, and "every candidate faced the same model of the
    future" is a fair-comparison guarantee this codebase already treats as
    structural rather than incidental.  Freezing makes accidental per-candidate
    mutation a ``TypeError`` instead of a silent ranking bias.
    """

    #: Per-category event statistics, in ``observed_event_stats`` shape.
    event_stats: Mapping[str, Mapping[str, Any]]
    drain_mult: float
    regen_mult: float
    max_steps_per_cycle: int
    #: ``"crude"`` or ``"inferred"`` — **DIAGNOSTIC ONLY**.  Nothing in the
    #: rollout may branch on this.  The difference between the two regimes must
    #: live entirely in the numbers; the moment any rollout logic reads this
    #: tag, the two arms diverge in code as well as in data.
    #:
    #: BOTH GOVERNMENTS EMIT ``"inferred"``, so this field is NOT
    #: arm-discriminating and must not be made so.  It answers exactly one
    #: question — which construction path built this object, the frozen crude
    #: singleton or a live :class:`ParameterEstimator` — and that question has
    #: exactly two answers.  Adding a third value to recover a regime tag would
    #: violate the DIAGNOSTIC-ONLY contract in spirit and would create an
    #: incentive for rollout logic to branch on it.  The arm is identified by
    #: ``government.name`` and by ``government_params.organization``.
    source: str

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready snapshot for the per-round decision log."""
        return {
            "source": self.source,
            "drain_mult": self.drain_mult,
            "regen_mult": self.regen_mult,
            "max_steps_per_cycle": self.max_steps_per_cycle,
            "event_rates": {
                category: float((self.event_stats.get(category) or {}).get("rate") or 0.0)
                for category in OBSERVED_EVENT_CATEGORIES
            },
        }


def _build_crude_event_stats() -> Mapping[str, Mapping[str, Any]]:
    """
    Zero-rate statistics with the same field shape ``observed_event_stats``
    emits at ``n == 0``.

    Every field is populated rather than left absent so that every consumer can
    index the dict unconditionally, exactly as the real implementation does.

    With ``rate == 0.0`` the evaluator's injection test
    (``if u >= cat_stats["rate"]: continue``) is unconditionally true, so ZERO
    phantom events are injected **while the injection RNG stream is consumed
    exactly as many times as before**.  Both the common-random-numbers property
    and the per-round stream position therefore survive untouched — which is
    what makes the crude regime a change of numbers rather than a change of
    stream, and is why the existing CRN test still guards it.
    """
    fallback_effect = {
        "drought": DROUGHT_FACTOR_PER_SEVERITY,
        "storm": STORM_DAMAGE_PER_SEVERITY,
        "epidemic": DISEASE_SPREAD_PER_SEVERITY,
    }
    return MappingProxyType({
        category: MappingProxyType({
            "n": 0,
            "rate": CRUDE_EVENT_RATE,
            "severity": 1.0,
            "effect": fallback_effect[category],
            "duration": float(FALLBACK_EVENT_DURATION[category]),
        })
        for category in OBSERVED_EVENT_CATEGORIES
    })


#: Zero-rate event statistics for the crude regime.  Wrapped in
#: ``MappingProxyType`` at both levels because this is a process-wide singleton
#: shared by every crude evaluation: a rollout that mutated it would silently
#: poison every subsequent evaluation in the process, which is the sort of bug
#: that shows up as an unreproducible result months later.
CRUDE_EVENT_STATS: Mapping[str, Mapping[str, Any]] = _build_crude_event_stats()

#: The crude baseline, as a module-level frozen singleton.
CRUDE_ENVIRONMENT = EnvironmentModel(
    event_stats=CRUDE_EVENT_STATS,
    drain_mult=CRUDE_DRAIN_MULT,
    regen_mult=CRUDE_REGEN_MULT,
    max_steps_per_cycle=CRUDE_MAX_STEPS_PER_CYCLE,
    source="crude",
)


# There is deliberately no `crude_environment()` accessor function returning
# CRUDE_ENVIRONMENT.  With both arms driving their own estimator there is no
# call site left that needs one, and keeping a one-line, docstring-advertised
# switch back to the crude environment would be exactly the kind of affordance
# a future "let's make this configurable" change reaches for.
#
# If a crude arm is ever wanted again, the honest construction is a NEW regime
# that passes CRUDE_ENVIRONMENT explicitly — one import, fully visible — not a
# re-flip of a primary regime.
#
# The CONSTANT and the four CRUDE_* values are emphatically NOT dead; see their
# definitions above.  The two are routinely conflated and do not share a fate.


# ---------------------------------------------------------------------------
# ParameterEstimator
# ---------------------------------------------------------------------------

class ParameterEstimator:
    """
    A government's evidence-based model of the environment's hidden dynamics
    parameters.

    Owned by both ``AdsGovernment`` and ``AutocracyLookaheadGovernment``.  Each
    constructs its own in ``__init__``, updates it exactly once per simulation
    cycle from an unconditional ``tick`` hook, and reads it once per decision
    round.

    THE TWO ARMS SHARE THIS CLASS AND NEVER SHARE AN INSTANCE.  Sharing one would
    be fatal: by a handful of cycles in, the two governments have enacted
    different laws and inhabit materially different worlds, so their evidence
    differs legitimately, and pooling it would destroy both the ablation and the
    reproducibility story.  Do not add a module-level cache.

    This class is not itself the epistemic asymmetry between ADS and
    ``autocracy_lookahead``: both arms build their estimate from the same class,
    so the ablation between them isolates organisation rather than information
    quality.

    Deliberately NOT part of ``EventSystem``: what a government believes is a
    property of the GOVERNMENT.  The engine stays regime-agnostic; it publishes
    observations and takes no view on who reads them.

    Namespaced away from ``ads.py``'s pre-existing and unrelated
    ``Food``/``Water``/``Health``/``TerrainEvidence`` dataclasses, which are
    per-cycle situational summaries for proposal heuristics and have nothing to
    do with parameter inference.

    Draws no randomness and mutates no simulation state.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger("sim.parameter_inference")
        self._drain = ShrunkRatioEstimator(
            "drain_mult", CRUDE_DRAIN_MULT, ESTIMATOR_PRIOR_CYCLES,
            DRAIN_MULT_MIN, DRAIN_MULT_MAX, self._logger,
        )
        self._regen = ShrunkRatioEstimator(
            "regen_mult", CRUDE_REGEN_MULT, ESTIMATOR_PRIOR_CYCLES,
            REGEN_MULT_MIN, REGEN_MULT_MAX, self._logger,
        )
        self._movement = FlooredMaxEstimator(
            "max_steps", MOVEMENT_PRIOR_CEILING, MOVEMENT_ESTIMATE_CAP,
            self._logger,
        )
        #: Previous cycle's ``(food, water)`` per cell, in ``grid.all_cells()``
        #: order.  Flat and index-ordered rather than keyed by ``(r, c)``:
        #: the grid's cell list is stable for a run's lifetime, so an index is a
        #: sufficient key and costs one tuple instead of one tuple plus one hash
        #: per cell per cycle.
        self._prev_stocks: Optional[List[Tuple[float, float]]] = None
        self._last_observed_cycle: Optional[int] = None

    # -- accessors ---------------------------------------------------------

    @property
    def drain_mult(self) -> float:
        """Current ``drain_mult`` estimate, in EVALUATOR units (see module
        docstring: this is not ``sim.drain_mult`` and is not trying to be)."""
        return self._drain.value

    @property
    def regen_mult(self) -> float:
        """Current ``regen_mult`` estimate, net of unmodelled global depletion."""
        return self._regen.value

    @property
    def max_steps_per_cycle(self) -> int:
        """Current lower bound on the engine's per-cycle movement cap."""
        return self._movement.value

    # -- the once-per-cycle update ----------------------------------------

    def observe(self, cycle: int, sim: "Simulation") -> None:
        """
        Fold this cycle's evidence into all three estimators.

        MUST be called from the owning government's ``tick`` on EVERY cycle,
        before anything in that tick can enact.  Idempotence is deliberately NOT
        provided: a second call for the same cycle raises, because a silently
        double-counted cycle would bias every estimate with no symptom.

        A skipped cycle raises too.  The movement and regen accumulators are
        per-cycle and un-diffable, so a cycle on which this does not run is a
        cycle of evidence silently lost — and for regen specifically, diffing
        stocks across a two-cycle gap would attribute two cycles of regrowth to
        one cycle of exposure.
        """
        cycle = int(cycle)
        if self._last_observed_cycle is not None:
            expected = self._last_observed_cycle + 1
            if cycle != expected:
                # THIS MESSAGE IS WHAT AN ENGINEER READS AT 3AM WHEN A SWEEP
                # DIES AT D100, so it names both arms and both known causes, in
                # order of likelihood.
                raise ValueError(
                    f"ParameterEstimator.observe() called for cycle {cycle}, "
                    f"expected {expected}. This hook must run exactly once per "
                    f"cycle, in order: a repeat double-counts, and a gap would "
                    f"diff resource stocks across more elapsed regeneration than "
                    f"the exposure term accounts for.\n"
                    f"Two known causes, in order of likelihood:\n"
                    f"  (1) The call sits DOWNSTREAM OF AN EARLY RETURN in the "
                    f"owning government's tick. AutocracyGovernment.tick returns "
                    f"early on no-sim, NO LIVING AGENTS and no-leader; the middle "
                    f"one fires routinely at D90-D100, which is why "
                    f"AutocracyLookaheadGovernment overrides tick and observes "
                    f"before calling super().\n"
                    f"  (2) The call has been moved inside a decision-interval "
                    f"gate. It must run every cycle; a round runs on a few.\n"
                    f"Both arms drive an estimator "
                    f"(AdsGovernment and AutocracyLookaheadGovernment) — check "
                    f"the tick of whichever government's logger this came from."
                )

        self._observe_drain(cycle, sim)
        self._observe_regen(cycle, sim)
        self._observe_movement(cycle, sim)
        self._last_observed_cycle = cycle

    def _observe_drain(self, cycle: int, sim: "Simulation") -> None:
        """
        Accumulate realized health drain against the evaluator's own predicted
        unit drain.

        Runs at ``tick`` time, which is step 6 of the cycle — after health
        dynamics (step 5).  That ordering is load-bearing and favourable: each
        agent's ``hunger``, ``thirst`` and ``position`` are still EXACTLY the
        values that produced the drain that just ran, because hunger and thirst
        are updated at the top of the health-dynamics loop, before the drain
        terms, and nothing touches them between step 5 and step 6.  So the
        exposure term is computed from the real inputs rather than a stale or
        extrapolated copy — which matters because reconstructing them would
        require ``metabolic_rate`` and ``hunger_build_rate``, both
        difficulty-scaled and therefore off-limits.
        """
        drains = getattr(sim, "last_health_drain", None) or {}
        storm_damage = sim.event_system.storm_damage_per_cycle()
        grid = sim.grid

        observed = 0.0
        exposure = 0.0
        for agent in sim.agents:
            # Agents that died this cycle are EXCLUDED from both sums: the
            # `max(0.0, ...)` clip at the end of _apply_health_dynamics truncated
            # their realized drain, so including them would understate the
            # numerator against a full-strength denominator.
            if not agent.alive or agent.position is None:
                continue
            realized = drains.get(agent.agent_id)
            if realized is None:
                # Never alive during step 5 of this cycle — no observation.
                continue

            hunger = agent.hunger
            thirst = agent.thirst
            unit_drain = 0.0
            if hunger > DRAIN_NEED_THRESHOLD:
                unit_drain += DRAIN_COEF_HUNGER * hunger
            if thirst > DRAIN_NEED_THRESHOLD:
                unit_drain += DRAIN_COEF_THIRST * thirst
            # KNOWN BOUNDED APPROXIMATION: len(epidemic_ids) read here can differ
            # by +/-1 from the value the drain actually used, because
            # `agent.infect()` fires later in the same per-agent block and
            # `tick_infections` fires after it.  Affects only agents whose
            # infection set changed on this exact cycle.  Accepted: the
            # alternative is having the engine compute the government's exposure
            # term, which puts the model on the wrong side of the layering.
            if agent.epidemic_ids:
                unit_drain += DRAIN_COEF_DISEASE * len(agent.epidemic_ids)

            r, c = agent.position
            if grid.in_bounds(r, c):
                cell = grid.cell(r, c)
                unit_drain += DRAIN_COEF_HAZARD * cell.hazard
                if storm_damage > 0 and not cell.shelter:
                    unit_drain += DRAIN_COEF_STORM * storm_damage

            observed += realized
            exposure += unit_drain

        self._drain.observe(cycle, observed, exposure)

    def _observe_regen(self, cycle: int, sim: "Simulation") -> None:
        """
        Accumulate realized resource regrowth against modelled regrowth at
        ``regen_mult = 1``.

        Food and water are POOLED into the one estimator, matching the single
        scalar that both the config and the evaluator use.

        Per-cell deltas are deliberately NOT clamped at zero.  A negative delta
        is real — it is the global depletion haircut the evaluator does not
        model — and under this module's estimand it belongs in the estimate.
        Only the final ``value`` is floored, at ``REGEN_MULT_MIN``.
        """
        grid = sim.grid
        cells = grid.all_cells()

        # Present-tense read, identical to the one `evaluate()` performs and to
        # the value `regenerate()` consumed at step 2 of this same cycle: an
        # event that started or ended this cycle did so in step 1, before both.
        drought_factor = max(
            (e.drought_factor for e in sim.event_system.active_events
             if e.event_type.value == "drought" and not e.cancelled),
            default=0.0,
        )
        drought_scale = max(0.0, 1.0 - drought_factor)

        prev = self._prev_stocks
        observed = 0.0
        exposure = 0.0

        # No previous snapshot (cycle 0), or the grid changed shape under us.
        # Either way there is nothing to diff, so this cycle is recorded as
        # carrying no information rather than skipped — see
        # ShrunkRatioEstimator.observe on why the distinction matters.
        if prev is not None and len(prev) == len(cells):
            collected = getattr(grid, "collected_cells_this_cycle", None) or frozenset()
            for index, cell in enumerate(cells):
                if cell.terrain == Terrain.WASTELAND:
                    continue                    # regenerate() skips it outright
                if (cell.row, cell.col) in collected:
                    continue                    # excludes the direct `-= taken`
                prev_food, prev_water = prev[index]
                if prev_food < REGEN_OBS_MAX_FILL * FOOD_CAP[cell.terrain]:
                    observed += cell.food - prev_food
                    exposure += cell.food_regen * drought_scale
                if prev_water < REGEN_OBS_MAX_FILL * WATER_CAP[cell.terrain]:
                    observed += cell.water - prev_water
                    exposure += cell.water_regen * drought_scale

        # A net-negative pooled delta is legitimate (heavy depletion), but the
        # estimator's own contract clamps a negative `observed` to 0 and counts
        # it as a channel fault.  Netting to zero here instead avoids logging a
        # spurious fault every depleted cycle, while still recording the cycle's
        # exposure so the shrinkage pulls the estimate down as it should.
        #
        # THE COST, STATED PLAINLY: this leaves the regen channel with NO fault
        # indicator whatsoever.  `ShrunkRatioEstimator.n_clamped` is the only
        # one it had, and this line guarantees it can never increment — so for
        # this channel `n_clamped == 0` carries no information at all.  A broken
        # `collected_cells_this_cycle`, a mis-signed stock diff, or an exposure
        # expression that drifts from `Grid.regenerate` would all be silently
        # zeroed here with no counter and no log line.
        #
        # Netting to zero does NOT keep the counter meaningful as a fault
        # indicator — it is precisely what makes the counter meaningless here.
        # Measured across every configuration tested, including 500 agents /
        # 50x50 / 150 cycles at D100, this branch fires 0 times out of 149
        # cycles, so it is not even buying spurious-fault suppression.
        #
        # Kept anyway, and kept UNINSTRUMENTED, by explicit decision: a real
        # regen-channel fault indicator is a design task in its own right (what
        # counts as a fault when the estimand tolerates a depletion haircut?)
        # and is deliberately deferred rather than bodged.
        # Do not read a clean `n_clamped` on this channel as a health signal.
        if observed < 0.0:
            observed = 0.0
        self._regen.observe(cycle, observed, exposure)

        self._prev_stocks = [(cell.food, cell.water) for cell in cells]

    def _observe_movement(self, cycle: int, sim: "Simulation") -> None:
        """
        Fold in the largest number of completed relocations any single agent
        achieved this cycle.

        Not a per-agent loop: the estimand is a population-wide cap, so the
        running max over the population's per-cycle maxima is the same quantity
        with less bookkeeping.
        """
        counts = getattr(sim, "last_move_counts", None) or {}
        self._movement.observe(cycle, max(counts.values(), default=0))

    # -- the once-per-round read -------------------------------------------

    def environment_model(self, sim: "Simulation", cycle: int) -> EnvironmentModel:
        """
        Package the current estimates plus the event statistics into an
        :class:`EnvironmentModel` tagged ``source="inferred"``.

        Event statistics are DELEGATED to ``EventSystem.observed_event_stats``
        rather than reimplemented here.  Three reasons: the existing
        implementation is already correct and already firewalled against reading
        the future schedule; its exact arithmetic is load-bearing for the
        published corpus; and re-deriving it would fork a formula whose only
        virtue is that there is one of it.  :class:`ShrunkRatioEstimator`
        generalises that formula, and a unit test pins the correspondence —
        which captures the benefit of the shared abstraction without taking on
        the risk of actually re-pointing the corpus at a second implementation.

        Called once per decision round; the frozen result is then shared across
        every candidate, so "every candidate faced the same model of the future"
        is enforced by immutability rather than by convention.
        """
        return EnvironmentModel(
            event_stats=sim.event_system.observed_event_stats(cycle),
            drain_mult=self._drain.value,
            regen_mult=self._regen.value,
            max_steps_per_cycle=self._movement.value,
            source="inferred",
        )

    def as_record(self) -> Dict[str, Any]:
        """Per-cycle JSON block for ``run_detail.jsonl``."""
        return {
            "drain_mult": self._drain.as_dict(),
            "regen_mult": self._regen.as_dict(),
            "max_steps": self._movement.as_dict(),
        }
