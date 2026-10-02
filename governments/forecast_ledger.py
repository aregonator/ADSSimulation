"""
The predicted/realized forecast-grading ledger, shared by both foresight arms.

WHY THIS MODULE EXISTS
----------------------
Both ``AdsGovernment`` and ``AutocracyLookaheadGovernment`` carry a forecast
ledger, so that the manuscript can say *"both arms forecast equally well; only
one coordinates."*

That sentence is only verifiable if the two arms **measure** forecast quality
with the same code.  Two copies would make it unverifiable the first time they
drifted — which is the identical argument that keeps ``EvaluatorNode`` shared and
unforked, and the identical argument that keeps ``ParameterEstimator`` in its
own module rather than embedded in ``ads.py``.  So the ledger is shared code
rather than duplicated per arm, and ADS's persisted output is held to an
additive-only + projection-identical gate to guarantee that sharing it changed
nothing about what ADS already emitted.

DEPENDENCY DIRECTION
--------------------
``engine`` <- { ``parameter_inference``, ``forecast_ledger`` } <- ``ads``
            <- ``autocracy_lookahead``

This module imports **nothing** from ``governments/`` — not even
``LOOKAHEAD_CYCLES``, which lives in ``ads.py`` and would close an import cycle.
The horizon is a constructor argument instead.  That is the better design
independently of the cycle: it makes the ledger table-testable with no
``Simulation`` and no horizon assumption, which is the property that lets
``_check_review_date`` and the partition-assignment test run directly against
this class.

WHAT IS COMPARED TO WHAT — read this before changing anything here.
-------------------------------------------------------------------
This documents the MECHANISM, not the regime: it lives here rather than in
``ads.py`` so that the next reader does not conclude the ledger is
ADS-specific.

This is a MEASUREMENT loop, not a control loop.  Nothing it computes is fed back
into ranking, scoring or the forecast.  The mechanism by which accuracy is
expected to improve over a run lives entirely upstream, in
``ParameterEstimator``'s inference on the evaluator's INPUTS.

The forecast is ``predicted``: the evaluator's projected mean health per agent of
the evaluated group after ``horizon`` cycles, dead agents contributing 0, exactly
as ``evaluate`` returned it.  ``realized`` is computed with the identical
functional form over the identical agents, exactly ``horizon`` cycles later.
Same units, same population, same elapsed horizon — which is the only thing that
makes their difference a *prediction error* rather than a units mismatch.

WHEN, and why it is a DATE and not a WINDOW.  The review date is fixed at
``open`` + ``horizon`` rather than tied to the originating law's own duration
(5-20 cycles, and essentially never equal to the projection horizon).  Pinning
the review to a duration-dependent expiry would difference a projection made
over one horizon against a snapshot taken after a shorter or longer interval,
so the residual would mix real forecast error with an interval mismatch.
Fixing the review date at ``open`` + ``horizon`` keeps the two quantities over
the same elapsed interval, so the residual is actual forecast error.

The review fires regardless of what became of the law — still active, expired,
repealed, superseded, or never enacted at all.  This is intentional, not an
oversight: the graded question is about the FORECAST FOR THESE AGENTS, and it
stays answerable however the intervention behind it ended.

Two places where the obvious shortcut is wrong.  Both are settled, not
provisional:

  1. NOT the crisis-boosted ranking score (``Candidate.adjusted_score``).  The
     boost is a priority multiplier in [1.0, 1.70], not a health forecast;
     differencing it against realized health would make the error series a
     measure of how often crises occurred rather than of predictive accuracy,
     and would bias every category in the same direction, which discriminates
     between none of them.  This is why :meth:`ForecastLedger.open` takes
     ``predicted`` as a positional argument and relegates the boosted figure to
     an opaque tag.
  2. NOT ``law.applies_to``.  For the REDISTRIBUTE_* law types ``applies_to`` is
     the list of *donors*, while the prediction was scored over the
     recipient-side group.  Measuring donors against a forecast made about
     recipients is not an error signal at all.  ``applies_to`` is still recorded
     for traceability; ``evaluated_agent_ids`` is what the error is computed
     over.  This is why the two are separate parameters rather than one.

THE TWO DERIVED TOLERANCES
--------------------------
See :data:`_ROUND6_HALF_QUANTUM` and :data:`_SUM_TOL` below.  They are exported
for the gate and the tests; neither may be replaced by a round number at a call
site.
"""

from __future__ import annotations

import logging
from typing import (
    AbstractSet, Any, Dict, List, Mapping, Optional, Sequence, Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:                                     # pragma: no cover
    from engine.agent import Agent
    from engine.simulation import Simulation


# ---------------------------------------------------------------------------
# Emission rounding, and the two tolerances derived from it
# ---------------------------------------------------------------------------

#: Decimal places the per-closure deltas are rounded to at emission.
#:
#: This is the single source of truth for the rounding that
#: :data:`_ROUND6_HALF_QUANTUM` is derived from.  Changing it without changing
#: that constant is the mistake the derivation below exists to prevent.
DELTA_ROUND_PLACES = 6

#: Half the quantum of ``round(x, DELTA_ROUND_PLACES)``.  The maximum
#: displacement a single observation suffers at emission, and therefore — since
#: the mean of values each displaced by at most d is itself displaced by at most
#: d — the exact tight bound on the gap between a mean of rounded deltas and the
#: same mean of unrounded deltas.  Attained only if every observation rounds the
#: same direction by the full half-quantum.  Measured on real data (the ADS
#: gate fixture) at n=94: 3.70e-08.
#:
#: Composed from its derivation rather than written as ``5e-7``: the literal
#: would silently stop matching if ``DELTA_ROUND_PLACES`` ever moved.
_ROUND6_HALF_QUANTUM = 0.5 * (10.0 ** -DELTA_ROUND_PLACES)

#: Naive summation of n non-negative floats has relative error bounded by
#: ``(n - 1) * eps``.  At this design's worst case — n = 1e4 closures in one
#: cell, delta magnitude <= 1.0 — that is 2.2e-12 on the mean.  The constant
#: below clears that bound by ~450x while sitting ~500x BELOW
#: :data:`_ROUND6_HALF_QUANTUM`, so it cannot mask a rounded-vs-unrounded
#: mismatch, which is the error class this tolerance must stay sensitive to.
_FLOAT_EPS = 2.220446049250313e-16          # sys.float_info.epsilon
_WORST_CASE_CELL_N = 10_000
_SUM_TOL = 1e-9
assert _SUM_TOL > (_WORST_CASE_CELL_N - 1) * _FLOAT_EPS, (
    "_SUM_TOL must clear the worst-case naive-summation error bound"
)
assert _SUM_TOL < _ROUND6_HALF_QUANTUM, (
    "_SUM_TOL must sit below the rounding half-quantum, or it can mask a "
    "rounded-vs-unrounded mismatch"
)

#: Tolerance for comparisons that CROSS the rounding boundary — one side derived
#: from the 6-dp-rounded leaves, the other from the unrounded running sum.
#: Applies to exactly one invariant — the ADS-side check that a rounded running
#: total matches its unrounded counterpart — and to nothing else.  The rule,
#: stated once: never compare a rounded quantity to an unrounded one where
#: there is a choice.
_CROSS_ROUNDING_TOL = _ROUND6_HALF_QUANTUM + _SUM_TOL


# ---------------------------------------------------------------------------
# ForecastLedger
# ---------------------------------------------------------------------------

class ForecastLedger:
    """
    Opens forecasts, grades them ``horizon`` cycles later, and reports.

    MEASURES ONLY.  Feeds nothing back.  Draws no randomness — that last
    property is load-bearing and is asserted by the A+L no-op gate: both arms'
    evaluators re-seed per ``evaluate()`` call, so a single stray draw inside
    this class would desynchronise every subsequent decision and change the
    trajectory of a run this class is only supposed to be watching.
    """

    def __init__(self, horizon: int, logger: logging.Logger) -> None:
        """
        *horizon* is the review lag in cycles: a forecast opened on cycle ``t``
        is graded on cycle ``t + horizon``.

        *logger* is REQUIRED, not defaulted.  Forecasts must log under the
        owning government's ``sim.<Gov>`` logger; a module-level default would
        silently re-route ADS's ``FORECAST_PREDICT`` / ``FORECAST_CLOSE`` lines
        for no reason.
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon!r}")
        self._horizon = int(horizon)
        self._logger = logger

        #: Keyed by CLOSE cycle, so the question asked on every one of the run's
        #: cycles is "what is due *now*" — a dict lookup — and not "which of the
        #: open predictions has come of age", which would be a scan of the whole
        #: ledger once per cycle.  A list per key because one round can open
        #: several forecasts and they therefore share a close cycle.
        #:
        #: Predictions whose close cycle falls past the end of the run are never
        #: popped and are reported as ``calibration_open_predictions_at_end``.
        self._due: Dict[int, List[Dict[str, Any]]] = {}

        #: Maintained alongside ``_due`` so "how many are open" stays O(1).
        #: Summing the bucket lengths would be cheap too, but this is read on
        #: every cycle by :meth:`as_record`.
        self._open_predictions: int = 0

        #: Every closed observation, as
        #: ``(cycle_closed, abs_err, outcome, scope_key)``.
        #:
        #: ONE list, from which BOTH the flat ``calibration_cycle_deltas`` series
        #: and its two-level partition are derived.  That is not a convenience:
        #: it is what makes the gate's projection identity — concatenate every
        #: partition leaf, sort, and it must equal the flat series element for
        #: element — true BY CONSTRUCTION rather than by two code paths happening
        #: to agree.  Do not add a second accumulator for the partition.
        #:
        #: The running totals below could be derived from this list, but
        #: recomputing a mean over it on every cycle is O(n) for a value needed
        #: once per cycle; all of them are updated at exactly one site
        #: (:meth:`close`) so they cannot drift apart.  Bounded by the number of
        #: forecasts a run opens (low thousands worst case) — no pruning needed.
        self._observations: List[Tuple[int, float, str, str]] = []

        self._abs_delta_sum: float = 0.0
        self._closed_total: int = 0

        #: Closures whose forecast was exactly 0.0 — a group projected extinct.
        #: COUNTED, NOT SKIPPED: with no ratio to compute there is nothing that
        #: could divide by zero, so such a closure is graded like any other and
        #: contributes to the error series.  A pure diagnostic — "how often did
        #: the evaluator write a group off entirely" is worth knowing and is not
        #: recoverable from the error series alone.
        self._n_degenerate: int = 0

        #: The originating law's fate at the review date, as a THREE-way split.
        #:
        #: ``no_law`` is why this is three buckets rather than two: A+L opens a
        #: forecast on every decision,
        #: including ``no_action`` and ``repealed``, which have no originating
        #: law at all (``tags["law_id"] is None``).  Filing those under "lifted"
        #: would conflate "the intervention ended" with "there was never an
        #: intervention".  ADS only ever opens a forecast on an enactment, so
        #: ``n_closed_no_law`` is a true, informative zero there — which is a
        #: different thing from a field that is constant for both arms.
        self._n_closed_law_active: int = 0
        self._n_closed_law_lifted: int = 0
        self._n_closed_no_law: int = 0

        #: Reset at the top of every :meth:`close` call, so a cycle in which
        #: nothing closed reports 0 rather than the last nonzero value.
        self._closed_this_cycle: int = 0

    # ------------------------------------------------------------------
    # Read-only accessors — same names and semantics as the AdsGovernment
    # fields they replace, so the tests that poke at ledger internals
    # re-point rather than change in intent.
    # ------------------------------------------------------------------

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def open_predictions(self) -> int:
        return self._open_predictions

    @property
    def closed_this_cycle(self) -> int:
        return self._closed_this_cycle

    @property
    def closed_total(self) -> int:
        return self._closed_total

    @property
    def n_degenerate(self) -> int:
        return self._n_degenerate

    @property
    def n_closed_law_active(self) -> int:
        return self._n_closed_law_active

    @property
    def n_closed_law_lifted(self) -> int:
        return self._n_closed_law_lifted

    @property
    def n_closed_no_law(self) -> int:
        return self._n_closed_no_law

    @property
    def observations(self) -> List[Tuple[int, float]]:
        """
        The closed series as ``(cycle, abs_err)`` pairs, UNROUNDED.

        Shape-compatible with the ``_calib_observations`` list this replaces, so
        callers that read ``observations[0][1]`` keep working.  A fresh list each
        call: the internal accumulator carries two more fields and must not be
        handed out for a caller to mutate.
        """
        return [(cycle, err) for cycle, err, _, _ in self._observations]

    def due_cycles(self) -> List[int]:
        """
        Close cycles that still hold at least one ungraded forecast, ascending.

        Exists so a test can assert the ledger drained — "no entry is left for a
        cycle already past" — without reaching into ``_due``.  That assertion is
        the regression test for this class's one silent failure mode: a skipped
        ``close()`` strands every forecast whose review date landed on it, the
        error series quietly shrinks, and
        ``calibration_open_predictions_at_end`` quietly inflates.  Loud failure
        you find in three minutes; silent failure you publish.
        """
        return sorted(self._due)

    def due_law_ids(self, cycle: int) -> List[Optional[str]]:
        """
        The ``law_id`` tag of every forecast whose review date is *cycle*, in
        open order.  ``None`` for a decision that enacted nothing.

        "What is due now" is a legitimate read, and exposing it here is what lets
        the T+horizon review-date test observe closures without reaching into
        ``_due`` — the coupling that made that test break when this code moved.
        Read-only: a fresh list, and the entries themselves are not handed out.
        """
        return [e["tags"].get("law_id") for e in self._due.get(int(cycle), ())]

    def mean_abs_delta(self) -> Optional[float]:
        """
        All-time mean ``abs(predicted - realized)``, or None before the first
        closure.

        UNFILTERED by construction, on both arms, and deliberately so — see the
        note on :meth:`as_summary`.  On ADS this is also what reaches
        ``final_stats.json`` as ``final_mean_prediction_error``, via
        ``get_ads_metrics`` and ``MetricsCollector.summary()``.  A+L does not
        join that hook.
        """
        if self._closed_total == 0:
            return None
        return self._abs_delta_sum / self._closed_total

    # ------------------------------------------------------------------
    # The two mutators
    # ------------------------------------------------------------------

    def open(
        self,
        cycle: int,
        predicted: float,
        evaluated_agent_ids: Sequence[str],
        tags: Mapping[str, Any],
    ) -> None:
        """
        Open one forecast, due to close at ``cycle + horizon``.

        *predicted* is the evaluator's own projection for *evaluated_agent_ids*
        — never a boosted ranking score.  *evaluated_agent_ids* is the population
        the error will be computed over; it is separate from any ``applies_to``
        tag for the donor/recipient reason given in the module docstring.

        *tags* is traceability metadata the ledger does not interpret, with
        these exceptions, which ARE interpreted and must be supplied by both
        arms so the two emit a shape-identical partition:

        ==================  ==========================================
        ``law_id``          ``str`` or ``None``.  ``None`` means "this
                            decision enacted nothing" and routes the
                            closure to ``n_closed_no_law``.
        ``outcome``         partition axis 1.  Constant ``"enacted"`` on
                            ADS; five-valued on A+L.
        ``n_groups``        partition axis 2 (the scope axis).  Variable
                            on ADS; constant ``1`` on A+L.  The two tags
                            are duals — each arm's informative axis is
                            the other's degenerate one — which is why one
                            partition keyed on both costs one copy of the
                            data rather than two.
        ``category``        interpolated into the two log lines.
        ``law_type``        interpolated into the two log lines.
        ``applies_to``      length interpolated into the PREDICT line.
        ``adjusted_at_enactment``  the boosted ranking score, PREDICT line
                            only.  Defaults to *predicted* when absent,
                            which is correct for an arm that applies no
                            boost — A+L's ``norm = raw / k`` IS its score.
        ==================  ==========================================

        No per-cycle contract: unlike :meth:`close`, this is called from a
        decision path and may be called any number of times, including zero, on
        any cycle.
        """
        close_cycle = int(cycle) + self._horizon
        entry = {
            # `predicted`, not `raw_predicted`: there is exactly one forecast
            # value, with no separate calibration/correction step, so a "raw"
            # prefix would be misleading.
            "predicted": float(predicted),
            "enacted_cycle": int(cycle),
            "close_cycle": close_cycle,
            "evaluated_agent_ids": list(evaluated_agent_ids),
            "tags": dict(tags),
        }
        self._due.setdefault(close_cycle, []).append(entry)
        self._open_predictions += 1

        tag = entry["tags"]
        # This format string matches ADS's own FORECAST_PREDICT log lines, so
        # log parsers keyed on it work unchanged regardless of which module
        # emits it.
        self._logger.debug(
            "cycle=%d FORECAST_PREDICT law_id=%s category=%s law_type=%s "
            "predicted=%.6f adjusted=%.6f "
            "close_cycle=%d n_evaluated=%d n_applies_to=%d open=%d",
            cycle, tag.get("law_id"), tag.get("category"), tag.get("law_type"),
            entry["predicted"],
            float(tag.get("adjusted_at_enactment", entry["predicted"])),
            close_cycle, len(entry["evaluated_agent_ids"]),
            len(tag.get("applies_to") or ()),
            self._open_predictions,
        )

    def close(
        self,
        cycle: int,
        sim: "Simulation",
        active_law_ids: AbstractSet[str],
    ) -> None:
        """
        Grade every forecast whose review date is *cycle*.  MEASURES ONLY.

        MUST be called on EVERY cycle, not only on decision cycles: the review
        date is ``open`` + ``horizon`` and need not land on a decision cadence.
        The lookup is a single dict ``pop``, so a cycle with nothing due costs
        one hash and touches no counter except the per-cycle reset.

        *active_law_ids* is read for the law-fate classification only, and is
        passed in rather than read off a government so that this class stays
        regime-agnostic and directly testable.  Its correctness depends on the
        CALLER placing this downstream of that cycle's law expiry — see each
        government's call site for why that placement is what it is.
        """
        self._closed_this_cycle = 0

        due = self._due.pop(int(cycle), None)
        if not due:
            return

        # Built once per cycle, and only if something actually closed.
        # `active_law_ids` is a caller-supplied parameter, not built here, so
        # this class stays regime-agnostic (see the docstring above).
        agents_by_id: Dict[str, "Agent"] = {a.agent_id: a for a in sim.agents}

        for entry in due:
            self._open_predictions -= 1
            tag = entry["tags"]
            law_id = tag.get("law_id")
            measured_ids: List[str] = entry["evaluated_agent_ids"]
            if not measured_ids:
                self._logger.debug(
                    "cycle=%d CALIB_CLOSE_SKIPPED law_id=%s "
                    "reason=no_measured_agents",
                    cycle, law_id,
                )
                continue

            # Dead agents contribute 0 to the numerator but still count in the
            # denominator — identical to how EvaluatorNode.evaluate scores a
            # projection, so predicted and realized punish mortality the same
            # way.  An id absent from the simulation entirely (should not happen;
            # agents are never removed from sim.agents) also contributes 0 rather
            # than silently shrinking the denominator.
            total = 0.0
            missing = 0
            for aid in measured_ids:
                agent = agents_by_id.get(aid)
                if agent is None:
                    missing += 1
                    continue
                if agent.alive:
                    total += agent.health
            realized = total / len(measured_ids)

            predicted = entry["predicted"]

            # THE reported metric, and the whole of it: the evaluator's own
            # projection differenced against the same agents' real outcome at the
            # same horizon.  No paired null exists within a run, because there is
            # no second number to difference against — the honest within-run
            # comparison is the first-half/second-half split.
            abs_err = abs(predicted - realized)

            # Written as `not (x > 0.0)` rather than `x == 0.0` so a negative or
            # NaN forecast — neither of which the evaluator can currently produce
            # — is also counted rather than passing silently.
            if not (predicted > 0.0):
                self._n_degenerate += 1

            self._closed_this_cycle += 1
            self._closed_total += 1
            self._abs_delta_sum += abs_err
            self._observations.append(
                (int(cycle), abs_err,
                 _outcome_key(tag), _scope_key(tag))
            )

            # The law's fate is recorded but NOT acted on.  "lifted" is a normal
            # outcome rather than an impossible one, which is exactly why it is
            # worth counting; "no law" is the third case, for closures with no
            # originating law at all.
            if law_id is None:
                self._n_closed_no_law += 1
                law_still_active = False
            elif law_id in active_law_ids:
                self._n_closed_law_active += 1
                law_still_active = True
            else:
                self._n_closed_law_lifted += 1
                law_still_active = False

            # This format string matches ADS's own FORECAST_CLOSE log lines.
            self._logger.debug(
                "cycle=%d FORECAST_CLOSE law_id=%s category=%s law_type=%s "
                "predicted=%.6f realized=%.6f abs_err=%.6f n=%d missing=%d "
                "window=%d law_still_active=%s",
                cycle, law_id, tag.get("category"), tag.get("law_type"),
                predicted, realized, abs_err, len(measured_ids), missing,
                cycle - entry["enacted_cycle"], law_still_active,
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def as_record(self) -> Dict[str, Any]:
        """
        The per-cycle block for ``run_detail.jsonl``, identical on both arms.

        Pure read of state already computed during the owning government's tick;
        no side effects, so it is safe to call on every cycle.  Dense by design —
        a consumer can plot ``mean_abs_delta_alltime`` against cycle without
        reindexing.

        Deliberately does NOT carry ``n_closed_no_law`` or either partition.
        Those are per-RUN summaries; emitting them once per cycle would inflate
        ``run_detail.jsonl`` and, on ADS, would break the no-op gate's clause
        holding that file unchanged.
        """
        return {
            "open_predictions": self._open_predictions,
            "closed_this_cycle": self._closed_this_cycle,
            "closed_total": self._closed_total,
            "n_degenerate": self._n_degenerate,
            "mean_abs_delta_alltime": self.mean_abs_delta(),
        }

    def as_summary(self, total_cycles: Optional[int] = None) -> Dict[str, Any]:
        """
        The per-run block merged into ``final_stats.json``, identical in shape on
        both arms.

        Splits the closed observations into halves of the run so that "did the
        forecast get better as evidence accumulated" is a subtraction rather than
        a re-run.  An observation is attributed to the cycle on which it *closed*
        (when the error became knowable), not the cycle on which the forecast was
        opened.

        *total_cycles* is the configured run length; the boundary is
        ``total_cycles // 2``, so a 150-cycle run splits 0-74 / 75-149.  When the
        caller does not know the run length, the last cycle on which an
        observation closed is used instead, which is a lower bound and will bias
        the split late if the tail of the run closed nothing.  Callers that have
        ``config.max_cycles`` should always pass it.

        The ``n_`` counts are not decoration: a mean over two observations and a
        mean over forty are not comparable, and a half with zero closures reports
        ``None`` rather than a misleading ``0.0``.

        THREE SCALARS HERE ARE UNFILTERED ALL-TIME MEANS AND MUST STAY THAT WAY:
        ``calibration_mean_abs_err`` and the two ``_half`` fields average over
        EVERY closure.  On ADS that is every enacted law.  On A+L it additionally
        includes ``retained``, ``repealed`` and ``no_action`` closures that ADS
        never grades at all, so plotting A+L's scalar beside ADS's compares two
        different populations.  The fix is NOT to filter here — changing ADS's
        values would break the no-op gate, and changing only A+L's would break
        shape-identity in substance while preserving it in form.  The fix is that
        no figure may read these for A+L: it must go through
        ``benchmark_core.filtered_calibration_mean`` /
        ``filtered_half_split``, which derive the filtered numbers from the
        partition below.  This docstring is the pointer; those two functions are
        the enforcement.

        Does NOT emit ``final_mean_prediction_error``.  That key is written by
        ``MetricsCollector.summary()`` from the last cycle's snapshot, and
        ``_merge_calibration_summary``'s ``summary.update(...)`` runs AFTER it —
        so a key of that name here would not error and would not duplicate, it
        would silently overwrite the metrics layer's value with no log line.
        Nor ``ads_forecast_schema_version`` / ``parameter_estimates_final``:
        those belong to the government (a provenance token and the estimator's
        state respectively), not to the ledger.
        """
        if total_cycles is not None and total_cycles > 0:
            boundary = total_cycles // 2
        elif self._observations:
            boundary = (max(c for c, _, _, _ in self._observations) + 1) // 2
        else:
            boundary = 0

        first: List[float] = []
        second: List[float] = []
        for closed_cycle, abs_delta, _, _ in self._observations:
            (first if closed_cycle < boundary else second).append(abs_delta)

        def _mean(values: List[float]) -> Optional[float]:
            return (sum(values) / len(values)) if values else None

        deltas, closures = self._partition()

        return {
            "calibration_mean_abs_delta_first_half": _mean(first),
            "calibration_mean_abs_delta_second_half": _mean(second),
            "calibration_n_closed_first_half": len(first),
            "calibration_n_closed_second_half": len(second),
            "calibration_open_predictions_at_end": self._open_predictions,
            "calibration_half_split_cycle": boundary,
            "calibration_mean_abs_err": self.mean_abs_delta(),
            "calibration_n_degenerate": self._n_degenerate,
            # How often the review outlived the law it was opened for, and how
            # often there was no law to outlive.
            "calibration_n_closed_law_active": self._n_closed_law_active,
            "calibration_n_closed_law_lifted": self._n_closed_law_lifted,
            "n_closed_no_law": self._n_closed_no_law,
            # The full flat series behind the two halves above, kept rather than
            # discarded after bucketing.  Tuples are not valid JSON, so each
            # observation is flattened to a ``[cycle, abs_delta]`` pair; the list
            # is already cycle-ascending because closures are appended in
            # `close()` as the simulation advances, so no caller needs to re-sort
            # it.  The post-burn-in trend reader and
            # `load_ads_calibration_cycle_deltas` read exactly this shape.
            "calibration_cycle_deltas": self.cycle_deltas(),
            # The same data, re-indexed by (outcome, scope).  See _partition.
            "calibration_cycle_deltas_by_outcome_then_scope": deltas,
            "calibration_closures_by_outcome_then_scope": closures,
        }

    def cycle_deltas(self) -> List[List[Any]]:
        """The flat ``[[cycle, delta_6dp], ...]`` series."""
        return [
            [cycle, round(abs_delta, DELTA_ROUND_PLACES)]
            for cycle, abs_delta, _, _ in self._observations
        ]

    def _partition(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Re-index the closed series by ``outcome`` then by ``str(n_groups)``.

        Returns ``(cycle_deltas_by_outcome_then_scope,
        closures_by_outcome_then_scope)``.

        FIVE PROPERTIES THIS GUARANTEES, each of which a gate clause checks and
        each of which is easy to lose in a rewrite:

        1. **Projection identity.**  Every leaf pair comes from the same
           ``self._observations`` walk that :meth:`cycle_deltas` uses, rounded by
           the same expression.  Concatenating the leaves and sorting therefore
           equals the flat series sorted, element for element with identical
           multiplicity — exactly, with no tolerance.  The partition is a
           lossless re-indexing of the same data the flat series carries, which
           is what makes relaxing the ADS gate from byte-identity to
           additive-only safe.

        2. **The rounded leaf is authoritative.**  Each cell's ``mean_abs_err``
           is the mean of the SAME rounded values that cell's leaf list carries,
           emitted unrounded.  One source of truth, and an exact aggregate-to-
           series tie-in rather than an approximate one.  (The four pre-existing
           unrounded scalars are untouched — the gate forbids moving them.)

        3. **Cells exist iff they hold at least one closure.**  Empty cells are
           omitted.  The alternative, padding the full ``outcome x scope``
           cross-product with ``n == 0`` entries, would be 30 mostly-empty cells
           per ADS run and would make the projection identity a statement about
           padding.  An arm that closed nothing emits ``{}``.

        4. **Both dicts have identical key structure at both levels**, because
           both are built in this one walk.  An emitter that built the series and
           the aggregate in two passes could disagree between them; this one
           cannot.

        5. **The scope keys are DERIVED FROM THE TAG, never written out.**  There
           is no ``["10","5","4","3","2","1"]`` anywhere in this module, and no
           import of ``GROUP_DISTRIBUTION_COUNTS`` — which this module could not
           reach anyway without closing an import cycle.  A hard-coded domain
           would pass the gate's Tier 3 subset check, pass the projection
           identity, and pass the archive cross-validation against an unpatched
           run; the only thing that catches it is a test that patches
           ``GROUP_DISTRIBUTION_COUNTS`` to a disjoint sentinel set and asserts
           the emitted keys follow.  Deriving from the tag makes that test pass
           by construction and is strictly stronger than generating from the
           constant.
        """
        deltas: Dict[str, Dict[str, List[List[Any]]]] = {}
        sums: Dict[str, Dict[str, float]] = {}

        for cycle, abs_delta, outcome, scope in self._observations:
            rounded = round(abs_delta, DELTA_ROUND_PLACES)
            deltas.setdefault(outcome, {}).setdefault(scope, []).append(
                [cycle, rounded]
            )
            bucket = sums.setdefault(outcome, {})
            bucket[scope] = bucket.get(scope, 0.0) + rounded

        closures: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for outcome, by_scope in deltas.items():
            cell: Dict[str, Dict[str, Any]] = {}
            for scope, pairs in by_scope.items():
                n = len(pairs)
                # n >= 1 by construction (a cell exists only because something
                # was appended to it), so mean_abs_err is always a real float.
                cell[scope] = {
                    "n": n,
                    "mean_abs_err": sums[outcome][scope] / n,
                }
            closures[outcome] = cell

        return deltas, closures


# ---------------------------------------------------------------------------
# Tag coercion
# ---------------------------------------------------------------------------

#: Partition key used when an arm opens a forecast without an ``outcome`` tag.
#: Not expected to occur — both arms tag every entry — but a KeyError inside a
#: measurement path would take down a run that is otherwise fine, and a silently
#: dropped closure would break the projection identity.  A visible bucket is the
#: least-bad third option: it fails the gate's enumerated outcome domain loudly
#: instead of vanishing.
UNTAGGED_OUTCOME = "untagged"

#: Likewise for a missing or non-integer ``n_groups``.
UNTAGGED_SCOPE = "untagged"


def _outcome_key(tag: Mapping[str, Any]) -> str:
    value = tag.get("outcome")
    return UNTAGGED_OUTCOME if value is None else str(value)


def _scope_key(tag: Mapping[str, Any]) -> str:
    """
    ``str(int(n_groups))``.

    Normalised through ``int`` so that a tag arriving as ``1.0`` or ``"1"``
    cannot mint a second cell for the same scope — a split cell would halve the
    apparent size of the whole-population subset and nothing downstream would
    say so.

    A reminder for every consumer, because it is loud rather than silent and it
    will happen anyway: ``"10"`` sorts before ``"5"`` lexically.  Call ``int(k)``
    before sorting the scope axis, or the whole-population cell lands in the
    wrong place on every figure.
    """
    value = tag.get("n_groups")
    if value is None:
        return UNTAGGED_SCOPE
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return UNTAGGED_SCOPE
