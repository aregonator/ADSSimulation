"""
Autocracy + 20-cycle lookahead — the "planning only" regime.

STATUS: this is one of the paper's **eight primary regimes** and also a clean
control condition at once.  It runs in the default sweep and appears in the
comparison figures alongside the other seven; the paragraphs below explain why
it was built, which is still the most useful thing to know when reading its
results.

WHY THIS REGIME EXISTS
----------------------
It exists so that one specific alternative explanation for the ADS result can
be ruled out.

The ADS differs from every other regime in the study along two dimensions at
once: it can *see ahead* (a 20-cycle forward projection of each candidate law),
and it is *organised* (evidence nodes, risk-banded agent groups, per-domain
hypothesis generation, multi-node role separation).  A reader is entitled to
ask which of the two does the work.  No other regime has foresight at all, so
without this one nothing in the design answers that question.

``AutocracyLookaheadGovernment`` is the leanest possible answer: plain autocracy,
with foresight bolted on and nothing else.  It reuses the ADS's own evaluator —
``EvaluatorNode``, imported unmodified from :mod:`governments.ads`, scoring on the
identical objective (total surviving health over ``LOOKAHEAD_CYCLES``) — so the
*quality* of its foresight is not a confound either.  What it deliberately does
**not** have:

  * no evidence nodes,
  * no risk-banding or per-domain agent grouping (one target group: everybody),
  * no Hypothesis Exchange, Level Gate, or Outcome Evaluation Tier,
  * no multi-node role separation.

Any ADS advantage that survives against this comparator is attributable to
coordination, not to planning.

**That interpretation depends on this regime staying one factor away from
`autocracy`.**  The `autocracy` -> `autocracy_lookahead` -> `ads` contrast is
the reason it is interesting, and adding ADS machinery here (risk banding,
per-domain grouping, role separation) would collapse the second step and cost
the comparison its meaning.  Improvements to the *shared* evaluator are a
different matter and are fine: they land on the ADS too, so the one-factor
difference is preserved — for example, teaching ``EvaluatorNode`` to model
agent movement (see THE STORM MENU below).

WHAT THE ABLATION ISOLATES
---------------------------
This regime:

  * drives its own ``ParameterEstimator`` and scores against an INFERRED
    environment model, rather than a frozen crude one; and
  * carries the same ``ForecastLedger`` the ADS does, so the two arms' forecast
    accuracy is measurable on a like-for-like basis.

Neither is ADS *organisation*.  Both are SHARED components reached through
neutral leaf modules (``parameter_inference``, ``forecast_ledger``) that neither
government imports from the other — the same discipline that keeps
``EvaluatorNode`` unforked, and for the same reason: if the two arms inferred, or
measured, with two copies of the code, "equally good foresight" would be
unverifiable the first time the copies drifted.

What this regime deliberately does NOT have, and what the one-factor
constraint protects:

  * no evidence nodes,
  * no risk-banding or per-domain agent grouping (one target group: everybody),
  * no Hypothesis Exchange, Level Gate, or Outcome Evaluation Tier,
  * no multi-node role separation,
  * no generated candidate menu — the hand-authored
    ``FOOD_RATION_CANDIDATES`` stands,
  * no correction of any kind applied to the projected score.

So the chain is
``autocracy`` (no foresight) -> ``autocracy_lookahead`` (good foresight, no
organisation) -> ``ads`` (good foresight + organisation), and any surviving ADS
advantage is attributable to COORDINATION rather than to either planning or
information quality.  This is a demanding ablation: it may show little or no
ADS advantage, and that possibility is expected rather than a sign of a fault.

TWO-TIER STRUCTURE (mirrors ``AdsGovernment``)
----------------------------------------------
The ADS does not lookahead-evaluate everything; it pairs an immediate reflex
(``AdsGovernment._fast_response``) with a periodic evaluated round
(``AdsGovernment._decision_round``).  This regime mirrors that split so the
comparison is like-for-like:

  * **Tier 1 — reflex (inherited, unchanged).**  ``AutocracyGovernment``'s canned
    event responses and its pre-emptive responses to warnings.  Same laws, same
    fixed parameters, same cycle, as plain autocracy.
  * **Tier 2 — evaluated decision round (new).**  Every ``T_DECISION_CALM``
    cycles (``T_DECISION_CRISIS`` while an event is active or warned — the same
    cadence rule ``AdsGovernment.tick`` uses), the leader scores a small menu of
    candidate responses per live event type with the 20-cycle lookahead and
    installs the winner, superseding the reflex's canned choice.

Resource extraction (the leader/loyalist skim) is **not** evaluated and is
inherited untouched: it fires on its own fixed schedule exactly as in plain
autocracy.  Only the event-response channel is subject to foresight.

The net effect on law volume is close to nil by construction: the decision round
chooses *which variant* of the response the reflex already produces should be in
force (or that none should be), rather than adding laws of its own.  That keeps
the ablation a clean one-factor change against ``autocracy``.

THE STORM MENU, AND ONE RESIDUAL CAVEAT
-----------------------------------------
The shared evaluator models one-cell-per-cycle agent movement toward the
relevant target (``ads.py`` ``_step``), which is what lets ``MANDATORY_SHELTER``
— whose entire benefit is that agents relocate to shelter — discriminate from
no mandate at all: without movement modelled, the two would score identically
and the storm decision would always fall through to the reflex via the tie rule
in ``_build_menu``.

With movement modelled, the tie is broken: measured in situ at production scale
(50x50, 500 agents, d=75, cycle 8, 40 agents off shelter), ``MANDATORY_SHELTER``
scores 180.07 against ``NO_ACTION``'s 178.11.  The storm channel carries real
information, for this regime and for the ADS alike — the evaluator is shared,
so the same movement model applies to both and the one-factor difference is
preserved.

**Residual caveat worth knowing.**  Agents are usually already on shelter by the
time a storm is active — the warning system and the inherited reflex get them
there — so the menu often still ties in practice simply because there is nothing
to improve.  Measured across four configurations, the fraction of living agents
off shelter *during storm cycles* was 0% at smoke scale (30x30, 60 agents) and
~8% at production scale (50x50, 500 agents).  So the storm menu discriminates at
production scale and frequently does not at smoke scale; do not conclude from a
quick-test log that the fix is inert.
"""

from __future__ import annotations

from typing import (
    Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple, TYPE_CHECKING,
)

from .ads import (
    ADS_FORECAST_SCHEMA_VERSION,
    EVENT_RATE_CAP,
    EVENT_RATE_PRIOR_CYCLES,
    LOOKAHEAD_CYCLES,
    PHANTOM_CATEGORIES,
    EvaluatorNode,
    ProposedLaw,
)
from .autocracy import AutocracyGovernment
from .base import DEFAULT_GOVERNMENT_SEED, Law
# Same dependency-direction argument as parameter_inference below: a neutral
# leaf module that imports nothing from governments/, so both arms reach the
# same grading code without either importing the other.
from .forecast_ledger import ForecastLedger
# Imported from parameter_inference, NOT from .ads.  The DEPENDENCY DIRECTION
# invariant is load-bearing:
#     engine <- parameter_inference <- ads <- autocracy_lookahead
# parameter_inference must never import governments.ads, which is what lets
# BOTH regimes reach the same estimator without either importing the other.
# The three-arm chain
#     autocracy (no foresight)
#       -> autocracy_lookahead (good foresight, no organisation)
#       -> ads (good foresight + organisation)
# isolates COORDINATION cleanly instead of confounding coordination with
# information quality.  The two arms share an estimator CLASS; they never
# share an estimator INSTANCE (see __init__).
from .parameter_inference import (
    CRUDE_DRAIN_MULT,
    CRUDE_EVENT_RATE,
    CRUDE_MAX_STEPS_PER_CYCLE,
    CRUDE_REGEN_MULT,
    DRAIN_MULT_MAX,
    DRAIN_MULT_MIN,
    ESTIMATOR_PRIOR_CYCLES,
    REGEN_MULT_MAX,
    REGEN_MULT_MIN,
    REGEN_OBS_MAX_FILL,
    EnvironmentModel,
    ParameterEstimator,
)
from engine.scenario_plan import derive_seed

if TYPE_CHECKING:
    from engine.agent import Agent
    from engine.events import EventWarning


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cycles between evaluated decision rounds when nothing is happening.  Matches
#: ``AdsGovernment.__init__``'s ``t_decision`` default so the two regimes
#: deliberate on the same schedule.
T_DECISION_CALM = 5

#: Cycles between rounds while an event is active or warned.  Matches the
#: ``interval = 2 if has_crisis`` rule in ``AdsGovernment.tick``.
T_DECISION_CRISIS = 2

#: Event types the leader will deliberate over, in a fixed order.  Ordered, not
#: a set: the order decides which event type is resolved first within a round,
#: and a set would order by string hash.  Reproducibility-critical for the same
#: reason ``EvaluatorNode.evaluate`` sorts its epidemic ids.
DECIDABLE_EVENT_TYPES: Tuple[str, ...] = ("drought", "storm", "epidemic")

#: Event types that shorten the decision cadence.  Mirrors ``AdsGovernment.tick``
#: exactly — including ``toxic_spill``, which shortens the cadence without being
#: decidable (this regime has no toxic-spill menu; the reflex handles it).
CRISIS_EVENT_TYPES = frozenset({"epidemic", "storm", "drought", "toxic_spill"})

#: Candidate ``FOOD_RATION.max_per_cycle`` values for a drought.
#:
#: 2.5 is plain autocracy's fixed value (``autocracy.py`` ``_enact_event_response``),
#: so "keep doing what autocracy does" is always on the menu and the ablation can
#: reduce to its base regime when foresight says the canned value is already best.
#: 1.5 is a tighter ration, 4.0 a looser one; both sit inside the range the ADS's
#: own reflex computes (``max(2.0, min(4.0, ...))`` in ``AdsGovernment._fast_response``),
#: so the menu spans the same decision space the ADS explores rather than a wider one.
#: Ascending order is deliberate — see ``_build_menu`` on tie-breaking.
FOOD_RATION_CANDIDATES: Tuple[float, ...] = (1.5, 2.5, 4.0)

#: Sentinel ``law_type`` for the "do nothing" candidate.  Never enacted; it exists
#: only to be *scored*, so that inaction competes on the same footing as action.
#: ``EvaluatorNode._step`` branches on specific law types and matches none of
#: these, so a ``NO_ACTION`` proposal yields exactly the unintervened
#: counterfactual — which is the whole point.
NO_ACTION = "NO_ACTION"

#: Recorded in ``Law.source`` for laws this tier produces, so a consumer can tell
#: an evaluated law from a reflex.  Parallels ADS's ``"fast_response"`` /
#: ``"decision_round"`` values.
SOURCE_DECISION_ROUND = "leader_decision_round"

#: ``trigger.kind`` values for :meth:`get_decision_record`.
#:
#: A real variable rather than a hardcoded ``"event_interval"`` constant,
#: because the warning-driven round would make a hardcoded value a lie the
#: first time one fires, and this project treats the ``trigger`` tag as
#: load-bearing.  ``crisis_interval`` is informative on its own even without
#: the warning round: the cadence has always distinguished calm from crisis
#: (``_is_decision_cycle``) but never recorded which fired.
TRIGGER_EVENT_INTERVAL = "event_interval"
TRIGGER_CRISIS_INTERVAL = "crisis_interval"
TRIGGER_WARNING_PREEMPT = "warning_preempt"

#: ``_law_expiry_reasons`` value for a law retired by the decision round.  Sits
#: alongside base.py's ``duration_elapsed`` / ``event_type_gone`` /
#: ``event_id_cleared``; a repeal is a fourth, distinct way for a law to stop.
EXPIRY_REASON_SUPERSEDED = "superseded_by_lookahead"


class AutocracyLookaheadGovernment(AutocracyGovernment):
    """
    Plain autocracy whose event responses are chosen by a 20-cycle lookahead.

    Inherits leader appointment, loyalist refresh, and resource extraction from
    :class:`~governments.autocracy.AutocracyGovernment` unchanged.  Overrides
    exactly one behavioural method, ``_enact_event_response`` (plus
    ``receive_event_warnings``, which also runs an out-of-band round), and adds
    the evaluated round itself.

    Two further overrides are NOT behavioural and exist only to place two
    per-cycle hooks upstream of the inherited tick's three early returns:
    :meth:`tick` (estimator evidence) and :meth:`_expire_laws` (forecast
    grading).  Both are documented at length at their definitions, because
    getting either placement wrong fails silently or fails only at D90+.
    """

    name = "AutocracyLookahead"

    def __init__(
        self,
        t_succession: int = 50,
        t_law: int = 20,
        seed: Optional[int] = None,
        t_decision: int = T_DECISION_CALM,
        eval_seed: Optional[int] = None,
    ):
        super().__init__(t_succession=t_succession, t_law=t_law, seed=seed)
        self.t_decision = t_decision

        #: The ADS's evaluator, imported and used unmodified.  Sharing the
        #: implementation is the point: if the two regimes scored candidates
        #: differently, a difference in outcome could be a difference in
        #: evaluation quality rather than in coordination.
        #:
        #: It is seeded here for the same reason ADS seeds its own: the
        #: evaluator now draws from a dedicated phantom-event stream, and an
        #: unseeded one would replay a single frozen future for every run of
        #: every difficulty.  The seed is kept SEPARATE from the leader's
        #: decision seed so a change in how often this regime deliberates cannot
        #: shift the lookahead's random realisations, and vice versa.
        #:
        #: Keyed on the government name by the harness (``gov.eval`` domain), so
        #: this regime and ADS get INDEPENDENT phantom futures.  That is
        #: deliberate: by a handful of cycles in, the two have enacted different
        #: laws and inhabit materially different worlds, so a shared phantom
        #: stream would pair nothing real — it would only create the chance that
        #: one shared realisation happens to suit one regime's actual event
        #: history better than the other's.  The control this arm provides is
        #: "identical evaluator, identical generative model of the future", not
        #: "identical realised draws", and the latter stops being meaningful the
        #: moment the worlds diverge.
        self._evaluator = EvaluatorNode(
            eval_seed=(
                eval_seed if eval_seed is not None
                else derive_seed(
                    DEFAULT_GOVERNMENT_SEED if seed is None else seed,
                    "gov.eval",
                )
            )
        )

        #: Warnings received this cycle.  Consumed and cleared by
        #: ``_enact_event_response``, which the base class calls as the last
        #: statement of ``tick`` — the same position in the cycle at which
        #: ``AdsGovernment.tick`` clears its own ``_pending_warnings``.
        self._pending_warnings: List["EventWarning"] = []

        #: Standing "do nothing" decisions, keyed by event type -> the set of
        #: event *instances* of that type that were live when the decision was
        #: taken.  ``{event_type: frozenset(instance_key, ...)}``.
        #:
        #: Load-bearing, and the reason is not obvious.  The inherited reflex
        #: re-enacts its canned response on *every* cycle the event type lacks an
        #: active law.  Without this memo, a round that repealed the ration at
        #: cycle 10 would see it re-enacted by the reflex at cycle 11 and repeal
        #: it again at cycle 12 — "do nothing" would never actually hold for more
        #: than alternate cycles, so the menu's cheapest option would be
        #: unimplementable and the decision round would produce pure churn.
        #:
        #: Keyed by instance, NOT by type alone.  A type-only memo is a real
        #: confound and was measured as one: a decision to tolerate
        #: epidemic *A* went on suppressing the reflex when unrelated epidemic
        #: *B* began, because "epidemic" was still live and the memo could not
        #: tell the two apart — so B got no quarantine at all.  That is a second,
        #: non-foresight difference from plain autocracy, which is precisely what
        #: this ablation may not have.  See ``_is_suppressed``: the memo lapses
        #: the moment an instance appears that the decision did not consider.
        self._suppressed_event_instances: Dict[str, FrozenSet[Any]] = {}

        #: This regime's OWN evidence-based model of the environment's hidden
        #: dynamics parameters.
        #:
        #: With both arms forecasting on inferred inputs, the
        #: `autocracy -> autocracy_lookahead -> ads` chain isolates ORGANISATION
        #: instead of confounding organisation with information quality.
        #:
        #: OWNERSHIP IS BY CONSTRUCTION, and the test asserts it anyway.  The two
        #: governments are separate objects built independently by
        #: `benchmark_core._run_one` and neither holds a reference to the other,
        #: so they cannot share an instance — but "by construction" is the kind
        #: of guarantee that survives right up until someone adds a module-level
        #: cache.  Sharing one would destroy both the ablation and the
        #: reproducibility story: by a handful of cycles in the two arms have
        #: enacted different laws and inhabit materially different worlds, so
        #: their evidence differs legitimately and must.
        self._estimator = ParameterEstimator(self._logger)

        #: This regime's forecast-grading ledger.
        #:
        #: WHY IT IS HERE AT ALL.  The paper's claim is that this arm has *good*
        #: foresight, and the only direct evidence for foresight QUALITY is a
        #: predicted-vs-realized error series.  Without one, the evidence is
        #: indirect (the estimator's inputs converge).
        #:
        #: It MEASURES ONLY and feeds nothing back — it cannot affect ranking,
        #: and it draws no randomness.  Both properties are gated: a stray draw
        #: in here would reseed nothing but would consume from no stream at all,
        #: yet any accidental draw would desynchronise the evaluator and shift
        #: every subsequent decision.  `verify_al_noop` compares this arm's
        #: `run_detail.jsonl` before and after the ledger landed, precisely to
        #: catch that.
        #:
        #: Same CLASS as ADS's, never the same INSTANCE — see `_estimator` above
        #: for why a shared instance would be fatal to the comparison.
        self._ledger = ForecastLedger(LOOKAHEAD_CYCLES, self._logger)

        # --- Instrumentation ------------------------------------------------
        self._decision_rounds = 0
        self._candidates_evaluated = 0
        self._lookahead_laws_enacted = 0
        self._lookahead_laws_repealed = 0
        self._lookahead_no_action_decisions = 0

        #: Cycle of the most recent round, or None.  ``_last_decision_details``
        #: persists between rounds; without this, ``get_decision_record`` would
        #: duplicate a stale record onto every intervening cycle.  Same contract
        #: as ``AdsGovernment._last_decision_cycle``.
        #: Rounds driven by a warning rather than by the cadence.  Quantifies a
        #: known and accepted loss: two rounds can run on one cycle, and the
        #: second overwrites `_last_decision_details`, so `get_decision_record`
        #: reports only the later one.  That matches ADS exactly, so it is
        #: accepted rather than fixed — but a reader can now tell how often a
        #: record is partial instead of having to assume it never is.
        self._warning_rounds = 0

        #: What triggered the round currently in progress.  Set at the top of
        #: `_leader_decision_round` and read by `_apply_decision` for the ledger
        #: tag, rather than threaded through two intermediate calls that have no
        #: other use for it.  Mirrors `AdsGovernment._last_decision_trigger`.
        self._last_decision_trigger_kind: str = TRIGGER_EVENT_INTERVAL

        #: Repeals taken OUT OF BAND, keyed by the cycle they were taken on,
        #: waiting to be re-merged once that cycle's expiry structures exist.
        #: See `_repeal_law` and `_expire_laws` for the full argument; in one
        #: line, a warning round runs at step 1 and the structures it would write
        #: into are reassigned wholesale at step 6.
        #:
        #: Cannot accumulate: an entry is written at step 1 of cycle t and popped
        #: at step 6 of the same cycle.  The stale-key sweep in `_expire_laws` is
        #: defensive only and logs if it ever fires.
        self._pending_repeals: Dict[int, List[Law]] = {}

        #: The last cycle `_expire_laws` has run for, or None.  This is how
        #: `_repeal_law` tells an in-band repeal (structures already rebuilt for
        #: this cycle -- write straight through) from an out-of-band one
        #: (structures still describe cycle-1 -- defer).
        self._expired_through_cycle: Optional[int] = None

        self._last_decision_cycle: Optional[int] = None
        self._last_decision_details: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Per-cycle hooks
    # ------------------------------------------------------------------

    def tick(self, cycle: int) -> None:
        """
        Inherited tick, preceded by the estimator's once-per-cycle update.

        EVIDENCE FIRST, UNCONDITIONALLY, and the placement is the single hardest
        constraint in this change rather than a stylistic choice.  Mirrors
        ``AdsGovernment.tick``'s first two statements, including its
        ``if not self._sim`` guard, for the identical reason.

        ``ParameterEstimator.observe()`` RAISES on a cycle gap — deliberately, so
        that a silently double-counted or silently skipped cycle cannot bias
        every estimate with no symptom.  ``AutocracyGovernment.tick`` has three
        early returns: no sim, **no living agents**, and no leader.  The middle
        one fires routinely at D90-D100 (the archived D90 pooled median survival
        is 14.6%, and zero-survivor runs exist).  So a hook placed where every
        other override in this class lives — ``_enact_event_response``, the last
        statement of the inherited tick — would crash a large fraction of the
        hardest cells in the sweep, and would do it only after 40+ hours of wall
        time.  That is why this method exists at all.

        This override adds no behaviour of its own.  Leader succession, loyalist
        refresh and resource extraction remain wholly inherited; the class
        docstring's "overrides exactly one behavioural method" still holds,
        because this one is not behavioural.
        """
        if self._sim is not None:
            self._estimator.observe(cycle, self._sim)
        super().tick(cycle)

    def _expire_laws(self, cycle: int) -> None:
        """
        Inherited expiry, plus this regime's forecast-grading hook.

        WHY HERE, and not in ``tick``.  This method is ``tick``'s FIRST
        statement in every one of the eight governments, so it is already an
        exactly-once-per-cycle hook that runs upstream of all three early
        returns in ``AutocracyGovernment.tick`` — including ``if not living:
        return``, which fires routinely at D90-D100.  It also runs AFTER the base
        class has rebuilt ``_laws_expired_this_cycle`` / ``_law_expiry_reasons``,
        which is what the hook below needs and what made this placement hard.

        ``close()`` has the same hard per-cycle requirement ``observe()`` has, but
        a WORSE failure mode: ``observe()`` raises on a gap, ``close()`` does not.
        A skipped cycle silently strands every forecast whose review date landed
        on it — the error series quietly shrinks and
        ``calibration_open_predictions_at_end`` quietly inflates.  Loud failure
        you find in three minutes; silent failure you publish.

        ``close()`` additionally has an ordering requirement ``observe()`` does
        not: it reads ``active_law_ids`` to classify whether the originating law
        outlived its review date, so it must run AFTER expiry.  Two rejected
        placements, recorded so they are not re-proposed:

          * *Before* ``super().tick()`` — runs before expiry, so a law expiring
            on the review cycle classifies as ``law_active`` here and
            ``law_lifted`` on ADS.  A one-cycle off-by-one in a field whose
            entire purpose is cross-arm comparison.
          * *Hoist* ``_expire_laws`` and call it early, then close, then
            ``super().tick()`` — ``_expire_laws`` REASSIGNS
            ``_laws_expired_this_cycle`` wholesale, so the second call wipes the
            first call's list and the recorder loses the cycle's expiries.

        The resulting order is bit-for-bit ``AdsGovernment.tick``'s:
        ``observe -> _expire_laws -> close -> (reflex / rounds)``.

        One consequence of the warning round, stated because it looks alarming
        and is not: a forecast can be opened at step 1 of cycle *t* and
        this method runs at step 6 of every cycle, so could a forecast be closed
        in the cycle it was opened?  No — the review date is ``t + horizon`` with
        horizon 20, so same-cycle closure is arithmetically impossible.  Same as
        ADS.
        """
        super()._expire_laws(cycle)
        self._expired_through_cycle = cycle

        # Re-merge repeals taken BEFORE this cycle's rebuild.  See
        # `_repeal_law` for why they were deferred.  This must run after
        # `super()._expire_laws`, which is what creates the structures being
        # merged into.
        for law in self._pending_repeals.pop(cycle, ()):
            self._record_repeal_bookkeeping(law)
        # Defensive only.  A key older than this cycle means a repeal was taken
        # on a cycle whose `_expire_laws` never ran, which should be impossible
        # -- `_expire_laws` is tick's first statement and tick runs every cycle.
        # Logged rather than silently dropped, because "impossible" states that
        # discard archive records are exactly the ones worth hearing about.
        for stale in [c for c in self._pending_repeals if c < cycle]:
            dropped = self._pending_repeals.pop(stale)
            self._logger.warning(
                "cycle=%d REPEAL_RECORD_STALE origin_cycle=%d n=%d law_ids=%s",
                cycle, stale, len(dropped),
                ",".join(law.law_id for law in dropped),
            )

        # Grade forecasts due this cycle.  MEASURES ONLY; feeds nothing back.
        if self._sim is not None:
            self._ledger.close(
                cycle, self._sim, {law.law_id for law in self.active_laws},
            )

    # ------------------------------------------------------------------
    # Tier 1 hook — reflex, plus the trigger for tier 2
    # ------------------------------------------------------------------

    def _enact_event_response(self, event_names: List[str], cycle: int) -> None:
        """
        Run the inherited reflex, then (on cadence) the evaluated decision round.

        Overriding this method rather than ``tick`` is deliberate: ``tick`` owns
        leader succession, loyalist refresh, and resource extraction, none of
        which this ablation touches.  ``tick`` calls this method as its last
        statement, so hooking here puts the decision round at the end of the
        cycle — the same slot ``AdsGovernment._decision_round`` occupies.
        """
        active_types = set(event_names)
        warned_types = {w.event_type.value for w in self._pending_warnings}

        # A suppression means "the leader has decided, for these specific live
        # occurrences, to do nothing".  Once the type is gone entirely the memo is
        # moot; a *new* occurrence of a still-live type is handled by
        # ``_is_suppressed``, which lapses the memo rather than swallowing it.
        self._prune_suppressions(active_types, warned_types)

        # --- Tier 1: reflex ------------------------------------------------
        # Filtering the argument, rather than modifying autocracy.py, keeps the
        # base class untouched: for an unsuppressed event type this is the exact
        # call plain autocracy makes.
        self._run_reflex(event_names, cycle)

        # --- Tier 2: evaluated round ---------------------------------------
        try:
            targets = [
                et for et in DECIDABLE_EVENT_TYPES
                if et in active_types or et in warned_types
            ]
            if targets and self._is_decision_cycle(cycle, active_types, warned_types):
                # The cadence rule already branched on crisis; this records
                # WHICH branch fired instead of labelling every in-band round
                # "event_interval".
                has_crisis = bool(
                    (active_types | warned_types) & CRISIS_EVENT_TYPES
                )
                self._leader_decision_round(
                    cycle, targets,
                    trigger_kind=(TRIGGER_CRISIS_INTERVAL if has_crisis
                                  else TRIGGER_EVENT_INTERVAL),
                )
        finally:
            # Cleared unconditionally: a warning is consumed by the cycle it
            # arrives in (Simulation dispatches warnings before tick), and a
            # raising round must not leave them to be re-counted next cycle.
            self._pending_warnings = []

    # ------------------------------------------------------------------
    # Suppression — standing "do nothing" decisions, keyed by event instance
    # ------------------------------------------------------------------

    def _live_instance_keys(self, event_type: str) -> FrozenSet[Any]:
        """
        Identify the live *occurrences* of ``event_type``, not just the type.

        ``ActiveEvent`` has no universal uid.  Epidemics carry ``epidemic_id``,
        which is already this codebase's notion of epidemic instance identity
        (``Law.event_id``, ``_expire_laws``'s ``event_id_cleared`` branch), so it
        is reused rather than shadowed by a parallel scheme.  Everything else is
        keyed by ``start_cycle``: two occurrences of one type cannot begin on the
        same cycle and be meaningfully distinct, and if they somehow did, treating
        them as one occurrence is the safe direction — it can only cause the memo
        to lapse early (an extra reflex), never to swallow a response.
        """
        sim = self._sim
        if sim is None:
            return frozenset()
        return frozenset(
            ev.epidemic_id if ev.epidemic_id is not None else ("t", ev.start_cycle)
            for ev in sim.event_system.active_events
            if ev.event_type.value == event_type and not ev.cancelled
        )

    def _is_suppressed(self, event_type: str, cycle: Optional[int] = None) -> bool:
        """
        Does a standing "do nothing" decision still cover this event type?

        Only while every live occurrence was one the deciding round actually saw.
        The instant an occurrence appears that the decision did not consider, the
        memo lapses and the inherited reflex handles the newcomer normally — until
        the next round re-decides with the newcomer in view.  Keying the memo by
        instance rather than by type is what makes this possible; a type-keyed
        memo would silently swallow the response to a new event.
        """
        remembered = self._suppressed_event_instances.get(event_type)
        if remembered is None:
            return False
        live = self._live_instance_keys(event_type)
        if live <= remembered:
            return True
        # A new occurrence appeared: the standing decision no longer applies.
        self._suppressed_event_instances.pop(event_type, None)
        self._logger.debug(
            "cycle=%s LOOKAHEAD_SUPPRESSION_LAPSED event_type=%s "
            "new_instances=%s reason=unconsidered_event_instance",
            "?" if cycle is None else cycle,
            event_type, ",".join(sorted(map(str, live - remembered))),
        )
        return False

    def _prune_suppressions(self, active_types: Set[str],
                            warned_types: Set[str]) -> None:
        """Drop memos for event types that are no longer live at all."""
        for event_type in list(self._suppressed_event_instances):
            if event_type not in active_types and event_type not in warned_types:
                self._suppressed_event_instances.pop(event_type, None)

    def _run_reflex(self, event_names: List[str], cycle: int) -> None:
        """Invoke the inherited canned response, minus any suppressed event type."""
        suppressed_now = {e for e in set(event_names)
                          if self._is_suppressed(e, cycle)}
        if suppressed_now:
            allowed = [e for e in event_names if e not in suppressed_now]
            self._logger.debug(
                "cycle=%d LOOKAHEAD_REFLEX_SUPPRESSED event_types=%s "
                "reason=standing_no_action_decision",
                cycle, ",".join(sorted(suppressed_now)),
            )
        else:
            allowed = event_names
        super()._enact_event_response(allowed, cycle)

    def receive_event_warnings(
        self, warnings: List["EventWarning"], cycle: int
    ) -> None:
        """
        Record warnings for the decision round, then run the inherited pre-emptive
        reflex over the warnings the leader has not already decided to ignore,
        then run an out-of-band evaluated round — see below.

        Note what this still does *not* do: it does not pre-simulate the warned
        event into the evaluation.  Pre-simulating warnings is an evidence-node
        capability (``FoodEvidenceNode`` and friends fold ``pending_warnings``
        into their statistics); importing it here would import exactly the
        coordination machinery this ablation exists to withhold.  The practical
        consequence is documented on ``_build_menu``: for a warned-but-not-yet-
        active event the lookahead sees an untroubled world, so inaction usually
        wins and pre-emption is left to the inherited reflex — which is precisely
        what plain autocracy does.

        Running an out-of-band decision round on a warning is a CADENCE change,
        not an import of ADS machinery: it uses the same evaluator, the same
        flat target group, and the same hand-authored menu as the regular
        decision round, just invoked on one more occasion.
        """
        self._pending_warnings.extend(warnings)

        # Inherited pre-emptive reflex, minus any suppressed type (UNCHANGED).
        if self._suppressed_event_instances:
            allowed = [w for w in warnings
                       if not self._is_suppressed(w.event_type.value, cycle)]
        else:
            allowed = warnings
        super().receive_event_warnings(allowed, cycle)

        # --- Out-of-band evaluated round -------------------------------
        #
        # The same evaluator, the same flat target group, the same hand-authored
        # menu, invoked on one more occasion.  Nothing about WHAT is deliberated
        # changes; only WHEN.
        #
        # THE ESTIMATOR IS DELIBERATELY NOT ADVANCED HERE, and this is a hard
        # contract rather than an omission.  `Simulation._step` dispatches
        # warnings at step 1 and calls `government.tick` at step 6, so a round
        # here reads estimator state through cycle-1 — which is exactly what
        # ADS's warning round does, so cross-arm parity on evidence recency is
        # automatic and needs no code.  An `observe()` call here would (a)
        # double-count the cycle and raise on the subsequent tick, and (b) read
        # pre-drain, pre-regen state, which `_observe_drain` explicitly relies on
        # not happening.
        #
        # NO PRE-EMPTIVE SHELTER LAW.  ADS enacts one on a warning; that is ADS's
        # REFLEX, and this regime's reflex is autocracy's, inherited via the
        # `super()` call above.  Adding ADS's would be an un-asked-for
        # behavioural change to the base regime and would break the one-factor
        # property against plain `autocracy`.
        if not any(w.event_type.value in CRISIS_EVENT_TYPES for w in warnings):
            return
        sim = self._sim
        if sim is None:
            return
        active_types = {e["type"] for e in sim.event_system.active_summary(cycle)}
        warned_types = {w.event_type.value for w in self._pending_warnings}
        targets = [et for et in DECIDABLE_EVENT_TYPES
                   if et in active_types or et in warned_types]
        if not targets:
            # A toxic_spill-only warning shortens ADS's cadence, but this regime
            # has no toxic-spill menu (see CRISIS_EVENT_TYPES' comment).  An
            # empty-target round would deliberate over nothing while still
            # incrementing the round counter and overwriting the cycle's decision
            # record — pure instrumentation noise.  ADS has no equivalent case
            # because it deliberates over the whole population, not only over
            # event responses.
            return
        self._warning_rounds += 1
        self._leader_decision_round(
            cycle, targets, trigger_kind=TRIGGER_WARNING_PREEMPT,
        )

    # ------------------------------------------------------------------
    # Seeding hooks
    # ------------------------------------------------------------------

    def set_eval_seed(self, eval_seed: int) -> None:
        """
        Set the root seed of the evaluator's rollout RNGs.

        Called by ``benchmark_core._run_one`` via ``getattr(gov,
        "set_eval_seed", None)``.  The harness constructs every regime through
        the same ``gov_cls(seed=...)`` call and duck-types this injection, so
        adding the method here is the whole of the wiring; the seed it receives
        is ``derive_seed(base, "gov.eval", gov_name, difficulty, run_idx)``,
        which differs from ADS's by the government name.

        Safe to call at any point before the first ``evaluate()``: the evaluator
        re-seeds from this root on every call, so there is no "already started"
        state to corrupt.

        NOTE what this method installs and what it does not.  It seeds the
        EVALUATOR only.  It does not touch ``_estimator`` (which draws no
        randomness at all — it reads observations) and it does not touch
        ``_ledger`` (which measures and draws nothing) — neither is seeded
        state.
        """
        self._evaluator.set_seed(eval_seed)

    def _is_decision_cycle(
        self, cycle: int, active_types: Set[str], warned_types: Set[str]
    ) -> bool:
        """Cadence rule, copied from ``AdsGovernment.tick`` for comparability."""
        has_crisis = bool((active_types | warned_types) & CRISIS_EVENT_TYPES)
        interval = T_DECISION_CRISIS if has_crisis else self.t_decision
        return cycle > 0 and interval > 0 and cycle % interval == 0

    # ------------------------------------------------------------------
    # Tier 2 — the evaluated decision round
    # ------------------------------------------------------------------

    def _leader_decision_round(
        self, cycle: int, targets: List[str], *, trigger_kind: str,
    ) -> None:
        """
        Score a small candidate menu per live event type and install the winners.

        One law, one target group (the whole living population), per event type —
        the coordination ceiling this ablation is built to respect.  Event types
        are resolved independently and in a fixed order, exactly as plain
        autocracy responds to each live event independently; constraining the
        round to a single law across *all* event types would make the ablation
        weaker than its own base regime during concurrent events, which would
        confound the comparison with a second difference.

        *trigger_kind* names the caller — cadence (calm or crisis) or warning.
        Passed in rather than inferred, because only the caller knows: by the
        time the round runs, the crisis state that chose the cadence has already
        been computed and a second derivation could disagree with the first.

        TWO ROUNDS ON ONE CYCLE ARE PERMITTED (ADS parity).  The second
        overwrites ``_last_decision_details``, so ``get_decision_record`` reports
        the LATER round only.  This matches ADS's existing behaviour exactly, so
        it is accepted rather than fixed — but the loss is quantified by
        ``_warning_rounds`` in ``get_audit_info``, so a reader can tell how often
        a record is partial.  The rejected alternative — accumulating a per-cycle
        list — would make this regime's decision-record schema differ in SHAPE
        from ADS's for a property both regimes share, creating a new asymmetry to
        solve the symptom of an old one.
        """
        sim = self._sim
        if sim is None:
            return
        living = sim.living_agents()
        if not living:
            return

        self._decision_rounds += 1
        self._last_decision_trigger_kind = trigger_kind
        decisions: List[Dict[str, Any]] = []

        # THE INFERRED ENVIRONMENT.
        #
        # Both arms score against ``ParameterEstimator``'s inferred model,
        # tagged ``source="inferred"``, so the surviving difference between
        # them is ORGANISATION alone, not information quality.  At D100 this
        # arm's evaluator uses roughly (1.74, 0.36, >=1, rate > 0) rather than
        # the frozen crude values (1.0, 1.0, 1, rate 0) — confidently-wrong
        # foresight becomes approximately-right foresight — so this arm's
        # results are markedly better than they would be under a crude,
        # uninformed world model.
        #
        # THE ESTIMATOR IS THIS INSTANCE'S OWN.  It is never ADS's: the two arms
        # inhabit different worlds within a handful of cycles, so their evidence
        # differs legitimately and a shared instance would destroy both the
        # ablation and the reproducibility story.
        #
        # Currently-active REAL events still apply and still expire at their true
        # ``cycles_remaining``.
        #
        # Fixed once per round and shared frozen across every candidate, exactly
        # as ADS does, so "every candidate in this round faced the same model of
        # the future" stays structural.
        env = self._estimator.environment_model(sim, cycle)

        for event_type in targets:
            decision = self._decide_event_type(
                event_type, cycle, living, env,
            )
            if decision is not None:
                decisions.append(decision)

        self._logger.info(
            "cycle=%d LOOKAHEAD_ROUND round=%d population=%d event_types=%s "
            "decisions=%d candidates_this_round=%d",
            cycle, self._decision_rounds, len(living), ",".join(targets),
            len(decisions), sum(d["candidates"] for d in decisions),
        )

        self._last_decision_details = {
            "cycle": cycle,
            "round": self._decision_rounds,
            "population": len(living),
            "event_types": list(targets),
            "decisions": decisions,
            "trigger_kind": trigger_kind,
            # Captured HERE, not read back in `get_decision_record`.  A
            # warning-driven round runs at step 1 and `_pending_warnings` is
            # cleared in `_enact_event_response`'s `finally` at step 6, long
            # before the recorder asks for the record — so reading it late would
            # report an empty list for precisely the rounds the field exists to
            # describe.
            "pending_warnings": [
                w.event_type.value for w in self._pending_warnings
            ],
        }
        self._last_decision_cycle = cycle

    def _decide_event_type(
        self, event_type: str, cycle: int, living: List["Agent"],
        env: EnvironmentModel,
    ) -> Optional[Dict[str, Any]]:
        """
        Build, score, and act on the candidate menu for one event type.

        Returns a JSON-safe record of the decision, or ``None`` when there was no
        decision to make (an empty menu — e.g. an epidemic whose infected agents
        have no computable quarantine region).
        """
        sim = self._sim
        if sim is None:                      # guarded by the caller; belt and braces
            return None

        menu = self._build_menu(event_type, cycle, living)
        if not menu:
            self._logger.debug(
                "cycle=%d LOOKAHEAD_MENU_EMPTY event_type=%s "
                "reason=no_actionable_candidate",
                cycle, event_type,
            )
            return None

        k = max(1, len(living))
        scored: List[Dict[str, Any]] = []
        best: Optional[ProposedLaw] = None
        best_raw = float("-inf")

        for proposal in menu:
            # No correction of any kind is applied to the projected score, here
            # or anywhere else in this class, and that is half of the arm's
            # defining property.  It is enforced by the ABSENCE OF CODE rather
            # than by a constant pinned to 1.0: there is no ledger, no closure
            # pass and no update rule in this file, so neutrality is not a
            # configuration that could drift or be "helpfully" made tunable
            # later — it is structural.  `norm` below therefore IS the ranking
            # score.  No correction is applied to the score, by absence of code;
            # that is what separates this from a control loop, and it is
            # structural.  The three-arm chain reads:
            # autocracy (no foresight) -> autocracy_lookahead (good foresight,
            # no organisation) -> ads (good foresight + organisation),
            # which isolates COORDINATION.
            raw = self._evaluator.evaluate(
                proposal, living, sim, cycle=cycle, env=env,
            )
            norm = raw / k
            proposal.evaluation_score = norm
            self._candidates_evaluated += 1

            self._logger.debug(
                "cycle=%d LOOKAHEAD_CANDIDATE event_type=%s law_type=%s "
                "params=%s group_size=%d horizon=%d raw_score=%.4f norm_score=%.4f",
                cycle, event_type, proposal.law_type,
                _params_repr(proposal.params), len(living), LOOKAHEAD_CYCLES,
                raw, norm,
            )
            scored.append({
                "law_type": proposal.law_type,
                "params": _jsonable(proposal.params),
                "raw_score": round(raw, 4),
                "norm_score": round(norm, 6),
            })

            # Strict ``>`` — ties go to the earlier candidate, and the menu is
            # ordered so that "act" precedes "do nothing" (see ``_build_menu``).
            if raw > best_raw:
                best_raw = raw
                best = proposal

        if best is None:                     # unreachable: the menu is non-empty
            return None

        norm_best = best_raw / k
        outcome = self._apply_decision(
            event_type, best, norm_best, cycle, living,
        )

        self._logger.info(
            "cycle=%d LOOKAHEAD_DECISION event_type=%s winner=%s params=%s "
            "norm_score=%.4f candidates=%d outcome=%s",
            cycle, event_type, best.law_type, _params_repr(best.params),
            norm_best, len(menu), outcome,
        )

        return {
            "event_type": event_type,
            "candidates": len(menu),
            "scored": scored,
            "winner": {
                "law_type": best.law_type,
                "params": _jsonable(best.params),
                "norm_score": round(norm_best, 6),
            },
            "outcome": outcome,
        }

    # ------------------------------------------------------------------
    # Candidate menus
    # ------------------------------------------------------------------

    def _build_menu(
        self, event_type: str, cycle: int, living: List["Agent"],
    ) -> List[ProposedLaw]:
        """
        Return the candidate menu for one event type, "do nothing" last.

        The menus are intentionally tiny.  Each varies exactly one decision
        variable — the ration cap for drought, and act/don't-act for storm and
        epidemic — over the whole living population.  No risk-banding, no
        per-domain split, no cross-product of law types.

        **Ordering is semantically load-bearing.**  ``NO_ACTION`` is always last,
        so that a tie resolves in favour of acting.  Ties are not hypothetical,
        and this matters most for the *pre-emptive* shelter order: a warning has
        been issued but the storm has not landed, so ``storm_damage`` is still
        zero and ``EvaluatorNode._step`` computes an identical trajectory with
        and without ``MANDATORY_SHELTER``.  A "do nothing wins ties" rule would
        have the round repeal the reflex's pre-emptive order on that tie —
        exactly when the order is most valuable.  Putting inaction last makes the
        lookahead strictly additive: it can override the reflex only on strict
        evidence that inaction is *better*, never on a coin-flip.

        Modelling movement narrows the tie but does not remove it.  Once a
        storm is actually in flight, the shelter mandate scores strictly
        better *for agents who are off shelter*.  Two
        cases still tie legitimately: a warned-but-not-landed storm (no damage to
        avoid yet, as above), and a storm during which every agent is already
        sheltered.  Both are ties because the mandate genuinely changes nothing,
        not because the evaluator is blind to it.  See the module docstring.
        """
        if event_type == "drought":
            menu = [
                ProposedLaw(
                    law_type="FOOD_RATION",
                    params={"max_per_cycle": cap},
                    # Resolved up front rather than passing ``duration=None`` at
                    # enactment, so the candidate record says what it would do.
                    # Identical value either way: ``_enact_law(duration=None)``
                    # calls this same helper.
                    duration=self._heuristic_duration(
                        "FOOD_RATION", {"max_per_cycle": cap}, cycle
                    ),
                    description=(
                        f"Leader decree (lookahead): food rationing at {cap:.1f}/cycle."
                    ),
                    source_category="drought",
                )
                for cap in FOOD_RATION_CANDIDATES
            ]
            return menu + [self._no_action_candidate("drought", "no food rationing")]

        if event_type == "storm":
            return [
                ProposedLaw(
                    law_type="MANDATORY_SHELTER",
                    params={},
                    duration=self._heuristic_duration("MANDATORY_SHELTER", {}, cycle),
                    description="Leader decree (lookahead): seek shelter immediately.",
                    source_category="storm",
                ),
                self._no_action_candidate("storm", "no shelter mandate"),
            ]

        if event_type == "epidemic":
            region = self._quarantine_region()
            if region is None:
                # Nothing to decide: with no computable region the only candidate
                # would be inaction, and "choose between one option" is not a
                # decision.  Leaving the menu empty also protects the reflex —
                # returning a lone NO_ACTION winner would suppress it.
                return []
            return [
                ProposedLaw(
                    law_type="QUARANTINE_EPIDEMIC",
                    params={"region": region},
                    duration=self._heuristic_duration(
                        "QUARANTINE_EPIDEMIC", {"region": region}, cycle
                    ),
                    description="Leader decree (lookahead): epidemic quarantine.",
                    source_category="epidemic",
                ),
                self._no_action_candidate("epidemic", "no quarantine"),
            ]

        return []

    @staticmethod
    def _no_action_candidate(event_type: str, label: str) -> ProposedLaw:
        """The counterfactual candidate.  Scored, never enacted."""
        return ProposedLaw(
            law_type=NO_ACTION,
            params={},
            duration=0,          # never enacted; no duration is ever read
            description=f"Leader decree (lookahead): {label}.",
            source_category=event_type,
        )

    def _quarantine_region(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute the quarantine region exactly as plain autocracy does.

        Reuses the base class's ``_compute_quarantine_region`` with the same
        arguments as ``AutocracyGovernment._enact_event_response``, so the region
        is *not* a decision variable in this ablation — only whether to quarantine
        at all is.  Deliberate: a better-targeted region would be a capability
        plain autocracy lacks, and the ablation must differ in foresight only.
        """
        sim = self._sim
        if sim is None:
            return None
        alive = sim.living_agents()
        infected = [a for a in alive if a.infected and a.position]
        return self._compute_quarantine_region(
            infected, len(alive), sim.grid.rows, sim.grid.cols
        )

    # ------------------------------------------------------------------
    # Enactment / repeal
    # ------------------------------------------------------------------

    def _apply_decision(
        self, event_type: str, winner: ProposedLaw, norm_score: float,
        cycle: int, living: List["Agent"],
    ) -> str:
        """
        Install the winning candidate, open a forecast on it, and report what
        happened.

        Returns one of ``"retained"``, ``"enacted"``, ``"replaced"``,
        ``"repealed"``, or ``"no_action"`` — the vocabulary the INFO line, the
        structured decision record and the ledger's ``outcome`` tag all use.

        THE SUBSTANTIVE DEPARTURE FROM ADS, and it is deliberate: **this regime
        opens a forecast on EVERY decision, including ``no_action`` and
        ``repealed``.  ADS opens one only on an enactment.**

        The metric is "the evaluator's projection for a named population,
        differenced against that population's real outcome at the same horizon".
        That is perfectly well-defined when the winning candidate was
        ``NO_ACTION``: it was scored, it won, and its score is a genuine forecast
        of what happens under inaction.  Grading only enactments would restrict
        this arm's error series to the cycles on which it changed something — a
        biased sample, biased in the direction that flatters it.  ``retained`` is
        graded for the same reason: the round re-affirmed a standing law on the
        strength of a fresh score, and that score is a forecast.

        A SUPERSET IS NOT A LIKE-FOR-LIKE COMPARISON, which is why ``outcome`` is
        a first-class partition axis rather than incidental traceability: an
        analyst filters to ``{enacted, replaced}`` for the strict cross-arm
        comparison and uses the full set for this arm's own accuracy.  The
        rejected alternative — grading only enactments, to mirror ADS — destroys
        information that cannot be recovered without another ~47-hour sweep.

        Corollary the reader should not smooth over: two rounds on one cycle open
        two forecasts per event type, both closing at t+20 against an identical
        realized value.  They are two genuinely distinct forecasts, both are
        graded, and the ``trigger`` tag is what lets an analyst dedupe.  Not a
        defect — a direct consequence of warning rounds being permitted on
        cadence cycles, which is itself ADS parity.
        """
        outcome, law_id = self._install_decision(
            event_type, winner, norm_score, cycle,
        )

        # ONE ledger call site, at the single exit, deliberately.  `_install_...`
        # has five return paths; opening the forecast inside it would mean five
        # call sites and a standing invitation for the sixth to forget.
        self._ledger.open(
            cycle=cycle,
            # `norm_score` — the winning candidate's raw/k.  The same quantity
            # ADS passes as `candidate.norm_score`, and for the same reason: it
            # is a health projection.  This arm applies no boost, so there is no
            # boosted score here to pass by mistake.
            predicted=norm_score,
            # Every living agent.  This arm's single flat target group IS the
            # scored population (`_decide_event_type` evaluates against
            # `living`), which is what makes the graded population well-defined
            # without an `applies_to` indirection.
            evaluated_agent_ids=[a.agent_id for a in living],
            tags={
                "law_id": law_id,
                "outcome": outcome,
                # CONSTANT 1 on this arm, and that constancy is informative
                # rather than dead weight: it is the dual of ADS's constant
                # `outcome`.  It is what lets an analyst select ADS's
                # single-group rounds and compare them to this arm at exactly
                # matched population scope.
                "n_groups": 1,
                "category": event_type,
                "law_type": winner.law_type,
                "trigger": self._last_decision_trigger_kind,
                # This arm's laws are always whole-population
                # (`applies_to is None`), so there are no donors to record.
                "applies_to": [],
            },
        )
        return outcome

    def _install_decision(
        self, event_type: str, winner: ProposedLaw, norm_score: float, cycle: int,
    ) -> Tuple[str, Optional[str]]:
        """
        Install the winning candidate.  Returns ``(outcome, law_id)``.

        ``law_id`` is the enacted or retained law's id, or ``None`` when the
        decision left no law standing (``no_action`` / ``repealed``).  The ledger
        routes a ``None`` to its ``n_closed_no_law`` bucket rather than
        conflating "the intervention ended" with "there was never one".
        """
        law_type = self._menu_law_type(event_type)
        if law_type is None:
            return "no_action", None
        incumbents = [
            law for law in self.active_laws
            if law.law_type == law_type and law.is_active(cycle)
        ]

        if winner.law_type == NO_ACTION:
            # Record WHICH occurrences this decision was taken about.  A later
            # occurrence the round never saw must not inherit the decision.
            self._suppressed_event_instances[event_type] = \
                self._live_instance_keys(event_type)
            self._lookahead_no_action_decisions += 1
            if not incumbents:
                return "no_action", None
            for law in incumbents:
                self._repeal_law(law, cycle, event_type)
            return "repealed", None

        # An action won, so any standing decision to do nothing is over.
        self._suppressed_event_instances.pop(event_type, None)

        # Retain rather than churn when an incumbent already implements the
        # winning decision.  Comparison is on the *decision variable* only (see
        # ``_decision_key``), so a quarantine whose region has drifted as the
        # infected moved is still "the same decision" and is left in force —
        # re-enacting it would reset its duration and mint a new law id every
        # round for no behavioural gain.
        winning_key = self._decision_key(winner.law_type, winner.params)
        for law in incumbents:
            if self._decision_key(law.law_type, law.params) == winning_key:
                self._logger.debug(
                    "cycle=%d LOOKAHEAD_RETAINED event_type=%s law_id=%s "
                    "law_type=%s params=%s",
                    cycle, event_type, law.law_id, law.law_type,
                    _params_repr(law.params),
                )
                return "retained", law.law_id

        for law in incumbents:
            self._repeal_law(law, cycle, event_type)

        law = self._enact_law(
            winner.law_type,
            dict(winner.params),
            cycle,
            duration=winner.duration,
            description=(
                f"{winner.description} "
                f"[lookahead norm={norm_score:.4f} over {LOOKAHEAD_CYCLES} cycles]"
            ),
            event_id=self._event_id_for(winner.law_type),
            event_type=self._event_type_for(winner.law_type),
            source=SOURCE_DECISION_ROUND,
        )
        self._lookahead_laws_enacted += 1
        return ("replaced" if incumbents else "enacted"), law.law_id

    @staticmethod
    def _menu_law_type(event_type: str) -> Optional[str]:
        """The single law type this event type's menu can produce."""
        return {
            "drought": "FOOD_RATION",
            "storm": "MANDATORY_SHELTER",
            "epidemic": "QUARANTINE_EPIDEMIC",
        }.get(event_type)

    @staticmethod
    def _decision_key(law_type: str, params: Dict[str, Any]) -> Tuple[Any, ...]:
        """
        Identity of a decision — law type plus whatever the menu actually varies.

        Rounded before comparison so a float that survived a round-trip through
        the law record still compares equal to the menu value it came from.
        """
        if law_type == "FOOD_RATION":
            try:
                cap = round(float(params.get("max_per_cycle", 0.0)), 6)
            except (TypeError, ValueError):
                cap = None
            return (law_type, cap)
        # Storm and epidemic menus are binary: the law's presence *is* the decision.
        return (law_type,)

    def _record_repeal_bookkeeping(self, law: Law) -> None:
        """
        Feed the two structures ``Government._expire_laws`` feeds, so the
        recorder and the audit trail see a repeal as an ordinary end-of-life with
        a distinguishable reason.

        Idempotent by construction: the append is guarded on membership and the
        reason assignment is a write of the same constant.  That matters because
        the deferral protocol in :meth:`_repeal_law` could, under a future
        change, route the same law through here twice.
        """
        if law not in self._laws_expired_this_cycle:
            self._laws_expired_this_cycle.append(law)
        self._law_expiry_reasons[law.law_id] = EXPIRY_REASON_SUPERSEDED

    def _repeal_law(self, law: Law, cycle: int, event_type: str) -> None:
        """
        Retire a law the decision round has superseded.

        WHY THE BOOKKEEPING IS CONDITIONAL.  ``Simulation._step`` dispatches
        warnings at **step 1** and calls ``tick`` at **step 6**, so a
        warning-driven repeal happens before ``_expire_laws`` has rebuilt this
        cycle's recorder structures.  Writing straight through unconditionally
        would write into the PREVIOUS cycle's structures, which
        ``_expire_laws`` then reassigns wholesale before the recorder reads
        them: the law would disappear from the archive with no expiry record
        and no reason.  ``run_recorder``'s enacted-side union with
        ``active_laws`` -- which does harden against out-of-band ENACTMENTS --
        cannot recover it either, because ``active_laws`` has already had the
        law removed by the time the write would be destroyed.

        The rejected alternative was forbidding the warning round to repeal.
        That would put a hidden behavioural divergence between the two round
        paths -- a second, invisible difference between the arms, which is
        precisely what this ablation may not have.

        Fixed government-side rather than in ``run_recorder``, deliberately: the
        recorder is regime-agnostic and the defect is this regime's.
        """
        if law in self.active_laws:
            self.active_laws.remove(law)
        if self._expired_through_cycle == cycle:
            # In-band round (step 6): `_expire_laws` has already rebuilt this
            # cycle's structures, so write straight through.
            self._record_repeal_bookkeeping(law)
        else:
            # Out-of-band round (step 1): the structures still describe cycle-1
            # and will be reassigned wholesale before the recorder reads them.
            # Defer to this cycle's `_expire_laws`.
            self._pending_repeals.setdefault(cycle, []).append(law)
        self._lookahead_laws_repealed += 1
        self._logger.debug(
            "cycle=%d LAW_REPEALED law_id=%s type=%s event_type=%s reason=%s "
            "description=%r",
            cycle, law.law_id, law.law_type, event_type,
            EXPIRY_REASON_SUPERSEDED, law.description,
        )

    def _event_type_for(self, law_type: str) -> Optional[str]:
        """
        Event-type linkage for auto-expiry, set only when the event is really live.

        Same rule ``AdsGovernment._decision_round`` applies: linking a law to an
        event type that is merely *warned* would have ``_expire_laws`` kill it on
        the very next cycle, because no event of that type is active yet.
        """
        wanted = {
            "FOOD_RATION": "drought",
            "MANDATORY_SHELTER": "storm",
            "QUARANTINE_EPIDEMIC": "epidemic",
        }.get(law_type)
        if wanted is None or self._sim is None:
            return None
        live = any(
            ev.event_type.value == wanted and not ev.cancelled
            for ev in self._sim.event_system.active_events
        )
        return wanted if live else None

    def _event_id_for(self, law_type: str) -> Optional[str]:
        """Epidemic id for quarantine laws, when an epidemic is actually active."""
        if law_type != "QUARANTINE_EPIDEMIC" or self._sim is None:
            return None
        for ev in self._sim.event_system.active_events:
            if ev.event_type.value == "epidemic" and not ev.cancelled:
                return ev.epidemic_id
        return None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Static tunables, recorded once in each run's detail header."""
        return {
            # This value is written into every run's detail header.  Archived
            # runs from when this regime was ablation-only record
            # "ablation_control" here instead of "primary"; any analysis keying
            # on this field must handle both.
            "regime_role": "primary",
            "base_regime": "autocracy",
            "t_decision": self.t_decision,
            "t_decision_crisis": T_DECISION_CRISIS,
            "lookahead_cycles": LOOKAHEAD_CYCLES,
            "decidable_event_types": list(DECIDABLE_EVENT_TYPES),
            "food_ration_candidates": list(FOOD_RATION_CANDIDATES),
            "target_groups": 1,
            # The shared evaluator's phantom-event model, recorded here as well
            # as in ADS's header so a stored run of THIS regime is
            # self-describing without a cross-reference.
            "phantom_categories": list(PHANTOM_CATEGORIES),
            "event_rate_prior_cycles": EVENT_RATE_PRIOR_CYCLES,
            "event_rate_cap": EVENT_RATE_CAP,
            # --- The three-arm ablation's self-description --------------------
            #
            # PRESENT ON BOTH ARMS.  This regime and ADS describe their own
            # epistemics with the same keys, so a reader never has to infer one
            # arm's position from an absent key.  Never ship one arm's copy
            # without the other's.
            #
            # `environment_model` answers "which construction path built the
            # evaluator's environment" and is NOT ARM-DISCRIMINATING: both arms
            # legitimately answer "inferred".  That is the correct report, not
            # information loss.
            "environment_model": "inferred",
            "parameter_inference_enabled": True,
            # `organization` is the axis the ablation isolates, and it is the
            # single EXPLICIT statement of it.  Do not prune it as redundant: it
            # is mechanically redundant (a reader can infer the arm from
            # `target_groups` here or `group_distribution_counts` on ADS), but
            # only BY IMPLICATION and by key presence — the same anti-pattern one
            # level down.
            "organization": "none",
            # The eight estimator-describing constants ADS already emits.
            # Mirrored here because this regime's `parameter_estimates` series is
            # uninterpretable without them: "drain 1.42" only means something
            # next to "crude would have been 1.0".  The four CRUDE_* values are
            # not this arm's environment, but they remain the estimators'
            # PRIOR MEANS, which is why the reference point is still the right
            # one to record.
            "crude_drain_mult": CRUDE_DRAIN_MULT,
            "crude_regen_mult": CRUDE_REGEN_MULT,
            "crude_max_steps_per_cycle": CRUDE_MAX_STEPS_PER_CYCLE,
            "crude_event_rate": CRUDE_EVENT_RATE,
            "estimator_prior_cycles": ESTIMATOR_PRIOR_CYCLES,
            "estimator_drain_bounds": [DRAIN_MULT_MIN, DRAIN_MULT_MAX],
            "estimator_regen_bounds": [REGEN_MULT_MIN, REGEN_MULT_MAX],
            "regen_obs_max_fill": REGEN_OBS_MAX_FILL,
        }

    def get_calibration_record(self, cycle: int) -> Dict[str, Any]:
        """
        Per-cycle forecast + parameter-inference block for ``run_detail.jsonl``.

        NAMED FOR THE HOOK, AND THE NAME IS REUSED ON PURPOSE.
        ``run_recorder._calibration_payload`` reaches this method by ``getattr``:
        a mismatched name does not raise, it silently drops the block from every
        record of every run.  That is the worst available failure mode, so the
        name stays even though "calibration" names a correction mechanism that
        does not exist in this codebase.

        SHAPE-IDENTICAL TO ``AdsGovernment``'s, key for key, because both are
        assembled from the same ``ForecastLedger`` and the same
        ``ParameterEstimator``.  That identity is the point: a script that reads
        one arm's series reads the other's unchanged, and a reader never has to
        infer which regime produced a record from which keys are ABSENT.

        There is deliberately no ``has_prediction_ledger`` discriminator.  An
        earlier draft added one, on the premise that this arm had no ledger.  With
        both arms carrying the full ledger it would be a constant ``true`` on
        both — a field that looks like a discriminator, is read as one, and
        discriminates nothing, which is worse than no field.  The arm is
        identified by ``government.name`` and by ``government_params.organization``.
        """
        return {
            **self._ledger.as_record(),
            "parameter_estimates": self._estimator.as_record(),
        }

    def get_calibration_summary(
        self, total_cycles: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Per-run forecast-accuracy summary, merged into ``final_stats.json``.

        Assembled exactly as ``AdsGovernment.get_calibration_summary`` assembles
        it — see :meth:`ForecastLedger.as_summary` for the arithmetic, and in
        particular for WHY three of the scalars it returns are unfiltered
        all-time means.

        THAT WARNING BITES HARDER HERE THAN ON ADS, and this is the one place a
        plausible-looking wrong figure could originate.  On ADS "every closure"
        means every enacted law, so the unfiltered scalars are correct by
        accident — its ``outcome`` tag is constant.  On this arm "every closure"
        additionally includes ``retained``, ``repealed`` and ``no_action``, a
        population ADS never grades at all.  Any figure putting the two arms side
        by side MUST derive this arm's number from
        ``calibration_closures_by_outcome_then_scope`` filtered to
        ``{enacted, replaced}`` — that is what
        ``benchmark_core.filtered_calibration_mean`` and
        ``filtered_half_split`` are for, and they are the only sanctioned readers.

        This method does NOT emit ``final_mean_prediction_error``, and this arm
        does not implement ``get_ads_metrics`` either.  Both absences are
        deliberate and are the same decision: that field is by construction an
        unfiltered all-time mean, so wiring this arm into it would manufacture
        precisely the number the paragraph above exists to stop anyone plotting.
        Leaving it ``None`` also keeps this arm's ``health_stats.csv``
        byte-identical, which is what makes its no-op gate assertable at all.
        """
        summary = self._ledger.as_summary(total_cycles)
        # Same name, same value (4) as ADS.  The `ads_` prefix is vestigial — a
        # provenance token minted when the metric was ADS-only.  This arm emits
        # it because provenance is a property of the RUN, not of the regime, and
        # this arm's metric genuinely IS generation 4's semantics: it is computed
        # by literally the same code.
        summary["ads_forecast_schema_version"] = ADS_FORECAST_SCHEMA_VERSION
        summary["parameter_estimates_final"] = self._estimator.as_record()
        return summary

    def get_audit_info(self, cycle: int) -> dict:
        info = super().get_audit_info(cycle)
        info.update({
            "lookahead_rounds_total": self._decision_rounds,
            "lookahead_candidates_evaluated_total": self._candidates_evaluated,
            "lookahead_laws_enacted_total": self._lookahead_laws_enacted,
            "lookahead_laws_repealed_total": self._lookahead_laws_repealed,
            "lookahead_no_action_decisions_total":
                self._lookahead_no_action_decisions,
            # Quantifies how often a cycle ran two rounds and therefore how often
            # `get_decision_record` reports only the later one.  ADS has the same
            # limitation and no equivalent counter, so the size of its hole is
            # not readable off a run; adding one there is a separate change.
            "lookahead_warning_rounds_total": self._warning_rounds,
            "lookahead_suppressed_event_types":
                ",".join(sorted(self._suppressed_event_instances)),
        })
        return info

    def get_decision_record(self, cycle: int) -> Optional[Dict[str, Any]]:
        """Structured record of a round that ran on THIS cycle, or None."""
        if self._last_decision_cycle != cycle or not self._last_decision_details:
            return None
        details = self._last_decision_details
        return {
            "mechanism": "autocracy_leader_lookahead",
            "trigger": {
                # A real variable over
                # {event_interval, crisis_interval, warning_preempt}, rather
                # than a hardcoded "event_interval", which would be a lie the
                # moment a warning round fires.
                "kind": details["trigger_kind"],
                "t_decision": self.t_decision,
                "t_decision_crisis": T_DECISION_CRISIS,
                "event_types": list(details["event_types"]),
                # Mirrors ADS's `trigger.pending_warnings`.
                "pending_warnings": list(details["pending_warnings"]),
            },
            "context": {
                "alive": details["population"],
                "lookahead_cycles": LOOKAHEAD_CYCLES,
                "target_groups": 1,
                "round": details["round"],
            },
            "summary": {
                "candidates_evaluated": sum(
                    d["candidates"] for d in details["decisions"]
                ),
                "decisions": len(details["decisions"]),
                "laws_enacted_total": self._lookahead_laws_enacted,
                "laws_repealed_total": self._lookahead_laws_repealed,
            },
            "decisions": details["decisions"],
        }

    def get_log_info(self, cycle: int) -> dict:
        info = super().get_log_info(cycle)
        if self._last_decision_cycle == cycle and self._last_decision_details:
            lines = [
                f"Lookahead decision round #{self._last_decision_details['round']} "
                f"({LOOKAHEAD_CYCLES}-cycle horizon, whole population):"
            ]
            for d in self._last_decision_details["decisions"]:
                lines.append(
                    f"    {d['event_type']}: {d['winner']['law_type']} "
                    f"{_params_repr(d['winner']['params'])} "
                    f"norm={d['winner']['norm_score']:.4f} "
                    f"({d['outcome']}, {d['candidates']} candidates)"
                )
            existing = info.get("decision_summary", "")
            info["decision_summary"] = (
                "\n".join(lines) if not existing
                else existing + "\n" + "\n".join(lines)
            )
        return info


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _params_repr(params: Dict[str, Any]) -> str:
    """Compact, stable ``k=v`` rendering of law params for log lines."""
    if not params:
        return "{}"
    return "{" + ",".join(f"{k}={params[k]}" for k in sorted(params)) + "}"


def _jsonable(value: Any) -> Any:
    """Convert tuples/sets to JSON-safe equivalents, recursively."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    return value


# ---------------------------------------------------------------------------
# Implementation summary
# ---------------------------------------------------------------------------
# AutocracyLookaheadGovernment subclasses AutocracyGovernment; autocracy.py is
# untouched.  __init__ inherits leader appointment, loyalist refresh and
# resource extraction unchanged; only _enact_event_response is overridden
# behaviourally (plus receive_event_warnings, to observe warnings and honour
# suppression).
#
# _enact_event_response is two-tier: inherited reflex as the fast path, then a
# periodic evaluated round, mirroring _fast_response / _decision_round in
# ads.py.  _is_decision_cycle fires every T_DECISION_CALM (5) cycles, or
# T_DECISION_CRISIS (2) while an event is active or warned — the same rule as
# AdsGovernment.tick.
#
# _build_menu, per active/warned event type: drought = FOOD_RATION at
# 1.5 / 2.5 / 4.0 plus "no ration"; epidemic = QUARANTINE_EPIDEMIC at the
# existing computed region plus "no quarantine"; storm = MANDATORY_SHELTER
# versus no shelter mandate.
#
# _decide_event_type scores every candidate with EvaluatorNode.evaluate(
# proposal, group_agents, sim) against sim.living_agents(), reusing
# ProposedLaw; both imported from governments/ads.py and left unforked, which
# is the property that matters — the ADS and this regime score with the
# *same* evaluator, so the one-factor difference between them is preserved.
#
# _apply_decision installs the single highest-scoring candidate, "do nothing"
# included, superseding the reflex's canned response for that event type.
#
# Resource extraction is not evaluated and not overridden; it fires on its
# own fixed schedule exactly as in AutocracyGovernment.
#
# Logging is one DEBUG line per candidate (event type, params, raw and
# normalised score) and INFO per decision and per round, following the
# key=value style of base.py's _enact_law / _expire_laws.
#
# One law, one target group per decision — applies_to is None (whole living
# population); no risk-banding, no per-domain grouping, no role separation.
#
# Registration, --list-governments and the README row are covered in
# governments/__init__.py, run_simulation.py, benchmark_core.py and README.md.
