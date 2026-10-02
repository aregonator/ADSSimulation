"""Environmental events that create survival pressure and test collective decision-making."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .difficulty import effective_difficulty

_log = logging.getLogger("sim.events")

if TYPE_CHECKING:
    from .grid import Grid
    from .agent import Agent


# ---------------------------------------------------------------------------
# Severity -> per-cycle effect conversions
# ---------------------------------------------------------------------------
#
# Named rather than inlined because THREE construction sites must agree on
# them: ``EventSystem._random_event`` below, and ``_random_event_dict`` /
# ``make_scheduled_event_dict`` in ``engine/scenario_plan.py`` (the benchmark
# builds every event through the latter two).  A drift between them would make
# the lookahead's phantom events a different species from the real ones.

#: ``ActiveEvent.drought_factor`` per unit of severity.
DROUGHT_FACTOR_PER_SEVERITY = 0.35
#: ``ActiveEvent.storm_damage`` per unit of severity.
STORM_DAMAGE_PER_SEVERITY = 0.04
#: ``ActiveEvent.disease_spread_rate`` per unit of severity, for events minted
#: by :meth:`EventSystem._random_event`.  ``scenario_plan`` uses 0.15 instead;
#: the two paths genuinely differ and this constant describes only this file's.
DISEASE_SPREAD_PER_SEVERITY = 0.20


# ---------------------------------------------------------------------------
# Observed-event statistics (consumed by the ADS / autocracy_lookahead evaluator)
# ---------------------------------------------------------------------------

#: The three categories :meth:`EventSystem.observed_event_stats` reports on, and
#: the three the lookahead evaluator models.  ``toxic_spill`` and
#: ``resource_rush`` are deliberately absent: the evaluator has no channel for
#: either (hazard is pinned to the agent's origin cell for the whole rollout,
#: and resource regeneration is modelled only through the drought factor), so
#: reporting them would invite a caller to inject an event that cannot be felt.
OBSERVED_EVENT_CATEGORIES: Tuple[str, ...] = ("drought", "storm", "epidemic")

#: Weak Gamma prior on the arrival rate, expressed as pseudo-cycles during
#: which no event was observed: ``rate = n / (cycle + EVENT_RATE_PRIOR_CYCLES)``.
#:
#: The naive ``n / cycle`` is violently unstable early — one storm at cycle 3
#: implies a 33%-per-cycle storm rate, an order of magnitude above the truth
#: (``event_frequency * weight``, i.e. 0.05 * 0.25 = 0.0125 for the random
#: generator below) — which would flood every early projection with phantom
#: storms at exactly the point where the government has the least evidence.
#: Shrinks toward zero early and is asymptotically negligible.
#:
#: Deliberately NOT derived from ``self.event_frequency``: reading the prior off
#: the generator's own knob would be an information leak from the world model
#: into the agent's estimate of it, and would make the estimator untestable
#: against a mis-specified world.
EVENT_RATE_PRIOR_CYCLES = 20

#: Hard ceiling on a reported per-cycle arrival rate.  Purely defensive: a
#: pathological history (a scripted scenario firing an event every cycle) must
#: not be able to drive the evaluator's injection probability to 1.0 and turn
#: every projection into a permanent storm.
EVENT_RATE_CAP = 0.5

#: Duration used for a category with no observed history.  Unreachable on the
#: normal path — ``rate`` is 0.0 when ``n == 0``, so nothing is ever injected —
#: but a defined value beats ``None`` propagating into arithmetic.  Values are
#: the midpoints of this file's own random-event duration ranges.
FALLBACK_EVENT_DURATION: Dict[str, int] = {"drought": 9, "storm": 5, "epidemic": 12}


class EventType(Enum):
    DROUGHT = "drought"
    STORM = "storm"
    EPIDEMIC = "epidemic"
    TOXIC_SPILL = "toxic_spill"
    RESOURCE_RUSH = "resource_rush"


@dataclass
class ActiveEvent:
    event_type: EventType
    start_cycle: int
    # duration is None for EPIDEMIC events (they never end naturally).
    # For all other event types it is a positive integer number of cycles.
    duration: Optional[int] = None
    severity: float = 1.0
    region: Optional[Tuple[int, int, int, int]] = None   # (r_min, c_min, r_max, c_max)
    description: str = ""

    # Per-cycle parameters derived from type
    drought_factor: float = 0.0      # 0–1, multiplies regen reduction
    storm_damage: float = 0.0        # health drain per cycle (for unsheltered agents)
    disease_spread_rate: float = 0.0 # probability of spread per cell-neighbour per cycle
    toxic_drain: float = 0.0         # extra hazard per cycle in affected cells
    resource_boost: float = 0.0      # extra resource regen in affected region

    # Epidemic identity — unique per epidemic instance
    epidemic_id: Optional[str] = None     # set on creation; persists for agent tracking
    epidemic_seed_row: Optional[int] = None
    epidemic_seed_col: Optional[int] = None

    cancelled: bool = False

    def is_active(self, cycle: int) -> bool:
        if self.cancelled:
            return False
        if self.duration is None:
            return True   # None duration → indefinite (used for epidemics)
        return cycle < self.start_cycle + self.duration

    def cycles_remaining(self, cycle: int) -> int:
        if self.duration is None:
            return -1   # indefinite
        return max(0, self.start_cycle + self.duration - cycle)


@dataclass
class EventWarning:
    """Advance warning broadcast to agents before an event starts."""
    event_type: EventType
    cycles_until: int
    severity: float
    region: Optional[Tuple[int, int, int, int]]
    description: str


class EventSystem:
    """Manages scheduled and random events, applies their effects each cycle."""

    def __init__(
        self,
        event_frequency: float = 0.05,
        event_severity: float = 1.0,
        event_warning_cycles: int = 1,
        seed: Optional[int] = None,
        difficulty: int = 25,
    ):
        self.event_frequency = event_frequency
        self.base_severity = event_severity
        self.warning_cycles = event_warning_cycles
        self.difficulty = difficulty
        self.rng = random.Random(seed)
        self.active_events: List[ActiveEvent] = []
        self.pending_warnings: List[EventWarning] = []
        self.scheduled_events: List[Tuple[int, ActiveEvent]] = []  # (trigger_cycle, event)
        self.history: List[ActiveEvent] = []
        #: Monotonic counter backing :meth:`_next_epidemic_id`.  See that method
        #: for why epidemic identifiers must not come from ``uuid.uuid4()``.
        self._epidemic_counter: int = 0

    # ------------------------------------------------------------------
    # Epidemic identity
    # ------------------------------------------------------------------

    def _next_epidemic_id(self) -> str:
        """
        Mint the next epidemic identifier for this EventSystem.

        REPRODUCIBILITY-CRITICAL — do not replace this with ``uuid.uuid4()``,
        which draws on OS entropy that no ``random.seed()`` can control.

        These identifiers are stored in ``Agent.epidemic_ids`` (a ``set``) and
        gathered into further sets by the governments, most notably
        ``EvaluatorNode.evaluate`` in
        ``governments/ads.py``, which iterates the set of active epidemics while
        drawing from an RNG inside the loop.  Iteration order of a ``set`` of
        strings is a function of those strings' hashes, so freshly random
        identifiers made the *order* random on every run — and therefore made
        run outcomes vary run-to-run even with an identical ``base_seed``.
        (``PYTHONHASHSEED`` is pinned to ``"0"`` for workers, which stabilises
        the hash *of a given string*; it cannot help when the strings themselves
        are new random values each run.)

        A per-EventSystem counter is used rather than a draw from ``self.rng``
        so that minting an identifier consumes no randomness and therefore does
        not perturb the event stream.  Identifiers need only be unique within a
        single simulation: they never leave one ``EventSystem`` / ``Agent`` set.
        """
        self._epidemic_counter += 1
        return f"epi{self._epidemic_counter:04d}"

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(self, cycle: int, event: ActiveEvent) -> None:
        self.scheduled_events.append((cycle, event))

    def schedule_sequence(self, events: List[Tuple[int, ActiveEvent]]) -> None:
        self.scheduled_events.extend(events)

    # ------------------------------------------------------------------
    # Per-cycle update
    # ------------------------------------------------------------------

    def update(self, cycle: int, grid: "Grid", agents: List["Agent"]) -> List[EventWarning]:
        """
        Called once per cycle. Emits advance warnings, starts events, applies
        effects, expires finished events, and returns new warnings for this cycle.

        Warning window:
          Scheduled events emit a warning each cycle from
            (trigger_cycle - warning_cycles)  through  (trigger_cycle - 1).
          At trigger_cycle the event starts and a final "ACTIVE" warning fires.
          Random events fire immediately so only an "ACTIVE" warning is emitted.
        """
        new_warnings: List[EventWarning] = []

        # Advance warnings for upcoming scheduled events
        for trigger_cycle, event in self.scheduled_events:
            cycles_until = trigger_cycle - cycle
            if 1 <= cycles_until <= self.warning_cycles:
                new_warnings.append(EventWarning(
                    event_type=event.event_type,
                    cycles_until=cycles_until,
                    severity=event.severity,
                    region=event.region,
                    description=(
                        f"[WARNING in {cycles_until} cycle(s)] {event.description}"
                    ),
                ))

        # Trigger scheduled events whose time has come
        for trigger_cycle, event in list(self.scheduled_events):
            if cycle >= trigger_cycle:
                self._start_event(event, cycle, grid, agents)
                self.scheduled_events.remove((trigger_cycle, event))
                new_warnings.append(EventWarning(
                    event_type=event.event_type,
                    cycles_until=0,
                    severity=event.severity,
                    region=event.region,
                    description=f"[ACTIVE] {event.description}",
                ))

        # Random event generation (no advance warning — random by definition)
        if self.rng.random() < self.event_frequency:
            event = self._random_event(cycle, grid)
            self._start_event(event, cycle, grid, agents)
            new_warnings.append(EventWarning(
                event_type=event.event_type,
                cycles_until=0,
                severity=event.severity,
                region=event.region,
                description=f"[ACTIVE] {event.description}",
            ))

        # Apply active events
        drought_factor = 0.0
        for event in list(self.active_events):
            if not event.is_active(cycle):
                self._end_event(event, grid)
                self.active_events.remove(event)
                continue
            drought_factor = max(drought_factor, event.drought_factor)
            self._apply_event(event, cycle, grid, agents)

        return new_warnings, drought_factor

    # ------------------------------------------------------------------
    # Event lifecycle
    # ------------------------------------------------------------------

    def _start_event(self, event: ActiveEvent, cycle: int, grid: "Grid",
                     agents: Optional[List["Agent"]] = None) -> None:
        event.start_cycle = cycle
        if event.event_type == EventType.EPIDEMIC and event.epidemic_id is None:
            event.epidemic_id = self._next_epidemic_id()
        self.active_events.append(event)
        self.history.append(event)
        _log.debug(
            "cycle=%d EVENT_START type=%s severity=%.2f duration=%s epidemic_id=%s region=%s",
            cycle, event.event_type.value, event.severity,
            event.duration, event.epidemic_id, event.region,
        )
        if event.event_type == EventType.TOXIC_SPILL and event.region:
            r_min, c_min, r_max, c_max = event.region
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if grid.in_bounds(r, c):
                        grid.cell(r, c).hazard_extra += event.toxic_drain
        if event.event_type == EventType.EPIDEMIC and event.epidemic_id:
            self._seed_epidemic(event, grid, agents or [])

    def _seed_epidemic(self, event: ActiveEvent, grid: "Grid",
                       agents: List["Agent"]) -> None:
        alive = [a for a in agents if a.alive and a.position]
        if not alive:
            return
        # Knee-folded difficulty — the one canonical difficulty schedule, shared
        # with SimulationConfig.from_difficulty and difficulty_multiplier.  NOTE
        # that unlike Grid, this class stores self.difficulty unclamped, so a
        # difficulty < 1 would otherwise pass straight through into a negative
        # infection_pct; effective_difficulty clamps to [1, 100] first.  No
        # caller in this repo supplies an out-of-range difficulty.
        d_eff = effective_difficulty(self.difficulty)
        infection_pct = 0.01 + (d_eff - 1) / 99 * 0.09
        k = max(1, int(len(alive) * infection_pct))
        r0 = event.epidemic_seed_row
        c0 = event.epidemic_seed_col
        if r0 is None or c0 is None:
            r0 = self.rng.randint(0, grid.rows - 1)
            c0 = self.rng.randint(0, grid.cols - 1)
        alive.sort(key=lambda a: abs(a.position[0] - r0) + abs(a.position[1] - c0))
        patient0 = alive[0]
        patient0.infect(event.epidemic_id)
        p0r, p0c = patient0.position
        remaining = [a for a in alive[1:] if event.epidemic_id not in a.epidemic_ids]
        remaining.sort(key=lambda a: abs(a.position[0] - p0r) + abs(a.position[1] - p0c))
        for agent in remaining[:k - 1]:
            agent.infect(event.epidemic_id)
        _log.debug(
            "EPIDEMIC_SEED epidemic_id=%s seed_origin=(%d,%d) initial_infected=%d",
            event.epidemic_id, r0, c0, k,
        )

    def _end_event(self, event: ActiveEvent, grid: "Grid") -> None:
        _log.debug(
            "EVENT_EXPIRE type=%s epidemic_id=%s duration=%s",
            event.event_type.value, event.epidemic_id, event.duration,
        )
        if event.event_type == EventType.TOXIC_SPILL and event.region:
            r_min, c_min, r_max, c_max = event.region
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if grid.in_bounds(r, c):
                        grid.cell(r, c).hazard_extra = max(
                            0.0, grid.cell(r, c).hazard_extra - event.toxic_drain
                        )
        # Epidemics never end naturally — _end_event is never called for them

    def _apply_event(
        self, event: ActiveEvent, cycle: int, grid: "Grid", agents: List["Agent"]
    ) -> None:
        if event.event_type == EventType.STORM:
            self._apply_storm(event, agents)
        elif event.event_type == EventType.EPIDEMIC:
            self._apply_epidemic(event, grid, agents)
        elif event.event_type == EventType.RESOURCE_RUSH and event.region:
            self._apply_resource_rush(event, grid)

    def _apply_storm(self, event: ActiveEvent, agents: List["Agent"]) -> None:
        for agent in agents:
            if not agent.alive:
                continue
            if agent.position:
                from .grid import Grid  # avoid circular at module level
                # We don't have direct grid access here; the damage is resolved in simulation
                pass
            # Damage is applied in simulation.py using event data

    def _apply_epidemic(self, event: ActiveEvent, grid: "Grid", agents: List["Agent"]) -> None:
        epidemic_id = event.epidemic_id
        if not epidemic_id:
            return
        # Spread from all agents carrying this epidemic ID to all 8 neighbours
        carriers = [a for a in agents if a.alive and epidemic_id in a.epidemic_ids]
        newly_infected: Set[str] = set()
        for carrier in carriers:
            if not carrier.position:
                continue
            r, c = carrier.position
            # 8-directional spread (Chebyshev distance 1)
            for neighbor_cell in grid.neighbors(r, c, radius=1):
                for candidate in neighbor_cell.agents:
                    if not candidate.alive or epidemic_id in candidate.epidemic_ids:
                        continue
                    if self.rng.random() < event.disease_spread_rate:
                        newly_infected.add(candidate.agent_id)
            # Contaminate the carrier's cell
            grid.cell(r, c).contaminated = True

        agent_map = {a.agent_id: a for a in agents}
        for aid in newly_infected:
            if aid in agent_map:
                agent_map[aid].infect(epidemic_id)

    def _apply_resource_rush(self, event: ActiveEvent, grid: "Grid") -> None:
        r_min, c_min, r_max, c_max = event.region
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if grid.in_bounds(r, c):
                    cell = grid.cell(r, c)
                    cell.food = min(100.0, cell.food + event.resource_boost)
                    cell.water = min(100.0, cell.water + event.resource_boost * 0.5)

    # ------------------------------------------------------------------
    # Random event factory
    # ------------------------------------------------------------------

    def _random_event(self, cycle: int, grid: "Grid") -> ActiveEvent:
        weights = [0.30, 0.25, 0.15, 0.20, 0.10]  # drought/storm/epidemic/toxic/rush
        event_type = self.rng.choices(list(EventType), weights=weights, k=1)[0]
        severity = self.base_severity * self.rng.uniform(0.7, 1.3)

        if event_type == EventType.DROUGHT:
            return ActiveEvent(
                event_type=EventType.DROUGHT,
                start_cycle=cycle,
                duration=int(self.rng.uniform(6, 12)),
                severity=severity,
                drought_factor=DROUGHT_FACTOR_PER_SEVERITY * severity,
                description=f"Drought (severity={severity:.1f}): resource regeneration reduced.",
            )

        if event_type == EventType.STORM:
            return ActiveEvent(
                event_type=EventType.STORM,
                start_cycle=cycle,
                duration=int(self.rng.uniform(3, 7)),
                severity=severity,
                storm_damage=STORM_DAMAGE_PER_SEVERITY * severity,
                description=f"Storm (severity={severity:.1f}): all unsheltered agents take damage.",
            )

        if event_type == EventType.EPIDEMIC:
            r0 = self.rng.randint(0, grid.rows - 1)
            c0 = self.rng.randint(0, grid.cols - 1)
            eid = self._next_epidemic_id()
            return ActiveEvent(
                event_type=EventType.EPIDEMIC,
                start_cycle=cycle,
                duration=None,           # epidemics have no end — duration is indefinite
                severity=severity,
                disease_spread_rate=DISEASE_SPREAD_PER_SEVERITY * severity,
                epidemic_id=eid,
                epidemic_seed_row=r0,
                epidemic_seed_col=c0,
                description=f"Epidemic[{eid}] (severity={severity:.1f}): spreading from ({r0},{c0}).",
            )

        if event_type == EventType.TOXIC_SPILL:
            r_min = self.rng.randint(0, grid.rows - 3)
            c_min = self.rng.randint(0, grid.cols - 3)
            region = (r_min, c_min, min(r_min + 2, grid.rows - 1), min(c_min + 2, grid.cols - 1))
            return ActiveEvent(
                event_type=EventType.TOXIC_SPILL,
                start_cycle=cycle,
                duration=int(self.rng.uniform(5, 15)),
                severity=severity,
                region=region,
                toxic_drain=0.08 * severity,
                description=f"Toxic spill (severity={severity:.1f}): region {region} hazardous.",
            )

        # RESOURCE_RUSH
        r_min = self.rng.randint(0, grid.rows - 4)
        c_min = self.rng.randint(0, grid.cols - 4)
        region = (r_min, c_min, min(r_min + 3, grid.rows - 1), min(c_min + 3, grid.cols - 1))
        return ActiveEvent(
            event_type=EventType.RESOURCE_RUSH,
            start_cycle=cycle,
            duration=int(self.rng.uniform(5, 10)),
            severity=severity,
            region=region,
            resource_boost=5.0 * severity,
            description=f"Resource rush in region {region}: bonus resources available.",
        )

    # ------------------------------------------------------------------
    # Status queries (for agent observation)
    # ------------------------------------------------------------------

    def active_summary(self, cycle: int) -> List[dict]:
        return [
            {
                "type": e.event_type.value,
                "cycles_remaining": e.cycles_remaining(cycle),
                "severity": round(e.severity, 2),
                "region": e.region,
                "description": e.description,
                "epidemic_id": e.epidemic_id,
            }
            for e in self.active_events if e.is_active(cycle)
        ]

    def has_active(self, event_type: EventType) -> bool:
        return any(e.event_type == event_type for e in self.active_events if not e.cancelled)

    def storm_damage_per_cycle(self) -> float:
        return max(
            (e.storm_damage for e in self.active_events
             if e.event_type == EventType.STORM and not e.cancelled),
            default=0.0,
        )

    def active_event_window(self, cycle: int, event_type: EventType) -> int:
        """
        How much longer events of *event_type* will keep acting, in cycles.

        Returns ``0`` when no such event is in force, ``-1`` when at least one
        is indefinite (``duration is None``, the epidemic convention), and
        otherwise the largest :meth:`ActiveEvent.cycles_remaining` across them.

        The filter is identical to :meth:`storm_damage_per_cycle`'s — same
        ``active_events`` list, same ``not cancelled`` test — on purpose: the
        lookahead evaluator reads the *magnitude* from one and the *window*
        from the other, and a disagreement between the two would silently
        apply a storm's damage for a drought's lifetime.  Keeping both here,
        adjacent, is what makes that pairing checkable.

        Used by ``EvaluatorNode`` so its lookahead projection respects an
        event's real cutoff instead of holding the effect for the entire
        lookahead horizon.  There is no gradual decay to model — ``storm_damage``
        and ``drought_factor`` are flat constants until
        :meth:`ActiveEvent.is_active` turns false — so knowing when it turns
        false is all the projection needs.
        """
        window = 0
        for event in self.active_events:
            if event.event_type != event_type or event.cancelled:
                continue
            remaining = event.cycles_remaining(cycle)
            if remaining < 0:
                return -1           # indefinite dominates any finite window
            if remaining > window:
                window = remaining
        return window

    def observed_event_stats(self, cycle: int) -> Dict[str, Dict[str, Any]]:
        """
        Cumulative arrival rate, typical severity/effect, and typical duration
        per event category, estimated from events that have **already started**.

        Returned shape, one entry per :data:`OBSERVED_EVENT_CATEGORIES`::

            {"n":        int,              # events of this type observed so far
             "rate":     float,            # est. P(one starts in a given cycle)
             "severity": float,            # mean raw ActiveEvent.severity
             "effect":   float,            # mean per-cycle effect magnitude
             "duration": Optional[float]}  # mean lifetime; None == indefinite

        ``effect`` is the category's own per-cycle channel — ``drought_factor``
        for droughts, ``storm_damage`` for storms, ``disease_spread_rate`` for
        epidemics — averaged over the observed events **directly**, rather than
        reconstructed as ``severity * CONVERSION``.  Both would agree for events
        this file mints, but the benchmark builds its events in
        ``engine/scenario_plan.py``, whose epidemic conversion differs (0.15 vs
        this file's 0.20).  Reading the realised field cannot drift from
        whatever actually produced the event, and needs no conversion constant
        at the call site.  ``severity`` is returned alongside it because it is
        the quantity the design spec names and the one worth logging; nothing
        in the projection consumes it.

        SCIENTIFIC-VALIDITY FIREWALL — do not "improve" this by reading
        ``self.scheduled_events`` or ``self.pending_warnings``.  This estimator
        exists so a government can anticipate *the kind of future the past
        implies*; letting it see the actual schedule would hand it the answer
        key and invalidate every comparison the benchmark draws between regimes
        with and without foresight.  ``self.history`` is append-only from
        :meth:`_start_event`, so every entry in it has already begun; the
        ``start_cycle <= cycle`` filter below is redundant with that invariant
        and is kept precisely so the property is enforced rather than assumed.

        Pure read.  No state is mutated and no randomness is drawn, so this is
        safe to call once per decision round (which is how the governments use
        it) or once per candidate (which would merely be wasteful).
        """
        # A negative cycle would make the prior denominator smaller than the
        # prior itself, which is meaningless; clamp rather than trust a caller.
        cycle = max(0, int(cycle))
        denominator = float(cycle + EVENT_RATE_PRIOR_CYCLES)

        effect_attr = {
            "drought": "drought_factor",
            "storm": "storm_damage",
            "epidemic": "disease_spread_rate",
        }
        fallback_effect = {
            "drought": DROUGHT_FACTOR_PER_SEVERITY,
            "storm": STORM_DAMAGE_PER_SEVERITY,
            "epidemic": DISEASE_SPREAD_PER_SEVERITY,
        }

        stats: Dict[str, Dict[str, Any]] = {}
        for category in OBSERVED_EVENT_CATEGORIES:
            observed = [
                e for e in self.history
                if e.event_type.value == category and e.start_cycle <= cycle
            ]
            n = len(observed)
            if n == 0:
                # rate 0.0 means nothing is ever injected for this category, so
                # the remaining fields are defensive defaults that no projection
                # can actually read.  They are still populated rather than left
                # absent so every consumer can index the dict unconditionally.
                stats[category] = {
                    "n": 0,
                    "rate": 0.0,
                    "severity": 1.0,
                    "effect": fallback_effect[category],
                    "duration": float(FALLBACK_EVENT_DURATION[category]),
                }
                continue

            finite_durations = [
                float(e.duration) for e in observed
                if e.duration is not None and e.duration > 0
            ]
            if finite_durations:
                # Mean over the finite ones.  A category is reported as
                # indefinite only when EVERY observed instance was indefinite:
                # one indefinite epidemic among nine finite ones says the world
                # mostly ends its epidemics, and flipping the whole category to
                # "never expires" on that single entry would be the wrong read.
                duration: Optional[float] = sum(finite_durations) / len(finite_durations)
            else:
                duration = None      # every observed instance was indefinite

            stats[category] = {
                "n": n,
                "rate": min(n / denominator, EVENT_RATE_CAP),
                "severity": sum(e.severity for e in observed) / n,
                "effect": sum(
                    getattr(e, effect_attr[category], 0.0) for e in observed
                ) / n,
                "duration": duration,
            }
        return stats
