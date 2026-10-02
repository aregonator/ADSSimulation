"""Abstract base class for all government types."""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

#: Seed for the fallback :attr:`Government.rng` installed by
#: :meth:`Government.__init__`.  Every seeded subclass overwrites ``self.rng``
#: immediately after calling ``super().__init__()``; this default exists so that
#: governments constructed without a seed (``anarchy``) still hold a *seeded*
#: generator rather than one keyed to OS entropy.  Any fixed integer will do —
#: what matters is that it is fixed.
DEFAULT_GOVERNMENT_SEED = 0

if TYPE_CHECKING:
    from engine.agent import Agent, Action
    from engine.events import EventWarning
    from engine.simulation import Simulation


@dataclass
class Law:
    """An active law enacted by the government."""
    law_id: str
    law_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    enacted_cycle: int = 0
    duration: int = 20           # cycles the law remains active
    description: str = ""
    applies_to: Optional[List[str]] = None  # agent_ids this law targets; None = all agents
    # Event linkage — when set, law auto-expires if the associated event ends
    event_id: Optional[str] = None    # epidemic_id or similar per-instance event identifier
    event_type: Optional[str] = None  # EventType.value string ("epidemic", "storm", "drought", …)
    # Which mechanism produced this law — e.g. "fast_response" (a reflex) vs
    # "decision_round" (an evaluated proposal).  Tracked as a structured field
    # rather than a free-text prefix inside `description` because it is
    # directly relevant to how much of ADS's behaviour comes from evaluation as
    # opposed to reflexes.  Last field and defaulted, so no positional breakage.
    source: Optional[str] = None

    def is_active(self, cycle: int) -> bool:
        return cycle < self.enacted_cycle + self.duration

    def cycles_remaining(self, cycle: int) -> int:
        return max(0, self.enacted_cycle + self.duration - cycle)

    def applies_to_agent(self, agent_id: str) -> bool:
        """Return True if this law applies to the given agent."""
        return self.applies_to is None or agent_id in self.applies_to


class Government(ABC):
    """
    Abstract interface for all government types.
    The simulation calls: bind(), tick(), filter_actions(), can_move(),
    food_collection_limit(), receive_vote(), receive_proposal(),
    receive_event_warnings().
    """

    name: str = "unknown"

    def __init__(self):
        self._sim: Optional["Simulation"] = None
        #: Decision RNG.  Seeded subclasses replace this right after calling
        #: ``super().__init__()``; declaring it here guarantees that *every*
        #: government — including unseeded ``anarchy`` — has a seeded generator,
        #: so shared code in this class (see ``filter_actions``) never has to
        #: reach for the process-global ``random`` module.
        self.rng = random.Random(DEFAULT_GOVERNMENT_SEED)
        self.active_laws: List[Law] = []
        self._law_counter = 0
        self._laws_enacted_this_cycle: List[Law] = []
        self._laws_expired_this_cycle: List[Law] = []
        # law_id -> why it expired this cycle.  "Why did the quarantine lift?"
        # is a first-order debugging question, so the answer is recorded here
        # instead of discarded.  Reset each cycle alongside the enacted list.
        self._law_expiry_reasons: Dict[str, str] = {}
        # Logger is keyed to the government instance name; set after name is resolved.
        # Subclasses that set self.name before calling super().__init__ will get the
        # correct logger immediately; others get it refreshed in bind().
        self._logger = logging.getLogger(f"sim.{self.name}")

    def bind(self, sim: "Simulation") -> None:
        """Called once when simulation is created; gives government access to sim."""
        self._sim = sim
        # Refresh logger now that the concrete subclass name is fully resolved.
        self._logger = logging.getLogger(f"sim.{self.name}")

    # ------------------------------------------------------------------
    # Per-cycle tick — government-specific logic (voting, ADS cycle, etc.)
    # ------------------------------------------------------------------

    @abstractmethod
    def tick(self, cycle: int) -> None:
        """Called once per cycle after all agents have acted."""
        ...

    # ------------------------------------------------------------------
    # Action mediation
    # ------------------------------------------------------------------

    def filter_actions(
        self, agent: "Agent", actions: List["Action"], cycle: int
    ) -> List["Action"]:
        """
        Apply active laws to an agent's action list.

        REDISTRIBUTE laws use pre-computed donor_amounts stored in law.params:
          - recipient_ids: list of K agent IDs who each receive amount_per_recipient
          - donor_amounts: {agent_id: total_to_give} pre-computed by the government
          Each donor injects a SHARE_RESOURCE action giving total_to_give / K to each recipient.

        REDISTRIBUTE_GENERIC covers all three resources in one law.

        Other laws handled: MOVE_TO_REGION, FLEE_CURRENT_LOCATION, SPREAD,
        EPIDEMIC_RESPONSE.
        """
        from engine.agent import Action, ActionType

        modified = list(actions)
        agent_laws = self.laws_for_agent(agent.agent_id, cycle)

        for law in agent_laws:
            lt = law.law_type

            # --- Redistribute single resource: K agents each get N from M donors --
            if lt in ("REDISTRIBUTE_FOOD", "REDISTRIBUTE_WATER", "REDISTRIBUTE_MEDICINE"):
                resource = lt.split("_", 1)[1].lower()
                donor_amounts: Dict[str, float] = law.params.get("donor_amounts", {})
                recipient_ids: List[str] = law.params.get("recipient_ids", [])
                amount = donor_amounts.get(agent.agent_id, 0.0)
                K = len(recipient_ids)
                if amount > 0 and K > 0:
                    modified.append(Action(ActionType.SHARE_RESOURCE, {
                        "resource": resource,
                        "target_ids": recipient_ids,
                        "amount": amount / K,   # simulation splits evenly among recipients
                        "law_directed": True,
                    }))

            # --- Redistribute generic (all three resources in one law) -----------
            elif lt == "REDISTRIBUTE_GENERIC":
                donor_amounts = law.params.get("donor_amounts", {})
                recipient_ids = law.params.get("recipient_ids", [])
                K = len(recipient_ids)
                for resource in ("food", "water", "medicine"):
                    amount = donor_amounts.get(agent.agent_id, {}).get(resource, 0.0) \
                             if isinstance(donor_amounts.get(agent.agent_id), dict) \
                             else 0.0
                    if amount > 0 and K > 0:
                        modified.append(Action(ActionType.SHARE_RESOURCE, {
                            "resource": resource,
                            "target_ids": recipient_ids,
                            "amount": amount / K,
                            "law_directed": True,
                        }))

            # --- Move toward a designated region (random point inside bounds) ---
            elif lt == "MOVE_TO_REGION" and agent.position:
                region = law.params.get("region")   # (r_min, c_min, r_max, c_max)
                target = law.params.get("assigned_target", {}).get(agent.agent_id)
                if region and not target:
                    # REPRODUCIBILITY-CRITICAL — must be ``self.rng``, never the
                    # process-global ``random`` module.  CPython seeds that module
                    # from OS entropy at interpreter start and nothing here
                    # re-seeds it, so drawing from it would make every
                    # MOVE_TO_REGION assignment unreproducible even with an
                    # identical ``base_seed`` — and this sits on the movement
                    # path, which feeds back into resource collection and
                    # survival.  ``self.rng`` is the government's seeded decision
                    # RNG (see ``__init__``); the distribution drawn from is
                    # unchanged, only its source.
                    r_min, c_min, r_max, c_max = region
                    target = (
                        self.rng.randint(r_min, r_max),
                        self.rng.randint(c_min, c_max),
                    )
                if target:
                    direction = agent.direction_toward(*target)
                    if direction:
                        modified.insert(0, Action(ActionType.MOVE, {"direction": direction}))

            # --- Flee current location: head toward least-hazard direction ------
            elif lt == "FLEE_CURRENT_LOCATION" and agent.position and self._sim:
                from engine.agent import DIRECTION_DELTAS
                r, c = agent.position
                visited: Set[Tuple[int, int]] = agent.memory.get("flee_visited", set())
                visited.add((r, c))
                agent.memory["flee_visited"] = visited

                best_dir = "north"
                best_score = float("inf")
                for d, (dr, dc) in DIRECTION_DELTAS.items():
                    nr, nc = r + dr, c + dc
                    if not self._sim.grid.in_bounds(nr, nc):
                        continue
                    if (nr, nc) in visited:
                        continue
                    cell = self._sim.grid.cell(nr, nc)
                    score = cell.hazard + getattr(cell, "hazard_extra", 0.0)
                    if score < best_score:
                        best_score = score
                        best_dir = d
                modified.insert(0, Action(ActionType.MOVE, {"direction": best_dir}))

            # --- Spread: move away from nearest other agent ------------------
            elif lt == "SPREAD" and agent.position and self._sim:
                from engine.agent import DIRECTION_DELTAS
                r, c = agent.position
                others = [
                    a for a in self._sim.living_agents()
                    if a.agent_id != agent.agent_id and a.position
                ]
                if others:
                    # Find nearest agent; move in opposite direction
                    nearest = min(others, key=lambda a: abs(a.position[0] - r) + abs(a.position[1] - c))
                    nr, nc = nearest.position
                    dr = r - nr
                    dc = c - nc
                    vert = "south" if dr < 0 else ("north" if dr > 0 else "")
                    horiz = "east" if dc < 0 else ("west" if dc > 0 else "")
                    direction = f"{vert}{horiz}" if vert and horiz else vert or horiz or "north"
                    modified.insert(0, Action(ActionType.MOVE, {"direction": direction}))

            # --- Quarantine epidemic: boundary avoidance + internal spreading ---
            elif lt == "QUARANTINE_EPIDEMIC" and agent.position and self._sim:
                region = law.params.get("region")
                if region:
                    from engine.agent import DIRECTION_DELTAS
                    r, c = agent.position
                    r_min, c_min, r_max, c_max = region

                    if self._in_region(r, c, region):
                        # Agent is INSIDE the quarantine region.
                        # Compute a repulsion vector away from all other agents also
                        # inside the region to spread them apart and reduce the chance
                        # of re-infection between quarantined agents.
                        inside_others = [
                            a for a in self._sim.living_agents()
                            if a.agent_id != agent.agent_id
                            and a.position
                            and self._in_region(*a.position, region)
                        ]
                        if inside_others:
                            # Sum weighted repulsion vectors: closer agents push harder.
                            rep_r: float = 0.0
                            rep_c: float = 0.0
                            for other in inside_others:
                                or_, oc = other.position
                                diff_r = r - or_
                                diff_c = c - oc
                                dist = max(1, abs(diff_r) + abs(diff_c))
                                rep_r += diff_r / dist
                                rep_c += diff_c / dist

                            # Map the continuous repulsion vector to a direction string.
                            vert = (
                                "south" if rep_r > 0
                                else ("north" if rep_r < 0 else "")
                            )
                            horiz = (
                                "east" if rep_c > 0
                                else ("west" if rep_c < 0 else "")
                            )
                            direction = (
                                f"{vert}{horiz}" if vert and horiz
                                else vert or horiz or "north"
                            )

                            # Pre-validate: only inject the move if it keeps the agent
                            # inside the region (can_move already enforces this via its
                            # QUARANTINE_EPIDEMIC branch, but checking here avoids
                            # prepending an action that will be immediately blocked).
                            dr, dc = DIRECTION_DELTAS[direction]
                            nr, nc = r + dr, c + dc
                            if (self._in_region(nr, nc, region)
                                    and self.can_move(agent, nr, nc, cycle)):
                                modified.insert(
                                    0, Action(ActionType.MOVE, {"direction": direction})
                                )
                    else:
                        # Agent is OUTSIDE the quarantine region.
                        # If the agent is within 1 cell (Chebyshev distance) of the
                        # boundary, push it one cell further away to avoid infection
                        # spillover from agents inside.
                        #
                        # Nearest boundary point: clamp agent coordinates into the
                        # region rectangle — this is the closest point that lies on
                        # or inside the boundary.
                        nearest_r = max(r_min, min(r, r_max))
                        nearest_c = max(c_min, min(c, c_max))
                        cheby_dist = max(abs(r - nearest_r), abs(c - nearest_c))

                        if cheby_dist <= 1:
                            # Direction is away from the nearest boundary point.
                            diff_r = r - nearest_r
                            diff_c = c - nearest_c
                            vert = (
                                "south" if diff_r > 0
                                else ("north" if diff_r < 0 else "")
                            )
                            horiz = (
                                "east" if diff_c > 0
                                else ("west" if diff_c < 0 else "")
                            )
                            direction = (
                                f"{vert}{horiz}" if vert and horiz
                                else vert or horiz or "north"
                            )

                            # Pre-validate: only inject if the destination is outside
                            # the region, within grid bounds, and permitted by can_move.
                            dr, dc = DIRECTION_DELTAS[direction]
                            nr, nc = r + dr, c + dc
                            if (not self._in_region(nr, nc, region)
                                    and self._sim.grid.in_bounds(nr, nc)
                                    and self.can_move(agent, nr, nc, cycle)):
                                modified.insert(
                                    0, Action(ActionType.MOVE, {"direction": direction})
                                )

            # --- Epidemic response: mandate treatment ------------------------
            elif lt == "EPIDEMIC_RESPONSE" and agent.infected:
                modified.insert(0, Action(ActionType.TREAT_SELF, {}))

        return modified

    def can_move(self, agent: "Agent", target_r: int, target_c: int, cycle: int) -> bool:
        """Return False if movement to (target_r, target_c) is prohibited for this agent."""
        for law in self.active_laws:
            if not law.is_active(cycle):
                continue
            lt = law.law_type

            if lt in ("QUARANTINE", "QUARANTINE_EPIDEMIC"):
                region = law.params.get("region")
                if not region:
                    continue
                quarantined_ids: Optional[List[str]] = law.applies_to
                if quarantined_ids is not None:
                    agent_is_quarantined = agent.agent_id in quarantined_ids
                else:
                    agent_is_quarantined = agent.infected
                if agent_is_quarantined and agent.position:
                    if (self._in_region(*agent.position, region)
                            and not self._in_region(target_r, target_c, region)):
                        return False
                else:
                    if self._in_region(target_r, target_c, region):
                        return False

            elif lt == "MANDATORY_SHELTER":
                if not law.applies_to_agent(agent.agent_id):
                    continue
                if self._sim and agent.position:
                    current_cell = self._sim.grid.cell(*agent.position)
                    target_cell = self._sim.grid.cell(target_r, target_c)
                    if current_cell.shelter and not target_cell.shelter:
                        return False

            elif lt == "LIMIT_STEPS":
                if not law.applies_to_agent(agent.agent_id):
                    continue
                if law.params.get("max_steps", 999) == 0:
                    return False

            elif lt == "MANDATORY_SHELTER_REGION":
                if not law.applies_to_agent(agent.agent_id):
                    continue
                region = law.params.get("region")
                if region and agent.position and self._in_region(*agent.position, region):
                    if self._sim:
                        current_cell = self._sim.grid.cell(*agent.position)
                        target_cell = self._sim.grid.cell(target_r, target_c)
                        if current_cell.shelter and not target_cell.shelter:
                            return False

        return True

    def food_collection_limit(self, agent: "Agent", cycle: int) -> float:
        """Return max food collectible per cycle for this agent (under active laws)."""
        return self._resource_collection_limit(agent, cycle, "food")

    def water_collection_limit(self, agent: "Agent", cycle: int) -> float:
        """Return max water collectible per cycle for this agent (under active laws)."""
        return self._resource_collection_limit(agent, cycle, "water")

    def medicine_collection_limit(self, agent: "Agent", cycle: int) -> float:
        """Return max medicine collectible per cycle for this agent (under active laws)."""
        return self._resource_collection_limit(agent, cycle, "medicine")

    def _resource_collection_limit(self, agent: "Agent", cycle: int, resource: str) -> float:
        """Return the minimum cap imposed on collecting `resource` this cycle."""
        law_types = {
            "food":     ("FOOD_RATION", "LIMIT_FOOD_CONSUMPTION"),
            "water":    ("LIMIT_WATER_CONSUMPTION",),
            "medicine": ("LIMIT_MEDICINE_CONSUMPTION",),
        }
        relevant = law_types.get(resource, ())
        limit = float("inf")
        for law in self.active_laws:
            if not law.is_active(cycle) or not law.applies_to_agent(agent.agent_id):
                continue
            if law.law_type in relevant:
                limit = min(limit, law.params.get("max_per_cycle", 5.0))
        return limit

    # ------------------------------------------------------------------
    # Input receivers
    # ------------------------------------------------------------------

    def receive_vote(self, agent: "Agent", option: Any, cycle: int) -> None:
        pass

    def receive_proposal(self, agent: "Agent", proposal: Any, cycle: int) -> None:
        pass

    def receive_event_warnings(self, warnings: List["EventWarning"], cycle: int) -> None:
        pass

    # ------------------------------------------------------------------
    # Law management
    # ------------------------------------------------------------------

    def _heuristic_duration(
        self,
        law_type: str,
        params: Dict[str, Any],
        cycle: int,
    ) -> int:
        """
        Return a context-appropriate duration for a law when the caller does not
        specify one explicitly.

        Values are calibrated against typical event lifetimes:
          - QUARANTINE_EPIDEMIC / EPIDEMIC_RESPONSE last until the epidemic is
            cleared (which can be long), so we use a generous base.
          - MANDATORY_SHELTER matches the storm window from warnings, plus a buffer.
          - FOOD_RATION scales with drought severity.
          - Redistribute / mobility laws are short to allow frequent re-evaluation.
        """
        if law_type == "QUARANTINE_EPIDEMIC":
            base = 20
            severity = params.get("severity", 1.0)
            if severity > 1.0:
                base += 10
            return min(50, base)

        if law_type == "MANDATORY_SHELTER":
            # Try to match the active storm's remaining duration + 3 buffer cycles.
            storm_duration = 10  # default if no sim reference
            if self._sim:
                for ev in self._sim.event_system.active_events:
                    if ev.event_type.value == "storm" and not ev.cancelled:
                        remaining = ev.cycles_remaining(cycle)
                        if remaining > 0:
                            storm_duration = remaining
                        elif ev.duration:
                            storm_duration = ev.duration
                        break
            return storm_duration + 3

        if law_type == "FOOD_RATION":
            base = 15
            if self._sim:
                drought_active = any(
                    ev.event_type.value == "drought" and not ev.cancelled
                    for ev in self._sim.event_system.active_events
                )
                if drought_active:
                    severity = max(
                        (ev.severity for ev in self._sim.event_system.active_events
                         if ev.event_type.value == "drought" and not ev.cancelled),
                        default=1.0,
                    )
                    base += max(0, int(10 * severity))
            return base

        if law_type == "EPIDEMIC_RESPONSE":
            return 25

        if law_type in ("REDISTRIBUTE_FOOD", "REDISTRIBUTE_WATER",
                        "REDISTRIBUTE_MEDICINE", "REDISTRIBUTE_GENERIC"):
            return 10

        if law_type == "MOVE_TO_REGION":
            return 8

        if law_type == "FLEE_CURRENT_LOCATION":
            return 5

        return 15  # safe default

    def _enact_law(
        self,
        law_type: str,
        params: Dict[str, Any],
        cycle: int,
        duration: Optional[int] = 20,
        description: str = "",
        applies_to: Optional[List[str]] = None,
        event_id: Optional[str] = None,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Law:
        resolved_duration = (
            self._heuristic_duration(law_type, params, cycle)
            if duration is None
            else duration
        )
        self._law_counter += 1
        law = Law(
            law_id=f"L-{self._law_counter:04d}",
            law_type=law_type,
            params=params,
            enacted_cycle=cycle,
            duration=resolved_duration,
            description=description,
            applies_to=applies_to,
            event_id=event_id,
            event_type=event_type,
            source=source,
        )
        self.active_laws.append(law)
        self._laws_enacted_this_cycle.append(law)
        self._logger.debug(
            "cycle=%d LAW_ENACTED law_id=%s type=%s duration=%d description=%r",
            cycle, law.law_id, law.law_type, resolved_duration, description,
        )
        return law

    def _expire_laws(self, cycle: int) -> None:
        """
        Remove all laws that should no longer be active:

          1. Time-based expiry: ``not law.is_active(cycle)`` — the law has outlived
             its enacted duration.

          2. Event-type expiry: if ``law.event_type`` is set and no active event of
             that type currently exists in the event system, the law is no longer
             relevant and is expired immediately.

          3. Epidemic-id expiry: if ``law.event_id`` is set (typically an
             ``epidemic_id``), the law expires as soon as no living agent still carries
             that specific epidemic.  This handles the case where all infected agents
             either die or are cured — the quarantine or treatment mandate becomes
             meaningless and should lift automatically.

        The three cases are distinguished in ``self._law_expiry_reasons``
        (law_id -> ``"duration_elapsed"`` / ``"event_type_gone"`` /
        ``"event_id_cleared"``) so a structured consumer can answer *why* a law
        lifted.  ``_laws_expired_this_cycle`` keeps its ``List[Law]`` type —
        ``engine/audit.py`` iterates it and must not break.
        """
        def _expiry_reason(law: Law) -> Optional[str]:
            # Standard time-based expiry
            if not law.is_active(cycle):
                return "duration_elapsed"

            if self._sim is None:
                return None

            # Event-type expiry: expire when the associated event class is gone
            if law.event_type is not None:
                still_active = any(
                    ev.event_type.value == law.event_type and not ev.cancelled
                    for ev in self._sim.event_system.active_events
                )
                if not still_active:
                    return "event_type_gone"

            # Epidemic-id expiry: expire when no agent carries this specific epidemic
            if law.event_id is not None:
                epidemic_id = law.event_id
                any_infected = any(
                    epidemic_id in agent.epidemic_ids
                    for agent in self._sim.agents
                    if agent.alive
                )
                if not any_infected:
                    return "event_id_cleared"

            return None

        expired: List[Law] = []
        reasons: Dict[str, str] = {}
        for law in self.active_laws:
            reason = _expiry_reason(law)
            if reason is not None:
                expired.append(law)
                reasons[law.law_id] = reason

        self._laws_expired_this_cycle = expired
        self._law_expiry_reasons = reasons
        self._laws_enacted_this_cycle = []   # reset for new cycle
        for law in expired:
            self.active_laws.remove(law)
            self._logger.debug(
                "cycle=%d LAW_EXPIRED law_id=%s type=%s description=%r",
                cycle, law.law_id, law.law_type, law.description,
            )

    def get_audit_info(self, cycle: int) -> dict:
        """Return government-specific audit data for this cycle. Override in subclasses."""
        return {}

    def get_decision_record(self, cycle: int) -> Optional[dict]:
        """
        Structured record of a decision round that ran on THIS cycle, or None.

        Default: None — a government with no explicit decision round produces no
        `decision` records.  Governments that deliberate (ADS today; democracy's
        elections and republic's parliamentary votes are natural future
        adopters) override this.  The consumer is government-agnostic: it writes
        whatever dict it receives, so adding a producer needs no schema change.

        Implementations MUST return None unless the round happened on *this*
        cycle — a record that lingers between rounds would be silently
        duplicated onto every intervening cycle.
        """
        return None

    def get_params(self) -> dict:
        """
        Static, JSON-safe configuration of this government (tunables, not state).

        Recorded once per run in the detail file's header so a reader can tell
        what the government was configured to do without reading the source at
        the matching revision.  Default: nothing to declare.
        """
        return {}

    def get_log_info(self, cycle: int) -> dict:
        """Return rich structured data for the simulation log.

        Subclasses should override to provide:
          - decision_summary: human-readable description of what the government did this cycle
          - decision_trigger: what prompted the action (event, scheduled vote, etc.)
          - vote_details: dict with vote breakdowns by cohort (for voting governments)
          - state_narrative: prose description of government-specific state

        Falls back to wrapping get_audit_info() if not overridden.
        """
        info = self.get_audit_info(cycle)
        lines = []
        for k, v in info.items():
            lines.append(f"  {k}: {v}")
        return {
            "decision_summary": "",
            "decision_trigger": "",
            "vote_details": {},
            "state_narrative": "\n".join(lines) if lines else "",
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _in_region(self, r: int, c: int, region: Tuple) -> bool:
        r_min, c_min, r_max, c_max = region
        return r_min <= r <= r_max and c_min <= c <= c_max

    def _compute_quarantine_region(
        self,
        infected: List["Agent"],
        n_alive: int,
        grid_rows: int,
        grid_cols: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute a quarantine region that covers all infected agents + buffer."""
        if not infected:
            return None
        rows = [a.position[0] for a in infected if a.position]
        cols = [a.position[1] for a in infected if a.position]
        if not rows:
            return None
        r_center = sum(rows) // len(rows)
        c_center = sum(cols) // len(cols)
        spread = max(max(rows) - min(rows), max(cols) - min(cols)) // 2
        frac_radius = int(len(infected) / max(1, n_alive) * max(grid_rows, grid_cols))
        radius = max(2, spread + 1, frac_radius)
        return (
            max(0, r_center - radius),
            max(0, c_center - radius),
            min(grid_rows - 1, r_center + radius),
            min(grid_cols - 1, c_center + radius),
        )

    def _compute_redistribution_amounts(
        self,
        agents_by_id: Dict[str, "Agent"],
        recipient_ids: List[str],
        donor_ids: List[str],
        resource: str,
        amount_per_recipient: float,
    ) -> Tuple[Dict[str, float], float]:
        """
        Pre-compute how much each M donor must give so K recipients each get N units.

        Returns (donor_amounts, actual_n) where actual_n may be less than
        amount_per_recipient if donors collectively lack sufficient stock.

        Overflow redistribution: if any donor can't cover their equal share,
        the deficit is redistributed to the richest remaining donors iteratively.
        """
        K = len(recipient_ids)
        N = amount_per_recipient
        stock_attr = f"{resource}_stock"

        if not donor_ids or K == 0 or N <= 0:
            return {}, N

        total_needed = K * N
        total_available = sum(
            getattr(agents_by_id.get(did), stock_attr, 0.0)
            for did in donor_ids
            if agents_by_id.get(did)
        )
        if total_available < total_needed:
            N = total_available / K
            total_needed = total_available

        base = total_needed / len(donor_ids)
        donor_amounts: Dict[str, float] = {did: base for did in donor_ids}

        for _ in range(len(donor_ids)):
            deficit = 0.0
            can_absorb: Dict[str, float] = {}
            for did in donor_ids:
                stock = getattr(agents_by_id.get(did), stock_attr, 0.0)
                if donor_amounts[did] > stock:
                    deficit += donor_amounts[did] - stock
                    donor_amounts[did] = stock
                else:
                    slack = stock - donor_amounts[did]
                    if slack > 0:
                        can_absorb[did] = slack
            if deficit <= 0 or not can_absorb:
                break
            sorted_rich = sorted(
                can_absorb.keys(),
                key=lambda did: getattr(agents_by_id.get(did), stock_attr, 0.0),
                reverse=True,
            )
            for did in sorted_rich:
                if deficit <= 0:
                    break
                take = min(deficit, can_absorb[did])
                donor_amounts[did] += take
                deficit -= take

        return {did: v for did, v in donor_amounts.items() if v > 0.001}, N

    def active_law_summary(self, cycle: int, agent_id: Optional[str] = None) -> List[dict]:
        return [
            {
                "law_id": l.law_id,
                "type": l.law_type,
                "cycles_remaining": l.cycles_remaining(cycle),
                "description": l.description,
                "params": l.params,
            }
            for l in self.active_laws
            if l.is_active(cycle) and (agent_id is None or l.applies_to_agent(agent_id))
        ]

    def laws_for_agent(self, agent_id: str, cycle: int) -> List[Law]:
        """Return all active laws that apply to a specific agent."""
        return [l for l in self.active_laws
                if l.is_active(cycle) and l.applies_to_agent(agent_id)]
