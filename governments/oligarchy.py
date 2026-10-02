"""Oligarchy: 1% elite hoard resources; 2% loyalists kept healthy; rest taxed."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .base import Government

if TYPE_CHECKING:
    from engine.agent import Agent, Action
    from engine.events import EventWarning


class OligarchyGovernment(Government):
    """
    A small elite (1% of population) extracts resources from the rest each
    cycle through laws — regardless of whether it benefits the population's
    collective health.  Their first priority is to hoard resources for
    themselves.

    Structure:
      Elite (1%)    — target ELITE_RESOURCE_SHARE (30%) of all resources collectively.
      Loyalists (2%) — target LOYALIST_RESOURCE_SHARE (30%) of all resources collectively.
      Rest           — subject to elite resource extraction + standard event laws.

    When an elite or loyalist dies the next-richest (by food+water stock) agent
    is immediately promoted and given a resource boost drawn from commons.
    """

    name = "Oligarchy"

    ELITE_FRACTION          = 0.01   # 1%
    LOYALIST_FRACTION       = 0.02   # 2%
    ELITE_RESOURCE_SHARE    = 0.30   # elite hold 30% of total resources
    LOYALIST_RESOURCE_SHARE = 0.30   # loyalists hold 30% of total resources
    # Stock caps are intentionally very high so that the 30% share can be reached
    # even with a large number of commons (e.g. 1 elite vs 48 commons requires
    # the elite to hold ~150 food when commons average 6 food each).
    ELITE_MAX_STOCK         = 500.0
    LOYALIST_MAX_STOCK      = 300.0
    T_RESOURCE_LAW          = 1      # extract every cycle for tight ratio control

    def __init__(
        self,
        t_law: int = 20,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.t_law = t_law
        self.rng = random.Random(seed)
        self.elite_ids: Set[str] = set()
        self.loyalist_ids: Set[str] = set()
        self._last_resource_law = -999
        self._newly_promoted_elite: List[str] = []
        self._newly_promoted_loyalists: List[str] = []

    # ------------------------------------------------------------------
    # Per-cycle tick
    # ------------------------------------------------------------------

    def tick(self, cycle: int) -> None:
        self._expire_laws(cycle)

        if not self._sim:
            return
        living = self._sim.living_agents()
        if not living:
            return

        # Initialise / refresh elite and loyalists (promotes next-richest on death)
        self._refresh_elite(living)
        self._refresh_loyalists(living)

        event_names = [e["type"] for e in self._sim.event_system.active_summary(cycle)]

        # Priority 1 & 2: resource extraction (refreshed periodically)
        if cycle - self._last_resource_law >= self.T_RESOURCE_LAW:
            self._enact_extraction(living, cycle)
            self._last_resource_law = cycle

        # Priority 3: event response for all non-elite / non-loyalist agents
        self._enact_event_response(event_names, cycle)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_elite(self, living: List["Agent"]) -> None:
        n = len(living)
        n_elite = max(1, round(n * self.ELITE_FRACTION))
        alive_ids = {a.agent_id for a in living}
        prev_elite = set(self.elite_ids)
        self.elite_ids &= alive_ids

        newly_added: List["Agent"] = []
        if len(self.elite_ids) < n_elite:
            # Rank by total resources (richest first, not healthiest)
            candidates = sorted(
                [a for a in living
                 if a.agent_id not in self.elite_ids
                 and a.agent_id not in self.loyalist_ids],
                key=lambda a: a.food_stock + a.water_stock,
                reverse=True,
            )
            for a in candidates[:n_elite - len(self.elite_ids)]:
                self.elite_ids.add(a.agent_id)
                if a.agent_id not in prev_elite:
                    newly_added.append(a)

        # Immediately boost newly promoted elite to their target share
        if newly_added:
            total_food = sum(a.food_stock for a in living)
            total_water = sum(a.water_stock for a in living)
            n_final = len(self.elite_ids)
            target_food = min(
                self.ELITE_MAX_STOCK,
                (self.ELITE_RESOURCE_SHARE * total_food) / max(1, n_final),
            )
            target_water = min(
                self.ELITE_MAX_STOCK,
                (self.ELITE_RESOURCE_SHARE * total_water) / max(1, n_final),
            )
            commons = [
                a for a in living
                if a.agent_id not in self.elite_ids
                and a.agent_id not in self.loyalist_ids
            ]
            for new_elite in newly_added:
                self._give_replacement_resources(new_elite, commons, target_food, target_water)

        self._newly_promoted_elite = [a.agent_id for a in newly_added]

    def _refresh_loyalists(self, living: List["Agent"]) -> None:
        n = len(living)
        n_loyalist = max(1, round(n * self.LOYALIST_FRACTION))
        alive_ids = {a.agent_id for a in living}
        prev_loyalists = set(self.loyalist_ids)
        self.loyalist_ids &= alive_ids
        self.loyalist_ids -= self.elite_ids  # elite cannot be loyalists

        newly_added: List["Agent"] = []
        if len(self.loyalist_ids) < n_loyalist:
            # Rank by total resources (richest first)
            candidates = sorted(
                [a for a in living
                 if a.agent_id not in self.elite_ids
                 and a.agent_id not in self.loyalist_ids],
                key=lambda a: a.food_stock + a.water_stock,
                reverse=True,
            )
            for a in candidates[:n_loyalist - len(self.loyalist_ids)]:
                self.loyalist_ids.add(a.agent_id)
                if a.agent_id not in prev_loyalists:
                    newly_added.append(a)

        # Immediately boost newly promoted loyalists to their target share
        if newly_added:
            total_food = sum(a.food_stock for a in living)
            total_water = sum(a.water_stock for a in living)
            n_final = len(self.loyalist_ids)
            target_food = min(
                self.LOYALIST_MAX_STOCK,
                (self.LOYALIST_RESOURCE_SHARE * total_food) / max(1, n_final),
            )
            target_water = min(
                self.LOYALIST_MAX_STOCK,
                (self.LOYALIST_RESOURCE_SHARE * total_water) / max(1, n_final),
            )
            commons = [
                a for a in living
                if a.agent_id not in self.elite_ids
                and a.agent_id not in self.loyalist_ids
            ]
            for new_loyal in newly_added:
                self._give_replacement_resources(new_loyal, commons, target_food, target_water)

        self._newly_promoted_loyalists = [a.agent_id for a in newly_added]

    def _give_replacement_resources(
        self,
        new_agent: "Agent",
        commons: List["Agent"],
        target_food: float,
        target_water: float,
    ) -> None:
        """Directly transfer resources from commons to a newly promoted agent."""
        needed_food = max(0.0, target_food - new_agent.food_stock)
        needed_water = max(0.0, target_water - new_agent.water_stock)

        if needed_food > 0 and commons:
            per_donor = needed_food / len(commons)
            total_given = 0.0
            for donor in commons:
                give = min(per_donor, donor.food_stock)
                donor.food_stock = max(0.0, donor.food_stock - give)
                total_given += give
            new_agent.food_stock += total_given

        if needed_water > 0 and commons:
            per_donor = needed_water / len(commons)
            total_given = 0.0
            for donor in commons:
                give = min(per_donor, donor.water_stock)
                donor.water_stock = max(0.0, donor.water_stock - give)
                total_given += give
            new_agent.water_stock += total_given

    def _enact_extraction(self, living: List["Agent"], cycle: int) -> None:
        """Directly redistribute resources from commons to elite and loyalists.

        Uses direct stock manipulation (not the law system) to avoid the
        double-extraction artifact that occurs when two separate redistribution
        laws both draw from the same donor pool in the same cycle.

        For each resource (food, water):
          1. Compute combined deficit for elite + loyalists vs. their 30% targets.
          2. Extract that deficit proportionally from commons (capped at 80%).
          3. Distribute the extracted amount between elite and loyalists
             proportionally to their individual deficits.
          4. Enact a descriptive law entry for audit visibility.
        """
        elite_agents = [a for a in living if a.agent_id in self.elite_ids]
        loyal_agents = [a for a in living if a.agent_id in self.loyalist_ids]
        non_special = [
            a for a in living
            if a.agent_id not in self.elite_ids
            and a.agent_id not in self.loyalist_ids
        ]
        if not non_special:
            return

        n_elite = len(elite_agents)
        n_loyalist = len(loyal_agents)

        for resource in ("food", "water"):
            attr = f"{resource}_stock"
            total = sum(getattr(a, attr) for a in living)
            if total <= 0:
                continue

            elite_current = sum(getattr(a, attr) for a in elite_agents)
            loyal_current = sum(getattr(a, attr) for a in loyal_agents)

            elite_target = min(
                self.ELITE_MAX_STOCK * n_elite,
                self.ELITE_RESOURCE_SHARE * total,
            )
            loyal_target = min(
                self.LOYALIST_MAX_STOCK * n_loyalist,
                self.LOYALIST_RESOURCE_SHARE * total,
            )

            elite_deficit = max(0.0, elite_target - elite_current)
            loyal_deficit = max(0.0, loyal_target - loyal_current)
            total_deficit = elite_deficit + loyal_deficit

            if total_deficit <= 0:
                continue

            # Extract from commons — capped at 80% of their current stock
            commons_total = sum(getattr(a, attr) for a in non_special)
            if commons_total <= 0:
                continue
            actual_extract = min(total_deficit, 0.80 * commons_total)
            extract_ratio = actual_extract / commons_total

            total_extracted = 0.0
            for a in non_special:
                take = extract_ratio * getattr(a, attr)
                setattr(a, attr, max(0.0, getattr(a, attr) - take))
                total_extracted += take

            # Distribute extracted resources to elite and loyalists
            if total_extracted > 0 and (elite_deficit + loyal_deficit) > 0:
                elite_share = elite_deficit / (elite_deficit + loyal_deficit)
                elite_gets = elite_share * total_extracted
                loyal_gets = total_extracted - elite_gets

                per_elite = elite_gets / max(1, n_elite)
                for a in elite_agents:
                    setattr(a, attr, getattr(a, attr) + per_elite)

                per_loyal = loyal_gets / max(1, n_loyalist)
                for a in loyal_agents:
                    setattr(a, attr, getattr(a, attr) + per_loyal)

            elite_pct = round(100 * (elite_current + total_extracted * elite_deficit / max(0.001, total_deficit)) / max(0.001, total), 1)
            self._enact_law(
                f"REDISTRIBUTE_{resource.upper()}",
                {
                    "resource": resource,
                    "extracted": round(total_extracted, 2),
                    "elite_deficit": round(elite_deficit, 2),
                    "loyal_deficit": round(loyal_deficit, 2),
                    "elite_target_pct": 30,
                    "loyal_target_pct": 30,
                },
                cycle, duration=1,
                description=(
                    f"Oligarchy {resource}: extracted {total_extracted:.1f} "
                    f"(elite_def={elite_deficit:.1f}, loyal_def={loyal_deficit:.1f})"
                ),
            )

    def _enact_event_response(self, event_names: List[str], cycle: int) -> None:
        active_types = {l.law_type for l in self.active_laws
                        if l.is_active(cycle) and l.applies_to is None}

        if "epidemic" in event_names and "QUARANTINE_EPIDEMIC" not in active_types:
            if self._sim:
                alive = self._sim.living_agents()
                infected = [a for a in alive if a.infected and a.position]
                region = self._compute_quarantine_region(
                    infected, len(alive), self._sim.grid.rows, self._sim.grid.cols
                )
                if region:
                    epidemic_id: Optional[str] = None
                    for ev in self._sim.event_system.active_events:
                        if ev.event_type.value == "epidemic" and not ev.cancelled:
                            epidemic_id = ev.epidemic_id
                            break
                    self._enact_law("QUARANTINE_EPIDEMIC", {"region": region}, cycle,
                                    duration=None,
                                    event_id=epidemic_id,
                                    event_type="epidemic",
                                    description="Oligarchy decree: epidemic quarantine.")

        if "storm" in event_names and "MANDATORY_SHELTER" not in active_types:
            self._enact_law("MANDATORY_SHELTER", {}, cycle, duration=None,
                            event_type="storm",
                            description="Oligarchy decree: shelter mandate.")

        if "drought" in event_names and "FOOD_RATION" not in active_types:
            self._enact_law("FOOD_RATION", {"max_per_cycle": 3.0}, cycle,
                            duration=None,
                            event_type="drought",
                            description="Oligarchy decree: food rationing.")

    def receive_event_warnings(self, warnings: List["EventWarning"], cycle: int) -> None:
        for w in warnings:
            active_types = {l.law_type for l in self.active_laws if l.is_active(cycle)}
            if w.event_type.value == "storm" and "MANDATORY_SHELTER" not in active_types:
                self._enact_law("MANDATORY_SHELTER", {}, cycle, duration=None,
                                event_type="storm",
                                description="Oligarchy pre-emptive shelter order.")

    # ------------------------------------------------------------------
    # Audit info
    # ------------------------------------------------------------------

    def get_audit_info(self, cycle: int) -> dict:
        if not self._sim:
            return {}
        living = self._sim.living_agents()
        elite_agents = [a for a in living if a.agent_id in self.elite_ids]
        loyal_agents = [a for a in living if a.agent_id in self.loyalist_ids]
        common_agents = [
            a for a in living
            if a.agent_id not in self.elite_ids and a.agent_id not in self.loyalist_ids
        ]

        total_food = sum(a.food_stock for a in living) or 1.0
        total_water = sum(a.water_stock for a in living) or 1.0

        elite_food = sum(a.food_stock for a in elite_agents)
        loyal_food = sum(a.food_stock for a in loyal_agents)
        common_food = sum(a.food_stock for a in common_agents)

        elite_water = sum(a.water_stock for a in elite_agents)
        loyal_water = sum(a.water_stock for a in loyal_agents)
        common_water = sum(a.water_stock for a in common_agents)

        return {
            "elite_count": len(elite_agents),
            "loyalist_count": len(loyal_agents),
            "common_count": len(common_agents),
            "elite_food_share_pct": round(100 * elite_food / total_food, 1),
            "loyalist_food_share_pct": round(100 * loyal_food / total_food, 1),
            "common_food_share_pct": round(100 * common_food / total_food, 1),
            "elite_water_share_pct": round(100 * elite_water / total_water, 1),
            "loyalist_water_share_pct": round(100 * loyal_water / total_water, 1),
            "common_water_share_pct": round(100 * common_water / total_water, 1),
            "elite_avg_food": round(elite_food / max(1, len(elite_agents)), 2),
            "loyalist_avg_food": round(loyal_food / max(1, len(loyal_agents)), 2),
            "common_avg_food": round(common_food / max(1, len(common_agents)), 2),
            "newly_promoted_elite": self._newly_promoted_elite,
            "newly_promoted_loyalists": self._newly_promoted_loyalists,
        }

    def get_log_info(self, cycle: int) -> dict:
        if not self._sim:
            return {"decision_summary": "", "decision_trigger": "", "vote_details": {}, "state_narrative": ""}
        living = self._sim.living_agents()
        elite_agents = [a for a in living if a.agent_id in self.elite_ids]
        loyal_agents = [a for a in living if a.agent_id in self.loyalist_ids]
        common_agents = [
            a for a in living
            if a.agent_id not in self.elite_ids and a.agent_id not in self.loyalist_ids
        ]

        total_food = sum(a.food_stock for a in living) or 1.0
        elite_food = sum(a.food_stock for a in elite_agents)
        loyal_food = sum(a.food_stock for a in loyal_agents)
        common_food = sum(a.food_stock for a in common_agents)

        summary_parts = []
        if self._newly_promoted_elite:
            summary_parts.append(f"Promoted to elite: {', '.join(self._newly_promoted_elite)}")
        if self._newly_promoted_loyalists:
            summary_parts.append(f"Promoted to loyalist: {', '.join(self._newly_promoted_loyalists)}")
        if not summary_parts:
            summary_parts.append("Routine resource extraction by elite.")

        event_names = [e["type"] for e in self._sim.event_system.active_summary(cycle)]
        trigger = "elite decrees each cycle"
        if event_names:
            trigger += f"; responding to: {', '.join(event_names)}"

        narrative = (
            f"  Elite: {len(elite_agents)} agents ({self.ELITE_FRACTION * 100:.0f}% target)\n"
            f"  Loyalists: {len(loyal_agents)} agents ({self.LOYALIST_FRACTION * 100:.0f}% target)\n"
            f"  Commons: {len(common_agents)} agents\n"
            f"  Resource distribution (food):\n"
            f"    Elite:    {100 * elite_food / total_food:.1f}% (target: {self.ELITE_RESOURCE_SHARE * 100:.0f}%) — avg {elite_food / max(1, len(elite_agents)):.1f}/agent\n"
            f"    Loyalist: {100 * loyal_food / total_food:.1f}% (target: {self.LOYALIST_RESOURCE_SHARE * 100:.0f}%) — avg {loyal_food / max(1, len(loyal_agents)):.1f}/agent\n"
            f"    Common:   {100 * common_food / total_food:.1f}% — avg {common_food / max(1, len(common_agents)):.1f}/agent"
        )

        return {
            "decision_summary": "\n".join(summary_parts),
            "decision_trigger": trigger,
            "vote_details": {},
            "state_narrative": narrative,
        }

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def is_elite(self, agent: "Agent") -> bool:
        return agent.agent_id in self.elite_ids

    def is_loyalist(self, agent: "Agent") -> bool:
        return agent.agent_id in self.loyalist_ids
