"""Autocracy: one leader makes all decisions; priority order is self → loyalists → rest."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .base import Government

if TYPE_CHECKING:
    from engine.agent import Agent, Action
    from engine.events import EventWarning


class AutocracyGovernment(Government):
    """
    One leader agent issues laws with no vote.

    Priority order each cycle:
      1. Enact laws that transfer resources from the rest of the population to
         the leader — keeping the leader at a comfortable stock level.
      2. Designate 1% of the population as loyalists; enact laws that keep them
         at LOYALIST_RESOURCE_SHARE (30%) of total resources collectively.
      3. Respond to environmental events (storm / drought / epidemic) for everyone
         else using standard response laws.

    If the leader dies, the healthiest surviving agent takes over (no severe
    resource redistribution for the new leader).
    If a loyalist dies, the next-richest (by food+water) non-leader agent
    is immediately promoted and given a resource boost from commons.
    """

    name = "Autocracy"

    LOYALIST_FRACTION       = 0.01   # 1% of population
    LOYALIST_RESOURCE_SHARE = 0.30   # loyalists collectively hold 30% of total resources
    LEADER_TARGET_STOCK     = 12.0   # leader target stock (comfortable, not extreme)
    LOYALIST_MAX_STOCK      = 300.0  # allow loyalists to accumulate enough for 30% share
    T_RESOURCE_LAW          = 1      # extract every cycle for tight ratio control

    def __init__(
        self,
        t_succession: int = 50,
        t_law: int = 20,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.t_succession = t_succession
        self.t_law = t_law
        self.rng = random.Random(seed)
        self.leader_id: Optional[str] = None
        self.loyalist_ids: Set[str] = set()
        self._last_resource_law = -999
        self._newly_promoted_loyalists: List[str] = []
        self._new_leader_this_cycle: Optional[str] = None

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

        # Ensure we have a leader
        self._new_leader_this_cycle = None
        if self.leader_id is None or all(a.agent_id != self.leader_id for a in living):
            self._appoint_leader(living, cycle)

        leader = next((a for a in living if a.agent_id == self.leader_id), None)
        if not leader:
            return

        # Ensure loyalists are 1% of current population
        self._refresh_loyalists(living, leader, cycle)

        event_names = [e["type"] for e in self._sim.event_system.active_summary(cycle)]

        # Priority 1 & 2: resource extraction laws (refresh periodically)
        if cycle - self._last_resource_law >= self.T_RESOURCE_LAW:
            self._enact_resource_extraction(living, leader, cycle)
            self._last_resource_law = cycle

        # Priority 3: environmental response for everyone
        self._enact_event_response(event_names, cycle)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _appoint_leader(self, living: List["Agent"], cycle: int) -> None:
        # Leader succession: healthiest surviving agent takes over.
        # No severe resource redistribution — the new leader earns resources
        # through the normal extraction cycle on the next T_RESOURCE_LAW.
        best = max(living, key=lambda a: a.health)
        self.leader_id = best.agent_id
        self._new_leader_this_cycle = best.agent_id
        self.loyalist_ids.clear()

    def _refresh_loyalists(
        self, living: List["Agent"], leader: "Agent", cycle: int
    ) -> None:
        n = len(living)
        n_loyalists = max(1, round(n * self.LOYALIST_FRACTION))
        alive_ids = {a.agent_id for a in living}
        prev_loyalists = set(self.loyalist_ids)
        self.loyalist_ids &= alive_ids
        self.loyalist_ids.discard(self.leader_id)

        newly_added: List["Agent"] = []
        if len(self.loyalist_ids) < n_loyalists:
            # Rank by total resources (richest first) to find replacements
            candidates = sorted(
                [a for a in living
                 if a.agent_id != self.leader_id
                 and a.agent_id not in self.loyalist_ids],
                key=lambda a: a.food_stock + a.water_stock,
                reverse=True,
            )
            for a in candidates[:n_loyalists - len(self.loyalist_ids)]:
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
                if a.agent_id != self.leader_id
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
        """Directly transfer resources from commons to newly promoted loyalist."""
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

    def _enact_resource_extraction(
        self, living: List["Agent"], leader: "Agent", cycle: int
    ) -> None:
        """
        1. Give leader a comfortable LEADER_TARGET_STOCK via law system.
        2. Directly redistribute resources from commons to loyalists to reach
           LOYALIST_RESOURCE_SHARE (30%) of total — using direct stock manipulation
           to avoid double-extraction artifacts.
        """
        non_special = [
            a for a in living
            if a.agent_id != self.leader_id
            and a.agent_id not in self.loyalist_ids
        ]
        if not non_special:
            return

        agents_by_id = {a.agent_id: a for a in living}
        donor_ids = [a.agent_id for a in non_special]
        loyal_agents = [a for a in living if a.agent_id in self.loyalist_ids]
        n_loyalist = len(loyal_agents)

        # --- Leader food (comfortable level via law system) ---
        n_food = max(0.0, self.LEADER_TARGET_STOCK - leader.food_stock)
        if n_food > 0:
            food_amounts, _ = self._compute_redistribution_amounts(
                agents_by_id, [self.leader_id], donor_ids, "food", n_food
            )
            if food_amounts:
                self._enact_law(
                    "REDISTRIBUTE_FOOD",
                    {"resource": "food", "recipient_ids": [self.leader_id],
                     "donor_amounts": food_amounts, "amount_per_recipient": n_food},
                    cycle, duration=2, applies_to=list(food_amounts.keys()),
                    description=f"Autocracy decree: food→leader (target {self.LEADER_TARGET_STOCK}).",
                )

        # --- Leader water ---
        n_water = max(0.0, self.LEADER_TARGET_STOCK - leader.water_stock)
        if n_water > 0:
            water_amounts, _ = self._compute_redistribution_amounts(
                agents_by_id, [self.leader_id], donor_ids, "water", n_water
            )
            if water_amounts:
                self._enact_law(
                    "REDISTRIBUTE_WATER",
                    {"resource": "water", "recipient_ids": [self.leader_id],
                     "donor_amounts": water_amounts, "amount_per_recipient": n_water},
                    cycle, duration=2, applies_to=list(water_amounts.keys()),
                    description=f"Autocracy decree: water→leader (target {self.LEADER_TARGET_STOCK}).",
                )

        if not loyal_agents:
            return

        # --- Loyalist food and water: direct stock manipulation for clean 30% ratio ---
        total_food = sum(a.food_stock for a in living)
        total_water = sum(a.water_stock for a in living)

        for resource, total in (("food", total_food), ("water", total_water)):
            attr = f"{resource}_stock"
            if total <= 0:
                continue

            loyal_current = sum(getattr(a, attr) for a in loyal_agents)
            loyal_target = min(
                self.LOYALIST_MAX_STOCK * n_loyalist,
                self.LOYALIST_RESOURCE_SHARE * total,
            )
            loyal_deficit = max(0.0, loyal_target - loyal_current)
            if loyal_deficit <= 0:
                continue

            commons_total = sum(getattr(a, attr) for a in non_special)
            if commons_total <= 0:
                continue
            actual_extract = min(loyal_deficit, 0.80 * commons_total)
            extract_ratio = actual_extract / commons_total

            total_extracted = 0.0
            for a in non_special:
                take = extract_ratio * getattr(a, attr)
                setattr(a, attr, max(0.0, getattr(a, attr) - take))
                total_extracted += take

            per_loyal = total_extracted / max(1, n_loyalist)
            for a in loyal_agents:
                setattr(a, attr, getattr(a, attr) + per_loyal)

            self._enact_law(
                f"REDISTRIBUTE_{resource.upper()}",
                {
                    "resource": resource,
                    "extracted": round(total_extracted, 2),
                    "loyal_deficit": round(loyal_deficit, 2),
                    "loyal_target_pct": 30,
                },
                cycle, duration=1,
                description=(
                    f"Autocracy {resource}→loyalists: "
                    f"extracted {total_extracted:.1f} (deficit {loyal_deficit:.1f})"
                ),
            )

    def _enact_event_response(self, event_names: List[str], cycle: int) -> None:
        active_types = {l.law_type for l in self.active_laws if l.is_active(cycle)
                        and l.applies_to is None}

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
                                    description="Leader decree: epidemic quarantine.")

        if "storm" in event_names and "MANDATORY_SHELTER" not in active_types:
            self._enact_law("MANDATORY_SHELTER", {}, cycle, duration=None,
                            event_type="storm",
                            description="Leader decree: seek shelter immediately.")

        if "drought" in event_names and "FOOD_RATION" not in active_types:
            self._enact_law("FOOD_RATION", {"max_per_cycle": 2.5}, cycle,
                            duration=None,
                            event_type="drought",
                            description="Leader decree: emergency food rationing.")

    def receive_event_warnings(self, warnings: List["EventWarning"], cycle: int) -> None:
        if not self._sim:
            return
        for w in warnings:
            active_types = {l.law_type for l in self.active_laws if l.is_active(cycle)}
            if w.event_type.value == "storm" and "MANDATORY_SHELTER" not in active_types:
                self._enact_law("MANDATORY_SHELTER", {}, cycle, duration=None,
                                event_type="storm",
                                description="Leader pre-emptive shelter order.")
            elif w.event_type.value == "epidemic" and "QUARANTINE_EPIDEMIC" not in active_types:
                if w.region:
                    # Pre-emptive: no epidemic event active yet, so no epidemic_id to link
                    self._enact_law("QUARANTINE_EPIDEMIC", {"region": w.region}, cycle,
                                    duration=None,
                                    event_type="epidemic",
                                    description=f"Leader pre-emptive quarantine {w.region}.")

    # ------------------------------------------------------------------
    # Audit info
    # ------------------------------------------------------------------

    def get_audit_info(self, cycle: int) -> dict:
        if not self._sim:
            return {}
        living = self._sim.living_agents()
        leader = next((a for a in living if a.agent_id == self.leader_id), None)
        loyal_agents = [a for a in living if a.agent_id in self.loyalist_ids]
        common_agents = [
            a for a in living
            if a.agent_id != self.leader_id and a.agent_id not in self.loyalist_ids
        ]

        total_food = sum(a.food_stock for a in living) or 1.0
        total_water = sum(a.water_stock for a in living) or 1.0

        leader_food = leader.food_stock if leader else 0.0
        leader_water = leader.water_stock if leader else 0.0
        loyal_food = sum(a.food_stock for a in loyal_agents)
        loyal_water = sum(a.water_stock for a in loyal_agents)
        common_food = sum(a.food_stock for a in common_agents)
        common_water = sum(a.water_stock for a in common_agents)

        return {
            "leader_id": self.leader_id,
            "leader_food": round(leader_food, 2),
            "leader_water": round(leader_water, 2),
            "leader_food_share_pct": round(100 * leader_food / total_food, 1),
            "loyalist_count": len(loyal_agents),
            "common_count": len(common_agents),
            "loyalist_food_share_pct": round(100 * loyal_food / total_food, 1),
            "loyalist_water_share_pct": round(100 * loyal_water / total_water, 1),
            "common_food_share_pct": round(100 * common_food / total_food, 1),
            "loyalist_avg_food": round(loyal_food / max(1, len(loyal_agents)), 2),
            "common_avg_food": round(common_food / max(1, len(common_agents)), 2),
            "new_leader_this_cycle": self._new_leader_this_cycle,
            "newly_promoted_loyalists": self._newly_promoted_loyalists,
        }

    def get_log_info(self, cycle: int) -> dict:
        if not self._sim:
            return {"decision_summary": "", "decision_trigger": "", "vote_details": {}, "state_narrative": ""}
        living = self._sim.living_agents()
        leader = next((a for a in living if a.agent_id == self.leader_id), None)
        loyal_agents = [a for a in living if a.agent_id in self.loyalist_ids]
        common_agents = [
            a for a in living
            if a.agent_id != self.leader_id and a.agent_id not in self.loyalist_ids
        ]

        total_food = sum(a.food_stock for a in living) or 1.0
        total_water = sum(a.water_stock for a in living) or 1.0
        leader_food = leader.food_stock if leader else 0.0
        loyal_food = sum(a.food_stock for a in loyal_agents)
        common_food = sum(a.food_stock for a in common_agents)

        summary_parts = []
        if self._new_leader_this_cycle:
            summary_parts.append(f"New leader appointed: {self._new_leader_this_cycle} (healthiest survivor)")
        if self._newly_promoted_loyalists:
            summary_parts.append(f"Promoted {len(self._newly_promoted_loyalists)} loyalist(s): {', '.join(self._newly_promoted_loyalists)}")

        event_names = [e["type"] for e in self._sim.event_system.active_summary(cycle)]
        trigger_parts = ["leader decrees each cycle"]
        if event_names:
            trigger_parts.append(f"responding to: {', '.join(event_names)}")

        leader_hp = f"{leader.health:.2f}" if leader else "N/A"
        narrative = (
            f"  Leader: {self.leader_id} (health={leader_hp}, food={leader_food:.1f})\n"
            f"  Loyalists: {len(loyal_agents)} agents | Commons: {len(common_agents)} agents\n"
            f"  Resource distribution:\n"
            f"    Leader food share:   {100 * leader_food / total_food:.1f}%\n"
            f"    Loyalist food share: {100 * loyal_food / total_food:.1f}% (target: {self.LOYALIST_RESOURCE_SHARE * 100:.0f}%)\n"
            f"    Common food share:   {100 * common_food / total_food:.1f}%\n"
            f"    Avg food — loyalists: {loyal_food / max(1, len(loyal_agents)):.1f}  commons: {common_food / max(1, len(common_agents)):.1f}"
        )

        return {
            "decision_summary": "\n".join(summary_parts) if summary_parts else "Routine resource extraction cycle.",
            "decision_trigger": "; ".join(trigger_parts),
            "vote_details": {},
            "state_narrative": narrative,
        }

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def is_leader(self, agent: "Agent") -> bool:
        return agent.agent_id == self.leader_id

    def is_loyalist(self, agent: "Agent") -> bool:
        return agent.agent_id in self.loyalist_ids
