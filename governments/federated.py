"""Federated government: grid divided into agent-population-proportional regions."""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .base import Government

if TYPE_CHECKING:
    from engine.agent import Agent, Action
    from engine.events import EventWarning


CROSS_REGION_HEALTH_PENALTY = 0.01  # health lost per step into another region


class Region:
    """A sub-grid region with its own local democratic government."""

    def __init__(self, region_id: str, r_min: int, c_min: int, r_max: int, c_max: int):
        self.region_id = region_id
        self.bounds = (r_min, c_min, r_max, c_max)
        self.votes: Dict[str, str] = {}

    def contains(self, r: int, c: int) -> bool:
        r_min, c_min, r_max, c_max = self.bounds
        return r_min <= r <= r_max and c_min <= c <= c_max


class FederatedGovernment(Government):
    """
    The grid is divided into up to N_REGIONS regions (default up to 10) such
    that the initial agent population is distributed as evenly as possible
    across regions.  Regions are rectangular strips.

    Each region votes independently every T_VOTE cycles.
    Agents who move into a region that already contains other agents pay a
    small health penalty (cross-region movement penalty).
    """

    name = "Federated"

    N_REGIONS_MAX = 10      # maximum number of regions

    def __init__(
        self,
        n_regions: int = 10,
        t_vote: int = 5,
        t_law: int = 20,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_regions = max(2, min(n_regions, self.N_REGIONS_MAX))
        self.t_vote = t_vote
        self.t_law = t_law
        self.rng = random.Random(seed)
        self.regions: List[Region] = []
        self._regions_initialized = False
        self._agent_region_map: Dict[str, str] = {}  # agent_id → home region_id
        # Per-region vote audit storage — populated after each regional election
        self._last_region_votes: Dict[str, dict] = {}  # region_id → audit record
        self._last_election_cycle: int = -1
        # Trade audit — updated each time _evaluate_trade_opportunities runs
        self._last_trades: List[dict] = []

    # ------------------------------------------------------------------
    # Initialization — partition grid to equalise initial agent population
    # ------------------------------------------------------------------

    def _init_regions(self) -> None:
        if not self._sim or self._regions_initialized:
            return
        grid = self._sim.grid
        agents = self._sim.living_agents()

        # Divide the grid into horizontal strips.
        # Assign strips so each has roughly equal numbers of agents.
        # Simple approach: sort agents by row, divide into N_REGIONS buckets.
        n = len(agents)
        n_regions = min(self.n_regions, max(1, n // 10 + 1))  # ensure ≥10 agents/region

        rows = grid.rows
        row_agent_counts = [0] * rows
        for agent in agents:
            if agent.position:
                row_agent_counts[agent.position[0]] += 1

        target_per_region = max(1, n // n_regions)
        region_row_starts = [0]
        cumulative = 0
        for r in range(rows):
            cumulative += row_agent_counts[r]
            if cumulative >= target_per_region and len(region_row_starts) < n_regions:
                region_row_starts.append(r + 1)
                cumulative = 0

        # Build regions from row boundaries
        self.regions = []
        for i, start_row in enumerate(region_row_starts):
            end_row = (
                region_row_starts[i + 1] - 1
                if i + 1 < len(region_row_starts)
                else rows - 1
            )
            region = Region(
                region_id=f"R{i}",
                r_min=start_row, c_min=0,
                r_max=end_row, c_max=grid.cols - 1,
            )
            self.regions.append(region)

        # Record each agent's home region
        for agent in agents:
            if agent.position:
                for region in self.regions:
                    if region.contains(*agent.position):
                        self._agent_region_map[agent.agent_id] = region.region_id
                        break

        self._regions_initialized = True

    def _agent_home_region(self, agent: "Agent") -> Optional[str]:
        return self._agent_region_map.get(agent.agent_id)

    def _region_of(self, r: int, c: int) -> Optional[Region]:
        for region in self.regions:
            if region.contains(r, c):
                return region
        return None

    # ------------------------------------------------------------------
    # Per-cycle tick
    # ------------------------------------------------------------------

    def tick(self, cycle: int) -> None:
        self._init_regions()
        self._expire_laws(cycle)

        if cycle % self.t_vote == 0:
            self._regional_elections(cycle)

        # Federation-mandated inter-region resource trades (not voted on)
        if cycle % 5 == 0:
            self._evaluate_trade_opportunities(cycle)

    # ------------------------------------------------------------------
    # Regional elections
    # ------------------------------------------------------------------

    def _regional_elections(self, cycle: int) -> None:
        if not self._sim:
            return
        event_names = [e["type"] for e in self._sim.event_system.active_summary(cycle)]
        self._last_region_votes = {}
        self._last_election_cycle = cycle

        for region in self.regions:
            regional_agents = [
                a for a in self._sim.living_agents()
                if a.position and region.contains(*a.position)
            ]
            if not regional_agents:
                self._last_region_votes[region.region_id] = {
                    "n_agents": 0,
                    "tally": {},
                    "winner": None,
                    "winner_pct": 0.0,
                    "decision_trigger": "no agents in region",
                    "median_health": 0.0,
                    "median_food": 0.0,
                    "median_water": 0.0,
                    "infected_count": 0,
                    "infected_pct": 0.0,
                }
                continue

            # Auto-vote all agents in this region
            for agent in regional_agents:
                if agent.agent_id not in region.votes:
                    region.votes[agent.agent_id] = self._region_heuristic(
                        region, event_names, cycle, agent
                    )

            counter = Counter(region.votes.values())
            region.votes = {}

            # Compute per-region agent stats for audit log
            healths = [a.health for a in regional_agents]
            foods = [a.food_stock for a in regional_agents]
            waters = [a.water_stock for a in regional_agents]
            infected_count = sum(1 for a in regional_agents if a.infected)
            n = len(regional_agents)

            median_health = statistics.median(healths)
            median_food = statistics.median(foods)
            median_water = statistics.median(waters)

            winner = counter.most_common(1)[0][0] if counter else "NO_ACTION"
            winner_votes = counter.get(winner, 0)
            total_votes = sum(counter.values()) or 1
            winner_pct = 100.0 * winner_votes / total_votes

            # Describe what event conditions drove this vote
            trigger = self._describe_regional_trigger(
                event_names, regional_agents, median_food, median_water
            )

            self._last_region_votes[region.region_id] = {
                "n_agents": n,
                "tally": dict(counter),
                "winner": winner,
                "winner_pct": winner_pct,
                "decision_trigger": trigger,
                "median_health": median_health,
                "median_food": median_food,
                "median_water": median_water,
                "infected_count": infected_count,
                "infected_pct": 100.0 * infected_count / max(1, n),
            }

            if counter:
                if winner != "NO_ACTION":
                    self._enact_regional_law(region, winner, cycle,
                                             regional_agents, event_names)

    def _region_heuristic(
        self,
        region: Region,
        event_names: List[str],
        cycle: int,
        agent: "Agent",
    ) -> str:
        """Derive a vote option for an agent based on region conditions."""
        if agent.infected:
            return "QUARANTINE_EPIDEMIC"
        if "storm" in event_names:
            return "MANDATORY_SHELTER"
        if "epidemic" in event_names:
            return "QUARANTINE_EPIDEMIC"
        if "drought" in event_names or agent.food_stock < 3.0:
            return "REDISTRIBUTE_FOOD"
        if agent.water_stock < 3.0:
            return "REDISTRIBUTE_WATER"
        return "NO_ACTION"

    def _evaluate_trade_opportunities(self, cycle: int) -> None:
        """Placeholder — inter-region federation trades not yet implemented."""

    def _describe_regional_trigger(
        self,
        event_names: List[str],
        regional_agents: List["Agent"],
        median_food: float,
        median_water: float,
    ) -> str:
        """Build a human-readable description of why the region voted the way it did."""
        parts = []
        if "epidemic" in event_names:
            parts.append("epidemic active")
        if "storm" in event_names:
            parts.append("storm active")
        if "drought" in event_names:
            parts.append("drought active")
        if "toxic_spill" in event_names:
            parts.append("toxic spill active")
        infected_count = sum(1 for a in regional_agents if a.infected)
        if infected_count > 0:
            parts.append(f"{infected_count} infected agent(s)")
        if median_food < 3.0:
            parts.append(f"low food (median={median_food:.1f})")
        if median_water < 3.0:
            parts.append(f"low water (median={median_water:.1f})")
        if not parts:
            parts.append(f"scheduled vote (every {self.t_vote} cycles)")
        return "; ".join(parts)

    def _enact_regional_law(
        self,
        region: Region,
        law_type: str,
        cycle: int,
        regional_agents: List["Agent"],
        event_names: List[str],
    ) -> None:
        n = len(regional_agents)
        agent_ids = [a.agent_id for a in regional_agents]

        if law_type == "FOOD_RATION":
            self._enact_law("FOOD_RATION", {"max_per_cycle": 3.5}, cycle,
                            duration=None, applies_to=agent_ids,
                            event_type="drought",
                            description=f"Region {region.region_id}: food ration.")

        elif law_type == "WATER_RATION":
            self._enact_law("LIMIT_WATER_CONSUMPTION", {"max_per_cycle": 3.5}, cycle,
                            duration=self.t_law, applies_to=agent_ids,
                            description=f"Region {region.region_id}: water ration.")

        elif law_type == "QUARANTINE_EPIDEMIC":
            infected = [a for a in regional_agents if a.infected and a.position]
            if infected:
                rows = [a.position[0] for a in infected]
                cols = [a.position[1] for a in infected]
                r0 = sum(rows) // len(rows)
                c0 = sum(cols) // len(cols)
                spread = max(max(rows) - min(rows), max(cols) - min(cols)) // 2
                frac_radius = int(len(infected) / max(1, n) * 8)
                radius = max(2, spread + 1, frac_radius)
                region_bounds = region.bounds
                sub_region = (
                    max(region_bounds[0], r0 - radius),
                    max(region_bounds[1], c0 - radius),
                    min(region_bounds[2], r0 + radius),
                    min(region_bounds[3], c0 + radius),
                )
                epidemic_id: Optional[str] = None
                if self._sim:
                    for ev in self._sim.event_system.active_events:
                        if ev.event_type.value == "epidemic" and not ev.cancelled:
                            epidemic_id = ev.epidemic_id
                            break
                self._enact_law("QUARANTINE_EPIDEMIC", {"region": sub_region}, cycle,
                                duration=None, applies_to=agent_ids,
                                event_id=epidemic_id,
                                event_type="epidemic",
                                description=f"Region {region.region_id}: epidemic quarantine.")

        elif law_type == "MANDATORY_SHELTER":
            self._enact_law("MANDATORY_SHELTER_REGION", {"region": region.bounds}, cycle,
                            duration=None, applies_to=agent_ids,
                            event_type="storm",
                            description=f"Region {region.region_id}: shelter mandate.")

        elif law_type in ("REDISTRIBUTE_FOOD", "REDISTRIBUTE_WATER"):
            resource = "food" if law_type == "REDISTRIBUTE_FOOD" else "water"
            attr = f"{resource}_stock"
            sorted_reg = sorted(regional_agents, key=lambda a: getattr(a, attr, 0.0))
            k_count = max(1, n * 30 // 100)
            recipient_ids = [a.agent_id for a in sorted_reg[:k_count]]
            donor_ids = [a.agent_id for a in sorted_reg[n * 70 // 100:]]
            if donor_ids and recipient_ids:
                agents_by_id = {a.agent_id: a for a in regional_agents}
                median_stock = getattr(sorted_reg[n // 2], attr, 5.0) if n > 1 else 5.0
                recip_agents = [agents_by_id[rid] for rid in recipient_ids if rid in agents_by_id]
                avg_need = sum(
                    max(0.0, median_stock - getattr(a, attr, 0.0))
                    for a in recip_agents
                ) / max(1, len(recip_agents))
                n_amount = max(1.0, avg_need)
                donor_amounts, actual_n = self._compute_redistribution_amounts(
                    agents_by_id, recipient_ids, donor_ids, resource, n_amount
                )
                if donor_amounts:
                    self._enact_law(law_type, {
                        "resource": resource, "recipient_ids": recipient_ids,
                        "donor_amounts": donor_amounts, "amount_per_recipient": actual_n,
                    }, cycle, duration=self.t_law, applies_to=list(donor_amounts.keys()),
                        description=f"Region {region.region_id}: redistribute {resource}.")

    # ------------------------------------------------------------------
    # Movement — apply cross-region penalty
    # ------------------------------------------------------------------

    def can_move(self, agent: "Agent", target_r: int, target_c: int, cycle: int) -> bool:
        # Apply base law checks first
        if not super().can_move(agent, target_r, target_c, cycle):
            return False
        return True

    def apply_cross_region_penalty(
        self, agent: "Agent", target_r: int, target_c: int
    ) -> None:
        """
        Called by the simulation after a successful move into a target cell.
        Deducts a health penalty if the target cell belongs to a different region
        that already has other agents present.
        """
        if not self._sim:
            return
        target_region = self._region_of(target_r, target_c)
        if target_region is None:
            return
        home_region_id = self._agent_home_region(agent)
        if home_region_id == target_region.region_id:
            return  # same region — no penalty
        # Penalty only if other agents are present in the target region
        others = [
            a for a in self._sim.living_agents()
            if a.agent_id != agent.agent_id
            and a.position
            and target_region.contains(*a.position)
        ]
        if others:
            agent.health = max(0.0, agent.health - CROSS_REGION_HEALTH_PENALTY)

    # ------------------------------------------------------------------
    # Vote reception
    # ------------------------------------------------------------------

    def receive_vote(self, agent: "Agent", option: Any, cycle: int) -> None:
        if not agent.position:
            return
        for region in self.regions:
            if region.contains(*agent.position):
                if isinstance(option, str):
                    region.votes[agent.agent_id] = option
                break

    def get_audit_info(self, cycle: int) -> dict:
        from collections import Counter as _Counter
        region_summaries = []
        for r in self.regions:
            tally = _Counter(r.votes.values())
            winner = tally.most_common(1)[0][0] if tally else "—"
            tally_str = " ".join(f"{k}:{v}" for k, v in tally.most_common())
            region_summaries.append(
                f"{r.region_id}[win={winner} tally={tally_str}]"
            )
        return {
            "n_regions": len(self.regions),
            "regions_initialized": self._regions_initialized,
            "regional_votes": " | ".join(region_summaries),
        }

    def get_log_info(self, cycle: int) -> dict:
        # Only emit the detailed election summary on the cycle when elections ran
        # (or the cycle immediately after, matching how democracy handles it)
        election_just_ran = (
            self._last_election_cycle == cycle
            or self._last_election_cycle == cycle - 1
        )

        decision_summary = ""
        vote_details: Dict[str, str] = {}

        if election_just_ran and self._last_region_votes:
            region_lines = ["Regional elections concluded."]
            for region_id, rec in self._last_region_votes.items():
                n = rec["n_agents"]
                if n == 0:
                    region_lines.append(f"    {region_id}: no agents present")
                    continue

                winner = rec["winner"] or "NO_ACTION"
                winner_pct = rec["winner_pct"]
                tally = rec["tally"]
                tally_parts = [
                    f"{opt}={cnt}"
                    for opt, cnt in sorted(tally.items(), key=lambda x: -x[1])
                ]
                trigger = rec["decision_trigger"]
                mh = rec["median_health"]
                mf = rec["median_food"]
                mw = rec["median_water"]
                inf_count = rec["infected_count"]
                inf_pct = rec["infected_pct"]

                region_lines.append(
                    f"    {region_id}: {n} agents | "
                    f"Winner: {winner} ({winner_pct:.1f}%) | "
                    f"Tally: {', '.join(tally_parts)}"
                )
                region_lines.append(
                    f"      Trigger: {trigger}"
                )
                region_lines.append(
                    f"      Stats: health={mh:.3f} food={mf:.1f} water={mw:.1f} "
                    f"infected={inf_count} ({inf_pct:.1f}%)"
                )

                # vote_details for audit table (one entry per region)
                vote_details[region_id] = (
                    f"winner={winner} ({winner_pct:.1f}%)  "
                    + "  ".join(tally_parts)
                    + f"  [trigger: {trigger}]"
                )

            decision_summary = "\n".join(region_lines)

        narrative_parts = [
            f"  Regions: {len(self.regions)} (initialized: {self._regions_initialized})",
            f"  Vote cycle: every {self.t_vote} cycles per region",
            f"  Cross-region movement penalty: {CROSS_REGION_HEALTH_PENALTY} health/step",
        ]

        if self._last_trades:
            narrative_parts.append("  Recent federation-mandated trades:")
            for trade in self._last_trades:
                narrative_parts.append(
                    f"    {trade['donor']} -> {trade['recipient']} | "
                    f"{trade['resource']}: {trade['amount']:.2f} units "
                    f"({trade['n_donors']} donors, {trade['n_recipients']} recipients)"
                )

        narrative = "\n".join(narrative_parts)

        return {
            "decision_summary": decision_summary,
            "decision_trigger": f"regional vote every {self.t_vote} cycles",
            "vote_details": vote_details,
            "state_narrative": narrative,
        }

    # ------------------------------------------------------------------
    # Inter-region trade (federation-mandated, not voted on by agents)
    # ------------------------------------------------------------------

    def _evaluate_trade_opportunities(self, cycle: int) -> None:
        """
        Evaluate and execute inter-region resource trades mandated by the federation.

        Steps:
          1. Compute per-region median and min for food, water (and medicine if present).
          2. Compute global medians across all living agents.
          3. Identify deficit regions: median < global_median * 0.7 AND median < 2.0.
          4. Identify surplus regions: median > 4.0 with positive offer capacity.
          5. Pair each deficit region with the best available surplus donor (greedy,
             highest capacity first).
          6. Execute transfers: each donor gives min(stock - 2.0, equal_share);
             total collected is distributed equally to all recipient agents.
          7. Record executed trades in self._last_trades.
          8. Register each trade as a REGION_TRADE law for the audit ledger.

        Constraints enforced:
          - No donor agent's stock drops below 2.0.
          - Donor region median must exceed 4.0 to be eligible.
          - Deficit trigger: recipient median < global_median * 0.7 AND < 2.0.
          - Trade cap: at most 30% of donor region's total stock for that resource.
          - Hard floor: no agent stock ever goes below 0 (max(0, ...) on deduction).
        """
        if not self._sim:
            return

        living = self._sim.living_agents()
        if not living:
            return

        # Map region_id -> list of living agents currently within that region.
        region_agents: Dict[str, List] = {r.region_id: [] for r in self.regions}
        for agent in living:
            if agent.position:
                for region in self.regions:
                    if region.contains(*agent.position):
                        region_agents[region.region_id].append(agent)
                        break

        # Compute global medians across all living agents.
        all_food = [a.food_stock for a in living]
        all_water = [a.water_stock for a in living]
        all_medicine = [
            a.medicine_stock for a in living if hasattr(a, "medicine_stock")
        ]

        if not all_food:
            return

        global_medians: Dict[str, float] = {
            "food": statistics.median(all_food),
            "water": statistics.median(all_water),
        }
        if all_medicine:
            global_medians["medicine"] = statistics.median(all_medicine)

        # Attribute name for each resource type.
        resource_attrs: Dict[str, str] = {
            "food": "food_stock",
            "water": "water_stock",
            "medicine": "medicine_stock",
        }

        # Compute per-region statistics for each resource.
        region_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        for region in self.regions:
            agents = region_agents[region.region_id]
            rstat: Dict[str, Dict[str, float]] = {}
            for resource, attr in resource_attrs.items():
                if resource not in global_medians:
                    continue
                stocks = [getattr(a, attr, 0.0) for a in agents if hasattr(a, attr)]
                if not stocks:
                    continue
                rstat[resource] = {
                    "median": statistics.median(stocks),
                    "min": min(stocks),
                    "total": sum(stocks),
                }
            region_stats[region.region_id] = rstat

        new_trades: List[dict] = []

        for resource, global_med in global_medians.items():
            attr = resource_attrs[resource]

            # Deficit regions: median below 70% of global AND below absolute floor of 2.0.
            deficit_regions: List[Tuple[float, str, Region]] = []
            for region in self.regions:
                rstat = region_stats.get(region.region_id, {})
                if resource not in rstat:
                    continue
                reg_med = rstat[resource]["median"]
                if reg_med < global_med * 0.7 and reg_med < 3.0:
                    deficit_regions.append((reg_med, region.region_id, region))
            deficit_regions.sort(key=lambda x: x[0])  # most critical first

            if not deficit_regions:
                continue

            # Surplus regions with computed offer capacity.
            surplus_regions: List[Tuple[float, str, Region]] = []
            for region in self.regions:
                rstat = region_stats.get(region.region_id, {})
                if resource not in rstat:
                    continue
                reg_med = rstat[resource]["median"]
                if reg_med <= 4.0:
                    continue  # must exceed donor eligibility threshold

                agents = region_agents[region.region_id]
                total_stock = rstat[resource]["total"]
                # Per-agent contribution = what they can spare above the 2.0 floor.
                raw_capacity = sum(
                    max(0.0, getattr(a, attr, 0.0) - 2.0)
                    for a in agents
                )
                # Cap donation at 30% of total regional stock for this resource.
                offer_cap = min(raw_capacity, total_stock * 0.30)
                if offer_cap > 0.0:
                    surplus_regions.append((offer_cap, region.region_id, region))
            surplus_regions.sort(key=lambda x: -x[0])  # highest capacity first

            if not surplus_regions:
                continue

            # Greedily pair each deficit region to the best available donor.
            used_donors: set = set()
            for _reg_med, deficit_rid, deficit_region in deficit_regions:
                donor_entry: Optional[Tuple[float, str, Region]] = None
                for cap, donor_rid, donor_region in surplus_regions:
                    if donor_rid != deficit_rid and donor_rid not in used_donors:
                        donor_entry = (cap, donor_rid, donor_region)
                        break
                if donor_entry is None:
                    continue

                offer_cap, donor_rid, donor_region = donor_entry
                donor_agents = region_agents[donor_rid]
                recipient_agents = region_agents[deficit_rid]

                if not donor_agents or not recipient_agents:
                    continue

                # Trade volume = min(what donors can offer, what recipients need).
                deficit_med = region_stats[deficit_rid][resource]["median"]
                need = (global_med - deficit_med) * len(recipient_agents)
                trade_amount = min(offer_cap, need)
                if trade_amount <= 0.0:
                    continue

                # Collect from donors proportionally; each floored at 2.0.
                per_agent_target = trade_amount / max(1, len(donor_agents))
                actual_contributions: Dict[str, float] = {}
                total_collected = 0.0
                for agent in donor_agents:
                    current = getattr(agent, attr, 0.0)
                    contribution = min(max(0.0, current - 2.0), per_agent_target)
                    if contribution > 0.0:
                        actual_contributions[agent.agent_id] = contribution
                        total_collected += contribution

                if total_collected <= 0.0:
                    continue

                # Apply deductions to donors (hard floor at 0.0 as safety net).
                for agent in donor_agents:
                    contrib = actual_contributions.get(agent.agent_id, 0.0)
                    if contrib > 0.0:
                        current = getattr(agent, attr, 0.0)
                        setattr(agent, attr, max(0.0, current - contrib))

                # Distribute collected amount equally to all recipient agents.
                per_recipient = total_collected / len(recipient_agents)
                for agent in recipient_agents:
                    current = getattr(agent, attr, 0.0)
                    setattr(agent, attr, current + per_recipient)

                trade_record = {
                    "donor": donor_rid,
                    "recipient": deficit_rid,
                    "resource": resource,
                    "amount": total_collected,
                    "n_donors": len(actual_contributions),
                    "n_recipients": len(recipient_agents),
                    "cycle": cycle,
                }
                new_trades.append(trade_record)

                # Register in the law ledger for audit (duration=1, already executed).
                self._trade_law(
                    cycle=cycle,
                    donor_region_id=donor_rid,
                    recipient_region_id=deficit_rid,
                    resource=resource,
                    amount=total_collected,
                )

                used_donors.add(donor_rid)

        self._last_trades = new_trades

    def _trade_law(
        self,
        cycle: int,
        donor_region_id: str,
        recipient_region_id: str,
        resource: str,
        amount: float,
    ) -> None:
        """
        Register a completed federation-mandated trade in the law ledger for audit.

        Type: REGION_TRADE
        Duration: 1 cycle (the transfer has already been executed this cycle).
        Not voted on — the federation government issues this directly.
        """
        self._enact_law(
            "REGION_TRADE",
            {
                "donor_region": donor_region_id,
                "recipient_region": recipient_region_id,
                "resource": resource,
                "amount": amount,
            },
            cycle,
            duration=1,
            description=(
                f"Federation trade: {donor_region_id} -> {recipient_region_id} | "
                f"{resource}: {amount:.2f} units (cycle {cycle})"
            ),
        )
