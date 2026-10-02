"""Direct democracy: all agents vote on proposals; majority wins."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import Government, Law

if TYPE_CHECKING:
    from engine.agent import Agent, Action
    from engine.events import EventWarning


PROPOSAL_OPTIONS = [
    # Food / water rationing
    "FOOD_RATION",
    "WATER_RATION",
    # Epidemic response
    "QUARANTINE_EPIDEMIC",
    "EPIDEMIC_RESPONSE",
    # Storm response
    "MANDATORY_SHELTER",
    # Resource management
    "RESOURCE_RESERVE",
    "REDISTRIBUTE_FOOD",
    "REDISTRIBUTE_WATER",
    "REDISTRIBUTE_MEDICINE",
    # Mobility
    "SPREAD",
    "FLEE_TOXIC",
    # Baseline
    "NO_ACTION",
]


class DemocracyGovernment(Government):
    """
    Every T_vote cycles, agents vote on a set of proposals.
    Majority wins and is enacted as a law for T_law cycles.
    """

    name = "Direct Democracy"

    def __init__(
        self,
        t_vote: int = 5,
        t_law: int = 20,
        num_options: int = 3,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.t_vote = t_vote
        self.t_law = t_law
        self.num_options = num_options
        self.rng = random.Random(seed)

        self._votes: Dict[str, str] = {}   # agent_id → option
        self._current_options: List[str] = []
        self._election_active: bool = False
        self._election_start_cycle: int = -1
        # Audit tracking
        self._last_vote_tallies: Dict[str, int] = {}
        self._last_vote_winner: str = ""
        self._last_vote_cycle: int = -1
        self._last_vote_breakdown: Dict[str, Dict[str, int]] = {}  # cohort -> {option: count}
        self._last_decision_trigger: str = ""

    def tick(self, cycle: int) -> None:
        self._expire_laws(cycle)

        # Start new election
        if cycle % self.t_vote == 0:
            self._start_election(cycle)

        # Close election (1 cycle after opening — agents vote in the same cycle)
        if self._election_active and cycle > self._election_start_cycle:
            self._close_election(cycle)

    def _start_election(self, cycle: int) -> None:
        # Generate options responsive to active events
        options = self._context_aware_options(cycle)
        self._current_options = options
        self._votes = {}
        self._election_active = True
        self._election_start_cycle = cycle

    def _context_aware_options(self, cycle: int) -> List[str]:
        if not self._sim:
            return ["NO_ACTION"] * self.num_options

        event_names = [e["type"] for e in self._sim.event_system.active_summary(cycle)]
        alive = self._sim.living_agents()
        options = []

        if "epidemic" in event_names:
            # Quarantine-vs-spread heuristic: if many agents are infected,
            # quarantine is more effective than spreading (spreading won't help
            # when already widespread). If few infected and agents are clustered,
            # spreading is worth offering.
            infected_count = sum(1 for a in alive if a.infected)
            infected_frac = infected_count / max(1, len(alive))
            options.append("QUARANTINE_EPIDEMIC")
            options.append("EPIDEMIC_RESPONSE")
            if infected_frac < 0.25:
                options.append("SPREAD")  # few infected → non-infected can flee
        if "storm" in event_names:
            options.append("MANDATORY_SHELTER")
        if "drought" in event_names:
            options.append("FOOD_RATION")
        if "toxic_spill" in event_names:
            options.append("FLEE_TOXIC")
        if alive:
            import statistics
            agent_foods = sorted([a.food_stock for a in alive])
            agent_waters = sorted([a.water_stock for a in alive])
            agent_meds = sorted([a.medicine_stock for a in alive])
            median_food = statistics.median(agent_foods)
            median_water = statistics.median(agent_waters)
            median_med = statistics.median(agent_meds)
            if median_food < 3.0:
                options.append("REDISTRIBUTE_FOOD")
                options.append("FOOD_RATION")
            if median_water < 3.0:
                options.append("REDISTRIBUTE_WATER")
                options.append("WATER_RATION")
            if median_med < 2.0:
                options.append("REDISTRIBUTE_MEDICINE")

        options.append("NO_ACTION")
        # Pad/trim to num_options
        while len(options) < self.num_options:
            options.append(self.rng.choice(PROPOSAL_OPTIONS))
        return list(dict.fromkeys(options))[:self.num_options]  # deduplicate

    def _close_election(self, cycle: int) -> None:
        # All surviving agents who have not voted yet cast a vote now
        if self._sim:
            event_names = [e["type"] for e in self._sim.event_system.active_summary(cycle)]
            for agent in self._sim.living_agents():
                if agent.agent_id not in self._votes and self._current_options:
                    chosen = self._auto_vote(agent, event_names)
                    if chosen:
                        self._votes[agent.agent_id] = chosen

        if not self._votes:
            self._election_active = False
            return
        counter = Counter(self._votes.values())
        winner = counter.most_common(1)[0][0]
        self._last_vote_tallies = dict(counter)
        self._last_vote_winner = winner
        self._last_vote_cycle = cycle

        # Build vote breakdown by cohort for log info
        self._last_vote_breakdown = self._compute_vote_breakdown()
        self._last_decision_trigger = self._describe_trigger(event_names)

        self._election_active = False
        self._votes = {}
        if winner != "NO_ACTION":
            already_active = any(
                l.law_type == winner and l.is_active(cycle)
                for l in self.active_laws
            )
            if not already_active:
                self._enact_winning(winner, cycle)

    def _compute_vote_breakdown(self) -> Dict[str, Dict[str, int]]:
        if not self._sim or not self._votes:
            return {}
        breakdown: Dict[str, Dict[str, int]] = {}
        agents_by_id = {a.agent_id: a for a in self._sim.agents if a.alive}
        for agent_id, vote in self._votes.items():
            agent = agents_by_id.get(agent_id)
            if not agent:
                continue
            if agent.infected:
                cohort = "infected"
            elif agent.health < 0.4:
                cohort = "critical_health"
            elif agent.food_stock < 2.0 or agent.water_stock < 2.0:
                cohort = "resource_scarce"
            else:
                cohort = "healthy"
            if cohort not in breakdown:
                breakdown[cohort] = {}
            breakdown[cohort][vote] = breakdown[cohort].get(vote, 0) + 1
        return breakdown

    def _describe_trigger(self, event_names: List[str]) -> str:
        parts = []
        if "epidemic" in event_names:
            parts.append("epidemic active")
        if "storm" in event_names:
            parts.append("storm active")
        if "drought" in event_names:
            parts.append("drought active")
        if "toxic_spill" in event_names:
            parts.append("toxic spill active")
        if not parts:
            parts.append("scheduled vote (every {} cycles)".format(self.t_vote))
        return "; ".join(parts)

    def _auto_vote(self, agent: "Agent", event_names: List[str]) -> Optional[str]:
        """Derive a vote for an agent that did not cast one explicitly."""
        from agents.citizen import CitizenAgent
        opts = self._current_options
        if not opts:
            return None
        if isinstance(agent, CitizenAgent):
            return agent._choose_vote(opts, event_names)
        # Fallback heuristic for non-citizen agents
        if "epidemic" in event_names:
            for o in ("QUARANTINE_EPIDEMIC", "EPIDEMIC_RESPONSE"):
                if o in opts:
                    return o
        if "storm" in event_names and "MANDATORY_SHELTER" in opts:
            return "MANDATORY_SHELTER"
        if "drought" in event_names and "FOOD_RATION" in opts:
            return "FOOD_RATION"
        return opts[-1]  # last option (NO_ACTION or similar)

    def _enact_winning(self, winner: str, cycle: int) -> None:
        if not self._sim:
            return
        sim = self._sim
        alive = sim.living_agents()
        n = len(alive)

        if winner == "FOOD_RATION":
            self._enact_law("FOOD_RATION", {"max_per_cycle": 3.0}, cycle,
                            duration=None,
                            event_type="drought",
                            description="Majority voted: ration food to 3 units/cycle.")

        elif winner == "WATER_RATION":
            self._enact_law("LIMIT_WATER_CONSUMPTION", {"max_per_cycle": 3.0}, cycle,
                            duration=self.t_law,
                            description="Majority voted: ration water to 3 units/cycle.")

        elif winner == "QUARANTINE_EPIDEMIC":
            infected = [a for a in alive if a.infected and a.position]
            region = self._compute_quarantine_region(
                infected, n, sim.grid.rows, sim.grid.cols
            )
            if not region:
                r0 = self.rng.randint(0, sim.grid.rows - 4)
                c0 = self.rng.randint(0, sim.grid.cols - 4)
                region = (r0, c0, r0 + 3, c0 + 3)
            epidemic_id: Optional[str] = None
            for ev in sim.event_system.active_events:
                if ev.event_type.value == "epidemic" and not ev.cancelled:
                    epidemic_id = ev.epidemic_id
                    break
            self._enact_law("QUARANTINE_EPIDEMIC", {"region": region}, cycle,
                            duration=None,
                            event_id=epidemic_id,
                            event_type="epidemic",
                            description=f"Majority voted: epidemic quarantine {region}.")

        elif winner == "EPIDEMIC_RESPONSE":
            epidemic_id = None
            for ev in sim.event_system.active_events:
                if ev.event_type.value == "epidemic" and not ev.cancelled:
                    epidemic_id = ev.epidemic_id
                    break
            self._enact_law("EPIDEMIC_RESPONSE", {"medicine_mandate": True}, cycle,
                            duration=None,
                            event_id=epidemic_id,
                            event_type="epidemic",
                            description="Majority voted: mandate medicine use for infected.")

        elif winner == "MANDATORY_SHELTER":
            self._enact_law("MANDATORY_SHELTER", {}, cycle,
                            duration=None,
                            event_type="storm",
                            description="Majority voted: all agents must seek shelter.")

        elif winner == "RESOURCE_RESERVE":
            r0 = self.rng.randint(0, sim.grid.rows - 3)
            c0 = self.rng.randint(0, sim.grid.cols - 3)
            region = (r0, c0, r0 + 2, c0 + 2)
            self._enact_law("RESOURCE_RESERVE", {"region": region}, cycle,
                            duration=self.t_law,
                            description=f"Majority voted: protect resource zone {region}.")

        elif winner in ("REDISTRIBUTE_FOOD", "REDISTRIBUTE_WATER", "REDISTRIBUTE_MEDICINE"):
            resource = winner.split("_", 1)[1].lower()
            attr = f"{resource}_stock"
            sorted_agents = sorted(alive, key=lambda a: getattr(a, attr, 0.0))
            # K = bottom 30% (recipients), M = top 30% (donors)
            k_count = max(1, n * 30 // 100)
            recipient_ids = [a.agent_id for a in sorted_agents[:k_count]]
            donor_ids = [a.agent_id for a in sorted_agents[n * 70 // 100:]]
            if donor_ids and recipient_ids:
                agents_by_id = {a.agent_id: a for a in alive}
                # N = average ask of bottom-30%: amount to reach the median stock
                median_stock = getattr(sorted_agents[n // 2], attr, 5.0) if n > 1 else 5.0
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
                    self._enact_law(winner, {
                        "resource": resource,
                        "recipient_ids": recipient_ids,
                        "donor_amounts": donor_amounts,
                        "amount_per_recipient": actual_n,
                    }, cycle, duration=self.t_law,
                        applies_to=list(donor_amounts.keys()),
                        description=f"Majority voted: redistribute {resource} (N={actual_n:.1f}).")

        elif winner == "SPREAD":
            # Only apply SPREAD to non-infected agents (infected should be quarantined)
            non_infected = [a.agent_id for a in alive if not a.infected]
            self._enact_law("SPREAD", {}, cycle,
                            duration=self.t_law,
                            applies_to=non_infected if non_infected else None,
                            description="Majority voted: non-infected agents spread out.")

        elif winner == "FLEE_TOXIC":
            self._enact_law("FLEE_CURRENT_LOCATION", {}, cycle,
                            duration=8,
                            description="Majority voted: flee toxic area.")

    def receive_vote(self, agent: "Agent", option: Any, cycle: int) -> None:
        if self._election_active and option in self._current_options:
            self._votes[agent.agent_id] = option

    def receive_event_warnings(self, warnings: List["EventWarning"], cycle: int) -> None:
        if not self._sim:
            return
        for w in warnings:
            if w.cycles_until > 0 and w.cycles_until <= 3:
                active_types = {l.law_type for l in self.active_laws if l.is_active(cycle)}
                if w.event_type.value == "storm" and "MANDATORY_SHELTER" not in active_types:
                    self._enact_law("MANDATORY_SHELTER", {}, cycle, duration=None,
                                    event_type="storm",
                                    description="Emergency decree: pre-emptive shelter (storm warning).")
                elif w.event_type.value == "epidemic" and "QUARANTINE_EPIDEMIC" not in active_types:
                    alive = self._sim.living_agents()
                    infected = [a for a in alive if a.infected and a.position]
                    if w.region:
                        region = tuple(w.region) if isinstance(w.region, list) else w.region
                        self._enact_law("QUARANTINE_EPIDEMIC", {"region": region}, cycle,
                                        duration=None, event_type="epidemic",
                                        description=f"Emergency decree: pre-emptive quarantine (epidemic warning).")

    def active_election_options(self, cycle: int) -> List[str]:
        if self._election_active:
            return self._current_options
        return []

    def is_election_active(self) -> bool:
        return self._election_active

    def get_audit_info(self, cycle: int) -> dict:
        tally_str = " ".join(f"{k}:{v}" for k, v in sorted(
            self._last_vote_tallies.items(), key=lambda x: -x[1]
        ))
        return {
            "election_active": self._election_active,
            "current_options": "|".join(self._current_options),
            "last_vote_cycle": self._last_vote_cycle,
            "last_vote_tallies": tally_str,
            "last_vote_winner": self._last_vote_winner,
        }

    def get_log_info(self, cycle: int) -> dict:
        total_votes = sum(self._last_vote_tallies.values()) or 1
        winner_votes = self._last_vote_tallies.get(self._last_vote_winner, 0)
        winner_pct = 100.0 * winner_votes / total_votes if total_votes > 0 else 0

        decision_summary = ""
        if self._last_vote_cycle == cycle or self._last_vote_cycle == cycle - 1:
            options_str = " | ".join(self._current_options) if self._current_options else "none"
            tally_lines = []
            for opt, count in sorted(self._last_vote_tallies.items(), key=lambda x: -x[1]):
                tally_lines.append(f"{opt}={count}")
            decision_summary = (
                f"Election held.\n"
                f"    Options: {options_str}\n"
                f"    Results: {'  '.join(tally_lines)}\n"
                f"    Winner: {self._last_vote_winner} ({winner_pct:.1f}% of votes)"
            )

        vote_details = {}
        for cohort, votes in self._last_vote_breakdown.items():
            cohort_total = sum(votes.values())
            parts = [f"{opt}={cnt}" for opt, cnt in sorted(votes.items(), key=lambda x: -x[1])]
            vote_details[f"{cohort} ({cohort_total})"] = ", ".join(parts)

        return {
            "decision_summary": decision_summary,
            "decision_trigger": self._last_decision_trigger,
            "vote_details": vote_details,
            "state_narrative": f"  Election cycle: every {self.t_vote} cycles | Law duration: {self.t_law} cycles",
        }
