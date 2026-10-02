"""Representative republic: citizens vote for parties; parliament passes laws every 2 cycles."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .base import Government

if TYPE_CHECKING:
    from engine.agent import Agent, Action
    from engine.events import EventWarning


# Parties and their platform weights per law type.
# Positive weight = party supports that law; negative = opposes.
PARTY_PLATFORMS: Dict[str, Dict[str, float]] = {
    "HealthParty":     {"QUARANTINE_EPIDEMIC": 0.9, "EPIDEMIC_RESPONSE": 0.9,
                        "MANDATORY_SHELTER": 0.7, "REDISTRIBUTE_MEDICINE": 0.8,
                        "FOOD_RATION": 0.3},
    "ResourceParty":   {"REDISTRIBUTE_FOOD": 0.9, "REDISTRIBUTE_WATER": 0.8,
                        "FOOD_RATION": 0.7, "WATER_RATION": 0.7,
                        "RESOURCE_RESERVE": 0.9, "QUARANTINE_EPIDEMIC": 0.2},
    "FreedomParty":    {"FOOD_RATION": 0.1, "QUARANTINE_EPIDEMIC": 0.1,
                        "MANDATORY_SHELTER": 0.1, "SPREAD": 0.8, "NO_ACTION": 0.9},
    "SolidarityParty": {"REDISTRIBUTE_FOOD": 0.9, "REDISTRIBUTE_WATER": 0.9,
                        "REDISTRIBUTE_MEDICINE": 0.9, "EPIDEMIC_RESPONSE": 0.6,
                        "QUARANTINE_EPIDEMIC": 0.5},
    "GreenParty":      {"SPREAD": 0.9, "RESOURCE_RESERVE": 0.8,
                        "FLEE_TOXIC": 0.9, "FOOD_RATION": 0.5},
    "SecurityParty":   {"QUARANTINE_EPIDEMIC": 0.8, "MANDATORY_SHELTER": 0.9,
                        "FLEE_TOXIC": 0.7, "LIMIT_STEPS": 0.5},
}


class RepublicGovernment(Government):
    """
    Citizens vote for parties every T_ELECTION cycles.
    Parliament holds sessions every T_SESSION cycles.
    Fixed structure: N_PARTIES parties, PARLIAMENT_SIZE seats, N_REPRESENTATIVES agents.
    All surviving agents automatically cast votes each election.
    """

    name = "Representative Republic"

    # Fixed structural parameters — independent of population size
    N_PARTIES        = 6    # number of active parties (must be <= len(PARTY_PLATFORMS))
    PARLIAMENT_SIZE  = 30   # fixed total parliament seats
    N_REPRESENTATIVES = 20  # fixed number of designated representative agents

    T_ELECTION = 5     # party election every 5 cycles
    T_SESSION  = 2     # parliament sessions every 2 cycles
    T_LAW      = 25    # law duration in cycles

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.rng = random.Random(seed)
        self.parties: List[str] = list(PARTY_PLATFORMS.keys())[:self.N_PARTIES]
        self.parliament: List[str] = []        # party names of seated representatives
        self.representative_agents: Set[str] = set()
        self._citizen_votes: Dict[str, str] = {}  # agent_id → party
        self._last_election = -(self.T_ELECTION + 1)   # trigger at cycle 0
        # Audit tracking
        self._last_election_tallies: Dict[str, int] = {}
        self._last_parliament_law: str = ""
        self._last_parliament_vote_cycle: int = -1

    # ------------------------------------------------------------------
    # Per-cycle tick
    # ------------------------------------------------------------------

    def tick(self, cycle: int) -> None:
        self._expire_laws(cycle)

        # Hold party election
        if cycle - self._last_election >= self.T_ELECTION:
            self._hold_election(cycle)

        # Parliament session (every 2 cycles)
        if self.parliament and cycle % self.T_SESSION == 0:
            self._parliament_session(cycle)

    # ------------------------------------------------------------------
    # Election
    # ------------------------------------------------------------------

    def _hold_election(self, cycle: int) -> None:
        if not self._sim:
            return
        self._last_election = cycle
        living = self._sim.living_agents()
        n = len(living)

        # Auto-vote all surviving agents who haven't voted
        event_names = [e["type"] for e in self._sim.event_system.active_summary(cycle)]
        for agent in living:
            if agent.agent_id not in self._citizen_votes:
                self._citizen_votes[agent.agent_id] = self._agent_choose_party(
                    agent, event_names
                )

        vote_counts = Counter(self._citizen_votes.values())
        self._last_election_tallies = dict(vote_counts)
        total = sum(vote_counts.values()) or 1

        # Proportional seat allocation into fixed parliament size
        n_seats = self.PARLIAMENT_SIZE
        self.parliament = []
        for party in self.parties:
            seats = max(0, round(vote_counts.get(party, 0) / total * n_seats))
            self.parliament.extend([party] * seats)
        self.parliament = self.parliament[:n_seats]

        # Designate fixed number of representative agents (randomly selected)
        self.rng.shuffle(living)
        n_reps = min(self.N_REPRESENTATIVES, n)
        self.representative_agents = {a.agent_id for a in living[:n_reps]}

        self._citizen_votes = {}

    def _agent_choose_party(self, agent: "Agent", event_names: List[str]) -> str:
        """Heuristic: choose the party whose platform best addresses the agent's needs."""
        from agents.citizen import CitizenAgent
        if isinstance(agent, CitizenAgent):
            return agent._choose_party(self.parties, event_names)
        # Fallback heuristic
        if "epidemic" in event_names:
            for p in self.parties:
                if "Health" in p:
                    return p
        if "drought" in event_names or agent.food_stock < 5:
            for p in self.parties:
                if "Resource" in p or "Solidarity" in p:
                    return p
        return self.rng.choice(self.parties)

    # ------------------------------------------------------------------
    # Parliament session
    # ------------------------------------------------------------------

    def _parliament_session(self, cycle: int) -> None:
        if not self._sim:
            return

        event_names = [e["type"] for e in self._sim.event_system.active_summary(cycle)]
        alive = self._sim.living_agents()

        # Accumulate weighted votes from parliament seats
        proposal_votes: Dict[str, float] = defaultdict(float)
        for party in self.parliament:
            platform = dict(PARTY_PLATFORMS.get(party, {}))
            # Contextual boost
            if "epidemic" in event_names:
                alive_ep = self._sim.living_agents() if self._sim else []
                infected_frac = (
                    sum(1 for a in alive_ep if a.infected) / max(1, len(alive_ep))
                )
                platform["QUARANTINE_EPIDEMIC"] = min(
                    1.0, platform.get("QUARANTINE_EPIDEMIC", 0) + 0.4
                )
                platform["EPIDEMIC_RESPONSE"] = min(
                    1.0, platform.get("EPIDEMIC_RESPONSE", 0) + 0.4
                )
                # If few infected, SPREAD is a valid escape option too
                if infected_frac < 0.25:
                    platform["SPREAD"] = min(1.0, platform.get("SPREAD", 0) + 0.3)
            if "storm" in event_names:
                platform["MANDATORY_SHELTER"] = min(
                    1.0, platform.get("MANDATORY_SHELTER", 0) + 0.5
                )
            if "drought" in event_names:
                platform["FOOD_RATION"] = min(
                    1.0, platform.get("FOOD_RATION", 0) + 0.4
                )
                platform["REDISTRIBUTE_FOOD"] = min(
                    1.0, platform.get("REDISTRIBUTE_FOOD", 0) + 0.3
                )
            if "toxic_spill" in event_names:
                platform["FLEE_TOXIC"] = min(
                    1.0, platform.get("FLEE_TOXIC", 0) + 0.5
                )

            for law_type, weight in platform.items():
                proposal_votes[law_type] += weight

        if not proposal_votes:
            return

        winner = max(proposal_votes, key=proposal_votes.__getitem__)
        self._last_parliament_law = winner
        self._last_parliament_vote_cycle = cycle
        if winner != "NO_ACTION":
            already_active = any(
                l.law_type == winner and l.is_active(cycle)
                for l in self.active_laws
            )
            if not already_active:
                self._pass_law(winner, cycle, event_names, alive)

    def _pass_law(
        self,
        law_type: str,
        cycle: int,
        event_names: List[str],
        alive: List["Agent"],
    ) -> None:
        n = len(alive)

        if law_type == "FOOD_RATION":
            self._enact_law("FOOD_RATION", {"max_per_cycle": 4.0}, cycle,
                            duration=None,
                            event_type="drought",
                            description="Parliament: food ration 4 units/cycle.")

        elif law_type == "WATER_RATION":
            self._enact_law("LIMIT_WATER_CONSUMPTION", {"max_per_cycle": 4.0}, cycle,
                            duration=self.T_LAW,
                            description="Parliament: water ration 4 units/cycle.")

        elif law_type == "QUARANTINE_EPIDEMIC":
            infected = [a for a in alive if a.infected and a.position]
            region = self._compute_quarantine_region(
                infected, n, self._sim.grid.rows, self._sim.grid.cols
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
                                description=f"Parliament: epidemic quarantine {region}.")

        elif law_type == "EPIDEMIC_RESPONSE":
            epidemic_id = None
            for ev in self._sim.event_system.active_events:
                if ev.event_type.value == "epidemic" and not ev.cancelled:
                    epidemic_id = ev.epidemic_id
                    break
            self._enact_law("EPIDEMIC_RESPONSE", {"medicine_mandate": True}, cycle,
                            duration=None,
                            event_id=epidemic_id,
                            event_type="epidemic",
                            description="Parliament: mandate medicine use for infected.")

        elif law_type == "MANDATORY_SHELTER":
            self._enact_law("MANDATORY_SHELTER", {}, cycle,
                            duration=None,
                            event_type="storm",
                            description="Parliament: mandatory shelter during storm.")

        elif law_type == "RESOURCE_RESERVE":
            r0 = self.rng.randint(0, self._sim.grid.rows - 3)
            c0 = self.rng.randint(0, self._sim.grid.cols - 3)
            region = (r0, c0, r0 + 2, c0 + 2)
            self._enact_law("RESOURCE_RESERVE", {"region": region}, cycle,
                            duration=self.T_LAW,
                            description=f"Parliament: resource reserve {region}.")

        elif law_type in ("REDISTRIBUTE_FOOD", "REDISTRIBUTE_WATER", "REDISTRIBUTE_MEDICINE"):
            resource = law_type.split("_", 1)[1].lower()
            attr = f"{resource}_stock"
            sorted_agents = sorted(alive, key=lambda a: getattr(a, attr, 0.0))
            # Bottom 30% are K recipients; top 30% are M donors
            k_count = max(1, n * 30 // 100)
            recipients = [a.agent_id for a in sorted_agents[:k_count]]
            donors_list = [a for a in sorted_agents[n * 70 // 100:]]
            donor_ids = [a.agent_id for a in donors_list]
            if donor_ids and recipients:
                agents_by_id = {a.agent_id: a for a in alive}
                # N = average deficit of bottom-30% to reach median stock
                median_stock = getattr(sorted_agents[n // 2], attr, 5.0) if n > 1 else 5.0
                recipient_agents = [agents_by_id[rid] for rid in recipients if rid in agents_by_id]
                avg_deficit = sum(
                    max(0.0, median_stock - getattr(a, attr, 0.0))
                    for a in recipient_agents
                ) / max(1, len(recipient_agents))
                n_amount = max(1.0, avg_deficit)
                donor_amounts, actual_n = self._compute_redistribution_amounts(
                    agents_by_id, recipients, donor_ids, resource, n_amount
                )
                if donor_amounts:
                    self._enact_law(law_type, {
                        "resource": resource,
                        "recipient_ids": recipients,
                        "donor_amounts": donor_amounts,
                        "amount_per_recipient": actual_n,
                    }, cycle, duration=self.T_LAW,
                        applies_to=list(donor_amounts.keys()),
                        description=f"Parliament: redistribute {resource} (N={actual_n:.1f}).")

        elif law_type == "SPREAD":
            self._enact_law("SPREAD", {}, cycle,
                            duration=self.T_LAW,
                            description="Parliament: agents must spread out.")

        elif law_type == "FLEE_TOXIC":
            self._enact_law("FLEE_CURRENT_LOCATION", {}, cycle,
                            duration=8,
                            description="Parliament: flee toxic area.")

    # ------------------------------------------------------------------
    # Vote reception (from explicit VOTE actions, in addition to auto-vote)
    # ------------------------------------------------------------------

    def receive_vote(self, agent: "Agent", option: Any, cycle: int) -> None:
        if isinstance(option, str) and option in self.parties:
            self._citizen_votes[agent.agent_id] = option

    def receive_event_warnings(self, warnings: List["EventWarning"], cycle: int) -> None:
        if not self._sim:
            return
        for w in warnings:
            if w.cycles_until > 0 and w.cycles_until <= 3:
                active_types = {l.law_type for l in self.active_laws if l.is_active(cycle)}
                if w.event_type.value == "storm" and "MANDATORY_SHELTER" not in active_types:
                    self._enact_law("MANDATORY_SHELTER", {}, cycle, duration=None,
                                    event_type="storm",
                                    description="Emergency parliament decree: pre-emptive shelter.")
                elif w.event_type.value == "epidemic" and "QUARANTINE_EPIDEMIC" not in active_types:
                    if w.region:
                        region = tuple(w.region) if isinstance(w.region, list) else w.region
                        self._enact_law("QUARANTINE_EPIDEMIC", {"region": region}, cycle,
                                        duration=None, event_type="epidemic",
                                        description="Emergency parliament decree: pre-emptive quarantine.")

    def is_representative(self, agent: "Agent") -> bool:
        return agent.agent_id in self.representative_agents

    def get_audit_info(self, cycle: int) -> dict:
        from collections import Counter as _Counter
        seat_counts = dict(_Counter(self.parliament))
        seats_str = " ".join(f"{p}:{n}" for p, n in sorted(seat_counts.items(), key=lambda x: -x[1]))
        election_tally_str = " ".join(
            f"{k}:{v}" for k, v in sorted(self._last_election_tallies.items(), key=lambda x: -x[1])
        )
        return {
            "parliament_seats": seats_str,
            "total_seats": len(self.parliament),
            "last_election_tally": election_tally_str,
            "last_parliament_law": self._last_parliament_law,
            "last_parliament_vote_cycle": self._last_parliament_vote_cycle,
            "n_representatives": len(self.representative_agents),
        }

    def get_log_info(self, cycle: int) -> dict:
        from collections import Counter as _Counter
        seat_counts = dict(_Counter(self.parliament))

        summary_parts = []
        if self._last_election == cycle:
            tally_parts = [f"{p}: {c} votes" for p, c in sorted(
                self._last_election_tallies.items(), key=lambda x: -x[1]
            )]
            summary_parts.append("Party election held.")
            summary_parts.append(f"    Citizen votes: {', '.join(tally_parts)}")
            seat_parts = [f"{p}: {n} seats" for p, n in sorted(seat_counts.items(), key=lambda x: -x[1])]
            summary_parts.append(f"    Parliament seats: {', '.join(seat_parts)}")
        if self._last_parliament_vote_cycle == cycle:
            summary_parts.append(f"Parliament session: enacted {self._last_parliament_law}")

        seat_lines = []
        for party, n in sorted(seat_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * n / max(1, len(self.parliament))
            seat_lines.append(f"    {party:<20} {n:2d} seats ({pct:.0f}%)")
        narrative = (
            f"  Parliament: {len(self.parliament)} seats across {len(seat_counts)} parties\n"
            + "\n".join(seat_lines) + "\n"
            f"  Representatives: {len(self.representative_agents)} designated agents\n"
            f"  Election cycle: every {self.T_ELECTION} | Session cycle: every {self.T_SESSION}"
        )

        return {
            "decision_summary": "\n".join(summary_parts),
            "decision_trigger": f"election every {self.T_ELECTION} cycles; parliament every {self.T_SESSION} cycles",
            "vote_details": {p: f"{c} votes" for p, c in self._last_election_tallies.items()} if self._last_election == cycle else {},
            "state_narrative": narrative,
        }
