"""
CitizenAgent — the general-purpose survival agent used by all governments.

Fully self-interested and rational: prioritises survival above all else.
Also participates in the government system (voting, proposing) when able.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from engine.agent import Agent, Action, ActionType

if TYPE_CHECKING:
    from engine.grid import Grid
    from governments.base import Government


class CitizenAgent(Agent):
    """
    A self-interested survival agent.

    Observation: scans visible cells for resources, threats, other agents.
    Action policy:
      Priority 1 — Treat infection if infected and has medicine.
      Priority 2 — Find water if critically thirsty.
      Priority 3 — Find food if critically hungry.
      Priority 4 — Seek shelter if storm active.
      Priority 5 — Move toward richest food cell if not at good food source.
      Priority 6 — Collect food/water.
      Priority 7 — Eat/drink from stock.
      Priority 8 — Vote intelligently in elections.
    """

    def __init__(self, agent_id: Optional[str] = None, seed: Optional[int] = None):
        super().__init__(agent_id=agent_id, role="citizen")
        self.rng = random.Random(seed)
        self._last_known_food: Optional[tuple] = None  # (r, c) of best food seen
        self._last_known_water: Optional[tuple] = None
        self._last_known_shelter: Optional[tuple] = None

    # ------------------------------------------------------------------
    # C lines of code — observe
    # ------------------------------------------------------------------

    def observe(self, grid: "Grid", government: "Government", cycle: int) -> Dict[str, Any]:
        if not self.position:
            return {}

        r, c = self.position
        visibility = getattr(government, '_sim', None)
        radius = 5  # default
        if visibility and hasattr(visibility, 'config'):
            radius = visibility.config.visibility_radius

        visible = grid.visible_cells(r, c, radius)
        current_cell = grid.cell(r, c)
        nearby_agents = [
            a.to_dict() for a in grid.agents_in_radius(r, c, radius)
            if a.agent_id != self.agent_id and a.alive
        ]

        # Track best known resource locations
        best_food_cell = max(visible, key=lambda x: x["food"], default=None)
        best_water_cell = max(visible, key=lambda x: x["water"], default=None)
        shelter_cells = [v for v in visible if v["shelter"]]

        if best_food_cell and best_food_cell["food"] > 10:
            self._last_known_food = (best_food_cell["row"], best_food_cell["col"])
        if best_water_cell and best_water_cell["water"] > 10:
            self._last_known_water = (best_water_cell["row"], best_water_cell["col"])
        if shelter_cells:
            sc = shelter_cells[0]
            self._last_known_shelter = (sc["row"], sc["col"])

        # Count infected nearby (disease signal)
        infected_nearby = sum(1 for a in nearby_agents if a.get("infected", False))

        sim = getattr(government, '_sim', None)
        max_steps = (
            getattr(getattr(sim, 'config', None), 'max_steps_per_cycle', 1)
            if sim else 1
        )

        return {
            "position": self.position,
            "health": self.health,
            "hunger": self.hunger,
            "thirst": self.thirst,
            "infected": self.infected,
            "food_stock": self.food_stock,
            "water_stock": self.water_stock,
            "medicine_stock": self.medicine_stock,
            "current_cell": current_cell.to_dict(),
            "visible_cells": visible,
            "nearby_agents": nearby_agents,
            "infected_nearby": infected_nearby,
            "last_known_food": self._last_known_food,
            "last_known_water": self._last_known_water,
            "last_known_shelter": self._last_known_shelter,
            "active_laws": government.active_law_summary(cycle),
            "max_steps_per_cycle": max_steps,
        }

    # ------------------------------------------------------------------
    # C lines of code — act
    # ------------------------------------------------------------------

    def act(self, obs: Dict[str, Any], government: "Government", cycle: int) -> List[Action]:
        actions: List[Action] = []

        if not obs:
            return [self.eat(), self.drink()]

        current_cell = obs.get("current_cell", {})
        active_events = obs.get("active_events", [])
        active_laws = obs.get("active_laws", [])
        event_warnings = obs.get("event_warnings", [])
        event_types = [e["type"] for e in active_events]
        law_types = [l["type"] for l in active_laws]
        warning_types = [w.get("event_type", w.get("type", "")) for w in event_warnings
                         if isinstance(w, dict)]
        k = obs.get("max_steps_per_cycle", 1)

        if self.food_stock > 0.5:
            actions.append(self.eat(min(2.0, self.food_stock)))
        if self.water_stock > 0.5:
            actions.append(self.drink(min(2.0, self.water_stock)))

        # --- Priority 1: Treat infection ---
        if self.infected and self.medicine_stock >= 1.0:
            actions.append(self.treat_self())
            return actions

        # --- Priority 2: Storm / shelter (active OR warned) ---
        storm_active = "storm" in event_types
        storm_warned = any("storm" in str(w) for w in warning_types)
        shelter_mandated = "MANDATORY_SHELTER" in law_types or "MANDATORY_SHELTER_REGION" in law_types
        need_shelter = (storm_active or shelter_mandated or
                        (storm_warned and not current_cell.get("shelter", False)))
        if need_shelter and not current_cell.get("shelter", False):
            visible = obs.get("visible_cells", [])
            shelter_cells = [v for v in visible if v["shelter"]]
            if shelter_cells:
                r0, c0 = self.position
                nearest = min(shelter_cells,
                              key=lambda v: abs(v["row"] - r0) + abs(v["col"] - c0))
                self._last_known_shelter = (nearest["row"], nearest["col"])
                actions[0:0] = self.steps_toward(nearest["row"], nearest["col"], k)
            elif obs.get("last_known_shelter"):
                actions[0:0] = self.steps_toward(*obs["last_known_shelter"], k)
            if storm_active or shelter_mandated:
                return actions

        # --- Priority 3: Critical thirst ---
        if self.urgently_needs_water():
            water_limit = government.water_collection_limit(self, cycle)
            if current_cell.get("water", 0) > 1.0:
                actions.insert(0, self.collect_water(min(5.0, water_limit)))
            elif obs.get("last_known_water"):
                actions[0:0] = self.steps_toward(*obs["last_known_water"], k)
            else:
                actions.insert(0, self._move_smart(obs))
            return actions

        # --- Priority 4: Critical hunger ---
        if self.urgently_needs_food():
            food_limit = government.food_collection_limit(self, cycle)
            if current_cell.get("food", 0) > 1.0:
                actions.insert(0, self.collect_food(min(5.0, food_limit)))
            elif obs.get("last_known_food"):
                actions[0:0] = self.steps_toward(*obs["last_known_food"], k)
            else:
                actions.insert(0, self._move_smart(obs))
            return actions

        # --- Priority 5: Share medicine with critical neighbours ---
        if self.medicine_stock > 3.0:
            nearby = obs.get("nearby_agents", [])
            for neighbour in nearby:
                if neighbour.get("infected") and neighbour.get("medicine_stock", 99) < 1.0:
                    actions.append(self.share_medicine(neighbour["id"]))
                    break

        # --- Priority 5b: Preventive medicine during epidemic ---
        epidemic_active = "epidemic" in event_types
        epidemic_warned = any("epidemic" in str(w) for w in warning_types)
        if (epidemic_active or epidemic_warned) and self.medicine_stock < 6.0:
            med_limit = government.medicine_collection_limit(self, cycle)
            if current_cell.get("medicine", 0) > 1.0:
                collect_amt = min(2.0, med_limit, 6.0 - self.medicine_stock)
                actions.insert(0, self.collect_medicine(collect_amt))
                self._add_vote_action(actions, obs, government, cycle, event_types)
                self._add_propose_action(actions, obs, government, cycle, event_types)
                return actions

        # --- Priority 6: Maintain stocks ---
        food_limit = government.food_collection_limit(self, cycle)
        water_limit = government.water_collection_limit(self, cycle)
        med_limit = government.medicine_collection_limit(self, cycle)

        cell_food = current_cell.get("food", 0)
        cell_water = current_cell.get("water", 0)
        cell_depleted = cell_food < 1.0 and cell_water < 1.0

        if self.food_stock < 10.0 and cell_food > 2.0:
            collect_amt = min(5.0, food_limit, 10.0 - self.food_stock)
            actions.insert(0, self.collect_food(collect_amt))
        elif self.water_stock < 10.0 and cell_water > 2.0:
            collect_amt = min(5.0, water_limit, 10.0 - self.water_stock)
            actions.insert(0, self.collect_water(collect_amt))
        elif current_cell.get("medicine", 0) > 1.0 and self.medicine_stock < 5.0:
            collect_amt = min(2.0, med_limit)
            actions.insert(0, self.collect_medicine(collect_amt))
        elif cell_depleted or (self.food_stock < 15.0 and cell_food < 2.0):
            if obs.get("last_known_food") and self.food_stock < 15.0:
                tf, tc = obs["last_known_food"]
                if (tf, tc) != self.position:
                    actions[0:0] = self.steps_toward(tf, tc, k)
                else:
                    actions.insert(0, self._move_smart(obs))
            else:
                actions.insert(0, self._move_smart(obs))
        else:
            if obs.get("last_known_food") and self.food_stock < 15.0:
                tf, tc = obs["last_known_food"]
                if (tf, tc) != self.position:
                    actions[0:0] = self.steps_toward(tf, tc, k)

        # --- Priority 7: Vote ---
        self._add_vote_action(actions, obs, government, cycle, event_types)

        # --- Priority 8: Propose in government systems ---
        self._add_propose_action(actions, obs, government, cycle, event_types)

        return actions

    # ------------------------------------------------------------------
    # Voting logic
    # ------------------------------------------------------------------

    def _add_vote_action(
        self,
        actions: List[Action],
        obs: Dict[str, Any],
        government: "Government",
        cycle: int,
        event_types: List[str],
    ) -> None:
        from governments.democracy import DemocracyGovernment
        from governments.republic import RepublicGovernment

        if isinstance(government, DemocracyGovernment):
            if government.is_election_active():
                options = government.active_election_options(cycle)
                if options:
                    vote = self._choose_vote(options, event_types)
                    actions.append(self.vote(vote))

        elif isinstance(government, RepublicGovernment):
            parties = government.parties
            if parties:
                # Vote for the party whose platform matches current threats
                chosen = self._choose_party(parties, event_types)
                actions.append(self.vote(chosen))

        elif hasattr(government, 'parties'):
            # Federated regions use party-like voting
            chosen = self._choose_from_list(
                ["FOOD_RATION", "QUARANTINE", "MANDATORY_SHELTER", "NO_ACTION"],
                event_types
            )
            actions.append(self.vote(chosen))

    def _choose_vote(self, options: List[str], event_types: List[str]) -> str:
        """
        Score each option by self-interest and pick the highest-scoring one.

        Each law is scored by how urgently the agent needs it right now.
        Scores are additive across multiple matching conditions.

        Ties are broken UNIFORMLY AT RANDOM from ``self.rng``.  This matters
        because ballot ORDER would otherwise decide the election whenever an
        agent is indifferent, and indifference is common (an agent with full
        stocks and no active event scores most options at 0.0).  Ballot order
        is set by the government, not the electorate, so a first-match
        tie-break would silently hand the proposer a tie-break vote in every
        regime that votes.  Same reasoning as ``_move_smart``; lower impact
        here, because only democracy / republic / federated reach this path.
        """

        def score(opt: str) -> float:
            s = 0.0
            food_need  = max(0.0, 1.0 - self.food_stock / 10.0)
            water_need = max(0.0, 1.0 - self.water_stock / 10.0)
            med_need   = max(0.0, 1.0 - self.medicine_stock / 5.0)
            health_gap = max(0.0, 1.0 - self.health)
            epidemic   = "epidemic" in event_types or self.infected

            if opt in ("REDISTRIBUTE_FOOD", "FOOD_RATION"):
                # Vote for redistribution if very low on food; vote for ration
                # only if drought active and we want to preserve future supply
                if opt == "REDISTRIBUTE_FOOD":
                    s += food_need * 2.0
                else:  # FOOD_RATION
                    if "drought" in event_types:
                        s += food_need * 0.8  # ration helps only if resources scarce
                    else:
                        s -= food_need * 0.5  # ration hurts when no drought

            elif opt in ("REDISTRIBUTE_WATER", "WATER_RATION", "LIMIT_WATER_CONSUMPTION"):
                if opt == "REDISTRIBUTE_WATER":
                    s += water_need * 2.0
                else:
                    if "toxic_spill" in event_types or "drought" in event_types:
                        s += water_need * 0.8
                    else:
                        s -= water_need * 0.5

            elif opt == "REDISTRIBUTE_MEDICINE":
                if self.infected:
                    s += 2.5  # critical: need medicine to cure
                s += med_need * 1.5

            elif opt == "REDISTRIBUTE_GENERIC":
                # Generic helps if all three are needed; prefer specific if only one is low
                worst = max(food_need, water_need, med_need)
                avg = (food_need + water_need + med_need) / 3
                s += avg * 1.8 if worst > 0.5 else avg * 0.9

            elif opt in ("QUARANTINE_EPIDEMIC", "EPIDEMIC_RESPONSE"):
                if self.infected:
                    # If infected: prefer EPIDEMIC_RESPONSE (treat me) over pure quarantine
                    s += 2.0 if opt == "EPIDEMIC_RESPONSE" else 1.2
                elif epidemic:
                    # Not infected but epidemic active: quarantine protects me
                    s += 1.8 if opt == "QUARANTINE_EPIDEMIC" else 0.5

            elif opt == "MANDATORY_SHELTER":
                if "storm" in event_types:
                    s += 2.0  # storm is immediately lethal without shelter
                if self.health < 0.5:
                    s += 0.5

            elif opt in ("FLEE_TOXIC", "FLEE_CURRENT_LOCATION"):
                if "toxic_spill" in event_types:
                    s += 2.0
                s += health_gap * 0.8

            elif opt == "SPREAD":
                if epidemic and not self.infected:
                    s += 1.5   # spreading away keeps me safe
                elif self.infected:
                    s -= 1.0   # spreading while infected is self-destructive

            elif opt == "MOVE_TO_REGION":
                # Government-proposed relocation; vote for if resource-starved
                s += (food_need + water_need) * 0.6

            elif opt == "RESOURCE_RESERVE":
                s += (food_need + water_need) * 0.4

            elif opt in ("LIMIT_STEPS",):
                s -= 0.5  # always against restricting movement

            elif opt == "NO_ACTION":
                # Only vote no-action if agent is healthy and no threats
                if health_gap < 0.1 and not epidemic and "storm" not in event_types:
                    s += 0.3
                else:
                    s -= 1.0

            return s

        if not options:
            raise ValueError("_choose_vote requires at least one option")

        best_opts: List[str] = []
        best_score = float("-inf")
        for opt in options:
            sc = score(opt)
            if sc > best_score:
                best_score = sc
                best_opts = [opt]
            elif sc == best_score:
                best_opts.append(opt)
        # One draw per call regardless of how many options tied — see the same
        # note in _move_smart for why the len == 1 shortcut is not taken.
        return self.rng.choice(best_opts)

    def _choose_party(self, parties: List[str], event_types: List[str]) -> str:
        """
        Score each party by how much its platform helps THIS agent's self-interest.

        For each law type in a party's platform the agent applies a situational
        multiplier: +1.0 if the law directly addresses the agent's worst current
        problem, +0.4 if the law helps a secondary concern, 0.0 if irrelevant,
        and -0.5 if the law restricts the agent when it is currently healthy
        (ration/quarantine laws that cost the agent without benefit).

        The agent knows all party platforms, so this is fully informed voting.
        """
        from governments.republic import PARTY_PLATFORMS

        # Determine how badly the agent needs each type of help (0.0–1.0 urgency)
        food_need    = max(0.0, 1.0 - self.food_stock / 10.0)
        water_need   = max(0.0, 1.0 - self.water_stock / 10.0)
        med_need     = 1.0 if self.infected else max(0.0, 1.0 - self.medicine_stock / 5.0)
        health_need  = max(0.0, 1.0 - self.health)
        shelter_need = 1.0 if "storm" in event_types else 0.0
        flee_need    = 1.0 if "toxic_spill" in event_types else 0.0
        epidemic     = "epidemic" in event_types or self.infected

        # Relevance of each law type to this agent's self-interest
        law_relevance: Dict[str, float] = {
            "REDISTRIBUTE_FOOD":      food_need * 1.5,
            "REDISTRIBUTE_WATER":     water_need * 1.5,
            "REDISTRIBUTE_MEDICINE":  med_need * 1.5,
            "REDISTRIBUTE_GENERIC":   (food_need + water_need + med_need) / 3 * 1.2,
            "FOOD_RATION":            -food_need * 0.8 if food_need < 0.3 else food_need * 0.3,
            "WATER_RATION":           -water_need * 0.8 if water_need < 0.3 else water_need * 0.3,
            "LIMIT_FOOD_CONSUMPTION": -food_need * 0.8 if food_need < 0.3 else food_need * 0.3,
            "QUARANTINE_EPIDEMIC":    (1.0 if epidemic else -0.3),
            "EPIDEMIC_RESPONSE":      (1.0 if (self.infected or med_need > 0.5) else 0.1),
            "MANDATORY_SHELTER":      shelter_need,
            "SPREAD":                 (1.0 if epidemic else 0.2),
            "FLEE_TOXIC":             flee_need,
            "RESOURCE_RESERVE":       (food_need + water_need) * 0.5,
            "NO_ACTION":              (0.8 if health_need < 0.2 and not epidemic else -0.4),
            "LIMIT_STEPS":            -0.3,  # restricts freedom, never in agent's interest
        }

        best_party = parties[0]
        best_score = float("-inf")
        for party in parties:
            platform = PARTY_PLATFORMS.get(party, {})
            score = sum(
                weight * law_relevance.get(law_type, 0.0)
                for law_type, weight in platform.items()
            )
            if score > best_score:
                best_score = score
                best_party = party
        return best_party

    def _choose_from_list(self, options: List[str], event_types: List[str]) -> str:
        if "epidemic" in event_types and "QUARANTINE" in options:
            return "QUARANTINE"
        if "storm" in event_types and "MANDATORY_SHELTER" in options:
            return "MANDATORY_SHELTER"
        if "drought" in event_types and "FOOD_RATION" in options:
            return "FOOD_RATION"
        return "NO_ACTION"

    def _add_propose_action(
        self,
        actions: List[Action],
        obs: Dict[str, Any],
        government: "Government",
        cycle: int,
        event_types: List[str],
    ) -> None:
        if cycle % 5 == 0 and "epidemic" in event_types:
            actions.append(self.propose({"type": "quarantine", "reason": "epidemic_detected"}))

    def _move_random(self) -> Action:
        directions = ["north", "south", "east", "west"]
        return self.move(self.rng.choice(directions))

    def _move_smart(self, obs: Dict[str, Any]) -> Action:
        """Move toward the lowest-hazard neighbour with the best resources.

        Ties are broken UNIFORMLY AT RANDOM from ``self.rng``, not by a strict
        ``if score > best_score`` scan over ``DIRECTION_DELTAS``.  That dict's
        insertion order begins ``north``, so a first-match rule resolves every
        tie north; at cycle 0, on a freshly generated grid with a clustered
        population, neighbouring cells tie constantly, which would produce a
        systematic northward drift that is an artifact of dict ordering rather
        than anything in the model, and identical in every run.  Sampling the
        argmax set removes the artifact and keeps the citizen RNG live
        (see ``scenarios/scenario_base._make_citizen_population``).
        """
        from engine.agent import DIRECTION_DELTAS
        if not self.position:
            return self._move_random()
        visible = obs.get("visible_cells", [])
        if not visible:
            return self._move_random()
        r, c = self.position
        # Index the visible cells once, keyed by position, rather than scanning
        # linearly per direction per agent per cycle over a visibility disc that
        # is ~80 cells at radius 5.
        by_pos = {(v["row"], v["col"]): v for v in visible}
        best_dirs: List[str] = []
        # -inf, not a -999.0 sentinel: with an argmax *set* a cell that scored
        # exactly -999.0 would have to be either a candidate or not, and -999.0
        # is a reachable score (hazard ~20 with resources present).  -inf is
        # unreachable, so "no candidate" is unambiguously the empty list.
        best_score = float("-inf")
        for direction, (dr, dc) in DIRECTION_DELTAS.items():
            cell = by_pos.get((r + dr, c + dc))
            if cell is None:
                continue
            hazard = cell.get("hazard", 0.0)
            food = cell.get("food", 0.0)
            water = cell.get("water", 0.0)
            score = (food * 0.3 + water * 0.3) - hazard * 50.0
            if score > best_score:
                best_score = score
                best_dirs = [direction]
            elif score == best_score:
                best_dirs.append(direction)
        if best_dirs:
            # `choice` even for a single candidate: branching on len == 1 to
            # "save a draw" would make the RNG's consumption depend on how many
            # cells happened to tie, so two agents in the same situation could
            # fall out of step for a reason that has nothing to do with their
            # behaviour.  One draw per call, always.
            return self.move(self.rng.choice(best_dirs))
        return self._move_random()
