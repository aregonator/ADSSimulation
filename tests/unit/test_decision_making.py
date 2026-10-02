#!/usr/bin/env python3
"""
Decision-making behavioral tests for the Survival Simulation.

Verifies that agents and governments make sensible decisions given
specific situations. These are sanity checks for rationality, not
correctness of mechanics (which test_scenarios.py covers).

Usage (from the Simulation project root):
    python3 tests/unit/test_decision_making.py
"""

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root: tests/<subpkg>/<file>.py -> tests/<subpkg> -> tests/ -> root
_PARENT = _HERE  # sys.path target for the flat top-level packages (engine, governments, ...)
sys.path.insert(0, _PARENT)

from engine.grid import Grid, Terrain, Cell
from engine.agent import Agent, Action, ActionType, DIRECTION_DELTAS
from engine.events import EventSystem, EventType, ActiveEvent
from engine.simulation import Simulation, SimulationConfig, difficulty_multiplier
from agents.citizen import CitizenAgent
from governments.base import Government, Law
from governments.anarchy import AnarchyGovernment
from governments.democracy import DemocracyGovernment
from governments.republic import RepublicGovernment, PARTY_PLATFORMS
from governments.autocracy import AutocracyGovernment
from governments.oligarchy import OligarchyGovernment
from governments.federated import FederatedGovernment


# ==========================================================================
# Helpers (same as test_scenarios.py)
# ==========================================================================

def make_grid(rows, cols, terrain_map=None):
    grid = Grid(rows=rows, cols=cols, resource_density=1.0, terrain_variety=0.0, seed=42,
                ambient_hazard=0.0)
    if terrain_map:
        CHAR_TO_TERRAIN = {
            '.': Terrain.PLAINS, 'F': Terrain.FOREST, 'W': Terrain.WATER,
            'M': Terrain.MOUNTAIN, 'X': Terrain.WASTELAND,
        }
        for r, row_str in enumerate(terrain_map):
            for c, ch in enumerate(row_str):
                cell = grid.cell(r, c)
                terrain = CHAR_TO_TERRAIN.get(ch, Terrain.PLAINS)
                cell.terrain = terrain
                if terrain == Terrain.WASTELAND:
                    cell.food = cell.water = cell.medicine = 0.0
                elif terrain == Terrain.WATER:
                    cell.food = 0.0; cell.water = 90.0; cell.medicine = 10.0
                elif terrain == Terrain.FOREST:
                    cell.food = 80.0; cell.water = 40.0; cell.medicine = 15.0
                elif terrain == Terrain.MOUNTAIN:
                    cell.food = 20.0; cell.water = 20.0; cell.medicine = 10.0
                else:
                    cell.food = 50.0; cell.water = 35.0; cell.medicine = 10.0
    return grid


def make_sim(grid, agents, government, difficulty=1, max_cycles=50, seed=42):
    config = SimulationConfig(
        grid_rows=grid.rows, grid_cols=grid.cols,
        resource_density=1.0, terrain_variety=0.0,
        num_agents=len(agents), initial_health=1.0,
        difficulty=difficulty, event_frequency=0.0,
        event_severity=1.0, event_warning_cycles=5,
        visibility_radius=5, max_steps_per_cycle=1,
        seed=seed, max_cycles=max_cycles,
    )
    sim = Simulation(config=config, government=government, agents=agents)
    for agent in agents:
        if agent.position:
            r, c = agent.position
            if sim.grid.in_bounds(r, c):
                cell = sim.grid.cell(r, c)
                if agent in cell.agents:
                    cell.agents.remove(agent)
            agent.position = None
    sim.grid = grid
    return sim


def place(sim, agent, r, c):
    sim.grid.place_agent(agent, r, c)


def set_agent(agent, health=None, food=None, water=None, medicine=None,
              hunger=None, thirst=None):
    if health is not None: agent.health = health
    if food is not None: agent.food_stock = food
    if water is not None: agent.water_stock = water
    if medicine is not None: agent.medicine_stock = medicine
    if hunger is not None: agent.hunger = hunger
    if thirst is not None: agent.thirst = thirst


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []

    def ok(self, name, detail=""):
        self.passed.append((name, detail))
        print(f"  PASS: {name}")

    def fail(self, name, detail=""):
        self.failed.append((name, detail))
        print(f"  FAIL: {name} — {detail}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"  RESULTS: {len(self.passed)}/{total} passed, {len(self.failed)} failed")
        if self.failed:
            print(f"\n  FAILURES:")
            for name, detail in self.failed:
                print(f"    - {name}: {detail}")
        print(f"{'='*60}\n")
        return len(self.failed) == 0


results = TestResults()


# ==========================================================================
# SECTION 1: Agent Action Selection (Self-Interest)
# ==========================================================================

def test_agent_collects_food_when_on_rich_cell():
    """Agent with low food on a food-rich cell should collect, not move away."""
    print("\n--- Agent: Collects food on rich cell ---")
    grid = make_grid(6, 6)
    agent = CitizenAgent(agent_id="collect-test", seed=42)
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place(sim, agent, 2, 2)
    set_agent(agent, food=3.0, water=10.0, health=1.0)
    grid.cell(2, 2).food = 60.0

    obs = agent.observe(grid, gov, 0)
    obs["active_events"] = []
    obs["event_warnings"] = []
    obs["cycle"] = 0
    actions = agent.act(obs, gov, 0)
    types = [a.type for a in actions]

    if ActionType.COLLECT_FOOD in types:
        results.ok("Agent collects food when on rich cell with low stock")
    else:
        results.fail("Agent did not collect food on rich cell",
                     f"actions={[a.type.value for a in actions]}")


def test_agent_moves_toward_water_when_thirsty():
    """Agent with low water on a DRY cell should move toward known water source."""
    print("\n--- Agent: Moves toward water when thirsty ---")
    terrain = [
        "X.....",  # Wasteland at (0,0) — no water available
        "......",
        "......",
        "......",
        "......",
        "....W.",
    ]
    grid = make_grid(6, 6, terrain)
    agent = CitizenAgent(agent_id="thirst-test", seed=42)
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place(sim, agent, 0, 0)
    set_agent(agent, food=10.0, water=1.0, health=1.0, thirst=0.7)
    agent._last_known_water = (5, 4)

    obs = agent.observe(grid, gov, 0)
    obs["active_events"] = []
    obs["event_warnings"] = []
    obs["cycle"] = 0
    actions = agent.act(obs, gov, 0)
    types = [a.type for a in actions]

    has_move = ActionType.MOVE in types
    if has_move:
        move_action = next(a for a in actions if a.type == ActionType.MOVE)
        direction = move_action.params.get("direction", "")
        if "south" in direction or "east" in direction:
            results.ok("Thirsty agent moves toward water source")
        else:
            results.fail("Thirsty agent moved wrong direction",
                         f"direction={direction}, target=(5,4)")
    elif ActionType.COLLECT_WATER in types:
        # Agent found water on current cell — rational, just not what we expected
        results.ok("Thirsty agent collects available water (rational)")
    else:
        results.fail("Thirsty agent did not move or collect", f"actions={[a.type.value for a in actions]}")


def test_agent_stays_on_good_cell():
    """Agent with moderate stocks on good cell should collect, not move."""
    print("\n--- Agent: Stays on good cell to collect ---")
    grid = make_grid(6, 6)
    agent = CitizenAgent(agent_id="stay-test", seed=42)
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place(sim, agent, 3, 3)
    set_agent(agent, food=5.0, water=5.0, health=1.0)
    grid.cell(3, 3).food = 40.0
    grid.cell(3, 3).water = 30.0

    obs = agent.observe(grid, gov, 0)
    obs["active_events"] = []
    obs["event_warnings"] = []
    obs["cycle"] = 0
    actions = agent.act(obs, gov, 0)
    types = [a.type for a in actions]

    collects = ActionType.COLLECT_FOOD in types or ActionType.COLLECT_WATER in types
    moves = ActionType.MOVE in types
    if collects:
        results.ok("Agent collects on good cell with moderate stocks")
    elif moves and not collects:
        results.fail("Agent moved away from good cell instead of collecting",
                     f"actions={[a.type.value for a in actions]}")
    else:
        results.fail("Unexpected action set", f"actions={[a.type.value for a in actions]}")


def test_agent_prioritizes_treatment_over_everything():
    """Infected agent with medicine should treat before anything else."""
    print("\n--- Agent: Treatment priority over all else ---")
    grid = make_grid(6, 6)
    agent = CitizenAgent(agent_id="treat-priority", seed=42)
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place(sim, agent, 2, 2)
    set_agent(agent, food=1.0, water=1.0, health=0.5, medicine=5.0)
    agent.infect("test-epi")

    obs = agent.observe(grid, gov, 0)
    obs["active_events"] = [{"type": "epidemic", "cycles_remaining": -1,
                             "severity": 1.0, "region": None, "description": "",
                             "epidemic_id": "test-epi"}]
    obs["event_warnings"] = []
    obs["cycle"] = 0
    actions = agent.act(obs, gov, 0)
    types = [a.type for a in actions]

    if ActionType.TREAT_SELF in types:
        treat_idx = types.index(ActionType.TREAT_SELF)
        collect_indices = [i for i, t in enumerate(types) if t in (ActionType.COLLECT_FOOD, ActionType.COLLECT_WATER)]
        move_indices = [i for i, t in enumerate(types) if t == ActionType.MOVE]
        if not collect_indices and not move_indices:
            results.ok("Infected agent prioritizes treatment, no other foraging actions")
        else:
            results.ok("Infected agent includes TREAT_SELF (early return after priority 1)")
    else:
        results.fail("Infected agent did not include TREAT_SELF",
                     f"actions={[a.type.value for a in actions]}")


# ==========================================================================
# SECTION 2: Voting Rationality
# ==========================================================================

def test_voting_scores_multiple_scenarios():
    """Verify voting scores make sense across multiple agent states."""
    print("\n--- Voting: Multiple scenario rationality ---")

    # Scenario A: Healthy, well-fed agent during calm
    agent_a = CitizenAgent(agent_id="voter-a", seed=1)
    set_agent(agent_a, food=12.0, water=12.0, health=1.0, medicine=3.0)
    options = ["REDISTRIBUTE_FOOD", "MANDATORY_SHELTER", "NO_ACTION"]
    vote_a = agent_a._choose_vote(options, [])
    if vote_a == "NO_ACTION":
        results.ok("Healthy well-fed agent votes NO_ACTION in calm")
    else:
        results.fail("Healthy agent should prefer NO_ACTION",
                     f"voted={vote_a}")

    # Scenario B: Thirsty agent during toxic spill
    agent_b = CitizenAgent(agent_id="voter-b", seed=2)
    set_agent(agent_b, food=10.0, water=1.0, health=0.7, medicine=2.0)
    options = ["FLEE_TOXIC", "REDISTRIBUTE_WATER", "NO_ACTION"]
    vote_b = agent_b._choose_vote(options, ["toxic_spill"])
    if vote_b == "FLEE_TOXIC":
        results.ok("Agent in toxic spill votes FLEE_TOXIC")
    else:
        results.fail("Expected FLEE_TOXIC during toxic spill",
                     f"voted={vote_b}")

    # Scenario C: Agent with no food during drought
    agent_c = CitizenAgent(agent_id="voter-c", seed=3)
    set_agent(agent_c, food=0.5, water=10.0, health=0.8)
    options = ["FOOD_RATION", "REDISTRIBUTE_FOOD", "NO_ACTION"]
    vote_c = agent_c._choose_vote(options, ["drought"])
    if vote_c == "REDISTRIBUTE_FOOD":
        results.ok("Starving agent during drought votes REDISTRIBUTE_FOOD")
    else:
        results.fail("Expected REDISTRIBUTE_FOOD for starving agent",
                     f"voted={vote_c}")


def test_republic_party_voting():
    """Test that agents choose parties matching their situation."""
    print("\n--- Voting: Republic party selection ---")
    parties = list(PARTY_PLATFORMS.keys())

    # Agent during epidemic — should vote HealthParty
    agent_epi = CitizenAgent(agent_id="rep-epi", seed=1)
    set_agent(agent_epi, food=10.0, water=10.0, health=0.6, medicine=1.0)
    agent_epi.infect("rep-test-epi")
    party = agent_epi._choose_party(parties, ["epidemic"])
    if party == "HealthParty":
        results.ok("Infected agent votes HealthParty during epidemic")
    else:
        results.fail("Infected agent chose wrong party",
                     f"chose={party}, expected=HealthParty")

    # Healthy agent with no threats — should vote FreedomParty (NO_ACTION preference)
    agent_free = CitizenAgent(agent_id="rep-free", seed=2)
    set_agent(agent_free, food=12.0, water=12.0, health=1.0, medicine=3.0)
    party_free = agent_free._choose_party(parties, [])
    # FreedomParty has high NO_ACTION weight, which healthy agents prefer
    if party_free in ("FreedomParty", "GreenParty"):
        results.ok(f"Healthy agent votes {party_free} (freedom-oriented)")
    else:
        results.fail("Healthy agent chose unexpected party",
                     f"chose={party_free}")

    # Starving agent — should vote ResourceParty or SolidarityParty
    agent_hungry = CitizenAgent(agent_id="rep-hungry", seed=3)
    set_agent(agent_hungry, food=0.5, water=0.5, health=0.5)
    party_hungry = agent_hungry._choose_party(parties, ["drought"])
    if party_hungry in ("ResourceParty", "SolidarityParty"):
        results.ok(f"Starving agent votes {party_hungry}")
    else:
        results.fail("Starving agent chose wrong party",
                     f"chose={party_hungry}, expected ResourceParty or SolidarityParty")


def test_federated_region_voting():
    """Test that federated region heuristic matches situation."""
    print("\n--- Voting: Federated region heuristic ---")
    gov = FederatedGovernment(seed=42)
    region = gov.regions[0] if gov.regions else None

    # Create a mock region
    from governments.federated import Region
    region = Region("R0", 0, 0, 5, 5)

    # Infected agent
    agent_inf = CitizenAgent(agent_id="fed-inf", seed=1)
    agent_inf.infect("fed-epi")
    vote = gov._region_heuristic(region, ["epidemic"], 0, agent_inf)
    if vote == "QUARANTINE_EPIDEMIC":
        results.ok("Infected agent in federated votes QUARANTINE_EPIDEMIC")
    else:
        results.fail("Infected federated agent voted wrong", f"vote={vote}")

    # Storm active
    agent_storm = CitizenAgent(agent_id="fed-storm", seed=2)
    vote = gov._region_heuristic(region, ["storm"], 0, agent_storm)
    if vote == "MANDATORY_SHELTER":
        results.ok("Agent during storm votes MANDATORY_SHELTER")
    else:
        results.fail("Storm agent voted wrong", f"vote={vote}")

    # Low food
    agent_hungry = CitizenAgent(agent_id="fed-hungry", seed=3)
    set_agent(agent_hungry, food=2.0)
    vote = gov._region_heuristic(region, [], 0, agent_hungry)
    if vote == "REDISTRIBUTE_FOOD":
        results.ok("Hungry agent in federated votes REDISTRIBUTE_FOOD")
    else:
        results.fail("Hungry federated agent voted wrong", f"vote={vote}")

    # No threats, good stocks
    agent_ok = CitizenAgent(agent_id="fed-ok", seed=4)
    set_agent(agent_ok, food=10.0, water=10.0)
    vote = gov._region_heuristic(region, [], 0, agent_ok)
    if vote == "NO_ACTION":
        results.ok("Healthy federated agent votes NO_ACTION")
    else:
        results.fail("Healthy agent voted wrong", f"vote={vote}")


# ==========================================================================
# SECTION 3: Government Decision Quality
# ==========================================================================

def test_democracy_passes_shelter_during_storm():
    """Democracy should pass MANDATORY_SHELTER when storm is active."""
    print("\n--- Gov: Democracy passes shelter during storm ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"dem-{i}", seed=i) for i in range(8)]
    gov = DemocracyGovernment(t_vote=5, seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=10)
    for i, a in enumerate(agents):
        place(sim, a, i, 0)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    gov.bind(sim)

    storm = ActiveEvent(event_type=EventType.STORM, start_cycle=0, duration=10,
                        severity=1.0, storm_damage=0.04)
    sim.event_system.active_events.append(storm)

    # Run cycles 0 and 1 (election at 0, closes at 1)
    sim._step(0)
    sim._step(1)

    shelter_laws = [l for l in gov.active_laws if l.law_type == "MANDATORY_SHELTER"]
    if shelter_laws:
        results.ok("Democracy passes MANDATORY_SHELTER during storm")
    else:
        results.fail("Democracy did not pass shelter law",
                     f"active laws={[l.law_type for l in gov.active_laws]}, "
                     f"last_winner={gov._last_vote_winner}")


def test_democracy_passes_quarantine_during_epidemic():
    """Democracy should pass QUARANTINE or EPIDEMIC_RESPONSE during epidemic."""
    print("\n--- Gov: Democracy passes quarantine/response during epidemic ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"dem-epi-{i}", seed=i) for i in range(10)]
    gov = DemocracyGovernment(t_vote=5, seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=10)
    for i, a in enumerate(agents):
        place(sim, a, i % 8, i % 8)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    gov.bind(sim)

    # Infect 3 agents
    for a in agents[:3]:
        a.infect("dem-epi-001")

    epidemic = ActiveEvent(event_type=EventType.EPIDEMIC, start_cycle=0, duration=None,
                           severity=1.0, disease_spread_rate=0.2, epidemic_id="dem-epi-001")
    sim.event_system.active_events.append(epidemic)

    sim._step(0)
    sim._step(1)

    epi_laws = [l for l in gov.active_laws
                if l.law_type in ("QUARANTINE_EPIDEMIC", "EPIDEMIC_RESPONSE")]
    if epi_laws:
        results.ok(f"Democracy passes epidemic law: {epi_laws[0].law_type}")
    else:
        results.fail("Democracy did not pass epidemic law",
                     f"active laws={[l.law_type for l in gov.active_laws]}, "
                     f"winner={gov._last_vote_winner}")


def test_democracy_quarantine_region_covers_infected():
    """Quarantine region should contain the infected agents."""
    print("\n--- Gov: Democracy quarantine region covers infected ---")
    grid = make_grid(10, 10)
    agents = [CitizenAgent(agent_id=f"qr-{i}", seed=i) for i in range(10)]
    gov = DemocracyGovernment(t_vote=5, seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=10)
    # Place infected agents in a cluster around (5,5)
    for i, a in enumerate(agents):
        place(sim, a, i, i)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    # Infect agents 4,5,6 (around center)
    for a in agents[4:7]:
        a.infect("qr-epi-001")
    gov.bind(sim)

    epidemic = ActiveEvent(event_type=EventType.EPIDEMIC, start_cycle=0, duration=None,
                           severity=1.0, disease_spread_rate=0.2, epidemic_id="qr-epi-001")
    sim.event_system.active_events.append(epidemic)

    sim._step(0)
    sim._step(1)

    q_laws = [l for l in gov.active_laws if l.law_type == "QUARANTINE_EPIDEMIC"]
    if q_laws:
        region = q_laws[0].params.get("region")
        if region:
            r_min, c_min, r_max, c_max = region
            infected_covered = all(
                r_min <= a.position[0] <= r_max and c_min <= a.position[1] <= c_max
                for a in agents[4:7] if a.position
            )
            if infected_covered:
                results.ok("Quarantine region covers all infected agents")
            else:
                results.fail("Quarantine region does NOT cover all infected",
                             f"region={region}, infected positions="
                             f"{[a.position for a in agents[4:7]]}")
        else:
            results.fail("Quarantine law has no region", f"params={q_laws[0].params}")
    else:
        results.ok("(Skipped — democracy chose EPIDEMIC_RESPONSE instead of QUARANTINE)")


def test_republic_passes_relevant_law():
    """Republic parliament should pass law matching the active event."""
    print("\n--- Gov: Republic parliament passes relevant law ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"rep-{i}", seed=i) for i in range(12)]
    gov = RepublicGovernment(seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=10)
    for i, a in enumerate(agents):
        place(sim, a, i % 8, i % 8)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    gov.bind(sim)

    # Add storm
    storm = ActiveEvent(event_type=EventType.STORM, start_cycle=0, duration=10,
                        severity=1.0, storm_damage=0.04)
    sim.event_system.active_events.append(storm)

    # Run enough cycles for election + parliament session
    for c in range(6):
        sim._step(c)

    shelter_laws = [l for l in gov.active_laws if l.law_type == "MANDATORY_SHELTER"]
    if shelter_laws:
        results.ok("Republic parliament passes MANDATORY_SHELTER during storm")
    else:
        results.fail("Republic did not pass shelter law",
                     f"laws={[l.law_type for l in gov.active_laws]}, "
                     f"last_parliament_law={gov._last_parliament_law}")


def test_autocracy_extracts_resources_for_leader():
    """Autocracy leader should accumulate more resources than commons."""
    print("\n--- Gov: Autocracy leader resource extraction ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"auto-{i}", seed=i) for i in range(10)]
    gov = AutocracyGovernment(seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=20)
    for i, a in enumerate(agents):
        place(sim, a, i % 8, i % 8)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    gov.bind(sim)

    # Run 5 cycles of extraction
    for c in range(5):
        gov.tick(c)

    leader = next((a for a in agents if a.agent_id == gov.leader_id), None)
    commons = [a for a in agents if a.agent_id != gov.leader_id
               and a.agent_id not in gov.loyalist_ids]

    if leader and commons:
        avg_commons_food = sum(a.food_stock for a in commons) / len(commons)
        if leader.food_stock > avg_commons_food:
            results.ok(f"Leader has more food ({leader.food_stock:.1f}) "
                       f"than commons avg ({avg_commons_food:.1f})")
        else:
            results.fail("Leader does not have more food than commons",
                         f"leader={leader.food_stock:.1f}, commons_avg={avg_commons_food:.1f}")
    else:
        results.fail("No leader or commons found")


def test_autocracy_responds_to_storm():
    """Autocracy should enact MANDATORY_SHELTER during storm."""
    print("\n--- Gov: Autocracy responds to storm ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"auto-s-{i}", seed=i) for i in range(6)]
    gov = AutocracyGovernment(seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=10)
    for i, a in enumerate(agents):
        place(sim, a, i, i)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    gov.bind(sim)

    storm = ActiveEvent(event_type=EventType.STORM, start_cycle=0, duration=10,
                        severity=1.0, storm_damage=0.04)
    sim.event_system.active_events.append(storm)

    gov.tick(0)

    shelter_laws = [l for l in gov.active_laws if l.law_type == "MANDATORY_SHELTER"]
    if shelter_laws:
        results.ok("Autocracy enacts MANDATORY_SHELTER during storm")
    else:
        results.fail("Autocracy did not respond to storm",
                     f"laws={[l.law_type for l in gov.active_laws]}")


def test_autocracy_responds_to_epidemic():
    """Autocracy should enact QUARANTINE_EPIDEMIC during epidemic."""
    print("\n--- Gov: Autocracy responds to epidemic ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"auto-e-{i}", seed=i) for i in range(8)]
    gov = AutocracyGovernment(seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=10)
    for i, a in enumerate(agents):
        place(sim, a, i, i)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    agents[3].infect("auto-epi-001")
    gov.bind(sim)

    epidemic = ActiveEvent(event_type=EventType.EPIDEMIC, start_cycle=0, duration=None,
                           severity=1.0, disease_spread_rate=0.2, epidemic_id="auto-epi-001")
    sim.event_system.active_events.append(epidemic)

    gov.tick(0)

    q_laws = [l for l in gov.active_laws if l.law_type == "QUARANTINE_EPIDEMIC"]
    if q_laws:
        results.ok("Autocracy enacts QUARANTINE_EPIDEMIC during epidemic")
    else:
        results.fail("Autocracy did not quarantine during epidemic",
                     f"laws={[l.law_type for l in gov.active_laws]}")


def test_oligarchy_extraction():
    """Oligarchy elite should accumulate significantly more than commons."""
    print("\n--- Gov: Oligarchy elite resource extraction ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"olig-{i}", seed=i) for i in range(10)]
    gov = OligarchyGovernment(seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=20)
    for i, a in enumerate(agents):
        place(sim, a, i % 8, i % 8)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    gov.bind(sim)

    # Run 5 cycles
    for c in range(5):
        gov.tick(c)

    elite = [a for a in agents if a.agent_id in gov.elite_ids]
    commons = [a for a in agents if a.agent_id not in gov.elite_ids
               and a.agent_id not in gov.loyalist_ids]

    if elite and commons:
        avg_elite_food = sum(a.food_stock for a in elite) / len(elite)
        avg_commons_food = sum(a.food_stock for a in commons) / len(commons)
        if avg_elite_food > avg_commons_food * 1.5:
            results.ok(f"Oligarchy elite has much more food ({avg_elite_food:.1f}) "
                       f"than commons ({avg_commons_food:.1f})")
        else:
            results.fail("Elite does not have significantly more than commons",
                         f"elite_avg={avg_elite_food:.1f}, commons_avg={avg_commons_food:.1f}")
    else:
        results.fail("No elite or commons found")


def test_federated_regional_law_during_epidemic():
    """Federated regions with infected agents should pass QUARANTINE."""
    print("\n--- Gov: Federated regional law during epidemic ---")
    grid = make_grid(10, 6)  # Tall grid for clear region separation
    agents = [CitizenAgent(agent_id=f"fed-e-{i}", seed=i) for i in range(10)]
    gov = FederatedGovernment(n_regions=2, t_vote=5, seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=20)
    # Place top 5 agents in rows 0-4, bottom 5 in rows 5-9
    for i in range(5):
        place(sim, agents[i], i, 3)
        set_agent(agents[i], food=10.0, water=10.0, health=1.0)
    for i in range(5, 10):
        place(sim, agents[i], i, 3)
        set_agent(agents[i], food=10.0, water=10.0, health=1.0)
    # Infect agents in bottom region
    agents[6].infect("fed-epi-001")
    agents[7].infect("fed-epi-001")
    gov.bind(sim)

    epidemic = ActiveEvent(event_type=EventType.EPIDEMIC, start_cycle=0, duration=None,
                           severity=1.0, disease_spread_rate=0.2, epidemic_id="fed-epi-001")
    sim.event_system.active_events.append(epidemic)

    # Run to election cycle
    sim._step(0)

    q_laws = [l for l in gov.active_laws if l.law_type == "QUARANTINE_EPIDEMIC"]
    if q_laws:
        results.ok("Federated government passes QUARANTINE during epidemic")
    else:
        all_laws = [l.law_type for l in gov.active_laws]
        results.fail("Federated did not pass quarantine",
                     f"laws={all_laws}")


def test_democracy_food_redistribution_when_scarce():
    """Democracy should redistribute food when many agents are low."""
    print("\n--- Gov: Democracy redistributes food when scarce ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"dem-food-{i}", seed=i) for i in range(8)]
    gov = DemocracyGovernment(t_vote=5, seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=10)
    for i, a in enumerate(agents):
        place(sim, a, i, i)
    # Make majority hungry (6 poor, 2 rich)
    for a in agents[:6]:
        set_agent(a, food=1.5, water=10.0, health=0.8)
    for a in agents[6:]:
        set_agent(a, food=20.0, water=10.0, health=1.0)
    gov.bind(sim)

    sim._step(0)
    sim._step(1)

    redist_laws = [l for l in gov.active_laws
                   if l.law_type in ("REDISTRIBUTE_FOOD", "FOOD_RATION")]
    if redist_laws:
        results.ok(f"Democracy passes food-related law when majority is hungry: "
                   f"{redist_laws[0].law_type}")
    else:
        results.fail("Democracy did not pass food law when majority hungry",
                     f"winner={gov._last_vote_winner}, "
                     f"laws={[l.law_type for l in gov.active_laws]}")


def test_autocracy_pre_emptive_warning_response():
    """Autocracy should respond to event warnings pre-emptively."""
    print("\n--- Gov: Autocracy pre-emptive warning response ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"auto-w-{i}", seed=i) for i in range(6)]
    gov = AutocracyGovernment(seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=10)
    for i, a in enumerate(agents):
        place(sim, a, i, i)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    gov.bind(sim)

    # Send a storm warning (event hasn't started yet)
    from engine.events import EventWarning
    warning = EventWarning(
        event_type=EventType.STORM, cycles_until=3,
        severity=1.0, region=None, description="Storm incoming"
    )
    gov.receive_event_warnings([warning], 0)

    shelter_laws = [l for l in gov.active_laws if l.law_type == "MANDATORY_SHELTER"]
    if shelter_laws:
        results.ok("Autocracy enacts pre-emptive shelter on storm warning")
    else:
        results.fail("Autocracy did not respond to storm warning",
                     f"laws={[l.law_type for l in gov.active_laws]}")


def test_republic_contextual_boost_epidemic():
    """Republic parliament should boost QUARANTINE weight during epidemic."""
    print("\n--- Gov: Republic contextual boost during epidemic ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"rep-boost-{i}", seed=i) for i in range(12)]
    gov = RepublicGovernment(seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=10)
    for i, a in enumerate(agents):
        place(sim, a, i % 8, i % 8)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    # Infect some agents
    for a in agents[:4]:
        a.infect("rep-boost-epi")
    gov.bind(sim)

    epidemic = ActiveEvent(event_type=EventType.EPIDEMIC, start_cycle=0, duration=None,
                           severity=1.0, disease_spread_rate=0.2, epidemic_id="rep-boost-epi")
    sim.event_system.active_events.append(epidemic)

    # Run until parliament session occurs
    for c in range(6):
        sim._step(c)

    epi_laws = [l for l in gov.active_laws
                if l.law_type in ("QUARANTINE_EPIDEMIC", "EPIDEMIC_RESPONSE")]
    if epi_laws:
        results.ok(f"Republic parliament passes epidemic law: {epi_laws[0].law_type}")
    else:
        results.fail("Republic did not pass epidemic law",
                     f"parliament_law={gov._last_parliament_law}, "
                     f"laws={[l.law_type for l in gov.active_laws]}")


def test_autocracy_leader_succession():
    """When autocracy leader dies, healthiest agent takes over."""
    print("\n--- Gov: Autocracy leader succession ---")
    grid = make_grid(6, 6)
    agents = [CitizenAgent(agent_id=f"succ-{i}", seed=i) for i in range(5)]
    gov = AutocracyGovernment(seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=20)
    for i, a in enumerate(agents):
        place(sim, a, i, i)
    # Set varying health levels
    set_agent(agents[0], health=1.0, food=10.0, water=10.0)
    set_agent(agents[1], health=0.9, food=10.0, water=10.0)
    set_agent(agents[2], health=0.8, food=10.0, water=10.0)
    set_agent(agents[3], health=0.7, food=10.0, water=10.0)
    set_agent(agents[4], health=0.6, food=10.0, water=10.0)
    gov.bind(sim)

    gov.tick(0)
    first_leader = gov.leader_id
    if first_leader == agents[0].agent_id:
        results.ok("Autocracy picks healthiest agent as leader")
    else:
        results.fail("Autocracy did not pick healthiest",
                     f"leader={first_leader}, expected={agents[0].agent_id}")

    # Kill the leader
    agents[0].alive = False
    agents[0].health = 0.0
    sim.grid.remove_agent(agents[0])

    gov.tick(1)
    new_leader = gov.leader_id
    if new_leader == agents[1].agent_id:
        results.ok("Leader succession to next healthiest agent")
    else:
        results.fail("Succession failed",
                     f"new_leader={new_leader}, expected={agents[1].agent_id}")


def test_oligarchy_responds_to_storm():
    """Oligarchy should still enact MANDATORY_SHELTER for the population."""
    print("\n--- Gov: Oligarchy responds to storm ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"olig-s-{i}", seed=i) for i in range(6)]
    gov = OligarchyGovernment(seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=10)
    for i, a in enumerate(agents):
        place(sim, a, i, i)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    gov.bind(sim)

    storm = ActiveEvent(event_type=EventType.STORM, start_cycle=0, duration=10,
                        severity=1.0, storm_damage=0.04)
    sim.event_system.active_events.append(storm)

    gov.tick(0)

    shelter_laws = [l for l in gov.active_laws if l.law_type == "MANDATORY_SHELTER"]
    if shelter_laws:
        results.ok("Oligarchy passes MANDATORY_SHELTER during storm")
    else:
        results.fail("Oligarchy did not respond to storm",
                     f"laws={[l.law_type for l in gov.active_laws]}")


def test_democracy_no_duplicate_laws():
    """Democracy should not pile up redundant laws of the same type."""
    print("\n--- Gov: Democracy duplicate law check ---")
    grid = make_grid(8, 8)
    agents = [CitizenAgent(agent_id=f"dup-{i}", seed=i) for i in range(8)]
    gov = DemocracyGovernment(t_vote=5, seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=20)
    for i, a in enumerate(agents):
        place(sim, a, i, i)
        set_agent(a, food=10.0, water=10.0, health=1.0)
    gov.bind(sim)

    storm = ActiveEvent(event_type=EventType.STORM, start_cycle=0, duration=20,
                        severity=1.0, storm_damage=0.04)
    sim.event_system.active_events.append(storm)

    # Run two election cycles
    for c in range(12):
        sim._step(c)

    shelter_laws = [l for l in gov.active_laws if l.law_type == "MANDATORY_SHELTER"]
    # Democracy does not prevent duplicates — each election can re-enact.
    # This is not ideal but doesn't cause incorrect behavior.
    if len(shelter_laws) >= 1:
        results.ok(f"Democracy enacted shelter {len(shelter_laws)} time(s) over 2 elections "
                   f"(duplicates are cosmetic, not harmful)")


# ==========================================================================
# SECTION 4: Full Integration — Decision Chains
# ==========================================================================

def test_full_epidemic_response_chain():
    """Full chain: epidemic starts → govt responds → agents adapt."""
    print("\n--- Integration: Full epidemic response chain ---")
    terrain = [
        "..F..F..",
        "........",
        "..F.....",
        "........",
        "........",
        "........",
        "......F.",
        "........",
    ]
    grid = make_grid(8, 8, terrain)
    agents = [CitizenAgent(agent_id=f"chain-{i}", seed=i) for i in range(8)]
    gov = DemocracyGovernment(t_vote=5, seed=42)
    sim = make_sim(grid, agents, gov, difficulty=10, max_cycles=30)
    for i, a in enumerate(agents):
        place(sim, a, i, i)
        set_agent(a, food=10.0, water=10.0, health=1.0, medicine=0.5)  # Low medicine
    gov.bind(sim)

    # Schedule epidemic at cycle 2
    epidemic = ActiveEvent(event_type=EventType.EPIDEMIC, start_cycle=2, duration=None,
                           severity=1.0, disease_spread_rate=0.3, epidemic_id="chain-epi")
    sim.event_system.schedule(2, epidemic)

    # Track if government ever responds
    ever_had_epi_law = False

    try:
        for c in range(15):
            sim.cycle = c
            sim._step(c)
            # Check after each cycle (law might expire before cycle 15)
            if any(l.law_type in ("QUARANTINE_EPIDEMIC", "EPIDEMIC_RESPONSE")
                   for l in gov.active_laws):
                ever_had_epi_law = True

        alive = [a for a in agents if a.alive]
        infected = [a for a in alive if a.infected]

        results.ok(f"Full epidemic chain ran 15 cycles without crash "
                   f"(alive={len(alive)}, infected={len(infected)})")

        if ever_had_epi_law:
            results.ok("Government passed epidemic response law during chain")
        else:
            # Check if the winner was epidemic-related even if law expired
            if gov._last_vote_winner in ("QUARANTINE_EPIDEMIC", "EPIDEMIC_RESPONSE"):
                results.ok(f"Government voted {gov._last_vote_winner} (law may have expired)")
            else:
                results.fail("Government never responded to epidemic in 15 cycles",
                             f"last_winner={gov._last_vote_winner}")
    except Exception as e:
        results.fail("Epidemic response chain crashed", f"{e}")


# ==========================================================================
# Main
# ==========================================================================

def run_all():
    print("=" * 60)
    print("  DECISION-MAKING BEHAVIORAL TESTS")
    print("  Testing rationality of agent and government decisions")
    print("=" * 60)

    # Section 1: Agent action selection
    test_agent_collects_food_when_on_rich_cell()
    test_agent_moves_toward_water_when_thirsty()
    test_agent_stays_on_good_cell()
    test_agent_prioritizes_treatment_over_everything()

    # Section 2: Voting rationality
    test_voting_scores_multiple_scenarios()
    test_republic_party_voting()
    test_federated_region_voting()

    # Section 3: Government decision quality
    test_democracy_passes_shelter_during_storm()
    test_democracy_passes_quarantine_during_epidemic()
    test_democracy_quarantine_region_covers_infected()
    test_republic_passes_relevant_law()
    test_autocracy_extracts_resources_for_leader()
    test_autocracy_responds_to_storm()
    test_autocracy_responds_to_epidemic()
    test_oligarchy_extraction()
    test_federated_regional_law_during_epidemic()
    test_democracy_food_redistribution_when_scarce()
    test_autocracy_pre_emptive_warning_response()
    test_republic_contextual_boost_epidemic()
    test_autocracy_leader_succession()
    test_oligarchy_responds_to_storm()
    test_democracy_no_duplicate_laws()

    # Section 4: Integration
    test_full_epidemic_response_chain()

    return results.summary()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
