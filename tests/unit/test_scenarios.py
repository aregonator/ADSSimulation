#!/usr/bin/env python3
"""
Test scenarios for the Decision-Making Survival Simulation.

Runs small-grid simulations with controlled setups to verify agent behavior,
government mechanics, event handling, and resource management.

Usage (from the Simulation project root):
    python3 tests/unit/test_scenarios.py
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


# ==========================================================================
# Helpers
# ==========================================================================

def make_grid_6x6(terrain_map=None):
    """Create a 6x6 grid with controlled terrain.
    terrain_map: list of 6 strings, each 6 chars. '.'=plains, 'F'=forest, 'W'=water, 'M'=mountain, 'X'=wasteland
    """
    grid = Grid(rows=6, cols=6, resource_density=1.0, terrain_variety=0.0, seed=42,
                ambient_hazard=0.0)
    # terrain_variety=0 makes all PLAINS. Now override terrain manually.
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
                # Reset resources based on terrain
                if terrain == Terrain.WASTELAND:
                    cell.food = cell.water = cell.medicine = 0.0
                elif terrain == Terrain.WATER:
                    cell.food = 0.0
                    cell.water = 90.0
                    cell.medicine = 10.0
                elif terrain == Terrain.FOREST:
                    cell.food = 80.0
                    cell.water = 40.0
                    cell.medicine = 15.0
                elif terrain == Terrain.MOUNTAIN:
                    cell.food = 20.0
                    cell.water = 20.0
                    cell.medicine = 10.0
                else:  # PLAINS
                    cell.food = 50.0
                    cell.water = 35.0
                    cell.medicine = 10.0
    return grid


def make_agent(agent_id, health=1.0, food=10.0, water=10.0, medicine=2.0, seed=None):
    """Create a CitizenAgent with specified stats."""
    a = CitizenAgent(agent_id=agent_id, seed=seed or 42)
    a.health = health
    a.food_stock = food
    a.water_stock = water
    a.medicine_stock = medicine
    return a


def make_sim(grid, agents, government, difficulty=1, max_cycles=20, seed=42):
    """Create a Simulation with pre-built grid and agents placed manually.

    NOTE: Simulation.__init__ calls _place_agents() which resets agent health,
    food_stock, and water_stock to config defaults. Callers must set agent
    stats AFTER calling make_sim + place_agent.
    """
    config = SimulationConfig(
        grid_rows=grid.rows,
        grid_cols=grid.cols,
        resource_density=1.0,
        terrain_variety=0.0,
        num_agents=len(agents),
        initial_health=1.0,
        difficulty=difficulty,
        event_frequency=0.0,
        event_severity=1.0,
        event_warning_cycles=5,
        visibility_radius=5,
        max_steps_per_cycle=1,
        seed=seed,
        max_cycles=max_cycles,
    )
    sim = Simulation(config=config, government=government, agents=agents)
    # Remove agents from the auto-generated grid (we'll re-place on custom grid)
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


def set_agent(agent, health=None, food=None, water=None, medicine=None,
              hunger=None, thirst=None):
    """Set agent stats after make_sim (which resets them via _place_agents)."""
    if health is not None:
        agent.health = health
    if food is not None:
        agent.food_stock = food
    if water is not None:
        agent.water_stock = water
    if medicine is not None:
        agent.medicine_stock = medicine
    if hunger is not None:
        agent.hunger = hunger
    if thirst is not None:
        agent.thirst = thirst


def place_agent(sim, agent, r, c):
    """Place an agent at specific grid coordinates."""
    sim.grid.place_agent(agent, r, c)


# ==========================================================================
# Test Results Tracking
# ==========================================================================

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
# TEST 1: Grid Boundary Enforcement
# ==========================================================================

def test_grid_boundary():
    print("\n--- Test 1: Grid Boundary Enforcement ---")
    grid = make_grid_6x6()
    agent = make_agent("boundary-1")
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 0, 0)

    # Try moving north (out of bounds)
    sim._do_move(agent, "north")
    if agent.position == (0, 0):
        results.ok("Agent blocked from moving north at row 0")
    else:
        results.fail("Agent moved north at row 0", f"position={agent.position}")

    # Try moving west (out of bounds)
    sim._do_move(agent, "west")
    if agent.position == (0, 0):
        results.ok("Agent blocked from moving west at col 0")
    else:
        results.fail("Agent moved west at col 0", f"position={agent.position}")

    # Move south (valid)
    sim._do_move(agent, "south")
    if agent.position == (1, 0):
        results.ok("Agent moved south successfully")
    else:
        results.fail("Agent failed to move south", f"position={agent.position}")

    # Move to corner (5,5) and try southeast
    place_agent(sim, agent, 5, 5)
    sim._do_move(agent, "southeast")
    if agent.position == (5, 5):
        results.ok("Agent blocked from moving southeast at (5,5)")
    else:
        results.fail("Agent moved southeast at (5,5)", f"position={agent.position}")


# ==========================================================================
# TEST 2: Movement Cost
# ==========================================================================

def test_movement_cost():
    print("\n--- Test 2: Movement Cost ---")
    grid = make_grid_6x6()
    agent = make_agent("cost-1", health=1.0)
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)

    h_before = agent.health
    sim._do_move(agent, "east")
    h_after = agent.health

    cost = h_before - h_after
    if abs(cost - 0.002) < 0.0001:
        results.ok("Movement cost is 0.002")
    else:
        results.fail("Movement cost incorrect", f"expected=0.002, got={cost:.6f}")


# ==========================================================================
# TEST 3: Storm / Shelter Behavior
# ==========================================================================

def test_storm_shelter():
    print("\n--- Test 3: Storm and Shelter ---")
    # Grid: agent on PLAINS (0,0), FOREST (shelter) at (0,2)
    terrain = [
        "..F...",
        "......",
        "......",
        "......",
        "......",
        "......",
    ]
    grid = make_grid_6x6(terrain)
    agent = make_agent("storm-1", health=1.0, food=10.0, water=10.0)
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 0, 0)

    # Inject active storm event
    storm = ActiveEvent(
        event_type=EventType.STORM, start_cycle=0, duration=10,
        severity=1.0, storm_damage=0.04,
    )
    sim.event_system.active_events.append(storm)

    # Run one step
    sim._step(0)

    # Agent should have moved toward the shelter at (0,2)
    if agent.position and agent.position[1] > 0:
        results.ok("Agent moved toward shelter during storm",
                   f"pos={agent.position}")
    else:
        results.fail("Agent did not move toward shelter during storm",
                     f"pos={agent.position}")

    # Test storm damage on unsheltered agent
    agent2 = make_agent("storm-2", health=1.0)
    grid2 = make_grid_6x6()  # all plains, no shelter
    gov2 = AnarchyGovernment()
    sim2 = make_sim(grid2, [agent2], gov2)
    place_agent(sim2, agent2, 3, 3)
    storm2 = ActiveEvent(
        event_type=EventType.STORM, start_cycle=0, duration=10,
        severity=1.0, storm_damage=0.04,
    )
    sim2.event_system.active_events.append(storm2)
    h_before = agent2.health
    sim2._apply_health_dynamics(0)
    h_after = agent2.health
    damage = h_before - h_after
    # At difficulty=1, drain_mult=0.10. Storm damage = 0.04 * 0.10 = 0.004
    # Plus terrain hazard for PLAINS = 0.00 * 0.10 = 0.0
    expected_damage = 0.04 * difficulty_multiplier(1)
    if abs(damage - expected_damage) < 0.001:
        results.ok("Storm damage correct for unsheltered agent",
                   f"damage={damage:.6f}, expected={expected_damage:.6f}")
    else:
        results.fail("Storm damage incorrect",
                     f"damage={damage:.6f}, expected={expected_damage:.6f}")

    # Test sheltered agent is protected
    terrain3 = [
        "......",
        "......",
        "......",
        "...F..",
        "......",
        "......",
    ]
    grid3 = make_grid_6x6(terrain3)
    agent3 = make_agent("storm-3", health=1.0)
    gov3 = AnarchyGovernment()
    sim3 = make_sim(grid3, [agent3], gov3)
    place_agent(sim3, agent3, 3, 3)  # On FOREST (shelter)
    storm3 = ActiveEvent(
        event_type=EventType.STORM, start_cycle=0, duration=10,
        severity=1.0, storm_damage=0.04,
    )
    sim3.event_system.active_events.append(storm3)
    h_before = agent3.health
    sim3._apply_health_dynamics(0)
    h_after = agent3.health
    damage = h_before - h_after
    # Only terrain hazard for FOREST = 0.01 * drain_mult
    forest_hazard = 0.01 * difficulty_multiplier(1)
    if abs(damage - forest_hazard) < 0.001:
        results.ok("Sheltered agent protected from storm damage",
                   f"damage={damage:.6f} (forest hazard only)")
    else:
        results.fail("Sheltered agent took storm damage",
                     f"damage={damage:.6f}, expected_only_hazard={forest_hazard:.6f}")


# ==========================================================================
# TEST 4: MANDATORY_SHELTER Bug Check
# ==========================================================================

def test_mandatory_shelter_bug():
    print("\n--- Test 4: MANDATORY_SHELTER Law (Bug Check) ---")
    terrain = [
        "....F.",
        "......",
        "......",
        "......",
        "......",
        "......",
    ]
    grid = make_grid_6x6(terrain)
    agent = make_agent("shelter-bug-1", health=1.0)
    gov = DemocracyGovernment(seed=42)
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 0, 0)
    gov.bind(sim)

    # Enact MANDATORY_SHELTER law
    gov._enact_law("MANDATORY_SHELTER", {}, 0, duration=20,
                   description="Test mandatory shelter")

    # Agent at (0,0) PLAINS wants to move east toward shelter at (0,4) FOREST.
    # First step would be to (0,1) which is PLAINS (not shelter).
    can_move_01 = gov.can_move(agent, 0, 1, 0)  # (0,1) is PLAINS
    can_move_04 = gov.can_move(agent, 0, 4, 0)  # (0,4) is FOREST (shelter)

    if not can_move_01:
        results.fail("BUG CONFIRMED: MANDATORY_SHELTER blocks movement to non-shelter cell (0,1)",
                     "Agent cannot walk TOWARD shelter through non-shelter cells")
    else:
        results.ok("MANDATORY_SHELTER allows movement to non-shelter cell (0,1)")

    if can_move_04:
        results.ok("MANDATORY_SHELTER allows movement to shelter cell (0,4)")
    else:
        results.fail("MANDATORY_SHELTER blocks movement to shelter cell (0,4)",
                     "Even shelter cells are blocked?!")


# ==========================================================================
# TEST 5: Quarantine Bug Check
# ==========================================================================

def test_quarantine_bug():
    print("\n--- Test 5: QUARANTINE_EPIDEMIC Law (Bug Check) ---")
    grid = make_grid_6x6()
    agent_infected = make_agent("q-infected", health=0.8)
    agent_healthy = make_agent("q-healthy", health=1.0)
    gov = DemocracyGovernment(seed=42)
    sim = make_sim(grid, [agent_infected, agent_healthy], gov)
    place_agent(sim, agent_infected, 3, 3)  # Inside region (2,2)-(4,4)
    place_agent(sim, agent_healthy, 0, 0)   # Outside region
    gov.bind(sim)

    # Infect the agent
    agent_infected.infect("test-epidemic-001")

    # Enact quarantine with region (2,2)-(4,4) — no applies_to (affects all)
    gov._enact_law("QUARANTINE_EPIDEMIC", {"region": (2, 2, 4, 4)}, 0,
                   duration=20, event_id="test-epidemic-001",
                   description="Test quarantine")

    # Check: Can infected agent (inside) move OUT of region?
    # Agent at (3,3), try to move to (1,3) which is outside
    can_infected_leave = gov.can_move(agent_infected, 1, 3, 0)
    if not can_infected_leave:
        results.ok("Infected agent inside quarantine blocked from leaving")
    else:
        results.fail("Infected agent can leave quarantine zone",
                     "Quarantine containment failed")

    # Check: Can healthy agent (outside) move INTO the region?
    # Agent at (0,0), try to move to (2,2) which is inside
    can_healthy_enter = gov.can_move(agent_healthy, 2, 2, 0)
    if not can_healthy_enter:
        results.ok("Healthy agent blocked from entering quarantine zone")
    else:
        results.fail("BUG CONFIRMED: Healthy agent CAN enter quarantine zone",
                     "applies_to=None makes all agents 'quarantined', else branch unreachable")

    # Check: If a healthy agent is trapped inside quarantine, can it leave?
    agent_trapped = make_agent("q-trapped", health=1.0)
    sim.agents.append(agent_trapped)
    sim.agent_map[agent_trapped.agent_id] = agent_trapped
    place_agent(sim, agent_trapped, 3, 4)  # Inside region

    can_trapped_leave = gov.can_move(agent_trapped, 1, 4, 0)
    if not can_trapped_leave:
        results.fail("BUG CONFIRMED: Healthy agent TRAPPED inside quarantine cannot leave",
                     "applies_to=None treats ALL agents as quarantined")
    else:
        results.ok("Healthy agent inside quarantine can leave")


# ==========================================================================
# TEST 6: Voluntary Sharing Bug Check
# ==========================================================================

def test_voluntary_sharing_bug():
    print("\n--- Test 6: Voluntary Medicine Sharing (Bug Check) ---")
    grid = make_grid_6x6()
    agent_giver = make_agent("giver", health=1.0, medicine=5.0)
    agent_receiver = make_agent("receiver", health=0.5, medicine=0.0)
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent_giver, agent_receiver], gov)
    place_agent(sim, agent_giver, 3, 3)
    place_agent(sim, agent_receiver, 3, 4)

    # Infect receiver so giver wants to share medicine
    agent_receiver.infect("vol-share-epidemic")

    # Create the share action directly (as the agent would)
    share_action = Action(ActionType.SHARE_MEDICINE, {
        "target_id": agent_receiver.agent_id, "amount": 3.0
    })

    # Execute it through the simulation
    med_before_giver = agent_giver.medicine_stock
    med_before_receiver = agent_receiver.medicine_stock
    sim._execute_actions(agent_giver, [share_action], 0)
    med_after_giver = agent_giver.medicine_stock
    med_after_receiver = agent_receiver.medicine_stock

    if med_after_receiver > med_before_receiver:
        results.ok("Voluntary medicine sharing works")
    else:
        results.fail("BUG CONFIRMED: Voluntary medicine sharing is silently dropped",
                     f"Receiver medicine: {med_before_receiver} -> {med_after_receiver} (no change). "
                     f"Action dropped because no 'law_directed' flag.")


# ==========================================================================
# TEST 7: Resource Collection
# ==========================================================================

def test_resource_collection():
    print("\n--- Test 7: Resource Collection ---")
    grid = make_grid_6x6()
    agent = make_agent("collect-1")
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)
    set_agent(agent, food=5.0, water=5.0)  # Set AFTER make_sim

    cell = grid.cell(2, 2)
    cell.food = 50.0
    food_before_agent = agent.food_stock  # 5.0

    actions = [Action(ActionType.COLLECT_FOOD, {"amount": 5.0})]
    sim._execute_actions(agent, actions, 0)

    # Expected: agent.food_stock = 5.0 + 5.0 = 10.0, cell.food = 50.0 - 5.0 = 45.0
    if abs(agent.food_stock - 10.0) < 0.01 and abs(cell.food - 45.0) < 0.01:
        results.ok("Food collection correct",
                   f"agent={agent.food_stock:.1f}, cell={cell.food:.1f}")
    else:
        results.fail("Food collection incorrect",
                     f"agent food={agent.food_stock:.2f} (expected 10.0), "
                     f"cell food={cell.food:.2f} (expected 45.0)")


# ==========================================================================
# TEST 8: Food Ration Law
# ==========================================================================

def test_food_ration():
    print("\n--- Test 8: Food Ration Law ---")
    grid = make_grid_6x6()
    agent = make_agent("ration-1")
    gov = DemocracyGovernment(seed=42)
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)
    set_agent(agent, food=5.0)  # Set AFTER make_sim
    gov.bind(sim)

    cell = grid.cell(2, 2)
    cell.food = 50.0

    # Enact food ration law: max 3.0 per cycle
    gov._enact_law("FOOD_RATION", {"max_per_cycle": 3.0}, 0, duration=20)

    # Try to collect 5.0 (should be capped at 3.0)
    limit = gov.food_collection_limit(agent, 0)
    if abs(limit - 3.0) < 0.01:
        results.ok("Food ration law limits collection to 3.0")
    else:
        results.fail("Food ration law limit incorrect",
                     f"expected=3.0, got={limit}")

    # Execute collection
    food_before = agent.food_stock  # 5.0
    actions = [Action(ActionType.COLLECT_FOOD, {"amount": 5.0})]
    sim._execute_actions(agent, actions, 0)
    collected = agent.food_stock - food_before
    if abs(collected - 3.0) < 0.01:
        results.ok("Agent collected only 3.0 due to ration",
                   f"collected={collected:.2f}")
    else:
        results.fail("Agent collected wrong amount",
                     f"collected={collected:.2f}, expected=3.0")


# ==========================================================================
# TEST 9: Eating Mechanics
# ==========================================================================

def test_eating():
    print("\n--- Test 9: Eating Mechanics ---")
    grid = make_grid_6x6()
    agent = make_agent("eat-1")
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)
    set_agent(agent, health=0.80, food=10.0, hunger=0.3)  # Set AFTER make_sim

    # Agent at health 0.80 eats. health_deficit=0.20, needed=0.20/0.05=4.0
    actions = [Action(ActionType.EAT, {"amount": 2.0})]
    sim._execute_actions(agent, actions, 0)

    # Expected: eats 4.0 units (min(needed=4.0, stock=10.0)). food=10->6. health=1.0
    if abs(agent.health - 1.0) < 0.001 and abs(agent.food_stock - 6.0) < 0.01:
        results.ok("Eating restores health correctly",
                   f"health={agent.health:.3f}, food={agent.food_stock:.1f}")
    else:
        results.fail("Eating mechanics incorrect",
                     f"health={agent.health:.3f} (expected 1.0), "
                     f"food={agent.food_stock:.2f} (expected 6.0)")


# ==========================================================================
# TEST 10: Health Dynamics — Hunger Buildup
# ==========================================================================

def test_hunger_buildup():
    print("\n--- Test 10: Hunger Buildup ---")
    grid = make_grid_6x6()
    agent = make_agent("hunger-1")
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)
    set_agent(agent, health=1.0, food=0.0, water=10.0, hunger=0.0)  # Set AFTER

    # After one cycle with no food: hunger += hunger_build_rate.  Pinned to the
    # literal calibrated value (engine/simulation.py, Simulation.__init__:
    # `self.hunger_build_rate = 0.05`) rather than read back off `sim`, so a
    # change to the calibrated constant is caught here instead of passing
    # unnoticed.
    HUNGER_BUILD_RATE = 0.05
    sim._apply_health_dynamics(0)
    if abs(agent.hunger - HUNGER_BUILD_RATE) < 0.001:
        results.ok(f"Hunger builds up at {HUNGER_BUILD_RATE}/cycle when no food")
    else:
        results.fail("Hunger buildup incorrect",
                     f"expected={HUNGER_BUILD_RATE}, got={agent.hunger:.4f}")

    # No health drain yet (hunger < 0.4 threshold)
    if abs(agent.health - 1.0) < 0.001:
        results.ok("No health drain when hunger below threshold (0.4)")
    else:
        results.fail("Unexpected health drain",
                     f"health={agent.health:.6f}")


# ==========================================================================
# TEST 11: Health Dynamics — Hunger Drain
# ==========================================================================

def test_hunger_drain():
    print("\n--- Test 11: Health Drain from Hunger ---")
    grid = make_grid_6x6()
    agent = make_agent("hunger-drain-1")
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)
    set_agent(agent, health=1.0, food=0.0, water=10.0, hunger=0.5)  # Set AFTER

    h_before = agent.health
    sim._apply_health_dynamics(0)
    # hunger -> 0.5 + hunger_build_rate (builds up).
    # Drain: hunger_drain * difficulty_multiplier(1) * hunger.
    # Plus terrain hazard for PLAINS = 0.0 * dm = 0.0
    #
    # All three factors are pinned to their literal calibrated values rather
    # than read back off `sim`/the library function, so a change to any of
    # them is caught here instead of passing unnoticed:
    #   HUNGER_BUILD_RATE = 0.05   (engine/simulation.py, Simulation.__init__)
    #   HUNGER_DRAIN      = 0.025  (engine/simulation.py, Simulation.__init__)
    #   DIFFICULTY_MULT_D1 = 0.5   (engine/difficulty.py:233, difficulty_multiplier(1))
    HUNGER_BUILD_RATE = 0.05
    HUNGER_DRAIN = 0.025
    DIFFICULTY_MULT_D1 = 0.5
    assert abs(difficulty_multiplier(1) - DIFFICULTY_MULT_D1) < 1e-9, (
        f"difficulty_multiplier(1) drifted from the pinned value: "
        f"got {difficulty_multiplier(1)}, test expects {DIFFICULTY_MULT_D1}"
    )
    expected_hunger = 0.5 + HUNGER_BUILD_RATE
    expected_drain = HUNGER_DRAIN * DIFFICULTY_MULT_D1 * expected_hunger
    actual_drain = h_before - agent.health

    if abs(actual_drain - expected_drain) < 0.0001:
        results.ok("Hunger drain correct",
                   f"drain={actual_drain:.6f}, expected={expected_drain:.6f}")
    else:
        results.fail("Hunger drain incorrect",
                     f"drain={actual_drain:.6f}, expected={expected_drain:.6f}")


# ==========================================================================
# TEST 12: Epidemic Seeding and Spread
# ==========================================================================

def test_epidemic_spread():
    print("\n--- Test 12: Epidemic Mechanics ---")
    grid = make_grid_6x6()
    agents = [make_agent(f"epi-{i}", seed=i) for i in range(4)]
    gov = AnarchyGovernment()
    sim = make_sim(grid, agents, gov)
    # Place agents: one will be infected, others adjacent
    place_agent(sim, agents[0], 3, 3)  # Patient zero
    place_agent(sim, agents[1], 3, 4)  # Adjacent
    place_agent(sim, agents[2], 3, 2)  # Adjacent
    place_agent(sim, agents[3], 0, 0)  # Far away

    # Infect agent 0
    agents[0].infect("test-epi-001")

    # Create epidemic event
    epidemic = ActiveEvent(
        event_type=EventType.EPIDEMIC, start_cycle=0, duration=None,
        severity=1.0, disease_spread_rate=1.0,  # 100% spread for deterministic test
        epidemic_id="test-epi-001",
    )
    sim.event_system.active_events.append(epidemic)

    # Apply epidemic spread
    sim.event_system._apply_epidemic(epidemic, sim.grid, sim.agents)

    # With spread_rate=1.0, adjacent agents should be infected
    adj_infected = sum(1 for a in [agents[1], agents[2]] if "test-epi-001" in a.epidemic_ids)
    far_infected = "test-epi-001" in agents[3].epidemic_ids

    if adj_infected == 2:
        results.ok("Epidemic spreads to adjacent agents (100% rate)")
    else:
        results.fail("Epidemic did not spread to all adjacent agents",
                     f"adjacent infected: {adj_infected}/2")

    if not far_infected:
        results.ok("Epidemic does NOT spread to distant agents")
    else:
        results.fail("Epidemic spread to distant agent",
                     "Agent at (0,0) should not be infected by agent at (3,3)")

    # Check cell contamination
    if grid.cell(3, 3).contaminated:
        results.ok("Carrier's cell marked as contaminated")
    else:
        results.fail("Carrier's cell not contaminated",
                     "Expected cell (3,3) to be contaminated=True")


# ==========================================================================
# TEST 13: Natural Recovery from Infection
# ==========================================================================

def test_natural_recovery():
    print("\n--- Test 13: Natural Recovery ---")
    grid = make_grid_6x6()
    agent = make_agent("recovery-1", health=1.0)
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov, difficulty=1)
    place_agent(sim, agent, 2, 2)

    # Infect agent
    agent.infect("recovery-test-001")
    agent.infection_cycles["recovery-test-001"] = 0

    # At difficulty=1, recovery_cycles = 15 + 0/99*35 = 15
    recovery_cycles = int(15 + (1-1)/99 * 35)

    # Simulate ticking infections for 14 cycles — should NOT recover
    for _ in range(14):
        agent.tick_infections(recovery_cycles)

    if "recovery-test-001" in agent.epidemic_ids:
        results.ok("Agent still infected after 14 cycles (recovery=15)")
    else:
        results.fail("Agent recovered too early",
                     "Expected to still be infected at cycle 14")

    # Tick once more — should recover at cycle 15
    agent.tick_infections(recovery_cycles)
    if "recovery-test-001" not in agent.epidemic_ids:
        results.ok("Agent recovered naturally at cycle 15")
    else:
        results.fail("Agent did not recover at cycle 15",
                     f"infection_cycles={agent.infection_cycles}")


# ==========================================================================
# TEST 14: Medicine Treatment
# ==========================================================================

def test_medicine_treatment():
    print("\n--- Test 14: Medicine Treatment ---")
    grid = make_grid_6x6()
    agent = make_agent("treat-1", health=0.8, medicine=5.0)
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)

    # Infect with 1 epidemic (needs 3.0 medicine)
    agent.infect("treat-epi-001")

    actions = [Action(ActionType.TREAT_SELF, {})]
    sim._execute_actions(agent, actions, 0)

    if not agent.infected and abs(agent.medicine_stock - 2.0) < 0.01:
        results.ok("Agent cured with medicine",
                   f"medicine={agent.medicine_stock:.1f}, infected={agent.infected}")
    else:
        results.fail("Treatment failed",
                     f"infected={agent.infected}, medicine={agent.medicine_stock:.2f}")


# ==========================================================================
# TEST 15: Resource Regeneration
# ==========================================================================

def test_regeneration():
    print("\n--- Test 15: Resource Regeneration ---")
    grid = make_grid_6x6()
    cell = grid.cell(2, 2)
    cell.terrain = Terrain.PLAINS
    cell.food = 20.0
    cell.water = 20.0

    # Normal regen (no drought)
    grid.regenerate(drought_factor=0.0, base_regen_mult=1.0)
    # PLAINS food_regen=0.5, so food = min(70, 20 + 0.5*1.0) = 20.5
    if abs(cell.food - 20.5) < 0.01:
        results.ok("Normal food regeneration correct (PLAINS)",
                   f"food={cell.food:.2f}")
    else:
        results.fail("Normal food regeneration incorrect",
                     f"food={cell.food:.3f}, expected=20.5")

    # Test with drought
    cell.food = 20.0
    grid.regenerate(drought_factor=0.7, base_regen_mult=1.0)
    # regen_mult = (1-0.7)*1.0 = 0.3. food = min(70, 20 + 0.5*0.3) = 20.15
    if abs(cell.food - 20.15) < 0.01:
        results.ok("Drought reduces regeneration correctly",
                   f"food={cell.food:.3f}")
    else:
        results.fail("Drought regeneration incorrect",
                     f"food={cell.food:.4f}, expected=20.15")


# ==========================================================================
# TEST 16: Voting — Storm Scenario
# ==========================================================================

def test_voting_storm():
    print("\n--- Test 16: Agent Voting During Storm ---")
    agent = make_agent("vote-1", health=0.8, food=10.0, water=10.0)
    options = ["MANDATORY_SHELTER", "REDISTRIBUTE_FOOD", "NO_ACTION"]
    event_types = ["storm"]

    vote = agent._choose_vote(options, event_types)
    if vote == "MANDATORY_SHELTER":
        results.ok("Agent votes MANDATORY_SHELTER during storm")
    else:
        results.fail("Agent did not vote MANDATORY_SHELTER",
                     f"voted={vote}, expected=MANDATORY_SHELTER")


# ==========================================================================
# TEST 17: Voting — Infected Agent
# ==========================================================================

def test_voting_infected():
    print("\n--- Test 17: Infected Agent Voting ---")
    agent = make_agent("vote-infected", health=0.7, food=5.0, water=5.0)
    agent.infect("vote-epi-001")
    options = ["QUARANTINE_EPIDEMIC", "EPIDEMIC_RESPONSE", "NO_ACTION"]
    event_types = ["epidemic"]

    vote = agent._choose_vote(options, event_types)
    if vote == "EPIDEMIC_RESPONSE":
        results.ok("Infected agent votes EPIDEMIC_RESPONSE over QUARANTINE")
    else:
        results.fail("Infected agent voted wrong",
                     f"voted={vote}, expected=EPIDEMIC_RESPONSE")


# ==========================================================================
# TEST 18: Voting — Hungry Agent
# ==========================================================================

def test_voting_hungry():
    print("\n--- Test 18: Hungry Agent Voting ---")
    agent = make_agent("vote-hungry", health=0.9, food=1.0, water=10.0)
    options = ["REDISTRIBUTE_FOOD", "MANDATORY_SHELTER", "NO_ACTION"]
    event_types = []  # no active events

    vote = agent._choose_vote(options, event_types)
    if vote == "REDISTRIBUTE_FOOD":
        results.ok("Hungry agent votes REDISTRIBUTE_FOOD",
                   f"food_stock={agent.food_stock}")
    else:
        results.fail("Hungry agent did not vote for food",
                     f"voted={vote}, food_stock={agent.food_stock}")


# ==========================================================================
# TEST 19: Democracy Election Timing
# ==========================================================================

def test_election_timing():
    print("\n--- Test 19: Democracy Election Timing ---")
    grid = make_grid_6x6()
    agents = [make_agent(f"dem-{i}", seed=i) for i in range(4)]
    gov = DemocracyGovernment(t_vote=5, seed=42)
    sim = make_sim(grid, agents, gov, max_cycles=12)
    for i, a in enumerate(agents):
        place_agent(sim, a, i, i)
    gov.bind(sim)

    # Cycle 0: election should start
    gov.tick(0)
    if gov._election_active:
        results.ok("Election starts at cycle 0")
    else:
        results.fail("Election did not start at cycle 0")

    # Cycle 1: election should close (cycle > start_cycle)
    gov.tick(1)
    if not gov._election_active:
        results.ok("Election closes at cycle 1 (one cycle after start)")
    else:
        results.fail("Election still active at cycle 1")

    # Cycle 5: next election
    gov.tick(5)
    if gov._election_active:
        results.ok("Next election starts at cycle 5")
    else:
        results.fail("Election did not start at cycle 5")


# ==========================================================================
# TEST 20: Event Warning System
# ==========================================================================

def test_event_warnings():
    print("\n--- Test 20: Event Warning System ---")
    grid = make_grid_6x6()
    agent = make_agent("warn-1")
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)

    # Schedule a storm at cycle 10
    storm = ActiveEvent(
        event_type=EventType.STORM, start_cycle=10, duration=5,
        severity=1.0, storm_damage=0.04,
    )
    sim.event_system.schedule(10, storm)
    sim.event_system.warning_cycles = 3

    # At cycle 7 (3 cycles before): should get warning
    warnings, _ = sim.event_system.update(7, grid, [agent])
    warnings_at_7 = [w for w in warnings if w.event_type == EventType.STORM]
    if warnings_at_7 and warnings_at_7[0].cycles_until == 3:
        results.ok("Warning emitted 3 cycles before event",
                   f"cycles_until={warnings_at_7[0].cycles_until}")
    else:
        results.fail("No warning at cycle 7 (3 before event)",
                     f"warnings={[(w.event_type.value, w.cycles_until) for w in warnings]}")

    # At cycle 10: event should activate
    warnings, _ = sim.event_system.update(10, grid, [agent])
    active_warnings = [w for w in warnings if w.cycles_until == 0]
    if active_warnings:
        results.ok("ACTIVE warning at event trigger cycle")
    else:
        results.fail("No ACTIVE warning at trigger cycle")

    if any(e.event_type == EventType.STORM for e in sim.event_system.active_events):
        results.ok("Storm added to active_events at trigger cycle")
    else:
        results.fail("Storm not in active_events after trigger")


# ==========================================================================
# TEST 21: Event Expiry (Non-Epidemic)
# ==========================================================================

def test_event_expiry():
    print("\n--- Test 21: Event Expiry (Storm) ---")
    grid = make_grid_6x6()
    agent = make_agent("expire-1")
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)

    # Storm starts at cycle 0, duration 5 → expires at cycle 5
    storm = ActiveEvent(
        event_type=EventType.STORM, start_cycle=0, duration=5,
        severity=1.0, storm_damage=0.04,
    )
    sim.event_system.active_events.append(storm)

    # At cycle 4: still active
    if storm.is_active(4):
        results.ok("Storm active at cycle 4 (duration=5, start=0)")
    else:
        results.fail("Storm expired too early at cycle 4")

    # At cycle 5: should expire
    if not storm.is_active(5):
        results.ok("Storm expires at cycle 5 (start+duration)")
    else:
        results.fail("Storm still active at cycle 5")


# ==========================================================================
# TEST 22: Epidemic Never Expires
# ==========================================================================

def test_epidemic_never_expires():
    print("\n--- Test 22: Epidemic Never Expires ---")
    epidemic = ActiveEvent(
        event_type=EventType.EPIDEMIC, start_cycle=0, duration=None,
        severity=1.0, disease_spread_rate=0.2, epidemic_id="permanent-001",
    )

    # Check at various cycles
    if epidemic.is_active(100) and epidemic.is_active(1000):
        results.ok("Epidemic with duration=None never expires")
    else:
        results.fail("Epidemic expired unexpectedly")


# ==========================================================================
# TEST 23: Anarchy — No Laws
# ==========================================================================

def test_anarchy_no_laws():
    print("\n--- Test 23: Anarchy Government ---")
    grid = make_grid_6x6()
    agent = make_agent("anarchy-1")
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)
    gov.bind(sim)

    # Run several cycles
    for cycle in range(10):
        gov.tick(cycle)

    if len(gov.active_laws) == 0:
        results.ok("Anarchy has no laws after 10 cycles")
    else:
        results.fail("Anarchy enacted laws",
                     f"laws={[l.law_type for l in gov.active_laws]}")

    # can_move always true
    if gov.can_move(agent, 5, 5, 0):
        results.ok("Anarchy allows all movement")
    else:
        results.fail("Anarchy blocked movement")


# ==========================================================================
# TEST 24: Simultaneous Storm + Epidemic
# ==========================================================================

def test_simultaneous_events():
    print("\n--- Test 24: Simultaneous Storm + Epidemic ---")
    terrain = [
        "...F..",
        "......",
        "......",
        "......",
        "......",
        "......",
    ]
    grid = make_grid_6x6(terrain)
    agents = [make_agent(f"simul-{i}", seed=i) for i in range(4)]
    gov = DemocracyGovernment(seed=42)
    sim = make_sim(grid, agents, gov)
    for i, a in enumerate(agents):
        place_agent(sim, a, 2, i)
    gov.bind(sim)

    # Infect agent 0
    agents[0].infect("simul-epi-001")

    # Add both events
    storm = ActiveEvent(
        event_type=EventType.STORM, start_cycle=0, duration=10,
        severity=1.0, storm_damage=0.04,
    )
    epidemic = ActiveEvent(
        event_type=EventType.EPIDEMIC, start_cycle=0, duration=None,
        severity=1.0, disease_spread_rate=0.3, epidemic_id="simul-epi-001",
    )
    sim.event_system.active_events.extend([storm, epidemic])

    # Run one cycle to see what happens
    sim._step(0)

    # Both events should be active
    active_types = [e.event_type for e in sim.event_system.active_events]
    if EventType.STORM in active_types and EventType.EPIDEMIC in active_types:
        results.ok("Both storm and epidemic active simultaneously")
    else:
        results.fail("Events not both active",
                     f"active={[t.value for t in active_types]}")


# ==========================================================================
# TEST 25: Law-Directed Redistribution
# ==========================================================================

def test_redistribution():
    print("\n--- Test 25: Law-Directed Redistribution ---")
    grid = make_grid_6x6()
    agents = [
        make_agent("poor-1", food=2.0),
        make_agent("poor-2", food=2.0),
        make_agent("rich-1", food=15.0),
        make_agent("rich-2", food=15.0),
    ]
    gov = DemocracyGovernment(seed=42)
    sim = make_sim(grid, agents, gov)
    for i, a in enumerate(agents):
        place_agent(sim, a, i, 0)
    gov.bind(sim)

    # Compute redistribution manually
    agents_by_id = {a.agent_id: a for a in agents}
    recipient_ids = [agents[0].agent_id, agents[1].agent_id]
    donor_ids = [agents[2].agent_id, agents[3].agent_id]

    donor_amounts, actual_n = gov._compute_redistribution_amounts(
        agents_by_id, recipient_ids, donor_ids, "food", 5.0
    )

    # With 2 recipients needing 5.0 each = 10.0 total, split between 2 donors = 5.0 each
    total_donated = sum(donor_amounts.values())
    if abs(total_donated - 10.0) < 0.01:
        results.ok("Redistribution amounts computed correctly",
                   f"donor_amounts={donor_amounts}")
    else:
        results.fail("Redistribution amounts wrong",
                     f"total_donated={total_donated:.2f}, expected=10.0, "
                     f"donor_amounts={donor_amounts}")

    # Now enact the law and run filter_actions for a donor
    gov._enact_law("REDISTRIBUTE_FOOD", {
        "resource": "food",
        "recipient_ids": recipient_ids,
        "donor_amounts": donor_amounts,
        "amount_per_recipient": actual_n,
    }, 0, duration=5, applies_to=list(donor_amounts.keys()))

    # Filter actions for rich-1
    donor = agents[2]
    actions = gov.filter_actions(donor, [], 0)
    share_actions = [a for a in actions if a.type == ActionType.SHARE_RESOURCE]
    if share_actions:
        results.ok("Redistribution law injects SHARE_RESOURCE action for donor")
        # Check it has law_directed
        if share_actions[0].params.get("law_directed"):
            results.ok("SHARE_RESOURCE action has law_directed=True")
        else:
            results.fail("SHARE_RESOURCE missing law_directed flag")
    else:
        results.fail("No SHARE_RESOURCE action injected for donor",
                     f"actions={[(a.type.value, a.params) for a in actions]}")


# ==========================================================================
# TEST 26: Agent Priority — Infection Treatment Takes Priority
# ==========================================================================

def test_priority_treat_infection():
    print("\n--- Test 26: Agent Priority — Treat Infection ---")
    grid = make_grid_6x6()
    agent = make_agent("priority-1", health=0.8, food=5.0, water=5.0, medicine=5.0)
    agent.infect("priority-epi-001")
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)
    gov.bind(sim)

    obs = agent.observe(grid, gov, 0)
    obs["active_events"] = [{"type": "epidemic", "cycles_remaining": -1, "severity": 1.0,
                             "region": None, "description": "", "epidemic_id": "priority-epi-001"}]
    obs["event_warnings"] = []
    obs["cycle"] = 0

    actions = agent.act(obs, gov, 0)
    action_types = [a.type for a in actions]

    if ActionType.TREAT_SELF in action_types:
        results.ok("Infected agent with medicine prioritizes TREAT_SELF")
    else:
        results.fail("Infected agent did not prioritize treatment",
                     f"actions={[a.type.value for a in actions]}")


# ==========================================================================
# TEST 27: Agent Priority — Storm/Shelter Over Hunger
# ==========================================================================

def test_priority_shelter_over_hunger():
    print("\n--- Test 27: Priority — Shelter Over Hunger ---")
    terrain = [
        "......",
        "......",
        "......",
        "...F..",  # Shelter at (3,3)
        "......",
        "......",
    ]
    grid = make_grid_6x6(terrain)
    agent = make_agent("priority-2", health=0.8, food=1.0, water=10.0)  # Low food
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 0, 0)
    gov.bind(sim)

    obs = agent.observe(grid, gov, 0)
    obs["active_events"] = [{"type": "storm", "cycles_remaining": 5, "severity": 1.0,
                             "region": None, "description": "", "epidemic_id": None}]
    obs["event_warnings"] = []
    obs["cycle"] = 0

    actions = agent.act(obs, gov, 0)
    action_types = [a.type for a in actions]

    # Priority 2 (storm) should trigger before Priority 4 (hunger)
    # Agent should try to MOVE toward shelter at (3,3) and then return (no food collection)
    has_move = ActionType.MOVE in action_types
    has_collect_food = ActionType.COLLECT_FOOD in action_types

    if has_move and not has_collect_food:
        results.ok("Agent moves toward shelter during storm, ignoring hunger")
    elif has_move and has_collect_food:
        results.fail("Agent both moves to shelter AND collects food",
                     "Shelter priority should short-circuit the act() method")
    else:
        results.fail("Unexpected action set during storm",
                     f"actions={[a.type.value for a in actions]}")


# ==========================================================================
# TEST 28: Death Threshold
# ==========================================================================

def test_death():
    print("\n--- Test 28: Agent Death ---")
    grid = make_grid_6x6()
    agent = make_agent("death-1")
    gov = AnarchyGovernment()
    sim = make_sim(grid, [agent], gov)
    place_agent(sim, agent, 2, 2)
    set_agent(agent, health=0.001, food=0.0, water=0.0, hunger=0.8, thirst=0.8)

    sim._apply_health_dynamics(0)

    if not agent.alive:
        results.ok("Agent dies when health drops to 0")
    else:
        results.fail("Agent still alive at near-zero health",
                     f"health={agent.health:.6f}")

    if agent.position is None:
        results.ok("Dead agent removed from grid")
    else:
        results.fail("Dead agent still has position",
                     f"position={agent.position}")


# ==========================================================================
# TEST 29: Context-Aware Election Options
# ==========================================================================

def test_context_aware_options():
    print("\n--- Test 29: Context-Aware Election Options ---")
    grid = make_grid_6x6()
    agents = [make_agent(f"ctx-{i}", seed=i) for i in range(4)]
    gov = DemocracyGovernment(seed=42)
    sim = make_sim(grid, agents, gov)
    for i, a in enumerate(agents):
        place_agent(sim, a, i, 0)
    gov.bind(sim)

    # Add epidemic event
    epidemic = ActiveEvent(
        event_type=EventType.EPIDEMIC, start_cycle=0, duration=None,
        severity=1.0, disease_spread_rate=0.2, epidemic_id="ctx-epi-001",
    )
    sim.event_system.active_events.append(epidemic)

    # Check options generated during epidemic
    options = gov._context_aware_options(0)
    if "QUARANTINE_EPIDEMIC" in options:
        results.ok("Epidemic triggers QUARANTINE_EPIDEMIC option")
    else:
        results.fail("QUARANTINE_EPIDEMIC not in options during epidemic",
                     f"options={options}")

    if "EPIDEMIC_RESPONSE" in options:
        results.ok("Epidemic triggers EPIDEMIC_RESPONSE option")
    else:
        results.fail("EPIDEMIC_RESPONSE not in options during epidemic",
                     f"options={options}")


# ==========================================================================
# TEST 30: Full Mini-Simulation Smoke Test
# ==========================================================================

def test_full_smoke():
    print("\n--- Test 30: Full 20-Cycle Smoke Test ---")
    terrain = [
        ".F....",
        "......",
        "..F...",
        "......",
        "....F.",
        "W.....",
    ]
    grid = make_grid_6x6(terrain)
    agents = [make_agent(f"smoke-{i}", seed=i) for i in range(5)]
    gov = DemocracyGovernment(t_vote=5, seed=42)
    sim = make_sim(grid, agents, gov, difficulty=10, max_cycles=20)
    positions = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    for a, pos in zip(agents, positions):
        place_agent(sim, a, *pos)
    gov.bind(sim)

    # Schedule a storm at cycle 5
    storm = ActiveEvent(
        event_type=EventType.STORM, start_cycle=5, duration=5,
        severity=1.0, storm_damage=0.04,
    )
    sim.event_system.schedule(5, storm)

    try:
        metrics = sim.run()
        alive = sum(1 for a in agents if a.alive)
        results.ok(f"Full simulation completed without crash (alive={alive}/5)")
    except Exception as e:
        results.fail("Simulation crashed",
                     f"error={e}\n{traceback.format_exc()}")


# ==========================================================================
# Main
# ==========================================================================

def run_all():
    print("=" * 60)
    print("  SIMULATION TEST SCENARIOS")
    print("  Grid: 6x6 | Agents: 1-5 per scenario | Difficulty: 1-10")
    print("=" * 60)

    test_grid_boundary()
    test_movement_cost()
    test_storm_shelter()
    test_mandatory_shelter_bug()
    test_quarantine_bug()
    test_voluntary_sharing_bug()
    test_resource_collection()
    test_food_ration()
    test_eating()
    test_hunger_buildup()
    test_hunger_drain()
    test_epidemic_spread()
    test_natural_recovery()
    test_medicine_treatment()
    test_regeneration()
    test_voting_storm()
    test_voting_infected()
    test_voting_hungry()
    test_election_timing()
    test_event_warnings()
    test_event_expiry()
    test_epidemic_never_expires()
    test_anarchy_no_laws()
    test_simultaneous_events()
    test_redistribution()
    test_priority_treat_infection()
    test_priority_shelter_over_hunger()
    test_death()
    test_context_aware_options()
    test_full_smoke()

    return results.summary()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
