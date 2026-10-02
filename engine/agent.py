"""Base agent class and action definitions."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .grid import Grid
    from governments.base import Government


class ActionType(Enum):
    MOVE = "move"
    COLLECT_FOOD = "collect_food"
    COLLECT_WATER = "collect_water"
    COLLECT_MEDICINE = "collect_medicine"
    EAT = "eat"
    DRINK = "drink"
    TREAT_SELF = "treat_self"
    SHARE_FOOD = "share_food"
    SHARE_WATER = "share_water"
    SHARE_MEDICINE = "share_medicine"
    SHARE_RESOURCE = "share_resource"   # generic multi-resource share
    REST = "rest"
    VOTE = "vote"
    PROPOSE = "propose"


@dataclass
class Action:
    type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)


DIRECTION_DELTAS = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
    "northeast": (-1, 1),
    "northwest": (-1, -1),
    "southeast": (1, 1),
    "southwest": (1, -1),
}


class Agent(ABC):
    """
    Base class for all agents. Subclasses implement observe() and act().
    These two methods together are the 'C lines of code' that define
    the agent's behavior.
    """

    def __init__(self, agent_id: Optional[str] = None, role: str = "citizen"):
        self.agent_id: str = agent_id or str(uuid.uuid4())[:8]
        self.role: str = role
        self.health: float = 0.8
        self.food_stock: float = 10.0
        self.water_stock: float = 10.0
        self.medicine_stock: float = 2.0
        self.hunger: float = 0.0
        self.thirst: float = 0.0
        self.epidemic_ids: Set[str] = set()   # IDs of active epidemic infections
        self.infection_cycles: Dict[str, int] = {}  # epidemic_id -> cycles infected
        self.position: Optional[Tuple[int, int]] = None
        self.alive: bool = True
        self.memory: Dict[str, Any] = {}
        self.age: int = 0  # cycles survived

    # ------------------------------------------------------------------
    # Infection helpers
    # ------------------------------------------------------------------

    @property
    def infected(self) -> bool:
        """True if the agent carries any epidemic."""
        return bool(self.epidemic_ids)

    @infected.setter
    def infected(self, value: bool) -> None:
        """Setter — clears all infections when set to False."""
        if not value:
            self.epidemic_ids.clear()

    def infect(self, epidemic_id: str) -> None:
        self.epidemic_ids.add(epidemic_id)
        if epidemic_id not in self.infection_cycles:
            self.infection_cycles[epidemic_id] = 0

    def cure_all(self) -> None:
        self.epidemic_ids.clear()
        self.infection_cycles.clear()

    def cure_one(self, epidemic_id: str) -> None:
        self.epidemic_ids.discard(epidemic_id)
        self.infection_cycles.pop(epidemic_id, None)

    def tick_infections(self, recovery_cycles: int) -> None:
        """Advance infection counters; naturally recover after recovery_cycles."""
        recovered = []
        for eid in list(self.infection_cycles):
            self.infection_cycles[eid] += 1
            if self.infection_cycles[eid] >= recovery_cycles:
                recovered.append(eid)
        for eid in recovered:
            self.epidemic_ids.discard(eid)
            del self.infection_cycles[eid]

    def medicine_needed_to_cure(self) -> float:
        """3 medicine units per active epidemic ID needed to treat all infections."""
        return len(self.epidemic_ids) * 3.0

    # ------------------------------------------------------------------
    # Core interface — subclasses implement these two methods
    # ------------------------------------------------------------------

    @abstractmethod
    def observe(self, grid: "Grid", government: "Government", cycle: int) -> Dict[str, Any]:
        """Return a structured observation dict from the agent's perspective."""
        ...

    @abstractmethod
    def act(self, obs: Dict[str, Any], government: "Government", cycle: int) -> List[Action]:
        """Return a list of actions to execute this cycle."""
        ...

    # ------------------------------------------------------------------
    # Survival helpers — convenience methods for subclasses
    # ------------------------------------------------------------------

    def is_hungry(self, threshold: float = 0.4) -> bool:
        return self.hunger > threshold

    def is_thirsty(self, threshold: float = 0.4) -> bool:
        return self.thirst > threshold

    def is_low_health(self, threshold: float = 0.3) -> bool:
        return self.health < threshold

    def urgently_needs_food(self) -> bool:
        return self.food_stock < 2.0 or self.hunger > 0.6

    def urgently_needs_water(self) -> bool:
        return self.water_stock < 2.5 or self.thirst > 0.55

    # ------------------------------------------------------------------
    # Action factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def move(direction: str) -> Action:
        return Action(ActionType.MOVE, {"direction": direction})

    @staticmethod
    def collect_food(amount: float = 5.0) -> Action:
        return Action(ActionType.COLLECT_FOOD, {"amount": amount})

    @staticmethod
    def collect_water(amount: float = 5.0) -> Action:
        return Action(ActionType.COLLECT_WATER, {"amount": amount})

    @staticmethod
    def collect_medicine(amount: float = 2.0) -> Action:
        return Action(ActionType.COLLECT_MEDICINE, {"amount": amount})

    @staticmethod
    def eat(amount: float = 2.0) -> Action:
        return Action(ActionType.EAT, {"amount": amount})

    @staticmethod
    def drink(amount: float = 2.0) -> Action:
        return Action(ActionType.DRINK, {"amount": amount})

    @staticmethod
    def treat_self() -> Action:
        return Action(ActionType.TREAT_SELF, {})

    @staticmethod
    def share_food(target_id: str, amount: float = 3.0) -> Action:
        return Action(ActionType.SHARE_FOOD, {"target_id": target_id, "amount": amount})

    @staticmethod
    def share_water(target_id: str, amount: float = 3.0) -> Action:
        return Action(ActionType.SHARE_WATER, {"target_id": target_id, "amount": amount})

    @staticmethod
    def share_medicine(target_id: str, amount: float = 3.0) -> Action:
        return Action(ActionType.SHARE_MEDICINE, {"target_id": target_id, "amount": amount})

    @staticmethod
    def share_resource(
        resource: str, target_ids: List[str], amount: float, fraction: float = 0.0
    ) -> Action:
        """Share any resource to one or more targets.
        resource: 'food' | 'water' | 'medicine'
        amount: flat amount per target; if fraction > 0, share fraction of stock instead.
        """
        return Action(ActionType.SHARE_RESOURCE, {
            "resource": resource,
            "target_ids": target_ids,
            "amount": amount,
            "fraction": fraction,
        })

    @staticmethod
    def rest() -> Action:
        return Action(ActionType.REST, {})

    @staticmethod
    def vote(option: Any) -> Action:
        return Action(ActionType.VOTE, {"option": option})

    @staticmethod
    def propose(proposal: Any) -> Action:
        return Action(ActionType.PROPOSE, {"proposal": proposal})

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def direction_toward(self, target_r: int, target_c: int) -> str:
        """Return the best cardinal/diagonal direction to move toward target."""
        if not self.position:
            return "north"
        r, c = self.position
        dr = target_r - r
        dc = target_c - c
        if dr == 0 and dc == 0:
            return "north"
        # Prefer diagonal movement
        vert = "south" if dr > 0 else ("north" if dr < 0 else "")
        horiz = "east" if dc > 0 else ("west" if dc < 0 else "")
        if vert and horiz:
            return f"{vert}{horiz}"
        return vert or horiz

    def step_toward(self, target_r: int, target_c: int) -> Action:
        return self.move(self.direction_toward(target_r, target_c))

    def steps_toward(self, target_r: int, target_c: int, k: int) -> List[Action]:
        """Return up to k MOVE actions greedily heading toward (target_r, target_c).

        Simulates position updates between steps so diagonal progress is
        represented correctly.  Stops early if the target is reached.
        """
        if not self.position or k <= 0:
            return []
        r, c = self.position
        actions: List[Action] = []
        for _ in range(k):
            dr = target_r - r
            dc = target_c - c
            if dr == 0 and dc == 0:
                break
            vert = "south" if dr > 0 else ("north" if dr < 0 else "")
            horiz = "east" if dc > 0 else ("west" if dc < 0 else "")
            direction = (f"{vert}{horiz}" if vert and horiz else vert or horiz)
            actions.append(self.move(direction))
            delta = DIRECTION_DELTAS[direction]
            r, c = r + delta[0], c + delta[1]
        return actions

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        pos = f"({self.position[0]},{self.position[1]})" if self.position else "?"
        return f"Agent[{self.agent_id}|{self.role}|h={self.health:.2f}|{pos}]"

    def to_dict(self) -> dict:
        # No round() — called ~35k times per cycle; precision is irrelevant for decisions.
        return {
            "id": self.agent_id,
            "role": self.role,
            "health": self.health,
            "hunger": self.hunger,
            "thirst": self.thirst,
            "infected": self.infected,
            "epidemic_ids": list(self.epidemic_ids),
            "position": self.position,
            "alive": self.alive,
            "food_stock": self.food_stock,
            "water_stock": self.water_stock,
            "medicine_stock": self.medicine_stock,
            "age": self.age,
        }
