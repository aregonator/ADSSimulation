"""Anarchy: no collective decision-making, pure self-interest."""

from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

from .base import Government

if TYPE_CHECKING:
    from engine.agent import Agent, Action
    from engine.events import EventWarning


class AnarchyGovernment(Government):
    """
    No laws, no proposals, no voting. Agents act entirely on self-interest.
    The only coordination is emergent from individual choices.
    """

    name = "Anarchy"

    def tick(self, cycle: int) -> None:
        # Nothing to coordinate
        self._expire_laws(cycle)

    def can_move(self, agent: "Agent", target_r: int, target_c: int, cycle: int) -> bool:
        return True

    def food_collection_limit(self, agent: "Agent", cycle: int) -> float:
        return 100.0  # No limits

    def get_audit_info(self, cycle: int) -> dict:
        return {"structure": "none"}

    def get_log_info(self, cycle: int) -> dict:
        return {
            "decision_summary": "",
            "decision_trigger": "",
            "vote_details": {},
            "state_narrative": "  No collective governance. Agents act purely on self-interest.",
        }
