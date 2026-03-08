"""ScoringRegistry — applies all registered factors and returns total score."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .base import AbstractScoringFactor

if TYPE_CHECKING:
    from neuralpipe.models.route import Route


class ScoringRegistry:
    """Maintains a list of scoring factors and computes weighted totals.

    Usage:
        registry = ScoringRegistry()
        registry.register(RouteLengthFactor())
        score, breakdown = registry.score(route, context)

    Adding a new factor = subclass AbstractScoringFactor + one register() call.
    """

    def __init__(self) -> None:
        self._factors: list[AbstractScoringFactor] = []

    def register(self, factor: AbstractScoringFactor) -> None:
        self._factors.append(factor)

    def unregister(self, name: str) -> None:
        self._factors = [f for f in self._factors if f.name != name]

    def score(
        self, route: "Route", context: dict[str, Any]
    ) -> tuple[float, dict[str, float]]:
        """Compute total score and per-factor breakdown.

        Returns:
            (total_score, breakdown) where breakdown maps factor name → weighted contribution.
        """
        breakdown: dict[str, float] = {}
        total = 0.0
        for factor in self._factors:
            raw = factor.score(route, context)
            weighted = factor.weight * raw
            breakdown[factor.name] = weighted
            total += weighted
        return total, breakdown

    @property
    def factors(self) -> list[AbstractScoringFactor]:
        return list(self._factors)
