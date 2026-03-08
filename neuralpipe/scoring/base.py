"""AbstractScoringFactor ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from neuralpipe.models.route import Route


class AbstractScoringFactor(ABC):
    """Base class for all route scoring factors.

    Subclass this, set name and weight, implement score().
    Register in ScoringRegistry — nothing else changes.

    Lower total score = better route (spec §8).
    """

    name: str = "unnamed_factor"
    weight: float = 1.0

    @abstractmethod
    def score(self, route: "Route", context: dict[str, Any]) -> float:
        """Return the raw (unweighted) score contribution for this factor.

        The registry applies weight automatically: contribution = weight * score().
        Return 0.0 if this factor does not apply to the route.
        """
