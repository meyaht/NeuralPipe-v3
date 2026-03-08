"""AbstractConstraint ABC and ConstraintResult."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neuralpipe.models.route import Route


@dataclass
class ConstraintResult:
    passed: bool
    constraint_name: str
    message: str = ""
    # Repaired route, if repair was possible
    repaired_route: Optional["Route"] = None
    # Structured data for diagnostics (clash point, span length, etc.)
    detail: dict[str, Any] = field(default_factory=dict)


class AbstractConstraint(ABC):
    """Base class for all routing constraints.

    Subclass this, set class-level attributes, implement check() and optionally
    repair(). Register the subclass in ConstraintRegistry — nothing else changes.
    """

    # --- class-level metadata (override in each subclass) ---
    name: str = "unnamed_constraint"
    scope: str = "GLOBAL"  # GLOBAL | UNIT | SERVICE_CLASS | LOCATION
    version: str = "1.0"
    priority: int = 100  # lower = checked first

    @abstractmethod
    def check(self, route: "Route", context: dict[str, Any]) -> ConstraintResult:
        """Evaluate the constraint against route.

        Args:
            route: the Route being evaluated
            context: dict with keys such as:
                - pipe_spec (PipeSpec)
                - voxel_grid (VoxelGridProvider)
                - nps_inches (float)
                - eod_m (float)
                - grade_routing_allowed (bool)
                - existing_routes (list[Route])

        Returns:
            ConstraintResult with passed=True if the route satisfies this constraint.
        """

    def repair(self, route: "Route", context: dict[str, Any]) -> Optional["Route"]:
        """Attempt to repair a failing route. Return None if repair is impossible.

        Default: no repair. Subclasses may override.
        """
        return None
