"""ConstraintRegistry — register and run constraints in priority order."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .base import AbstractConstraint, ConstraintResult

if TYPE_CHECKING:
    from neuralpipe.models.route import Route


class ConstraintRegistry:
    """Maintains an ordered list of constraints and runs them against a route.

    Usage:
        registry = ConstraintRegistry()
        registry.register(OrthogonalityConstraint())
        results = registry.check_all(route, context)

    Adding a new constraint = instantiate it, call registry.register(). Nothing else.
    """

    def __init__(self) -> None:
        self._constraints: list[AbstractConstraint] = []

    def register(self, constraint: AbstractConstraint) -> None:
        """Add a constraint. Re-sorts by priority after insertion."""
        self._constraints.append(constraint)
        self._constraints.sort(key=lambda c: c.priority)

    def unregister(self, name: str) -> None:
        """Remove a constraint by name."""
        self._constraints = [c for c in self._constraints if c.name != name]

    def check_all(
        self, route: "Route", context: dict[str, Any]
    ) -> list[ConstraintResult]:
        """Run all registered constraints. Returns a list of results in priority order."""
        results = []
        for constraint in self._constraints:
            result = constraint.check(route, context)
            results.append(result)
        return results

    def check_all_pass(
        self, route: "Route", context: dict[str, Any]
    ) -> tuple[bool, list[ConstraintResult]]:
        """Return (all_passed, results). Short-circuits on first hard failure if fast=True."""
        results = self.check_all(route, context)
        all_passed = all(r.passed for r in results)
        return all_passed, results

    def apply_repairs(
        self, route: "Route", context: dict[str, Any], max_iterations: int = 5
    ) -> tuple["Route", list[ConstraintResult]]:
        """Iteratively attempt repairs until all constraints pass or max_iterations reached.

        Returns the (possibly repaired) route and final constraint results.
        """
        current_route = route
        for _ in range(max_iterations):
            all_passed, results = self.check_all_pass(current_route, context)
            if all_passed:
                return current_route, results
            for constraint, result in zip(self._constraints, results):
                if not result.passed:
                    repaired = constraint.repair(current_route, context)
                    if repaired is not None:
                        current_route = repaired
                        break
            else:
                break
        return current_route, self.check_all(current_route, context)

    @property
    def constraints(self) -> list[AbstractConstraint]:
        return list(self._constraints)
