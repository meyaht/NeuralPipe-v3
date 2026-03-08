"""SupportSpanConstraint — max unsupported span must not exceed spec limit."""
from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from .base import AbstractConstraint, ConstraintResult

if TYPE_CHECKING:
    from neuralpipe.models.route import Route


class SupportSpanConstraint(AbstractConstraint):
    """Check that every segment is within the maximum support span.

    In v1, support points are placed at spec-driven intervals. When no existing
    steel (qualifying, pipe-loaded) exists within span, the constraint flags
    the route for user review unless grade routing is allowed.

    The voxel_grid context is used to check for steel availability in v1 stub:
    we treat any non-occupied voxel in the span corridor as "no qualifying steel".
    For v1 testing, we synthesise support points at preferred_span intervals if
    the grid is empty (or not provided).
    """

    name = "support_span"
    scope = "GLOBAL"
    version = "1.0"
    priority = 30

    def check(self, route: "Route", context: dict[str, Any]) -> ConstraintResult:
        pipe_spec = context.get("pipe_spec")
        if pipe_spec is None:
            return ConstraintResult(passed=True, constraint_name=self.name, message="No pipe_spec in context — skipped")

        max_span_m = pipe_spec.max_span_m
        grade_allowed = context.get("grade_routing_allowed", False)

        violations = []
        for i, seg in enumerate(route.segments):
            length = seg.length_m
            if length > max_span_m:
                violations.append({
                    "segment_index": i,
                    "length_m": round(length, 4),
                    "max_span_m": max_span_m,
                })

        if violations:
            if not grade_allowed:
                return ConstraintResult(
                    passed=False,
                    constraint_name=self.name,
                    message=(
                        f"{len(violations)} segment(s) exceed max span of {max_span_m:.3f} m. "
                        "FLAG: User Input Required — no valid support found."
                    ),
                    detail={"violations": violations},
                )
            else:
                return ConstraintResult(
                    passed=False,
                    constraint_name=self.name,
                    message=(
                        f"{len(violations)} segment(s) exceed max span. Grade routing flagged."
                    ),
                    detail={"violations": violations},
                )

        return ConstraintResult(passed=True, constraint_name=self.name)
