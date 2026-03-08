"""All 7 scoring factors from spec §8. Lower = better."""
from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy as np

from .base import AbstractScoringFactor

if TYPE_CHECKING:
    from neuralpipe.models.route import Route


class RouteLengthFactor(AbstractScoringFactor):
    """Total route length. 1.0× per metre. Shorter is preferred."""
    name = "route_length"
    weight = 1.0

    def score(self, route: "Route", context: dict[str, Any]) -> float:
        return route.total_length_m


class ElbowCountFactor(AbstractScoringFactor):
    """Number of elbows. 15× per elbow. Each direction change costs pressure drop."""
    name = "elbow_count"
    weight = 15.0

    def score(self, route: "Route", context: dict[str, Any]) -> float:
        return float(route.num_elbows)


class SupportCountFactor(AbstractScoringFactor):
    """Number of supports. 5× per support. Fewer = less civil/structural work."""
    name = "support_count"
    weight = 5.0

    def score(self, route: "Route", context: dict[str, Any]) -> float:
        return float(route.num_supports)


class GradeRoutingFactor(AbstractScoringFactor):
    """Grade routing segments. 50× per metre. Strong penalty — last resort."""
    name = "grade_routing"
    weight = 50.0

    def score(self, route: "Route", context: dict[str, Any]) -> float:
        grade_length = sum(
            route.segments[i].length_m
            for i in route.grade_segment_indices
            if i < len(route.segments)
        )
        return grade_length


class DeviationFactor(AbstractScoringFactor):
    """Deviation from direct path. 2× per metre of excess. Penalises wandering."""
    name = "deviation_from_direct"
    weight = 2.0

    def score(self, route: "Route", context: dict[str, Any]) -> float:
        if len(route.waypoints) < 2:
            return 0.0
        start = route.waypoints[0]
        end = route.waypoints[-1]
        dx = end.x - start.x
        dy = end.y - start.y
        dz = end.z - start.z
        direct_length = math.sqrt(dx**2 + dy**2 + dz**2)
        excess = max(0.0, route.total_length_m - direct_length)
        return excess


class ElevationChangeFactor(AbstractScoringFactor):
    """Vertical run length. 0.5× per metre. Mild preference for flat runs."""
    name = "elevation_changes"
    weight = 0.5

    def score(self, route: "Route", context: dict[str, Any]) -> float:
        vertical_length = 0.0
        for seg in route.segments:
            dz = abs(seg.end.z - seg.start.z)
            if dz > 1e-6:
                vertical_length += seg.length_m
        return vertical_length


class CongestionFactor(AbstractScoringFactor):
    """Congestion factor. 10× per nearby line. Routes through dense corridors score worse.

    'Nearby' is defined as: any existing route waypoint within congestion_radius_m
    of any waypoint on this route. The context must supply 'existing_routes'.
    """
    name = "congestion"
    weight = 10.0

    def __init__(self, congestion_radius_m: float = 0.5) -> None:
        self.congestion_radius_m = congestion_radius_m

    def score(self, route: "Route", context: dict[str, Any]) -> float:
        existing_routes: list = context.get("existing_routes", [])
        if not existing_routes:
            return 0.0

        nearby_lines: set[str] = set()
        for wp in route.waypoints:
            wp_arr = np.array([wp.x, wp.y, wp.z])
            for existing in existing_routes:
                if existing.route_id == route.route_id:
                    continue
                for ex_wp in existing.waypoints:
                    ex_arr = np.array([ex_wp.x, ex_wp.y, ex_wp.z])
                    if float(np.linalg.norm(wp_arr - ex_arr)) <= self.congestion_radius_m:
                        nearby_lines.add(existing.line_number or existing.route_id)
                        break

        return float(len(nearby_lines))


def build_default_scoring_registry():
    """Return a ScoringRegistry pre-loaded with all 7 spec-mandated factors."""
    from neuralpipe.scoring.registry import ScoringRegistry
    registry = ScoringRegistry()
    registry.register(RouteLengthFactor())
    registry.register(ElbowCountFactor())
    registry.register(SupportCountFactor())
    registry.register(GradeRoutingFactor())
    registry.register(DeviationFactor())
    registry.register(ElevationChangeFactor())
    registry.register(CongestionFactor())
    return registry
