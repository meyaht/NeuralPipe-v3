"""LR Elbow geometry — envelope computation at each waypoint.

The spec requires: LR elbow centerline radius = 1.5 × NPS (inches → mm).
The routing geometry must respect the physical envelope of the LR elbow at
every direction change, not just the centerline waypoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ElbowEnvelope:
    """Bounding sphere approximation of a 90° LR elbow sweep.

    For v1 we use a conservative bounding sphere centred at the waypoint.
    The sphere radius is the hypotenuse of the elbow tangent length in each
    run direction: sqrt(2) * R where R is the centerline bend radius.
    This is conservative (the actual swept volume is smaller) but safe for
    clash detection.
    """
    centre: tuple[float, float, float]  # waypoint in metres
    radius_m: float  # conservative bounding radius in metres


def lr_elbow_radius_m(nps_inches: float) -> float:
    """LR elbow centerline radius in metres for a given NPS (inches)."""
    return 1.5 * nps_inches * 0.0254  # 1 inch = 0.0254 m


def compute_elbow_envelope(
    waypoint: tuple[float, float, float],
    nps_inches: float,
    eod_m: float,
) -> ElbowEnvelope:
    """Return bounding sphere for LR elbow at waypoint.

    Args:
        waypoint: (x, y, z) in metres — the direction-change point
        nps_inches: nominal pipe size in inches
        eod_m: effective outside diameter in metres (pipe + insulation)

    Returns:
        ElbowEnvelope with conservative sphere radius = sqrt(2) * R_cl + EOD/2
    """
    R_cl = lr_elbow_radius_m(nps_inches)
    # Tangent length from waypoint to tangent points = R_cl (for 90° elbows)
    # Conservative bounding sphere: cover both tangent points
    bounding_radius = math.sqrt(2) * R_cl + eod_m / 2
    return ElbowEnvelope(centre=waypoint, radius_m=bounding_radius)


def elbow_tangent_length(nps_inches: float) -> float:
    """Distance from elbow apex (waypoint) to tangent point on each run (metres).

    For a 90° LR elbow, the tangent length = R_cl (the geometry is a quarter-circle).
    """
    return lr_elbow_radius_m(nps_inches)


def check_elbow_clearance(
    waypoint: tuple[float, float, float],
    nps_inches: float,
    eod_m: float,
    obstacle_centre: tuple[float, float, float],
    obstacle_radius_m: float = 0.0,
    clearance_m: float = 0.05,
) -> bool:
    """Return True if the elbow envelope is clear of an obstacle sphere.

    Args:
        waypoint: LR elbow apex in metres
        nps_inches: NPS in inches
        eod_m: effective OD in metres
        obstacle_centre: centre of obstacle in metres
        obstacle_radius_m: radius of obstacle bounding sphere
        clearance_m: required separation (default 50 mm)
    """
    envelope = compute_elbow_envelope(waypoint, nps_inches, eod_m)
    wx, wy, wz = envelope.centre
    ox, oy, oz = obstacle_centre
    dist = math.sqrt((wx - ox) ** 2 + (wy - oy) ** 2 + (wz - oz) ** 2)
    return dist >= (envelope.radius_m + obstacle_radius_m + clearance_m)


def direction_unit_vector(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> np.ndarray:
    """Return unit vector from p1 to p2."""
    v = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        raise ValueError(f"Zero-length segment between {p1} and {p2}")
    return v / norm


def angle_between_segments_deg(
    p_in: tuple[float, float, float],
    apex: tuple[float, float, float],
    p_out: tuple[float, float, float],
) -> float:
    """Return the turn angle at apex in degrees (0 = straight, 90 = elbow)."""
    d_in = direction_unit_vector(p_in, apex)
    d_out = direction_unit_vector(apex, p_out)
    dot = float(np.dot(d_in, d_out))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))
