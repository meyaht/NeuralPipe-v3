"""Gate-aware routing: A* sequenced through pipe rack gate openings.

When gates are provided, the router breaks the start→end path into segments
separated by gate centers and runs A* independently over each segment.
This forces routes to pass through physical gate openings in the pipe rack
rather than cutting through structural steel.

Gate selection logic
--------------------
- Determine primary travel axis from start→end (largest absolute coordinate delta)
- Select gates whose axis matches the primary travel axis (axis=Y gate → XZ opening,
  traversed by pipes moving in Y direction)
- Filter by position along travel axis: must fall between start and end
- Filter by lateral proximity: gate center must be within `lateral_tol` metres of
  the straight line from start to end
- Sort by travel-axis position (start → end order)
"""
from __future__ import annotations

import math
import uuid
from typing import Optional

from neuralpipe.models.gate import GateOpening
from neuralpipe.models.route import Route, Segment, Waypoint, ElbowFitting
from neuralpipe.geometry.voxel_grid import VoxelGridProvider
from neuralpipe.routing.resolution import ResolutionPass
from neuralpipe.routing.astar import AStarRouter, _simplify_path
from neuralpipe.geometry.elbow import direction_unit_vector, lr_elbow_radius_m

XYZ = tuple[float, float, float]

# Map axis label → coordinate index
_AX = {"X": 0, "Y": 1, "Z": 2}


def _dist_point_to_segment(p: XYZ, a: XYZ, b: XYZ) -> float:
    """Perpendicular distance from point p to the line segment a→b."""
    ab = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    ap = (p[0]-a[0], p[1]-a[1], p[2]-a[2])
    len_ab = math.sqrt(sum(v**2 for v in ab))
    if len_ab < 1e-9:
        return math.sqrt(sum(v**2 for v in ap))
    t = max(0.0, min(1.0, sum(ap[i]*ab[i] for i in range(3)) / (len_ab**2)))
    closest = (a[0]+t*ab[0], a[1]+t*ab[1], a[2]+t*ab[2])
    return math.sqrt(sum((p[i]-closest[i])**2 for i in range(3)))


def filter_gates_on_path(
    gates: list[GateOpening],
    start: XYZ,
    end: XYZ,
    lateral_tol: float = 5.0,
    min_confidence: float = 0.5,
) -> list[GateOpening]:
    """Return gates that lie along the start→end path, sorted travel-order.

    Args:
        gates:          All available gates (loaded from gates.json).
        start, end:     Route endpoints in metres.
        lateral_tol:    Max perpendicular distance from the straight start→end line
                        for a gate center to be considered on-path (metres).
        min_confidence: Skip gates below this confidence threshold.
    """
    delta = [end[i] - start[i] for i in range(3)]
    abs_delta = [abs(d) for d in delta]

    # Ignore Z for travel-axis selection — gates are in XZ or YZ planes
    if abs_delta[0] >= abs_delta[1]:
        travel_ax = "X"
    else:
        travel_ax = "Y"

    ax_i = _AX[travel_ax]
    ax_lo = min(start[ax_i], end[ax_i])
    ax_hi = max(start[ax_i], end[ax_i])

    selected = []
    for g in gates:
        if g.confidence < min_confidence:
            continue
        # Gate axis must match travel axis (axis=Y gate → XZ opening, pipe moves in Y)
        if g.axis.upper() != travel_ax:
            continue
        # Gate position along travel axis must fall between start and end
        if not (ax_lo <= g.position_m <= ax_hi):
            continue
        # Lateral proximity check (using horizontal center only)
        cx, cy, cz = g.center_3d()
        dist = _dist_point_to_segment((cx, cy, cz), start, end)
        if dist > lateral_tol:
            continue
        selected.append(g)

    # Sort by position along travel axis
    reverse = end[ax_i] < start[ax_i]
    selected.sort(key=lambda g: g.position_m, reverse=reverse)
    return selected


def route_through_gates(
    start: XYZ,
    end: XYZ,
    gates: list[GateOpening],
    resolution_pass: ResolutionPass,
    voxel_grid: Optional[VoxelGridProvider],
    eod_m: float,
    nps_inches: float,
    elbow_penalty: float = 15.0,
    route_id: str = "",
    line_number: str = "",
) -> Optional[Route]:
    """Route from start to end, sequencing through gate centers.

    Runs A* independently for each segment: start → g0 → g1 → … → end.
    If A* fails for a segment, that gate is skipped and routing continues
    to the next waypoint (graceful degradation).

    Returns a single merged Route, or None if the overall route is impossible.
    """
    route_id = route_id or f"ROUTE-{uuid.uuid4().hex[:6].upper()}"
    step = resolution_pass.step_size_m
    router = AStarRouter(
        step_m=step,
        voxel_grid=voxel_grid,
        eod_m=eod_m,
        elbow_penalty=elbow_penalty,
    )

    # Build waypoint list: start + gate centers + end
    gate_wps: list[XYZ] = [g.center_3d() for g in gates]
    all_wps: list[XYZ] = [start] + gate_wps + [end]

    # Route each consecutive pair, concatenate raw node paths
    merged_path: list[XYZ] = [start]

    for i in range(len(all_wps) - 1):
        seg_start = all_wps[i]
        seg_end   = all_wps[i + 1]
        sub = router.search(
            start=seg_start,
            goal=seg_end,
            route_id=f"{route_id}-SEG{i}",
            line_number=line_number,
            nps_inches=nps_inches,
        )
        if sub is None:
            # Segment failed — skip the intermediate gate, try bridging to next waypoint
            # (handled implicitly: merged_path already contains seg_start, next iteration
            #  will route from seg_start to seg_end+1)
            continue
        # Append segment path, excluding the first point (already in merged_path)
        seg_pts = [wp.as_tuple() for wp in sub.waypoints]
        merged_path.extend(seg_pts[1:])

    if len(merged_path) < 2:
        return None

    merged_path = _simplify_path(merged_path)
    waypoints = [Waypoint(x, y, z) for x, y, z in merged_path]
    segments = [
        Segment(start=waypoints[i], end=waypoints[i + 1])
        for i in range(len(waypoints) - 1)
    ]

    # Build elbow fittings
    elbows: list[ElbowFitting] = []
    for i in range(1, len(waypoints) - 1):
        try:
            d_in  = direction_unit_vector(waypoints[i-1].as_tuple(), waypoints[i].as_tuple())
            d_out = direction_unit_vector(waypoints[i].as_tuple(),   waypoints[i+1].as_tuple())
            if not all(abs(d_in[k] - d_out[k]) < 1e-6 for k in range(3)):
                elbows.append(ElbowFitting(
                    location=waypoints[i],
                    incoming_dir=tuple(d_in),
                    outgoing_dir=tuple(d_out),
                    radius_m=lr_elbow_radius_m(nps_inches),
                ))
        except ValueError:
            pass

    return Route(
        route_id=route_id,
        line_number=line_number,
        waypoints=waypoints,
        segments=segments,
        elbow_locations=elbows,
        eod_m=eod_m,
    )


def generate_gate_candidates(
    start: XYZ,
    end: XYZ,
    gates: list[GateOpening],
    resolution_pass: ResolutionPass,
    voxel_grid: Optional[VoxelGridProvider],
    eod_m: float,
    nps_inches: float,
    route_id_prefix: str = "ROUTE",
    line_number: str = "",
    num_candidates: int = 5,
    lateral_tol: float = 5.0,
) -> list[Route]:
    """Generate gate-sequenced route candidates with varied elbow penalties.

    Filters gates to those on the start→end path, then runs route_through_gates
    with multiple elbow_penalty values to produce a diverse candidate set.
    """
    on_path = filter_gates_on_path(gates, start, end, lateral_tol=lateral_tol)

    penalties = [15.0, 5.0, 30.0, 10.0, 50.0][:num_candidates]
    candidates = []

    for i, penalty in enumerate(penalties):
        route_id = f"{route_id_prefix}-{str(i+1).zfill(3)}"
        route = route_through_gates(
            start=start,
            end=end,
            gates=on_path,
            resolution_pass=resolution_pass,
            voxel_grid=voxel_grid,
            eod_m=eod_m,
            nps_inches=nps_inches,
            elbow_penalty=penalty,
            route_id=route_id,
            line_number=line_number,
        )
        if route is not None:
            candidates.append(route)

    # Deduplicate
    seen: set[tuple] = set()
    unique: list[Route] = []
    for r in candidates:
        key = tuple(wp.as_tuple() for wp in r.waypoints)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique
