"""3D A* search on voxel grid for pipe route generation.

The search operates on a discrete grid aligned to the current resolution pass
step size. Only orthogonal moves (±X, ±Y, ±Z) are considered, enforcing the
spec requirement for rectilinear routing.

The heuristic is Manhattan distance to the goal (admissible for orthogonal moves).

Each candidate route returned is a list of Waypoints; the caller is responsible
for converting them into Segments and attaching elbow/support metadata.
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Optional

from neuralpipe.models.route import Route, Segment, Waypoint, ElbowFitting, SupportPoint
from neuralpipe.geometry.voxel_grid import VoxelGridProvider
from neuralpipe.geometry.elbow import direction_unit_vector
from neuralpipe.routing.resolution import ResolutionPass


XYZ = tuple[float, float, float]

# Six orthogonal neighbours
_DIRECTIONS: list[XYZ] = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


def _snap(value: float, step: float) -> float:
    """Snap value to nearest multiple of step."""
    return round(value / step) * step


def _snap_xyz(xyz: XYZ, step: float) -> XYZ:
    return (_snap(xyz[0], step), _snap(xyz[1], step), _snap(xyz[2], step))


def _manhattan(a: XYZ, b: XYZ, step: float) -> float:
    return (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) / step


def _count_turns(path: list[XYZ]) -> int:
    """Count direction changes in a node path."""
    if len(path) < 3:
        return 0
    turns = 0
    for i in range(1, len(path) - 1):
        dx1, dy1, dz1 = path[i][0]-path[i-1][0], path[i][1]-path[i-1][1], path[i][2]-path[i-1][2]
        dx2, dy2, dz2 = path[i+1][0]-path[i][0], path[i+1][1]-path[i][1], path[i+1][2]-path[i][2]
        if (dx1, dy1, dz1) != (dx2, dy2, dz2):
            turns += 1
    return turns


class AStarRouter:
    """3D orthogonal A* router operating at a fixed step size.

    Attributes:
        step_m: grid step size in metres
        max_nodes: abort search after this many nodes (prevents runaway)
        elbow_penalty: cost added per direction change (encourages fewer elbows)
    """

    def __init__(
        self,
        step_m: float,
        voxel_grid: Optional[VoxelGridProvider] = None,
        eod_m: float = 0.0,
        clearance_m: float = 0.050,
        elbow_penalty: float = 15.0,
        max_nodes: int = 100_000,
    ) -> None:
        self.step_m = step_m
        self.voxel_grid = voxel_grid
        self.eod_m = eod_m
        self.clearance_m = clearance_m
        self.elbow_penalty = elbow_penalty
        self.max_nodes = max_nodes

    def _is_free(self, xyz: XYZ) -> bool:
        """Return True if the voxel at xyz is not occupied by an obstacle."""
        if self.voxel_grid is None:
            return True
        # Check a small neighbourhood around the pipe centreline
        r = self.eod_m / 2 + self.clearance_m
        # For v1: check the centre point only (voxel grid resolution handles the rest)
        return not self.voxel_grid.is_occupied(xyz)

    def search(
        self,
        start: XYZ,
        goal: XYZ,
        route_id: str = "ROUTE-UNKNOWN",
        line_number: str = "",
        nps_inches: float = 6.0,
    ) -> Optional[Route]:
        """Run A* from start to goal.

        Returns a Route with waypoints and segments, or None if no path found.
        """
        step = self.step_m
        s = _snap_xyz(start, step)
        g = _snap_xyz(goal, step)

        # Priority queue: (f, g_cost, node, parent)
        open_heap: list[tuple[float, float, XYZ, Optional[XYZ]]] = []
        # g_costs[node] = best cost to reach node
        g_costs: dict[XYZ, float] = {s: 0.0}
        # came_from[node] = (parent_node, direction_from_parent)
        came_from: dict[XYZ, Optional[tuple[XYZ, XYZ]]] = {s: None}

        h0 = _manhattan(s, g, step)
        heapq.heappush(open_heap, (h0, 0.0, s, None))

        nodes_expanded = 0
        prev_dir: dict[XYZ, Optional[XYZ]] = {s: None}

        while open_heap:
            f, g_cost, current, parent = heapq.heappop(open_heap)
            nodes_expanded += 1

            if nodes_expanded > self.max_nodes:
                break

            if current == g:
                return self._reconstruct_route(
                    came_from, g, start, goal, route_id, line_number, nps_inches
                )

            if g_cost > g_costs.get(current, math.inf):
                continue

            cur_dir = prev_dir.get(current)

            for dx, dy, dz in _DIRECTIONS:
                nx = _snap(current[0] + dx * step, step)
                ny = _snap(current[1] + dy * step, step)
                nz = _snap(current[2] + dz * step, step)
                neighbour: XYZ = (nx, ny, nz)

                if not self._is_free(neighbour):
                    continue

                new_dir: XYZ = (dx, dy, dz)
                move_cost = step  # 1 step = step_m cost
                if cur_dir is not None and new_dir != cur_dir:
                    move_cost += self.elbow_penalty

                tentative_g = g_cost + move_cost

                if tentative_g < g_costs.get(neighbour, math.inf):
                    g_costs[neighbour] = tentative_g
                    prev_dir[neighbour] = new_dir
                    came_from[neighbour] = (current, new_dir)
                    h = _manhattan(neighbour, g, step) * step
                    heapq.heappush(open_heap, (tentative_g + h, tentative_g, neighbour, current))

        return None  # No path found

    def _reconstruct_route(
        self,
        came_from: dict[XYZ, Optional[tuple[XYZ, XYZ]]],
        goal: XYZ,
        start_orig: XYZ,
        goal_orig: XYZ,
        route_id: str,
        line_number: str,
        nps_inches: float,
    ) -> Route:
        """Reconstruct waypoint list from came_from map."""
        path: list[XYZ] = []
        node: Optional[XYZ] = goal
        while node is not None:
            path.append(node)
            entry = came_from.get(node)
            node = entry[0] if entry is not None else None
        path.reverse()

        # Replace first/last with exact coordinates
        if path:
            path[0] = start_orig
            path[-1] = goal_orig

        # Simplify: merge consecutive collinear points
        path = _simplify_path(path)

        waypoints = [Waypoint(x, y, z) for x, y, z in path]
        segments = [
            Segment(start=waypoints[i], end=waypoints[i + 1])
            for i in range(len(waypoints) - 1)
        ]

        # Build elbow fittings at direction changes
        elbows: list[ElbowFitting] = []
        for i in range(1, len(waypoints) - 1):
            prev_wp = waypoints[i - 1]
            curr_wp = waypoints[i]
            next_wp = waypoints[i + 1]
            try:
                d_in = direction_unit_vector(prev_wp.as_tuple(), curr_wp.as_tuple())
                d_out = direction_unit_vector(curr_wp.as_tuple(), next_wp.as_tuple())
                # Check if it's actually a turn (not collinear)
                if not all(abs(d_in[k] - d_out[k]) < 1e-6 for k in range(3)):
                    from neuralpipe.geometry.elbow import lr_elbow_radius_m
                    elbows.append(ElbowFitting(
                        location=curr_wp,
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
            eod_m=self.eod_m,
        )


def _simplify_path(path: list[XYZ]) -> list[XYZ]:
    """Remove intermediate points that are collinear with their neighbours."""
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        prev = simplified[-1]
        curr = path[i]
        nxt = path[i + 1]
        # Direction prev→curr
        d1 = (curr[0] - prev[0], curr[1] - prev[1], curr[2] - prev[2])
        # Direction curr→next
        d2 = (nxt[0] - curr[0], nxt[1] - curr[1], nxt[2] - curr[2])
        # Normalise
        n1 = math.sqrt(sum(x**2 for x in d1))
        n2 = math.sqrt(sum(x**2 for x in d2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        d1n = tuple(x / n1 for x in d1)
        d2n = tuple(x / n2 for x in d2)
        if all(abs(d1n[k] - d2n[k]) < 1e-6 for k in range(3)):
            continue  # collinear — skip
        simplified.append(curr)
    simplified.append(path[-1])
    return simplified


def generate_candidates(
    start: XYZ,
    goal: XYZ,
    resolution_pass: ResolutionPass,
    voxel_grid: Optional[VoxelGridProvider],
    eod_m: float,
    nps_inches: float,
    route_id_prefix: str = "ROUTE",
    line_number: str = "",
    num_candidates: int = 5,
) -> list[Route]:
    """Generate multiple route candidates using A* with varied elbow penalties.

    Varying the elbow penalty biases the search toward routes with different
    numbers of direction changes, producing a diverse candidate set.
    """
    penalties = [15.0, 5.0, 30.0, 10.0, 50.0][:num_candidates]
    candidates = []

    for i, penalty in enumerate(penalties):
        router = AStarRouter(
            step_m=resolution_pass.step_size_m,
            voxel_grid=voxel_grid,
            eod_m=eod_m,
            elbow_penalty=penalty,
        )
        route_id = f"{route_id_prefix}-{str(i + 1).zfill(3)}"
        route = router.search(
            start=start,
            goal=goal,
            route_id=route_id,
            line_number=line_number,
            nps_inches=nps_inches,
        )
        if route is not None:
            candidates.append(route)

    # Deduplicate: remove routes with identical waypoints
    seen: set[tuple] = set()
    unique: list[Route] = []
    for r in candidates:
        key = tuple(wp.as_tuple() for wp in r.waypoints)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique
