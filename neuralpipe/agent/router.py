"""NeuralPipeAgent — main orchestration class (the Python API surface)."""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from neuralpipe.models.pipe import PipeSpecRegistry
from neuralpipe.models.route import Route, Waypoint
from neuralpipe.models.feedback import DQRecord
from neuralpipe.geometry.voxel_grid import VoxelGridProvider, InMemoryVoxelGrid
from neuralpipe.constraints.registry import ConstraintRegistry
from neuralpipe.constraints.orthogonality import OrthogonalityConstraint
from neuralpipe.constraints.support_span import SupportSpanConstraint
from neuralpipe.constraints.clearance import EODClearanceConstraint, SurfaceClearanceConstraint
from neuralpipe.constraints.exclusion import ExclusionZoneConstraint
from neuralpipe.scoring.factors import build_default_scoring_registry
from neuralpipe.feedback.store import FeedbackStore, SQLiteFeedbackStore
from neuralpipe.routing.resolution import RESOLUTION_LADDER, get_pass
from neuralpipe.routing.astar import generate_candidates
from neuralpipe.routing.gate_router import generate_gate_candidates
from neuralpipe.models.gate import GateOpening

logger = logging.getLogger(__name__)

# Default config path, relative to the project root
_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "pipe_specs.json"
_DEFAULT_DB = Path(__file__).parent.parent.parent / "neuralpipe_dq.db"


def _compute_eod(od_m: float, insulation_thickness_mm: Optional[float]) -> float:
    """EOD = bare_pipe_OD + (2 × insulation_thickness) + 3mm jacket allowance."""
    if insulation_thickness_mm:
        return od_m + 2 * (insulation_thickness_mm / 1000.0) + 0.003
    return od_m


def _make_route_id(line_number: str, index: int) -> str:
    short = (line_number or "LINE")[:8].replace(" ", "").upper()
    return f"ROUTE-{short}-{str(index + 1).zfill(3)}"


class NeuralPipeAgent:
    """Main public API for NeuralPipe.

    Typical usage:
        agent = NeuralPipeAgent()
        candidates = agent.route(start=..., end=..., nominal_diameter=6, ...)
        agent.disqualify(route_id=candidates[1].route_id, reason="Too close to crane")
        history = agent.get_dq_history(line_number="6-HGO-1001-A1A")

    The agent is extensible:
        - Inject a custom VoxelGridProvider via voxel_grid parameter
        - Add constraints via agent.constraint_registry.register(...)
        - Add scoring factors via agent.scoring_registry.register(...)
        - Swap the feedback store via feedback_store parameter
    """

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_CONFIG,
        voxel_grid: Optional[VoxelGridProvider] = None,
        feedback_store: Optional[FeedbackStore] = None,
        db_path: str | Path = _DEFAULT_DB,
        num_candidates: int = 5,
    ) -> None:
        # Pipe spec registry
        self.pipe_specs = PipeSpecRegistry(config_path)

        # Voxel grid (stub for v1; swap with COPCVoxelGrid for real scan data)
        self.voxel_grid: VoxelGridProvider = voxel_grid or InMemoryVoxelGrid()

        # Constraint registry — pre-loaded with all v1 constraints
        self.constraint_registry = ConstraintRegistry()
        self.constraint_registry.register(ExclusionZoneConstraint())   # priority 10
        self.constraint_registry.register(OrthogonalityConstraint())   # priority 20
        self.constraint_registry.register(SupportSpanConstraint())     # priority 30
        self.constraint_registry.register(EODClearanceConstraint())    # priority 50
        self.constraint_registry.register(SurfaceClearanceConstraint())  # priority 55

        # Scoring registry — pre-loaded with all 7 spec factors
        self.scoring_registry = build_default_scoring_registry()

        # Feedback store
        self.feedback_store: FeedbackStore = feedback_store or SQLiteFeedbackStore(db_path)

        self.num_candidates = num_candidates

        # In-memory store of route candidates (most recent call)
        self._last_candidates: dict[str, Route] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        nominal_diameter: float,
        pipe_spec: str,
        fluid_service: str,
        grade_routing_allowed: bool,
        line_number: str,
        insulation_type: Optional[str] = None,
        insulation_thickness_mm: Optional[float] = None,
        existing_routes: Optional[list[Route]] = None,
        exclusion_zones: Optional[list[dict]] = None,
        surfaces: Optional[list[dict]] = None,
        gates: Optional[list[GateOpening]] = None,
        gate_lateral_tol: float = 5.0,
    ) -> list[Route]:
        """Generate ranked route candidates.

        Args:
            start: (x, y, z) in metres — origin nozzle/tie-in
            end:   (x, y, z) in metres — destination nozzle/tie-in
            nominal_diameter: NPS in inches (e.g. 6.0)
            pipe_spec: piping class string (e.g. "A1A")
            fluid_service: service tag (e.g. "HGO")
            grade_routing_allowed: whether grade routing is permitted for this line
            line_number: full line number tag
            insulation_type: "PERS" | "HOTC" | "COLD" | "TRACE" | None
            insulation_thickness_mm: thickness in mm if insulated
            existing_routes: other Route objects already in the space
            exclusion_zones: list of exclusion zone dicts (see ExclusionZoneConstraint)
            surfaces: list of surface dicts (see SurfaceClearanceConstraint)

        Returns:
            List of Route candidates, sorted by score (lowest first).
            Empty list if no valid routes found.
        """
        # Resolve pipe spec
        try:
            pspec = self.pipe_specs.get_or_interpolate(nominal_diameter)
        except ValueError as e:
            logger.error(f"Pipe spec not found: {e}")
            return []

        # Compute EOD
        eod_m = _compute_eod(pspec.od_m, insulation_thickness_mm)

        # Build context dict passed to constraints and scoring
        context: dict = {
            "pipe_spec": pspec,
            "nps_inches": nominal_diameter,
            "eod_m": eod_m,
            "grade_routing_allowed": grade_routing_allowed,
            "voxel_grid": self.voxel_grid,
            "existing_routes": existing_routes or [],
            "exclusion_zones": exclusion_zones or [],
            "surfaces": surfaces or [],
            "pipe_spec_str": pipe_spec,
            "fluid_service": fluid_service,
            "line_number": line_number,
        }

        # Select routing resolution pass — use pass 5 (1ft/305mm) for v1 main routing
        # This balances candidate quality with speed. Fine-tuning happens in constraint repair.
        routing_pass = get_pass(5)

        # Generate raw candidates — gate-sequenced if gates provided, direct A* otherwise
        prefix = f"ROUTE-{(line_number or 'LINE')[:6].upper()}"
        if gates:
            raw_candidates = generate_gate_candidates(
                start=start,
                end=end,
                gates=gates,
                resolution_pass=routing_pass,
                voxel_grid=self.voxel_grid,
                eod_m=eod_m,
                nps_inches=nominal_diameter,
                route_id_prefix=prefix,
                line_number=line_number,
                num_candidates=self.num_candidates,
                lateral_tol=gate_lateral_tol,
            )
        else:
            raw_candidates = generate_candidates(
                start=start,
                goal=end,
                resolution_pass=routing_pass,
                voxel_grid=self.voxel_grid,
                eod_m=eod_m,
                nps_inches=nominal_diameter,
                route_id_prefix=prefix,
                line_number=line_number,
                num_candidates=self.num_candidates,
            )

        if not raw_candidates:
            logger.warning(f"A* found no paths from {start} to {end}")
            return []

        # Run support-point placement (simple: place at preferred_span intervals)
        for route in raw_candidates:
            self._place_supports(route, pspec, eod_m)

        # Evaluate constraints (pass 5 onwards: orthogonality + elbow; pass 8: fine clearance)
        context["resolution_pass"] = 5
        valid_routes = []
        for route in raw_candidates:
            passed, results = self.constraint_registry.check_all_pass(route, context)
            failing = [r for r in results if not r.passed]

            if not passed:
                # Check if any failures are hard-fail (exclusion zone)
                hard_fails = [r for r in failing if r.constraint_name == "exclusion_zone"]
                if hard_fails:
                    route.status = "FAILED"
                    route.failure_reason = "; ".join(r.message for r in hard_fails)
                    logger.debug(f"Route {route.route_id} FAILED: {route.failure_reason}")
                    continue
                else:
                    # Soft failures → FLAGGED
                    route.status = "FLAGGED"
                    for r in failing:
                        route.flags.append(f"[{r.constraint_name}] {r.message}")

            # Score the route
            score, breakdown = self.scoring_registry.score(route, context)
            route.score = score
            route.score_breakdown = breakdown
            valid_routes.append(route)

        # Sort by score (lower = better)
        valid_routes.sort(key=lambda r: r.score)

        # Store for later DQ reference
        self._last_candidates = {r.route_id: r for r in valid_routes}

        return valid_routes

    def disqualify(self, route_id: str, reason: str) -> DQRecord:
        """Disqualify a route candidate with a free-text reason.

        Claude parses the reason into a structured DQRecord. Falls back to a
        minimal record if Claude is unavailable.

        Args:
            route_id: the route_id of the candidate to disqualify
            reason: engineer's free-text explanation

        Returns:
            The stored DQRecord.
        """
        route = self._last_candidates.get(route_id)
        if route is None:
            # Build a minimal stub route for parsing
            from neuralpipe.models.route import Route
            route = Route(
                route_id=route_id,
                line_number=route_id,
            )

        from neuralpipe.agent.claude_client import parse_dq
        record = parse_dq(free_text=reason, route=route)
        self.feedback_store.save(record)
        logger.info(f"DQ recorded: {record.dq_id} for route {route_id}")
        return record

    def get_dq_history(self, line_number: str) -> list[DQRecord]:
        """Return all DQ records for a given line number."""
        return self.feedback_store.get_by_line(line_number)

    def explain_route(self, route_id: str, context: Optional[dict] = None) -> str:
        """Return a Claude-generated markdown explanation for a route."""
        route = self._last_candidates.get(route_id)
        if route is None:
            return f"Route {route_id} not found in current candidate set."
        from neuralpipe.agent.claude_client import explain_route
        return explain_route(route, context or {})

    def suggest_constraint_from_dq(self, dq_id: str) -> Optional[str]:
        """Ask Claude whether a DQ should become a permanent spec rule."""
        all_records = self.feedback_store.get_all()
        record = next((r for r in all_records if r.dq_id == dq_id), None)
        if record is None:
            return None
        from neuralpipe.agent.claude_client import suggest_constraint
        return suggest_constraint(record)

    def promote_to_spec(self, dq_id: str, rule_text: str, spec_path: Optional[Path] = None) -> bool:
        """Append a derived rule to ROUTING_SPEC.md and mark the DQ as applied.

        Args:
            dq_id: the DQRecord.dq_id to mark as applied
            rule_text: the plain-English rule to append
            spec_path: path to ROUTING_SPEC.md (defaults to project root)

        Returns:
            True on success, False if spec file not found.
        """
        if spec_path is None:
            spec_path = Path(__file__).parent.parent.parent / "ROUTING_SPEC.md"

        if not spec_path.exists():
            logger.error(f"ROUTING_SPEC.md not found at {spec_path}")
            return False

        existing = spec_path.read_text()
        # Bump version number
        import re
        version_match = re.search(r"\*\*Version:\*\* (\d+\.\d+)", existing)
        if version_match:
            old_ver = version_match.group(1)
            major, minor = old_ver.split(".")
            new_ver = f"{major}.{int(minor) + 1}"
            existing = existing.replace(f"**Version:** {old_ver}", f"**Version:** {new_ver}")

        appendix = f"\n\n---\n\n## Derived Rule (from DQ {dq_id})\n\n{rule_text}\n"
        spec_path.write_text(existing + appendix)

        self.feedback_store.mark_applied_to_spec(dq_id)
        logger.info(f"Promoted DQ {dq_id} to spec: {rule_text}")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_supports(self, route: Route, pspec, eod_m: float) -> None:
        """Place support points at preferred_span intervals along each segment.

        v1 stub: we don't check for qualifying steel; we place at preferred span
        intervals to give the scoring factors meaningful data. The SupportSpanConstraint
        checks that no single span exceeds max_span.
        """
        from neuralpipe.models.route import SupportPoint, Waypoint
        import math

        route.support_points.clear()
        route.grade_segment_indices.clear()

        for i, seg in enumerate(route.segments):
            seg_len = seg.length_m
            if seg_len < 1e-6:
                continue

            # Place supports at preferred_span intervals
            span = pspec.preferred_span_m
            num_spans = max(1, math.ceil(seg_len / span))
            for j in range(1, num_spans):
                t = j * span / seg_len
                if t >= 1.0:
                    break
                x = seg.start.x + t * (seg.end.x - seg.start.x)
                y = seg.start.y + t * (seg.end.y - seg.start.y)
                z = seg.start.z + t * (seg.end.z - seg.start.z)
                route.support_points.append(
                    SupportPoint(location=Waypoint(x, y, z), steel_member_id="STUB-STEEL")
                )

            # Flag grade segments (z ≈ 0)
            if seg.start.z < 0.01 and seg.end.z < 0.01:
                seg.is_grade = True
                route.grade_segment_indices.append(i)
