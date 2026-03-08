"""Claude API wrapper — explain routes, parse DQs, suggest constraints.

All three functions degrade gracefully if the Anthropic API key is absent or
the API call fails. The agent continues to function without Claude integration.

Model: claude-sonnet-4-6
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neuralpipe.models.route import Route
    from neuralpipe.models.feedback import DQRecord

logger = logging.getLogger(__name__)

_SPEC_PATH = Path(__file__).parent.parent.parent / "ROUTING_SPEC.md"
_MODEL = "claude-sonnet-4-6"


def _load_spec() -> str:
    try:
        return _SPEC_PATH.read_text()
    except FileNotFoundError:
        return "(ROUTING_SPEC.md not found)"


def _get_client():
    """Return an Anthropic client or None if SDK/key not available."""
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        return None


def _call_claude(system_prompt: str, user_message: str, max_tokens: int = 1024) -> Optional[str]:
    """Make a Claude API call. Return text or None on failure."""
    client = _get_client()
    if client is None:
        logger.warning("Claude API not available — API key missing or anthropic SDK not installed.")
        return None
    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
    except Exception as exc:
        logger.warning(f"Claude API call failed: {exc}")
        return None


def _build_system_prompt(extra_sections: str = "") -> str:
    spec = _load_spec()
    return f"""You are NeuralPipe, an AI assistant specialised in industrial pipe routing.
You have deep knowledge of the following authoritative routing specification:

<routing_spec>
{spec}
</routing_spec>

{extra_sections}

Always respond concisely and reference specific spec sections when relevant.
"""


def explain_route(route: "Route", context: dict) -> str:
    """Return a markdown explanation of why this route was chosen.

    Describes constraints that drove each decision and the score breakdown.
    Returns a fallback string if Claude is unavailable.
    """
    score_breakdown = json.dumps(route.score_breakdown, indent=2)
    waypoints_str = "\n".join(
        f"  WP{i}: ({wp.x:.3f}, {wp.y:.3f}, {wp.z:.3f})"
        for i, wp in enumerate(route.waypoints)
    )

    pipe_spec = context.get("pipe_spec")
    nps_str = f"{pipe_spec.nps_inches}\" NPS" if pipe_spec else "unknown NPS"

    user_msg = f"""Please explain this pipe route to the reviewing engineer.

**Route ID:** {route.route_id}
**Line Number:** {route.line_number}
**Pipe Size:** {nps_str}
**Status:** {route.status}
**Total Score:** {route.score:.2f} (lower = better)
**EOD:** {route.eod_m * 1000:.1f} mm

**Waypoints:**
{waypoints_str}

**Number of Elbows:** {route.num_elbows}
**Number of Supports:** {route.num_supports}
**Total Length:** {route.total_length_m:.3f} m

**Score Breakdown:**
```json
{score_breakdown}
```

**Flags:**
{chr(10).join(route.flags) if route.flags else "None"}

Provide a clear, concise markdown explanation of:
1. Why this path was chosen over a more direct route (if it deviates)
2. What spec constraints drove the key waypoints
3. What the score breakdown means for the engineer
4. Any flags they should pay attention to
"""

    result = _call_claude(_build_system_prompt(), user_msg, max_tokens=800)
    if result is None:
        # Fallback: generate a simple explanation without Claude
        lines = [
            f"## Route {route.route_id}",
            f"",
            f"**Total length:** {route.total_length_m:.2f} m  ",
            f"**Elbows:** {route.num_elbows}  ",
            f"**Supports:** {route.num_supports}  ",
            f"**Score:** {route.score:.2f}",
            f"",
            "### Score Breakdown",
        ]
        for k, v in route.score_breakdown.items():
            lines.append(f"- **{k}:** {v:.2f}")
        if route.flags:
            lines.append("\n### Flags")
            for flag in route.flags:
                lines.append(f"- {flag}")
        return "\n".join(lines)
    return result


def parse_dq(free_text: str, route: "Route") -> "DQRecord":
    """Parse a free-text DQ reason into a structured DQRecord.

    Claude extracts: dq_category, derived_rule, constraint_scope, suggested_permanence.
    Falls back to a minimal DQRecord if Claude is unavailable.
    """
    from neuralpipe.models.feedback import DQRecord, DQ_CATEGORIES

    system = _build_system_prompt(
        extra_sections="""
When parsing disqualification reasons, extract:
- dq_category: one of CLEARANCE_VIOLATION, SUPPORT_UNAVAILABLE, MAINTENANCE_ACCESS,
  PROCESS_CONFLICT, CONSTRUCTABILITY, OPERABILITY, EXCLUSION_ZONE, COST, OTHER
- derived_rule: a concise constraint statement that could be added to the spec (or null)
- constraint_scope: GLOBAL | UNIT | SERVICE_CLASS | LOCATION
- suggested_permanence: "permanent" if this should become a spec rule, "one-off" otherwise

Respond ONLY with a JSON object containing these keys. No markdown, no extra text.
"""
    )

    user_msg = f"""Parse this disqualification reason for route {route.route_id} (line {route.line_number}):

"{free_text}"

Respond with JSON only.
"""

    result = _call_claude(system, user_msg, max_tokens=400)

    if result:
        try:
            # Strip any accidental markdown fencing
            cleaned = result.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(cleaned)
            category = parsed.get("dq_category", "OTHER")
            if category not in DQ_CATEGORIES:
                category = "OTHER"
            scope = parsed.get("constraint_scope", "GLOBAL")
            if scope not in {"GLOBAL", "UNIT", "SERVICE_CLASS", "LOCATION"}:
                scope = "GLOBAL"
            return DQRecord(
                route_id=route.route_id,
                line_number=route.line_number,
                dq_reason_text=free_text,
                dq_category=category,
                derived_rule=parsed.get("derived_rule"),
                constraint_scope=scope,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"Claude DQ parse failed to produce valid JSON: {exc}")

    # Fallback: minimal record with OTHER category
    return DQRecord(
        route_id=route.route_id,
        line_number=route.line_number,
        dq_reason_text=free_text,
        dq_category="OTHER",
    )


def suggest_constraint(dq_record: "DQRecord") -> Optional[str]:
    """Review a DQRecord and suggest whether it should become a permanent spec rule.

    Returns a plain-English rule suggestion or None if it's a one-off.
    """
    if not dq_record.derived_rule:
        return None

    system = _build_system_prompt(
        extra_sections="""
Your task is to evaluate whether a disqualification record warrants a new permanent
constraint in the routing specification. Consider:
- Is this likely to recur in other routes?
- Is the reasoning site-specific (one-off) or project/discipline-wide (permanent)?
- Does an equivalent rule already exist in the spec?

If permanent: respond with a concise plain-English rule statement that could be
added to the spec (1-2 sentences). Start with "RULE:".
If one-off: respond with "ONE-OFF: <brief reason>".
"""
    )

    user_msg = f"""Evaluate this DQ record:

**Route:** {dq_record.route_id}
**Line:** {dq_record.line_number}
**Category:** {dq_record.dq_category}
**Scope:** {dq_record.constraint_scope}
**Engineer's reason:** {dq_record.dq_reason_text}
**Derived rule:** {dq_record.derived_rule}

Should this become a permanent constraint in the routing spec?
"""

    result = _call_claude(system, user_msg, max_tokens=300)
    if result is None:
        return None

    result = result.strip()
    if result.upper().startswith("RULE:"):
        return result[5:].strip()
    if result.upper().startswith("ONE-OFF:"):
        return None
    # Ambiguous — return as-is for the engineer to decide
    return result
