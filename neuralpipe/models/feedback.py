"""DQRecord dataclass — disqualification feedback schema."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid

# Valid DQ categories per spec Section 11
DQ_CATEGORIES = {
    "CLEARANCE_VIOLATION",
    "SUPPORT_UNAVAILABLE",
    "MAINTENANCE_ACCESS",
    "PROCESS_CONFLICT",
    "CONSTRUCTABILITY",
    "OPERABILITY",
    "EXCLUSION_ZONE",
    "COST",
    "OTHER",
}

CONSTRAINT_SCOPES = {"GLOBAL", "UNIT", "SERVICE_CLASS", "LOCATION"}


@dataclass
class DQRecord:
    route_id: str
    line_number: str
    dq_reason_text: str
    dq_category: str = "OTHER"
    dq_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    derived_rule: Optional[str] = None
    constraint_scope: str = "GLOBAL"
    applied_to_spec: bool = False

    def __post_init__(self) -> None:
        if self.dq_category not in DQ_CATEGORIES:
            raise ValueError(f"Invalid dq_category: {self.dq_category!r}. Must be one of {DQ_CATEGORIES}")
        if self.constraint_scope not in CONSTRAINT_SCOPES:
            raise ValueError(f"Invalid constraint_scope: {self.constraint_scope!r}. Must be one of {CONSTRAINT_SCOPES}")
