"""ResolutionLadder — 15-step coarse-to-fine pass sequence per spec Section 5."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolutionPass:
    pass_number: int      # 1-15
    step_size_mm: float   # step size in mm
    imperial_label: str   # human-readable imperial

    @property
    def step_size_m(self) -> float:
        return self.step_size_mm / 1000.0

    @property
    def enforce_orthogonality(self) -> bool:
        """Orthogonality is not enforced during coarse passes 1-4."""
        return self.pass_number >= 5

    @property
    def check_elbow_envelope(self) -> bool:
        """Elbow envelope checking begins at pass 5."""
        return self.pass_number >= 5

    @property
    def check_fine_clearance(self) -> bool:
        """Fine clearance checking (EOD tolerance) begins at pass 8."""
        return self.pass_number >= 8


# Authoritative ladder from spec §5
RESOLUTION_LADDER: list[ResolutionPass] = [
    ResolutionPass(1,  3048, "10 ft"),
    ResolutionPass(2,  1524, "5 ft"),
    ResolutionPass(3,   914, "3 ft"),
    ResolutionPass(4,   610, "2 ft"),
    ResolutionPass(5,   305, "1 ft"),
    ResolutionPass(6,   229, "9 in"),
    ResolutionPass(7,   203, "8 in"),
    ResolutionPass(8,   152, "6 in"),
    ResolutionPass(9,   102, "4 in"),
    ResolutionPass(10,   76, "3 in"),
    ResolutionPass(11,   51, "2 in"),
    ResolutionPass(12,   25, "1 in"),
    ResolutionPass(13,   13, "0.5 in"),
    ResolutionPass(14,    6, "0.25 in"),
    ResolutionPass(15,    3, "0.125 in"),
]

_BY_PASS_NUMBER: dict[int, ResolutionPass] = {p.pass_number: p for p in RESOLUTION_LADDER}


def get_pass(pass_number: int) -> ResolutionPass:
    return _BY_PASS_NUMBER[pass_number]


def passes_for_routing(coarse_only: bool = False) -> list[ResolutionPass]:
    """Return passes to use for a routing run.

    Args:
        coarse_only: if True, return only passes 1-4 (coarse skeleton).
                     Useful for fast initial candidate generation.
    """
    if coarse_only:
        return RESOLUTION_LADDER[:4]
    return RESOLUTION_LADDER
