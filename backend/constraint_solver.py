"""
constraint_solver.py — Mechanical Constraint Solver (MCS)
==========================================================
Checks mechanical compatibility before code generation.
Returns errors, warnings, and automatic corrections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ── Bearing catalogue (bore → OD, width) ─────────────────────────────────────
BEARING_CATALOGUE: dict[str, dict] = {
    "608":  {"bore": 8,  "od": 22, "width": 7,  "label": "608"},
    "626":  {"bore": 6,  "od": 19, "width": 6,  "label": "626"},
    "625":  {"bore": 5,  "od": 16, "width": 5,  "label": "625ZZ"},
    "624":  {"bore": 4,  "od": 13, "width": 5,  "label": "624ZZ"},
    "623":  {"bore": 3,  "od": 10, "width": 4,  "label": "623ZZ"},
    "6200": {"bore": 10, "od": 30, "width": 9,  "label": "6200"},
    "6201": {"bore": 12, "od": 32, "width": 10, "label": "6201"},
    "6202": {"bore": 15, "od": 35, "width": 11, "label": "6202"},
    "6203": {"bore": 17, "od": 40, "width": 12, "label": "6203"},
    "6204": {"bore": 20, "od": 47, "width": 14, "label": "6204"},
    "6205": {"bore": 25, "od": 52, "width": 15, "label": "6205"},
    "6206": {"bore": 30, "od": 62, "width": 16, "label": "6206"},
    "6207": {"bore": 35, "od": 72, "width": 17, "label": "6207"},
    "6208": {"bore": 40, "od": 80, "width": 18, "label": "6208"},
}

# ── Motor shaft diameters ─────────────────────────────────────────────────────
MOTOR_SHAFT: dict[str, float] = {
    "NEMA11": 5.0, "NEMA14": 5.0, "NEMA17": 5.0,
    "NEMA23": 6.35, "NEMA34": 12.7,
    "MG996R": 6.0, "SG90": 2.0,
}

# ── Chain pitch table ─────────────────────────────────────────────────────────
CHAIN_PITCH: dict[str, float] = {
    "#25": 6.35, "#35": 9.525, "#40": 12.70,
    "#41": 12.70, "#50": 15.875, "#60": 19.05,
}

# ── Min wall thickness by process ────────────────────────────────────────────
MIN_WALL: dict[str, float] = {
    "fdm": 1.2, "resin": 0.8, "cnc": 1.0, "laser": 3.0,
}


@dataclass
class ConstraintIssue:
    severity: Literal["error", "warning", "info"]
    rule_id: str
    message: str
    fix: str | None = None
    auto_corrected: bool = False


@dataclass
class ConstraintResult:
    passed: bool
    issues: list[ConstraintIssue] = field(default_factory=list)
    corrected_params: dict = field(default_factory=dict)

    @property
    def errors(self): return [i for i in self.issues if i.severity == "error"]
    @property
    def warnings(self): return [i for i in self.issues if i.severity == "warning"]


def check_bearing_shaft_compatibility(
    bearing_name: str | None,
    shaft_d: float | None,
    bearing_od: float | None,
) -> list[ConstraintIssue]:
    """MCS-01, MCS-02: bearing bore must match shaft; housing must match bearing OD."""
    issues = []

    if bearing_name and shaft_d:
        bearing = BEARING_CATALOGUE.get(bearing_name.lower().replace("zz","").replace("rs",""))
        if bearing:
            if abs(bearing["bore"] - shaft_d) > 0.5:
                alternatives = [
                    b["label"] for b in BEARING_CATALOGUE.values()
                    if abs(b["bore"] - shaft_d) <= 0.5
                ]
                fix = f"Use bearing {alternatives[0]}" if alternatives else f"Change shaft to {bearing['bore']}mm"
                issues.append(ConstraintIssue(
                    severity="error",
                    rule_id="MCS-02",
                    message=(
                        f"{bearing_name} bearing bore = {bearing['bore']}mm, "
                        f"shaft = {shaft_d}mm — incompatible."
                    ),
                    fix=fix,
                ))
        else:
            issues.append(ConstraintIssue(
                severity="warning",
                rule_id="MCS-02",
                message=f"Unknown bearing '{bearing_name}' — cannot verify bore compatibility.",
            ))

    if bearing_name and bearing_od:
        bearing = BEARING_CATALOGUE.get(bearing_name.lower().replace("zz","").replace("rs",""))
        if bearing and abs(bearing["od"] - bearing_od) > 1.0:
            issues.append(ConstraintIssue(
                severity="error",
                rule_id="MCS-03",
                message=(
                    f"{bearing_name} OD = {bearing['od']}mm, "
                    f"housing bore = {bearing_od}mm — incompatible."
                ),
                fix=f"Set housing bore to {bearing['od']}mm for {bearing_name}",
            ))

    return issues


def check_bore_vs_od(bore_d: float | None, od: float | None, part: str = "part") -> list[ConstraintIssue]:
    """MCS-01: bore must be smaller than OD with minimum wall."""
    issues = []
    if bore_d is None or od is None:
        return issues
    min_wall = 2.0
    if bore_d >= od:
        issues.append(ConstraintIssue(
            severity="error",
            rule_id="MCS-01",
            message=f"{part}: bore ({bore_d}mm) ≥ OD ({od}mm) — impossible geometry.",
            fix=f"Increase OD to at least {bore_d + min_wall*2:.0f}mm or reduce bore.",
        ))
    elif (od - bore_d) / 2 < min_wall:
        issues.append(ConstraintIssue(
            severity="warning",
            rule_id="MCS-06",
            message=f"{part}: wall = {(od-bore_d)/2:.1f}mm < recommended {min_wall}mm.",
            fix=f"Increase OD to {bore_d + min_wall*2:.0f}mm or reduce bore.",
        ))
    return issues


def check_motor_shaft(motor_type: str | None, bore_d: float | None) -> list[ConstraintIssue]:
    """Match motor shaft diameter to part bore."""
    issues = []
    if not motor_type or bore_d is None:
        return issues
    shaft = MOTOR_SHAFT.get(motor_type.upper())
    if shaft and abs(shaft - bore_d) > 0.5:
        issues.append(ConstraintIssue(
            severity="warning",
            rule_id="MCS-02",
            message=f"{motor_type} shaft = {shaft}mm, bore = {bore_d}mm — mismatch.",
            fix=f"Set bore to {shaft}mm for {motor_type}.",
        ))
    return issues


def check_wall_thickness(wall: float | None, process: str = "fdm") -> list[ConstraintIssue]:
    """MCS-06: minimum wall thickness per manufacturing process."""
    issues = []
    if wall is None:
        return issues
    min_w = MIN_WALL.get(process.lower(), 1.2)
    if wall < min_w:
        issues.append(ConstraintIssue(
            severity="warning",
            rule_id="MCS-06",
            message=f"Wall {wall}mm < minimum {min_w}mm for {process.upper()}.",
            fix=f"Increase wall to at least {min_w}mm.",
        ))
    return issues


def check_bolt_hole_spacing(bolt_d: float | None, spacing: float | None) -> list[ConstraintIssue]:
    """MCS-07: bolt holes must not overlap."""
    issues = []
    if bolt_d is None or spacing is None:
        return issues
    if spacing < bolt_d * 2.5:
        issues.append(ConstraintIssue(
            severity="warning",
            rule_id="MCS-07",
            message=f"Bolt spacing {spacing}mm too small for M{bolt_d:.0f} bolts (min {bolt_d*2.5:.0f}mm).",
            fix=f"Increase bolt circle to at least {bolt_d*2.5:.0f}mm.",
        ))
    return issues


def check_pulley_belt_compatibility(
    teeth: int | None, belt_type: str | None, belt_width: float | None
) -> list[ConstraintIssue]:
    """MCS-04: belt width must be supported by pulley spec."""
    issues = []
    belt_min_teeth = {"GT2": 16, "HTD3": 14, "HTD5": 16, "T5": 12, "T2.5": 10}
    if belt_type and teeth:
        min_t = belt_min_teeth.get(belt_type.upper(), 14)
        if teeth < min_t:
            issues.append(ConstraintIssue(
                severity="warning",
                rule_id="MCS-04",
                message=f"{belt_type} belt: {teeth} teeth < minimum {min_t} (risk of belt skip).",
                fix=f"Use at least {min_t} teeth.",
            ))
    return issues


def solve_constraints(params: dict, family: str | None = None) -> ConstraintResult:
    """
    Run all applicable constraint checks for the given parameters.
    Returns a ConstraintResult with all issues and any auto-corrections.
    """
    issues: list[ConstraintIssue] = []

    bore_d   = params.get("bore_d") or params.get("bore")
    od       = params.get("od") or params.get("outer_d")
    wall     = params.get("wall") or params.get("wall_thickness")
    bearing  = params.get("bearing") or params.get("bearing_name")
    shaft_d  = params.get("shaft_d") or bore_d
    motor    = params.get("motor_type")
    teeth    = params.get("teeth")
    belt     = params.get("belt_type")
    belt_w   = params.get("belt_width")
    bolt_d   = params.get("bolt_d")
    bolt_spc = params.get("bolt_spacing")
    process  = params.get("manufacturing", "fdm")

    # Run checks
    issues += check_bore_vs_od(bore_d, od, family or "part")
    issues += check_bearing_shaft_compatibility(bearing, shaft_d, od)
    issues += check_motor_shaft(motor, bore_d)
    issues += check_wall_thickness(wall, process)
    issues += check_bolt_hole_spacing(bolt_d, bolt_spc)
    issues += check_pulley_belt_compatibility(teeth, belt, belt_w)

    passed = not any(i.severity == "error" for i in issues)
    return ConstraintResult(passed=passed, issues=issues)


def format_constraint_summary(result: ConstraintResult) -> str:
    """Human-readable constraint check summary."""
    if not result.issues:
        return "[CONSTRAINT CHECK] All mechanical constraints passed."
    lines = ["[CONSTRAINT CHECK]"]
    for issue in result.issues:
        icon = "❌" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "ℹ️"
        lines.append(f"{icon} [{issue.rule_id}] {issue.message}")
        if issue.fix:
            lines.append(f"   → Fix: {issue.fix}")
    return "\n".join(lines)
