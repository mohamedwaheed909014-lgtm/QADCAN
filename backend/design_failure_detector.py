"""
design_failure_detector.py — Design Failure Detector (FAIL)
============================================================
Detects mechanically bad designs that pass syntax validation but
are geometrically or functionally wrong.
Assigns a mechanical quality score (0–100).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FailureIssue:
    rule_id: str
    severity: Literal["error", "warning", "info"]
    message: str
    suggestion: str | None = None


@dataclass
class FailureReport:
    score: int                           # 0–100 mechanical quality
    grade: str                           # A / B / C / D / F
    issues: list[FailureIssue] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)

    @property
    def errors(self): return [i for i in self.issues if i.severity == "error"]
    @property
    def warnings(self): return [i for i in self.issues if i.severity == "warning"]


def _score_to_grade(score: int) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"


def detect_failures(code: str, params: dict | None = None, family: str | None = None) -> FailureReport:
    """
    Analyze generated OpenSCAD code for mechanical design failures.
    Returns a FailureReport with score, grade, and issues.
    """
    issues: list[FailureIssue] = []
    passed: list[str] = []
    params = params or {}

    lower = code.lower()
    deductions = 0

    # ── FAIL-01: Thin walls ─────────────────────────────────────────────────
    # Detect near-zero difference operations (e.g. same diameter for body and bore)
    wall_vals = re.findall(r'wall\s*=\s*(\d+(?:\.\d+)?)', code, re.IGNORECASE)
    for wv in wall_vals:
        if float(wv) < 1.5:
            issues.append(FailureIssue(
                rule_id="FAIL-01",
                severity="warning",
                message=f"Wall thickness {wv}mm is very thin (< 1.5mm). Risk of breakage.",
                suggestion="Increase wall to at least 2mm for FDM, 1.2mm for resin.",
            ))
            deductions += 10

    if not wall_vals:
        passed.append("No critically thin walls detected")

    # ── FAIL-02: Weak hub detection ─────────────────────────────────────────
    bore_vals  = [float(v) for v in re.findall(r'bore(?:_d)?\s*=\s*(\d+(?:\.\d+)?)', code, re.IGNORECASE)]
    hub_vals   = [float(v) for v in re.findall(r'hub(?:_od|_d)?\s*=\s*(\d+(?:\.\d+)?)', code, re.IGNORECASE)]
    if bore_vals and hub_vals:
        bore = bore_vals[0]
        hub  = hub_vals[0]
        wall = (hub - bore) / 2
        if wall < 3.0:
            issues.append(FailureIssue(
                rule_id="FAIL-02",
                severity="warning",
                message=f"Hub wall = {wall:.1f}mm (bore {bore}mm, hub OD {hub}mm). Minimum recommended: 3mm.",
                suggestion=f"Set hub_od ≥ {bore + 6:.0f}mm.",
            ))
            deductions += 10
        else:
            passed.append("Hub wall thickness adequate")

    # ── FAIL-03: Missing set screw ─────────────────────────────────────────
    has_shaft = "shaft" in lower or "bore" in lower
    has_set_screw = "set_screw" in lower or "grub" in lower or "m3" in lower or "m4" in lower
    if has_shaft and not has_set_screw and family not in (
        "gear_reference", "sprocket_chain_reference", "bearing_housing_reference"
    ):
        issues.append(FailureIssue(
            rule_id="FAIL-03",
            severity="info",
            message="No set screw or retention feature detected for shaft bore.",
            suggestion="Add M3 set screw hole perpendicular to shaft bore for positive retention.",
        ))
        deductions += 5
    else:
        passed.append("Shaft retention feature present")

    # ── FAIL-04: Missing mounting holes ────────────────────────────────────
    mounting_families = {
        "bracket_and_motor_mount_reference", "bearing_housing_reference",
        "enclosure_box_reference", "cooling_fan_mount_reference",
        "flange_pipe_fitting_reference",
    }
    has_mounting = any(kw in lower for kw in ["bolt_hole", "mount_hole", "mounting", "screw_hole", "m3", "m4", "m5", "m6"])
    if family in mounting_families and not has_mounting:
        issues.append(FailureIssue(
            rule_id="FAIL-04",
            severity="warning",
            message=f"No mounting holes detected for {family.replace('_reference','').replace('_',' ')}.",
            suggestion="Add 4× M4 or M5 mounting holes for assembly.",
        ))
        deductions += 15
    else:
        passed.append("Mounting holes present")

    # ── FAIL-05: Oversized holes ────────────────────────────────────────────
    od_vals   = [float(v) for v in re.findall(r'(?:outer_d|od)\s*=\s*(\d+(?:\.\d+)?)', code, re.IGNORECASE)]
    bore_all  = [float(v) for v in re.findall(r'd\s*=\s*(\d+(?:\.\d+)?)', code, re.IGNORECASE)]
    if bore_vals and od_vals:
        if bore_vals[0] > od_vals[0] * 0.85:
            issues.append(FailureIssue(
                rule_id="FAIL-05",
                severity="error",
                message=f"Bore ({bore_vals[0]}mm) is >85% of OD ({od_vals[0]}mm). Insufficient material.",
                suggestion="Reduce bore or increase OD.",
            ))
            deductions += 25

    # ── FAIL-06: Disconnected geometry check ────────────────────────────────
    # If there are multiple top-level union() calls without a containing module
    union_count = len(re.findall(r'\bunion\s*\(', code))
    module_count = len(re.findall(r'\bmodule\s+\w+', code))
    if union_count > 3 and module_count == 0:
        issues.append(FailureIssue(
            rule_id="FAIL-06",
            severity="warning",
            message="Multiple union() calls without enclosing module — may produce disconnected geometry.",
            suggestion="Wrap all geometry in a single module and call it once.",
        ))
        deductions += 10

    # ── FAIL-07: Decorative-only check ────────────────────────────────────
    functional_keywords = ["bore", "shaft", "hole", "thread", "bolt", "slot", "mount", "bearing", "gear", "hub"]
    if not any(kw in lower for kw in functional_keywords):
        issues.append(FailureIssue(
            rule_id="FAIL-07",
            severity="info",
            message="No functional mechanical features detected (bore, shaft, holes, threads).",
            suggestion="Add functional features appropriate to the part family.",
        ))
        deductions += 5

    # ── FAIL-08: Quality score ─────────────────────────────────────────────
    # Check for parametric variables (good practice)
    param_count = len(re.findall(r'^\s*\w+\s*=\s*\d+', code, re.MULTILINE))
    if param_count >= 5:
        passed.append(f"Good parametric style ({param_count} named parameters)")
    else:
        issues.append(FailureIssue(
            rule_id="FAIL-08",
            severity="info",
            message=f"Only {param_count} named parameters. Increase parametric quality.",
            suggestion="Move all magic numbers to named variables at the top.",
        ))
        deductions += 5

    score = max(0, 100 - deductions)
    return FailureReport(
        score=score,
        grade=_score_to_grade(score),
        issues=issues,
        passed_checks=passed,
    )


def format_failure_report(report: FailureReport) -> str:
    lines = [f"[MECHANICAL QUALITY SCORE: {report.score}/100  Grade: {report.grade}]"]
    for issue in report.issues:
        icon = "❌" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "ℹ️"
        lines.append(f"{icon} [{issue.rule_id}] {issue.message}")
        if issue.suggestion:
            lines.append(f"   → {issue.suggestion}")
    if report.passed_checks:
        lines.append("✅ Passed: " + " | ".join(report.passed_checks))
    return "\n".join(lines)
