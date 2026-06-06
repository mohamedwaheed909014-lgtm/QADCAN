"""
manufacturing_rules.py — Manufacturing-Aware Generator (MFG)
=============================================================
DFM (Design for Manufacturing) rules injected into prompts.
Generates manufacturing notes and printability warnings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ManufacturingProcess = Literal["fdm", "resin", "cnc", "laser"]


@dataclass
class MfgRule:
    rule_id: str
    description: str
    openscad_hint: str | None = None


@dataclass
class MfgProfile:
    process: str
    rules: list[MfgRule]
    material_recommendations: list[str]
    notes: list[str]


# ── DFM rules per process ─────────────────────────────────────────────────────
DFM_RULES: dict[str, list[MfgRule]] = {
    "fdm": [
        MfgRule("MFG-FDM-01", "Minimum wall thickness ≥ 1.2mm (ideally ≥ 2mm).",
                "wall = max(wall, 2.0);"),
        MfgRule("MFG-FDM-02", "Avoid unsupported overhangs >45° without supports.",
                "// Use chamfers instead of sharp horizontal overhangs"),
        MfgRule("MFG-FDM-03", "Add fillets r≥0.5mm to all inside corners to prevent stress cracks.",
                "// Use minkowski() or offset() for internal fillets"),
        MfgRule("MFG-FDM-04", "Bolt holes should be oriented vertically for best accuracy.",
                "// Rotate part so bolt holes are parallel to Z axis"),
        MfgRule("MFG-FDM-05", "Add 0.3mm clearance to shaft bores for running fit.",
                "bore_d = shaft_d + 0.3;  // FDM running clearance"),
        MfgRule("MFG-FDM-06", "Bearing seats: subtract 0.1mm from bearing OD for press fit.",
                "bearing_seat = bearing_od - 0.1;  // FDM press fit"),
        MfgRule("MFG-FDM-07", "Minimum feature size ≥ 0.4mm (nozzle diameter).",
                "// Fine features smaller than 0.4mm will not print"),
        MfgRule("MFG-FDM-08", "Add chamfer to bottom edges to reduce elephant foot.",
                "// chamfer = 0.3 on bottom perimeter edges"),
    ],
    "resin": [
        MfgRule("MFG-RES-01", "Minimum wall ≥ 0.8mm for structural parts.",
                "wall = max(wall, 0.8);"),
        MfgRule("MFG-RES-02", "Hollow large parts to save resin and reduce warping.",
                "// Use shell() or manual hollow with wall = 1.5mm"),
        MfgRule("MFG-RES-03", "Add drain holes ≥2mm to hollow parts.",
                "// Drain holes prevent resin pooling inside"),
        MfgRule("MFG-RES-04", "Shaft bores: add 0.15mm clearance.",
                "bore_d = shaft_d + 0.15;"),
        MfgRule("MFG-RES-05", "Bearing seats: subtract 0.05mm for press fit.",
                "bearing_seat = bearing_od - 0.05;"),
    ],
    "cnc": [
        MfgRule("MFG-CNC-01", "Inside corners have minimum radius = end mill radius (typically 1–4mm).",
                "corner_r = 2.0;  // minimum inside corner radius for CNC"),
        MfgRule("MFG-CNC-02", "Avoid features smaller than 3mm in depth-to-width ratio > 5:1.",
                "// Thin deep slots require special tooling"),
        MfgRule("MFG-CNC-03", "Hole diameters should match standard drill sizes.",
                "// Use M3=3.3, M4=4.3, M5=5.3, M6=6.5 clearance holes"),
        MfgRule("MFG-CNC-04", "Shaft tolerances: H7/g6 for running fit, H7/p6 for press fit.",
                "shaft_bore_d = shaft_d + 0.021;  // H7 upper tolerance for 20mm shaft"),
    ],
    "laser": [
        MfgRule("MFG-LAS-01", "Design as 2D profile — all features must be through-cuts.",
                "// Laser cutting is 2D only; use linear_extrude for 3D"),
        MfgRule("MFG-LAS-02", "Minimum kerf compensation: add 0.1–0.2mm to hole diameters.",
                "laser_kerf = 0.15;  // compensate for beam width"),
        MfgRule("MFG-LAS-03", "Minimum tab width ≥ 3mm for sheet metal.",
                "tab_width = max(tab_width, 3.0);"),
        MfgRule("MFG-LAS-04", "Add press-fit tabs for assembly without fasteners.",
                "// Slot width = material_thickness - 0.1mm for snap fit"),
    ],
}

MATERIAL_RECS: dict[str, list[str]] = {
    "fdm": ["PLA (prototypes, display)", "PETG (functional, food-safe)", "ABS/ASA (high temp, UV)", "TPU (flexible parts)", "PA12 Nylon (high load)"],
    "resin": ["Standard resin (detail, brittle)", "ABS-like resin (functional)", "Tough resin (impact)", "Flexible resin (seals)"],
    "cnc":  ["Aluminium 6061 (lightweight structural)", "Steel 1045 (high strength)", "Brass (corrosion resistant)", "Delrin/POM (low friction)"],
    "laser": ["Acrylic (display, light duty)", "Plywood (structural)", "Aluminium sheet (heat)", "Mild steel (heavy duty)"],
}


def get_mfg_profile(process: ManufacturingProcess, part_family: str | None = None) -> MfgProfile:
    rules = DFM_RULES.get(process, DFM_RULES["fdm"])
    mats  = MATERIAL_RECS.get(process, MATERIAL_RECS["fdm"])
    notes = [r.description for r in rules[:4]]  # top 4 most important
    return MfgProfile(process=process, rules=rules, material_recommendations=mats, notes=notes)


def format_mfg_context(process: ManufacturingProcess, part_family: str | None = None) -> str:
    """Generate manufacturing context string for prompt injection."""
    profile = get_mfg_profile(process, part_family)
    lines = [f"[MANUFACTURING: {process.upper()} DFM RULES]"]
    for rule in profile.rules:
        lines.append(f"  • {rule.description}")
        if rule.openscad_hint:
            lines.append(f"    Code: {rule.openscad_hint}")
    lines.append(f"Recommended materials: {', '.join(profile.material_recommendations[:3])}")
    return "\n".join(lines)


def check_dfm(code: str, process: ManufacturingProcess = "fdm") -> list[str]:
    """Quick DFM check on generated code, returns list of warnings."""
    import re
    warnings = []
    rules = DFM_RULES.get(process, [])

    if process == "fdm":
        # Check wall thickness
        walls = [float(v) for v in re.findall(r'wall\s*=\s*(\d+(?:\.\d+)?)', code, re.IGNORECASE)]
        if walls and min(walls) < 1.2:
            warnings.append(f"MFG-FDM-01: Wall {min(walls)}mm < 1.2mm minimum for FDM")
        # Check for fillets
        if "fillet" not in code.lower() and "minkowski" not in code.lower():
            warnings.append("MFG-FDM-03: No fillets detected — add to inside corners")

    if process == "laser":
        if "linear_extrude" not in code:
            warnings.append("MFG-LAS-01: Laser design should use linear_extrude for 2D profile")

    return warnings
