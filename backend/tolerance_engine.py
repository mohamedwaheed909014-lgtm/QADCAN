"""
tolerance_engine.py — Tolerance Engine (TOL)
=============================================
Applies manufacturing-aware tolerances to generated dimensions.
Supports FDM, resin, CNC, and laser cutting processes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ManufacturingProcess = Literal["fdm", "resin", "cnc", "laser"]

# ── Tolerance tables ──────────────────────────────────────────────────────────
# (process) → (fit_type) → offset in mm (positive = larger hole / looser)
TOLERANCE_TABLE: dict[str, dict[str, float]] = {
    "fdm": {
        "clearance_hole":   0.30,   # bolt/shaft slides through freely
        "press_fit":       -0.10,   # bearing press-fit into housing
        "close_fit":        0.15,   # shaft in bushing, minimal play
        "running_fit":      0.25,   # rotating shaft, moderate clearance
        "thread_clearance": 0.40,   # bolt thread in printed hole
        "nut_pocket":       0.20,   # hex nut in pocket (each side)
        "pin_clearance":    0.20,   # pivot/pin through hole
    },
    "resin": {
        "clearance_hole":   0.15,
        "press_fit":       -0.05,
        "close_fit":        0.08,
        "running_fit":      0.15,
        "thread_clearance": 0.20,
        "nut_pocket":       0.10,
        "pin_clearance":    0.10,
    },
    "cnc": {
        "clearance_hole":   0.10,
        "press_fit":       -0.02,
        "close_fit":        0.05,
        "running_fit":      0.10,
        "thread_clearance": 0.15,
        "nut_pocket":       0.05,
        "pin_clearance":    0.05,
    },
    "laser": {
        "clearance_hole":   0.20,
        "press_fit":        0.00,
        "close_fit":        0.10,
        "running_fit":      0.20,
        "thread_clearance": 0.25,
        "nut_pocket":       0.10,
        "pin_clearance":    0.15,
    },
}

# ── Feature patterns requiring tolerance ─────────────────────────────────────
TOLERANCE_COMMENTS = {
    "clearance_hole":   "clearance — shaft/bolt passes through freely",
    "press_fit":        "press fit — bearing/bushing seats with interference",
    "close_fit":        "close fit — minimal play, snug assembly",
    "running_fit":      "running fit — rotating shaft clearance",
    "thread_clearance": "thread clearance — bolt/screw in printed hole",
    "nut_pocket":       "nut pocket — hex nut captured in recess",
    "pin_clearance":    "pin clearance — pivot pin / hinge pin",
}


@dataclass
class TolerancedDimension:
    feature: str           # e.g. "shaft_bore", "bearing_seat"
    nominal_mm: float      # design nominal value
    fit_type: str          # clearance_hole, press_fit, etc.
    offset_mm: float       # applied tolerance offset
    final_mm: float        # nominal + offset
    comment: str


def apply_tolerance(
    nominal: float,
    fit_type: str,
    process: ManufacturingProcess = "fdm",
) -> TolerancedDimension:
    """Return a toleranced dimension for a given fit type and process."""
    offset = TOLERANCE_TABLE.get(process, TOLERANCE_TABLE["fdm"]).get(fit_type, 0.0)
    comment = TOLERANCE_COMMENTS.get(fit_type, fit_type)
    return TolerancedDimension(
        feature=fit_type,
        nominal_mm=nominal,
        fit_type=fit_type,
        offset_mm=offset,
        final_mm=round(nominal + offset, 3),
        comment=comment,
    )


def generate_tolerance_block(
    params: dict,
    process: ManufacturingProcess = "fdm",
) -> str:
    """
    Generate OpenSCAD tolerance variable block to prepend to generated code.
    Reads common dimension keys from params dict and applies tolerances.
    """
    lines = [
        f"// ── Tolerances for {process.upper()} manufacturing ────────────────",
        f'manufacturing = "{process}";',
        "",
    ]

    tol = TOLERANCE_TABLE.get(process, TOLERANCE_TABLE["fdm"])

    # Always emit standard tolerance variables
    lines += [
        f"clearance      = {tol['clearance_hole']};   // {TOLERANCE_COMMENTS['clearance_hole']}",
        f"press_fit_adj  = {tol['press_fit']};  // {TOLERANCE_COMMENTS['press_fit']}",
        f"close_fit_adj  = {tol['close_fit']};   // {TOLERANCE_COMMENTS['close_fit']}",
        f"running_fit_adj= {tol['running_fit']};   // {TOLERANCE_COMMENTS['running_fit']}",
        f"thread_clear   = {tol['thread_clearance']};   // {TOLERANCE_COMMENTS['thread_clearance']}",
        f"nut_tol        = {tol['nut_pocket']};   // {TOLERANCE_COMMENTS['nut_pocket']}",
        "",
    ]

    # Emit toleranced versions of provided dimensions
    if "bore_d" in params or "shaft_d" in params:
        shaft = params.get("bore_d") or params.get("shaft_d")
        lines.append(f"shaft_bore_d   = {shaft} + running_fit_adj;   // shaft bore with running clearance")

    if "bearing_od" in params:
        bod = params["bearing_od"]
        lines.append(f"bearing_seat_d = {bod} + press_fit_adj;  // bearing seat (press fit)")

    if "bolt_d" in params:
        bd = params["bolt_d"]
        lines.append(f"bolt_hole_d    = {bd} + thread_clear;   // bolt clearance hole")

    return "\n".join(lines)


def get_tolerance_table_html(process: ManufacturingProcess = "fdm") -> list[dict]:
    """Return tolerance table as list of dicts for frontend display."""
    tol = TOLERANCE_TABLE.get(process, TOLERANCE_TABLE["fdm"])
    return [
        {
            "fit_type": ft,
            "offset_mm": offset,
            "description": TOLERANCE_COMMENTS.get(ft, ft),
        }
        for ft, offset in tol.items()
    ]
