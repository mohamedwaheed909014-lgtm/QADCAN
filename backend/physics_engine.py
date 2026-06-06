"""
physics_engine.py — Physics and Formula Engine (PHY)
=====================================================
Mechanical calculations: gear ratios, belt lengths, lead screw torque,
shaft stress, motor sizing recommendations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ── Motor torque catalogue (Nm) ───────────────────────────────────────────────
MOTOR_TORQUE: dict[str, float] = {
    "NEMA11": 0.10,
    "NEMA14": 0.14,
    "NEMA17_40": 0.40,   # standard 40mm NEMA17
    "NEMA17_48": 0.59,   # 48mm NEMA17
    "NEMA23_57": 1.26,
    "NEMA23_76": 2.20,
    "NEMA34": 4.00,
    "NEMA17": 0.45,      # default
    "NEMA23": 1.26,
}


@dataclass
class PhysicsResult:
    calculations: dict[str, Any]
    warnings: list[str]
    recommendations: list[str]


def gear_ratio(driver_teeth: int, driven_teeth: int) -> float:
    return driven_teeth / driver_teeth


def gear_train_ratio(teeth_list: list[int]) -> float:
    """teeth_list alternates driver/driven: [d1, d2, d3, d4] = (d1→d2)×(d3→d4)"""
    ratio = 1.0
    for i in range(0, len(teeth_list) - 1, 2):
        ratio *= teeth_list[i + 1] / teeth_list[i]
    return ratio


def belt_length(center_dist: float, d1: float, d2: float) -> float:
    """Approximate belt length for two-pulley drive (mm)."""
    return (
        2 * center_dist
        + math.pi * (d1 + d2) / 2
        + (d2 - d1) ** 2 / (4 * center_dist)
    )


def pulley_speed_ratio(driver_teeth: int, driven_teeth: int, driver_rpm: float) -> float:
    return driver_rpm * driver_teeth / driven_teeth


def lead_screw_torque(
    load_n: float,
    lead_mm: float,
    efficiency: float = 0.35,
) -> float:
    """Required torque (Nm) to move a load with a lead screw."""
    return (load_n * lead_mm / 1000) / (2 * math.pi * efficiency)


def lead_screw_linear_speed(rpm: float, lead_mm: float) -> float:
    """Linear speed in mm/s for given screw RPM and lead."""
    return rpm * lead_mm / 60


def shaft_torsional_stress(torque_nm: float, shaft_d_mm: float) -> float:
    """Maximum torsional shear stress in MPa (solid round shaft)."""
    r = shaft_d_mm / 2 / 1000  # convert to meters
    J = math.pi * r ** 4 / 2   # polar moment of inertia
    return (torque_nm * r / J) / 1e6  # MPa


def shaft_min_diameter(torque_nm: float, material_shear_mpa: float = 50) -> float:
    """Minimum shaft diameter in mm for given torque and material shear strength."""
    # τ = 16T / (π d³)  →  d = (16T / (π τ))^(1/3)
    d_m = (16 * torque_nm / (math.pi * material_shear_mpa * 1e6)) ** (1 / 3)
    return d_m * 1000  # mm


def recommend_motor(required_torque_nm: float) -> list[str]:
    """Recommend suitable NEMA motors for required torque."""
    return [
        name for name, torque in MOTOR_TORQUE.items()
        if torque >= required_torque_nm * 1.25  # 25% safety factor
    ]


def analyze_physics(params: dict) -> PhysicsResult:
    """Run relevant physics calculations based on available params."""
    calcs: dict[str, Any] = {}
    warnings: list[str] = []
    recs: list[str] = []

    # ── Lead screw analysis ──────────────────────────────────────────────────
    load_kg = params.get("load_kg")
    lead_mm = params.get("lead_mm") or params.get("lead")
    motor = params.get("motor_type", "NEMA17")

    if load_kg and lead_mm:
        load_n = load_kg * 9.81
        torque_req = lead_screw_torque(load_n, lead_mm)
        calcs["load_force_N"] = round(load_n, 2)
        calcs["required_torque_Nm"] = round(torque_req, 4)
        motor_torque = MOTOR_TORQUE.get(motor, 0.45)
        safety_factor = motor_torque / torque_req if torque_req > 0 else 999
        calcs["motor_torque_Nm"] = motor_torque
        calcs["safety_factor"] = round(safety_factor, 2)
        if safety_factor < 1.5:
            warnings.append(f"Safety factor {safety_factor:.1f}x is too low (min 1.5×). Upgrade motor.")
            recs += recommend_motor(torque_req)
        if params.get("rpm"):
            calcs["linear_speed_mm_s"] = round(lead_screw_linear_speed(params["rpm"], lead_mm), 1)

    # ── Gear ratio ───────────────────────────────────────────────────────────
    driver = params.get("driver_teeth") or params.get("teeth_driver")
    driven = params.get("driven_teeth") or params.get("teeth_driven")
    if driver and driven:
        ratio = gear_ratio(int(driver), int(driven))
        calcs["gear_ratio"] = f"{ratio:.2f}:1"
        if params.get("input_rpm"):
            calcs["output_rpm"] = round(params["input_rpm"] / ratio, 1)

    # ── Belt drive ───────────────────────────────────────────────────────────
    center = params.get("center_distance")
    d1 = params.get("pulley1_od")
    d2 = params.get("pulley2_od")
    if center and d1 and d2:
        bl = belt_length(center, d1, d2)
        calcs["belt_length_mm"] = round(bl, 1)
        if params.get("driver_teeth") and params.get("driven_teeth"):
            ratio = gear_ratio(params["driver_teeth"], params["driven_teeth"])
            calcs["speed_ratio"] = f"{ratio:.2f}:1"

    # ── Shaft stress ─────────────────────────────────────────────────────────
    torque = params.get("torque_nm")
    shaft_d = params.get("shaft_d") or params.get("bore_d")
    if torque and shaft_d:
        stress = shaft_torsional_stress(torque, shaft_d)
        calcs["torsional_stress_MPa"] = round(stress, 1)
        if stress > 60:
            warnings.append(f"Shaft stress {stress:.0f}MPa exceeds steel yield (~60MPa). Increase diameter.")
            min_d = shaft_min_diameter(torque)
            recs.append(f"Minimum safe shaft diameter: {min_d:.1f}mm for this torque.")

    return PhysicsResult(calculations=calcs, warnings=warnings, recommendations=recs)


def format_physics_summary(result: PhysicsResult) -> str:
    if not result.calculations and not result.warnings:
        return ""
    lines = ["[PHYSICS CALCULATIONS]"]
    for k, v in result.calculations.items():
        lines.append(f"  {k.replace('_',' ').title()}: {v}")
    for w in result.warnings:
        lines.append(f"  ⚠️ {w}")
    for r in result.recommendations:
        lines.append(f"  → {r}")
    return "\n".join(lines)
