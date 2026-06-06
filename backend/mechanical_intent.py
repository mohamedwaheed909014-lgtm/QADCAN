"""
mechanical_intent.py — Mechanical Intent Analyzer (MIA)
========================================================
Extracts structured design intent from a natural-language prompt:
  - Detected part family
  - Extracted numeric/string parameters
  - Missing critical parameters (with safe defaults)
  - Design intent class (visual / functional / manufacturing / optimization)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ── Parameter extraction patterns ────────────────────────────────────────────
_NUM_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mm|cm|m|inch|in|teeth|tooth|starts?|"
    r"tooth_count|N|Nm|rpm|kg|g|°|deg|degrees?|%|V|A|W)\b",
    re.IGNORECASE,
)
_NAMED_PARAM = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:mm)?\s+(?:bore|shaft|diameter|od|id|width|"
    r"length|height|depth|thick(?:ness)?|pitch|module|teeth|spokes?)\b",
    re.IGNORECASE,
)


# ── Critical parameters per family ───────────────────────────────────────────
FAMILY_CRITICAL_PARAMS: dict[str, dict[str, Any]] = {
    "gear_reference": {
        "required": ["teeth", "module", "bore"],
        "defaults": {"teeth": 20, "module": 2, "bore": 8, "width": 10, "pressure_angle": 20},
    },
    "pulley_belt_drive_reference": {
        "required": ["teeth", "bore", "belt_width"],
        "defaults": {"teeth": 20, "bore": 5, "belt_width": 6, "belt_type": "GT2"},
    },
    "bearing_housing_reference": {
        "required": ["bearing_od", "shaft_d"],
        "defaults": {"bearing_od": 22, "shaft_d": 8, "wall": 6},
    },
    "shaft_reference": {
        "required": ["shaft_d", "length"],
        "defaults": {"shaft_d": 10, "length": 80},
    },
    "shaft_coupler_reference": {
        "required": ["bore_a", "bore_b"],
        "defaults": {"bore_a": 5, "bore_b": 8, "length": 25},
    },
    "sprocket_chain_reference": {
        "required": ["teeth", "bore"],
        "defaults": {"teeth": 17, "bore": 10, "chain": "#40"},
    },
    "bracket_and_motor_mount_reference": {
        "required": ["motor_type"],
        "defaults": {"motor_type": "NEMA17", "wall": 4, "bolt_size": "M3"},
    },
    "lead_screw_actuator_reference": {
        "required": ["screw_d", "stroke"],
        "defaults": {"screw_d": 8, "stroke": 200, "lead": 8, "screw_type": "T8"},
    },
    "enclosure_box_reference": {
        "required": ["length", "width", "height"],
        "defaults": {"length": 100, "width": 60, "height": 40, "wall": 3},
    },
    "linear_rail_carriage_reference": {
        "required": ["rail_type"],
        "defaults": {"rail_type": "MGN12", "wall": 4},
    },
    "flange_pipe_fitting_reference": {
        "required": ["pipe_od", "bolt_count"],
        "defaults": {"pipe_od": 25, "bolt_count": 4, "bolt_size": "M6"},
    },
    "gearbox_housing_reference": {
        "required": ["shaft_d"],
        "defaults": {"shaft_d": 10, "wall": 5},
    },
}

# ── Intent classification keywords ───────────────────────────────────────────
INTENT_KEYWORDS = {
    "visual":         ["preview", "display", "show", "visualize", "mock", "model"],
    "functional":     ["load", "torque", "force", "pressure", "stress", "support", "carry"],
    "manufacturing":  ["print", "fdm", "cnc", "laser", "machine", "3d print", "resin", "cast"],
    "optimization":   ["lighter", "reduce weight", "stronger", "stiffer", "optimize", "minimum"],
}

# ── Part family detection keywords ───────────────────────────────────────────
FAMILY_KEYWORDS: dict[str, list[str]] = {
    "gear_reference":               ["gear", "spur gear", "helical gear", "bevel gear", "worm gear", "planetary gear"],
    "pulley_belt_drive_reference":  ["pulley", "belt", "gt2", "timing pulley", "idler", "htd"],
    "sprocket_chain_reference":     ["sprocket", "chain", "chain drive", "roller chain"],
    "bearing_housing_reference":    ["bearing housing", "pillow block", "bearing block", "plummer block"],
    "shaft_reference":              ["shaft", "axle", "spindle"],
    "shaft_coupler_reference":      ["coupler", "shaft coupler", "coupling", "flexible coupler"],
    "bracket_and_motor_mount_reference": ["bracket", "motor mount", "nema", "l-bracket", "angle bracket"],
    "lead_screw_actuator_reference": ["lead screw", "leadscrew", "t8", "actuator", "nut carriage"],
    "linear_rail_carriage_reference": ["linear rail", "carriage", "mgn12", "linear guide", "v-slot"],
    "enclosure_box_reference":      ["enclosure", "box", "case", "project box", "electronics box"],
    "hinge_joint_snapfit_reference": ["hinge", "snap fit", "snap-fit", "living hinge"],
    "flange_pipe_fitting_reference": ["flange", "pipe flange", "pipe fitting"],
    "gearbox_housing_reference":    ["gearbox", "gear housing", "gearbox housing"],
    "cam_crank_linkage_reference":  ["cam", "crank", "linkage", "scotch yoke", "four bar"],
    "fastener_nut_trap_reference":  ["nut trap", "heat insert", "through hole", "counterbore"],
    "threads_knobs_shaft_features_reference": ["knob", "thread", "standoff", "spacer"],
    "robotics_servo_reference":     ["servo", "mg996r", "servo mount", "servo bracket"],
    "cooling_fan_mount_reference":  ["fan mount", "fan duct", "cooling fan", "heatsink"],
    "clamp_jig_fixture_reference":  ["clamp", "jig", "fixture", "v-block", "toggle clamp"],
    "structural_profiles_reference": ["i-beam", "h-beam", "ipe", "heb", "structural profile"],
    "common_trusses_reference":     ["truss", "king post", "queen post", "pratt", "warren", "fink"],
}


@dataclass
class IntentResult:
    part_family: str | None
    intent_class: str                        # visual / functional / manufacturing / optimization
    extracted_params: dict[str, Any]
    missing_params: list[str]
    defaults_applied: dict[str, Any]
    is_assembly: bool
    confidence: float                        # 0.0–1.0
    warnings: list[str] = field(default_factory=list)


def analyze_intent(prompt: str) -> IntentResult:
    """Parse a user prompt into a structured mechanical design intent."""
    lower = prompt.lower()

    # ── Detect part family ────────────────────────────────────────────────────
    family_hits: dict[str, int] = {}
    for family, kws in FAMILY_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in lower)
        if hits:
            family_hits[family] = hits
    detected_family = max(family_hits, key=family_hits.get) if family_hits else None
    confidence = min(1.0, max(family_hits.values()) / 3.0) if family_hits else 0.0

    # ── Detect intent class ───────────────────────────────────────────────────
    intent_class = "functional"
    for cls, kws in INTENT_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            intent_class = cls
            break

    # ── Detect assembly request ───────────────────────────────────────────────
    assembly_kws = ["assembly", "system", "actuator", "drive", "gearbox", "mechanism", "stage"]
    is_assembly = any(kw in lower for kw in assembly_kws)

    # ── Extract numeric parameters ────────────────────────────────────────────
    extracted: dict[str, Any] = {}
    for m in _NUM_UNIT.finditer(prompt):
        val_str, unit = m.group(1), m.group(2).lower()
        val = float(val_str)
        if unit in ("mm", "cm", "m"):
            # context-assign based on nearby words
            context = prompt[max(0, m.start()-30):m.end()+20].lower()
            if any(w in context for w in ["bore", "shaft", "hole"]):
                extracted["bore_d"] = val
            elif any(w in context for w in ["outer", " od", "outside diameter"]):
                extracted["od"] = val
            elif any(w in context for w in ["width", "wide"]):
                extracted["width"] = val
            elif any(w in context for w in ["height", "tall", "high"]):
                extracted["height"] = val
            elif any(w in context for w in ["length", "long"]):
                extracted["length"] = val
            elif any(w in context for w in ["thick", "wall"]):
                extracted["wall_thickness"] = val
        elif unit in ("teeth", "tooth"):
            extracted["teeth"] = int(val)
        elif unit in ("nm",):
            extracted["torque_nm"] = val
        elif unit in ("rpm",):
            extracted["rpm"] = val
        elif unit in ("kg", "g"):
            extracted["load_kg"] = val if unit == "kg" else val / 1000

    # ── Extract named dimensions ──────────────────────────────────────────────
    teeth_re = re.search(r"(\d+)\s*(?:-|–)?\s*tooth|(\d+)\s+teeth", lower)
    if teeth_re:
        extracted["teeth"] = int(teeth_re.group(1) or teeth_re.group(2))

    module_re = re.search(r"\bmodule\s+(\d+(?:\.\d+)?)\b|\bm(\d+(?:\.\d+)?)\b", lower)
    if module_re:
        extracted["gear_module"] = float(module_re.group(1) or module_re.group(2))

    bore_re = re.search(r"(\d+(?:\.\d+)?)\s*mm\s+bore|bore\s+(\d+(?:\.\d+)?)\s*mm", lower)
    if bore_re:
        extracted["bore_d"] = float(bore_re.group(1) or bore_re.group(2))

    # ── Determine missing critical params ────────────────────────────────────
    family_spec = FAMILY_CRITICAL_PARAMS.get(detected_family or "", {})
    required = family_spec.get("required", [])
    defaults = family_spec.get("defaults", {})
    missing = [p for p in required if p not in extracted]
    defaults_applied = {p: defaults[p] for p in missing if p in defaults}

    warnings = []
    if missing and not defaults_applied:
        warnings.append(f"Missing critical parameters: {', '.join(missing)}")

    return IntentResult(
        part_family=detected_family,
        intent_class=intent_class,
        extracted_params=extracted,
        missing_params=missing,
        defaults_applied=defaults_applied,
        is_assembly=is_assembly,
        confidence=confidence,
        warnings=warnings,
    )


def format_intent_summary(result: IntentResult) -> str:
    """Human-readable intent summary to inject into the prompt."""
    lines = ["[MECHANICAL INTENT ANALYSIS]"]
    if result.part_family:
        lines.append(f"Detected family: {result.part_family.replace('_reference','').replace('_',' ').title()}")
    lines.append(f"Design intent: {result.intent_class}")
    if result.is_assembly:
        lines.append("Mode: assembly (multiple components)")
    if result.extracted_params:
        params_str = ", ".join(f"{k}={v}" for k, v in result.extracted_params.items())
        lines.append(f"Extracted parameters: {params_str}")
    if result.defaults_applied:
        defs_str = ", ".join(f"{k}={v}" for k, v in result.defaults_applied.items())
        lines.append(f"Defaults applied for missing params: {defs_str}")
    if result.warnings:
        for w in result.warnings:
            lines.append(f"Warning: {w}")
    return "\n".join(lines)
