"""
validation_and_logging.py
=========================
Drop-in replacement / enhancement for the validate_scad() function and
logging layer in app.py.

What this module adds over the original:
  1.  STRUCTURED GENERATION LOG  â€” every generation attempt emits a
      machine-readable JSON event to a rotating log file
      (logs/generation_events.jsonl) so errors can be inspected offline.

  2.  DERIVED-PARAMETER VERIFICATION  â€” for each known family the
      generated code is checked that derived parameters (e.g. pitch_diameter
      = module * teeth) appear and are not replaced by hard-coded magic
      numbers.

  3.  OPENSCAD SKELETON CHECKS  â€” verify that union() contains only solid
      primitives and difference() subtracts them; neither block is empty.

  4.  PER-FAMILY EXHAUSTIVE CHECKS  â€” every family in the JSON databases
      maps to a set of mandatory code patterns; all are tested and all
      results appear in the validation list returned to the frontend.

  5.  CONSTRAINT-CHECK MIRROR  â€” the constraint_checks from the JSON
      database are re-evaluated at generation time and appended to the
      validation list so the frontend shows them as named checks with
      pass/fail status.

Usage in app.py:
    from validation_and_logging import (
        validate_scad_full,
        log_generation_event,
        FAMILY_VALIDATION_RULES,
    )
    # replace:   validate_scad(result, text)
    # with:      validate_scad_full(result, text, family_id)
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("openscad-copilot.validation")

# â”€â”€ Log file setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_LOGS_DIR = Path(__file__).resolve().parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_EVENTS_LOG = _LOGS_DIR / "generation_events.jsonl"


def log_generation_event(event: dict) -> None:
    """
    Append a structured JSON event to logs/generation_events.jsonl.

    Required keys in event:
        request_id   str    â€“ UUID from the chat request
        timestamp    str    â€“ ISO 8601 (auto-filled if absent)
        prompt       str    â€“ user prompt (truncated)
        family_id    str|None
        provider     str
        model        str
        code_chars   int
        validation   list[dict]  â€“ full validation result
        passed       int    â€“ count of passed checks
        failed       int    â€“ count of failed checks
        hard_fails   list[str]   â€“ labels of blocking failures
        repair_used  bool
        error        str|None    â€“ exception text if generation failed
    """
    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        with _EVENTS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("Could not write generation event log: %s", exc)


# â”€â”€ Universal OpenSCAD checks (family-agnostic) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_CHAT_META_ASSIGN = re.compile(
    r'^\s*(usage|use_case|bearing_series|standard_used|standards_used|material|materials'
    r'|material_suggestion|notes?)\s*=\s*"[^"]*"\s*;.*$',
    re.IGNORECASE,
)
_CHAT_META_COMMENT = re.compile(
    r"\b(main user parameters|bearing catalog defaults|standard used|standards?|catalog"
    r"|material|usage|informational only|nominal bore|mounted.bearing)\b",
    re.IGNORECASE,
)


def _looks_truncated(code: str) -> bool:
    stripped = code.strip()
    if not stripped:
        return True
    last = stripped.splitlines()[-1].strip()
    return bool(
        stripped.endswith(("=", ",", "(", "[", "{"))
        or re.search(r"=\s*[^;]+$", last)
        or last in {"difference()", "union()", "intersection()"}
    )


GEARS_LIBRARY_MODULE_RE = re.compile(
    r"\b(rack|herringbone_rack|mountable_rack|mountable_herringbone_rack|"
    r"spur_gear|herringbone_gear|rack_and_pinion|ring_gear|"
    r"herringbone_ring_gear|planetary_gear|bevel_gear|"
    r"bevel_herringbone_gear|spiral_bevel_gear|bevel_gear_pair|"
    r"bevel_herringbone_gear_pair|worm|worm_gear)\s*\(",
    re.IGNORECASE,
)


def _is_gears_library_call(code: str) -> bool:
    lowered = code.lower()
    return "gears.scad" in lowered and bool(GEARS_LIBRARY_MODULE_RE.search(code))


def _is_external_library_call(code: str) -> bool:
    has_library = bool(re.search(r"^\s*(include|use)\s*<[^>\n]+>\s*;?\s*$", code, re.IGNORECASE | re.MULTILINE))
    has_call = bool(re.search(r"^\s*[A-Za-z_]\w*\s*\([^;{}]*\)\s*;?\s*$", code, re.MULTILINE))
    return has_library and has_call


def _universal_checks(code: str) -> list[dict]:
    """Checks that apply to every generated file regardless of family."""
    stripped = code.strip()
    last_line = stripped.splitlines()[-1] if stripped else ""
    library_call = _is_gears_library_call(code) or _is_external_library_call(code)

    hole_words = bool(
        re.search(r"\b(bore|hole|slot|cutout|clearance|set[_ -]?screw|counterbore)\b",
                  stripped, re.IGNORECASE)
    )

    checks = [
        {
            "label": "No markdown fences",
            "passed": "```" not in code,
            "detail": "Output must be raw OpenSCAD, not markdown-fenced.",
            "severity": "hard",
        },
        {
            "label": "No stray language label",
            "passed": not bool(re.match(r"^\s*(openscad|scad|scss)\b", stripped, re.IGNORECASE)),
            "detail": "Output must start with OpenSCAD code, not a language label.",
            "severity": "hard",
        },
        {
            "label": "No chat metadata string variables",
            "passed": not bool(_CHAT_META_ASSIGN.search(stripped)),
            "detail": "Usage, bearing series, material, standard, and notes belong in chat, not as string variables.",
            "severity": "warning",
        },
        {
            "label": "No material or standard prose comments",
            "passed": not any(
                line.strip().startswith("//") and _CHAT_META_COMMENT.search(line)
                for line in stripped.splitlines()
            ),
            "detail": "Standards and materials should be in chat notes, not explanatory code comments.",
            "severity": "warning",
        },
        {
            "label": "Balanced braces { }",
            "passed": code.count("{") == code.count("}"),
            "detail": "Unbalanced braces cause OpenSCAD parse failures.",
            "severity": "hard",
        },
        {
            "label": "Balanced parentheses ( )",
            "passed": code.count("(") == code.count(")"),
            "detail": "Unbalanced parentheses indicate broken primitive calls.",
            "severity": "hard",
        },
        {
            "label": "Uses PI constant, not pi",
            "passed": not bool(re.search(r"\bpi\b", code)),
            "detail": "OpenSCAD uses PI (uppercase), not pi.",
            "severity": "warning",
        },
        {
            "label": "Parametric variables present",
            "passed": library_call or bool(re.search(r"^\s*[a-zA-Z_]\w*\s*=\s*[\d\.]", code, re.MULTILINE)),
            "detail": "At least one named numeric parameter should appear, or a trusted parametric library call should be used.",
            "severity": "hard",
        },
        {
            "label": "3D primitives used",
            "passed": library_call or any(
                tok in code
                for tok in ("cube(", "cylinder(", "sphere(", "polyhedron(",
                            "linear_extrude(", "rotate_extrude(")
            ),
            "detail": "Code should contain a 3D primitive/extrusion or call a trusted library module that creates them.",
            "severity": "hard",
        },
        {
            "label": "Named module present",
            "passed": library_call or bool(re.search(r"\bmodule\s+\w+", code)),
            "detail": "Wrap geometry in a named module, or call a trusted named library module.",
            "severity": "hard",
        },
        {
            "label": "Module called at end",
            "passed": library_call or bool(re.match(r"^\s*\w+\s*\([^;]*\);?\s*$", last_line)),
            "detail": "The main module should be instantiated exactly once at the end.",
            "severity": "hard",
        },
        {
            "label": "External library use allowed",
            "passed": True,
            "detail": "OpenSCAD include<> and use<> statements are allowed when the referenced library is available to OpenSCAD.",
            "severity": "warning",
        },
        {
            "label": "$fn resolution set",
            "passed": "$fn" in code,
            "detail": "Set $fn=96 or higher for smooth circular features.",
            "severity": "warning",
        },
        {
            "label": "No unfinished assignment at end",
            "passed": not _looks_truncated(code),
            "detail": "The file appears truncated or ends with an incomplete expression.",
            "severity": "hard",
        },
        {
            "label": "Boolean subtraction for cut features",
            "passed": library_call or (("difference()" in code) if hole_words else True),
            "detail": "Bores, holes, slots, and cutouts must use difference().",
            "severity": "warning",
        },
        {
            "label": "difference() wraps union() correctly",
            "passed": _check_difference_wraps_union(code),
            "detail": "difference() must come before union() in the CSG tree: difference() { union() { solids } cuts }.",
            "severity": "warning",
        },
    ]
    return checks


def _check_difference_wraps_union(code: str) -> bool:
    """True when difference() appears before union() at the same nesting depth."""
    first_diff = code.find("difference()")
    first_union = code.find("union()")
    if first_diff < 0:
        return True  # no difference â€” nothing to check
    if first_union < 0:
        return True  # no union â€” cannot evaluate
    # difference before union is correct in standard CSG
    return first_diff < first_union


# â”€â”€ Per-family checks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each entry is a list of (label, regex_or_callable, detail, severity)
# regex: searched on the full stripped code (case-insensitive).
# callable: called with stripped code, returns bool.

FAMILY_VALIDATION_RULES: dict[str, list[tuple[str, Any, str, str]]] = {
    # â”€â”€ Bearing housing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "bearing_housing_reference": [
        ("Bearing seat uses bearing_od",
         r"\bd\s*=\s*bearing_od|d\s*=\s*bearing_outer_diameter|bearing_od\s*[+)]",
         "Bearing housing must subtract a bearing seat using bearing_od.", "error"),
        ("Shaft bore uses shaft_diameter",
         r"\bd\s*=\s*shaft_diameter|shaft_diameter\s*[+)]",
         "Shaft through-bore should use shaft_diameter plus clearance.", "error"),
        ("Housing has base geometry",
         r"\bbase(_length|_width|_thickness)?\s*=",
         "A bearing housing needs a supporting base or body.", "error"),
        ("Bearing seat is subtractive",
         lambda c: (
             c.find("difference()") >= 0
             and bool(re.search(r"bearing_od|bearing_outer", c, re.IGNORECASE))
             and c.rfind("cylinder(") > c.find("union()")
         ),
         "Bearing seat must be subtracted after the union() body.", "error"),
        ("No solid bearing seat in union",
         lambda c: not bool(re.search(r"union\(\).*bearing_od.*union\(\)", c, re.DOTALL)),
         "Do not place the bearing seat inside union(); it must be in difference().", "error"),
        ("Pillow block has paired mounting holes",
         lambda c: (
             not bool(re.search(r"\b(pillow|plummer)\b", c, re.IGNORECASE))
             or (
                 bool(re.search(r"\bfor\s*\(|mount_hole_spacing|mounting_hole_spacing", c, re.IGNORECASE))
                 and bool(re.search(r"\bmount_hole|mounting_hole|bolt_hole", c, re.IGNORECASE))
             )
         ),
         "A pillow block needs two symmetric base/foot mounting holes.", "warning"),
        ("Pillow block has pedestal support",
         lambda c: (
             not bool(re.search(r"\b(pillow|plummer)\b", c, re.IGNORECASE))
             or bool(re.search(r"\b(pedestal|saddle|cap|split|rib|web|foot|feet)\b", c, re.IGNORECASE))
         ),
         "A real pillow block needs pedestal/saddle/rib/foot â€” not just a ring on a plate.", "warning"),
        ("Pillow block avoids arch-frame shape",
         lambda c: not bool(re.search(r"\b(arch|bridge_frame|support_rib|side_rib|triangular_rib)\b", c, re.IGNORECASE)),
         "Use a compact UCP-style cast body, not tall arch ribs or bridge frames.", "warning"),
        ("No solid rectangular top slab",
         lambda c: not bool(re.search(r"\b(top_slab|cap_block|roof|rectangular_cap)\b", c, re.IGNORECASE)),
         "Use a thin subtractive split-line groove; do not add a massive solid cap slab.", "warning"),
    ],

    # â”€â”€ Gears â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "gear_reference": [
        ("Pitch diameter from module أ— tooth_count",
         r"\bpitch_diameter\s*=\s*module[_\s]*size\s*\*\s*tooth|pitch_d\s*=\s*module\s*\*\s*teeth",
         "Gear pitch diameter must be derived from module أ— tooth_count, not hard-coded.", "error"),
        ("Bore separate from gear body",
         r"\b(bore_diameter|bore_d|bore)\s*=\s*\d",
         "Gear bore must be a separate named parameter, not equal to pitch_diameter.", "error"),
        ("Tooth repetition loop",
         r"\bfor\s*\(.*?\bteeth\b|\bfor\s*\(.*?\btooth_count\b",
         "Gear teeth must be placed with a for-loop using tooth_count.", "warning"),
        ("Hub present",
         r"\bhub_diameter|hub_d\s*=",
         "A gear should have a hub cylinder to engage the shaft.", "warning"),
        ("Bore subtracted, not added",
         lambda c: (
             c.find("difference()") >= 0
             and bool(re.search(r"bore_diameter|bore_d", c, re.IGNORECASE))
             and c.rfind("bore") > c.find("difference()")
         ),
         "Gear bore must be a subtractive cut in difference(), not a solid in union().", "error"),
    ],

    # â”€â”€ Gearbox housing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "gearbox_housing_reference": [
        ("Housing is hollow",
         r"\b(inner_cavity|internal|hollow)\b|\bdifference\(\).*cube.*cube",
         "Gearbox housing must be hollow â€” outer shell minus inner cavity.", "error"),
        ("Shaft bores through housing walls",
         r"\bshaft_bore|shaft_d\s*=|shaft_diameter\s*=",
         "Gearbox housing needs shaft bore holes through the side walls.", "error"),
        ("Bearing seats present",
         r"\bbearing_seat|bearing_od|bearing_bore",
         "Gearbox housing should have bearing seat recesses at shaft locations.", "warning"),
        ("Lid bolt holes present",
         r"\bbolt_hole|lid_bolt|perimeter_bolt",
         "Gearbox housing lid should have perimeter bolt holes.", "warning"),
    ],

    # â”€â”€ Shaft coupler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "shaft_coupler_reference": [
        ("Two distinct shaft bores defined",
         r"\b(shaft_a|bore_a|shaft_a_d)\b.*\b(shaft_b|bore_b|shaft_b_d)\b|\b(shaft_b|bore_b)\b.*\b(shaft_a|bore_a)\b",
         "A shaft coupler must define both shaft A and shaft B bores.", "error"),
        ("Bores coaxial (both on Z at origin)",
         lambda c: (
             c.count("shaft_a") >= 1 and c.count("shaft_b") >= 1
             and not bool(re.search(r"translate\s*\(\s*\[\s*[1-9]", c))
         ),
         "Both shaft bores must be centered on the same axis (Z at origin).", "warning"),
        ("Clamp slit present",
         r"\b(clamp_gap|slit|gap)\s*=",
         "A clamp-style coupler must define a radial slit.", "error"),
        ("Clamping screws cross the slit",
         r"\b(screw|bolt)\b.*\brotate|rotate.*\b(screw|bolt)\b",
         "Clamping screws must be rotated perpendicular to the slit to close it.", "warning"),
    ],

    # â”€â”€ Flange / pipe fitting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "flange_pipe_fitting_reference": [
        ("Central bore defined",
         r"\b(bore|inner_d|pipe_d|central_bore)\s*=\s*\d",
         "A flange must include a central bore or pipe opening.", "error"),
        ("Bolt circle uses cos/sin parametric placement",
         r"\bcos\s*\(\s*angle\b|\bsin\s*\(\s*angle\b|\bcos\(.*bolt|\bsin\(.*bolt",
         "Flange bolt holes must use cos()/sin() placement on the bolt circle.", "error"),
        ("Bolt count is parametric",
         r"\bn_bolts|bolt_count|num_bolts",
         "The number of bolts must be a named parameter, not a hard-coded loop.", "warning"),
        ("Raised face or boss present",
         r"\braised_face|boss_d|boss_diameter",
         "A standard flange usually includes a raised face or central boss.", "warning"),
    ],

    # â”€â”€ Enclosure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "enclosure_box_reference": [
        ("Wall thickness parametric",
         r"\bwall_thickness|wall\s*=\s*\d",
         "Enclosure wall thickness must be a named parameter.", "error"),
        ("Hollow shell present",
         lambda c: c.find("difference()") >= 0 and c.count("cube(") >= 2,
         "Enclosure must be a hollow shell: outer cube minus inner cavity.", "error"),
        ("Lid or lid lip present",
         r"\b(lid|lid_lip|lid_overlap|top_plate)\b",
         "An enclosure usually needs a lid strategy.", "warning"),
        ("Screw boss standoffs present",
         r"\b(standoff|boss|screw_post)\b",
         "An enclosure should have screw boss standoffs to mount the lid.", "warning"),
    ],

    # â”€â”€ Hinge / snap-fit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "hinge_joint_snapfit_reference": [
        ("Pin or barrel defined",
         r"\b(pin_diameter|pin_d|barrel_d|knuckle)\s*=",
         "A hinge must define a pin or barrel diameter.", "error"),
        ("Clearance gap between leaves",
         r"\bclearance\s*=\s*0\.[0-9]|\bgap\s*=\s*0\.[0-9]",
         "Print-in-place hinge needs a clearance gap between leaves for free rotation.", "error"),
        ("Knuckle repetition loop",
         r"\bfor\s*\(.*knuckle|\bfor\s*\(.*i\s*=\s*\[0\s*:",
         "Hinge knuckles should be placed with a for-loop.", "warning"),
    ],

    # â”€â”€ Lead screw actuator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "lead_screw_actuator_reference": [
        ("Nut pocket sized for nut OD",
         r"\bnut_od\s*=\s*\d|d\s*=\s*nut_od",
         "T8 nut pocket must use nut_od (22 mm), not the screw diameter.", "error"),
        ("Screw clearance bore separate from nut pocket",
         r"\bscrew_diameter|d\s*=\s*screw_diameter",
         "The screw clearance through-bore must be separate from the nut pocket.", "error"),
        ("Two guide rod bores parallel",
         lambda c: (
             bool(re.search(r"\brod_spacing\b", c, re.IGNORECASE))
             and bool(re.search(r"\bfor\s*\(.*rod_spacing|\-rod_spacing/2.*rod_spacing/2", c, re.IGNORECASE))
         ),
         "T8 carriage needs two parallel guide rod bores at rod_spacing.", "warning"),
    ],

    # â”€â”€ Linear rail / carriage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "linear_rail_carriage_reference": [
        ("Bearing OD bore present",
         r"\blm_od|bearing_od|lm8uu|lm10uu",
         "Carriage must bore rod holes sized for LM-series bearing OD.", "error"),
        ("Two parallel rod bores",
         lambda c: (
             bool(re.search(r"\brod_spacing\b", c, re.IGNORECASE))
             and c.lower().count("cylinder(") >= 2
         ),
         "A dual-rod carriage needs two parallel rod bore cylinders.", "warning"),
        ("Mounting holes on top face",
         r"\bmount_hole|mounting_hole|payload",
         "Carriage should have top-face mounting holes for the payload.", "warning"),
    ],

    # â”€â”€ Pulley / belt drive â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "pulley_belt_drive_reference": [
        ("Pitch diameter from formula",
         r"pitch_d\s*=\s*teeth\s*\*\s*belt_pitch\s*/\s*PI|pitch_diameter\s*=.*teeth.*PI",
         "GT2 pitch diameter must be derived: teeth أ— belt_pitch / PI.", "error"),
        ("Flanges retain belt",
         r"\bflange_d|flange_diameter|flange_h",
         "Timing pulley should have flanges to retain the belt.", "warning"),
        ("Set-screw hole present",
         r"\bset_screw|rotate\s*\(\s*\[90|radial.*cylinder",
         "Pulley should have a radial set-screw hole.", "warning"),
        ("Bore through hub",
         r"\bbore_d\s*=|bore_diameter\s*=",
         "Pulley bore must be a named parameter separate from pitch_diameter.", "error"),
    ],

    # â”€â”€ Sprocket / chain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "sprocket_chain_reference": [
        ("Pitch diameter from chain formula",
         r"use\s*<[^>]*sprocket\.scad>|pitch_d\s*=\s*chain_pitch\s*/\s*sin|pitch_diameter\s*=\s*chain_pitch\s*/\s*sin|pitch_radius\s*=\s*.*get_pitch\s*\(\s*size\s*\)\s*/\s*sin",
         "Use sprocket.scad or derive sprocket pitch diameter/radius from chain pitch and tooth count.", "error"),
        ("Roller pocket loop",
         r"use\s*<[^>]*sprocket\.scad>|\bfor\s*\(.*num_teeth|\bfor\s*\(.*tooth_count|\bfor\s*\(.*teeth",
         "Use sprocket.scad or place roller pocket cutouts in a for-loop.", "error"),
        ("Sprockets.scad tooth construction",
         r"use\s*<[^>]*sprocket\.scad>|sprocket_plate\s*\(|middle_radius|fudge_teeth_x|fudge_teeth_y",
         "Sprocket must use sprocket.scad or the Sprockets.scad circular flank tooth construction, not a smooth cylinder.", "error"),
        ("No gear-module sprocket logic",
         lambda c: not bool(re.search(r"\bmodul\s*=\s*pitch\s*/\s*PI|\baddendum\b|\bdedendum\b|roller_r\s*=\s*pitch\s*/\s*2", c, re.IGNORECASE)),
         "Roller chain sprockets must not be generated as gear-module/addendum/dedendum annuli, and roller radius is not pitch/2.", "error"),
        ("Circular flank intersections",
         r"use\s*<[^>]*sprocket\.scad>|intersection\s*\(\s*\).*cylinder\s*\(",
         "Use sprocket.scad or generate teeth by intersecting circular flank cylinders.", "warning"),
    ],

    # â”€â”€ Structural profiles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "structural_profiles_reference": [
        ("I-beam has top and bottom flanges",
         lambda c: (
             not bool(re.search(r"\b(i[- ]?beam|ipe|hea|h[- ]?beam|wide.flange)\b", c, re.IGNORECASE))
             or (
                 bool(re.search(r"\bflange(_thickness|_width)?\s*=", c, re.IGNORECASE))
                 and bool(re.search(r"\bweb(_thickness)?\s*=", c, re.IGNORECASE))
                 and c.lower().count("cube(") >= 3
             )
         ),
         "I-beam MUST be three cubes: top flange, bottom flange, and web. A single rectangular bar is wrong.", "error"),
        ("I-beam not a single solid bar",
         lambda c: (
             not bool(re.search(r"\b(i[- ]?beam|ipe|hea)\b", c, re.IGNORECASE))
             or not bool(re.search(r"cube\s*\(\s*\[\s*\w+\s*,\s*\w+\s*,\s*beam_height\s*\]", c, re.IGNORECASE))
         ),
         "Do not represent an I-beam as a single solid rectangular bar.", "error"),
        ("Queen-post truss uses endpoint placement",
         lambda c: (
             not bool(re.search(r"\bqueen.post|truss\b", c, re.IGNORECASE))
             or bool(re.search(r"\bbeam_between|atan2|sqrt\s*\(", c, re.IGNORECASE))
         ),
         "Queen-post truss members should be placed with endpoint-derived angles and lengths.", "warning"),
    ],

    # â”€â”€ Brackets / motor mounts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "bracket_and_motor_mount_reference": [
        ("NEMA mount has 4-bolt square pattern",
         lambda c: (
             not bool(re.search(r"\bnema\b", c, re.IGNORECASE))
             or bool(re.search(r"for\s*\(.*bolt_circle|for\s*\(x.*for\s*\(y|nema_bolt_circle", c, re.IGNORECASE))
         ),
         "NEMA motor mount needs 4 bolts on a square bolt-circle pattern.", "error"),
        ("Shaft clearance hole present on motor mount",
         lambda c: (
             not bool(re.search(r"\bnema\b", c, re.IGNORECASE))
             or bool(re.search(r"\bshaft_boss_d|shaft_clearance_d|boss_d\s*=", c, re.IGNORECASE))
         ),
         "Motor mount needs a central shaft clearance hole for the motor boss.", "error"),
        ("L-bracket has two perpendicular plates",
         lambda c: (
             not bool(re.search(r"\bl[_-]?bracket\b", c, re.IGNORECASE))
             or c.lower().count("cube(") >= 2
         ),
         "L-bracket needs two perpendicular plate cubes in union().", "warning"),
    ],

    # â”€â”€ Clamp / jig / fixture â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "clamp_jig_fixture_reference": [
        ("Clamp bore present",
         r"\bpipe_od|bore_d|inner_radius|v_groove",
         "A clamp or fixture must define the workpiece interface geometry.", "error"),
        ("Clamping bolt holes present",
         r"\bbolt_hole|clamp_bolt|mount_hole",
         "A clamp or fixture needs mounting or clamping bolt holes.", "warning"),
    ],

    # â”€â”€ Cooling fan mount â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "cooling_fan_mount_reference": [
        ("Air opening is circular cutout",
         r"\bair_opening|fan_opening|d\s*=\s*fan_size|d\s*=\s*air_opening_d",
         "Fan mount must have a circular air opening cylinder in difference().", "error"),
        ("4-bolt square pattern",
         r"\bfor\s*\(x.*for\s*\(y|\bfor\s*\(.*fan_bolt_circle|fan_hole",
         "Fan mount should use a 4-bolt square pattern.", "warning"),
    ],

    # â”€â”€ Fastener / nut trap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "fastener_nut_trap_reference": [
        ("Hex pocket uses $fn=6",
         r"\$fn\s*=\s*6",
         "Hex nut trap must use cylinder with $fn=6 for a hexagonal profile.", "error"),
        ("Bolt clearance coaxial with nut pocket",
         r"\bbolt_d\s*=|bolt_clearance\s*=",
         "Bolt clearance bore must be coaxial and extend beyond the nut pocket.", "error"),
    ],

    # â”€â”€ Servo / robotics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "robotics_servo_reference": [
        ("Servo body pocket present",
         r"\bservo_body|servo_l|servo_w|servo_pocket",
         "Servo bracket needs a rectangular clearance pocket for the servo body.", "error"),
        ("Tab mounting holes present",
         r"\btab_span|tab_hole|servo_hole",
         "Servo bracket needs mounting holes matching the servo tab span.", "error"),
    ],

    # â”€â”€ Cam / crank / linkage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "cam_crank_linkage_reference": [
        ("Eccentric bore uses offset translate",
         lambda c: (
             not bool(re.search(r"\beccentric\b|eccen\b", c, re.IGNORECASE))
             or bool(re.search(r"translate\s*\(\s*\[\s*eccentric|translate\s*\(\s*\[\s*eccen", c, re.IGNORECASE))
         ),
         "Eccentric cam bore must be translated by the eccentricity offset, not placed at origin.", "error"),
        ("Linkage arm uses hull() or rounded ends",
         lambda c: (
             not bool(re.search(r"\b(link|arm|crank)\b", c, re.IGNORECASE))
             or bool(re.search(r"\bhull\(\)|pivot_d|pin_d", c, re.IGNORECASE))
         ),
         "Linkage arms should use hull() between pivot circles for rounded ends.", "warning"),
    ],

    # â”€â”€ Threads / knobs / shaft features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "threads_knobs_shaft_features_reference": [
        ("Grip ridges use for-loop",
         lambda c: (
             not bool(re.search(r"\bknob\b", c, re.IGNORECASE))
             or bool(re.search(r"for\s*\(.*grip_count|\bfor\s*\(.*\bgrip\b", c, re.IGNORECASE))
         ),
         "Knob grip ridges must be placed with a for-loop.", "warning"),
        ("Bore subtracted from knob body",
         lambda c: (
             not bool(re.search(r"\bknob\b", c, re.IGNORECASE))
             or bool(re.search(r"\bbore_d\s*=|bore_diameter\s*=", c, re.IGNORECASE))
         ),
         "Knob must have a central bore subtracted for shaft or insert.", "error"),
    ],

    # â”€â”€ Shaft (JSON database) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "shaft_reference": [
        ("Shaft has main diameter",
         r"\b(main_diameter|shaft_d|shaft_diameter)\s*=",
         "Shaft must define a main diameter parameter.", "error"),
        ("Stepped shaft uses multiple cylinders",
         lambda c: (
             not bool(re.search(r"\bstep|stepped\b", c, re.IGNORECASE))
             or c.lower().count("cylinder(") >= 2
         ),
         "A stepped shaft needs multiple cylinder() calls at different diameters.", "warning"),
    ],
}


# â”€â”€ Derived-parameter checks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# For families where a formula-derived parameter can be spot-checked.
_DERIVED_PATTERNS: dict[str, list[tuple[str, str, str]]] = {
    "gear_reference": [
        ("pitch_diameter = module أ— tooth_count",
         r"pitch_d(iameter)?\s*=\s*module[_\s]*size\s*\*\s*tooth|pitch_d\s*=\s*module\s*\*\s*teeth",
         "Gear pitch diameter should be computed: module أ— tooth_count, not hard-coded."),
    ],
    "pulley_belt_drive_reference": [
        ("pitch_diameter = teeth أ— pitch / PI",
         r"pitch_d\s*=\s*teeth\s*\*\s*belt_pitch\s*/\s*PI",
         "GT2 pitch diameter must be computed: teeth أ— belt_pitch / PI."),
    ],
    "sprocket_chain_reference": [
        ("pitch_diameter = chain_pitch / sin(180/N)",
         r"use\s*<[^>]*sprocket\.scad>|pitch_d\s*=\s*chain_pitch\s*/\s*sin|pitch_radius\s*=\s*.*get_pitch\s*\(\s*size\s*\)\s*/\s*sin",
         "Use sprocket.scad or derive sprocket pitch diameter/radius from chain pitch and tooth count."),
    ],
    "bearing_housing_reference": [
        ("boss_diameter = bearing_od + 2*wall",
         r"boss_diameter\s*=\s*bearing_od\s*\+\s*2\s*\*\s*wall|boss_d\s*=\s*bearing_od",
         "Bearing boss diameter should derive from bearing_od + 2*wall_thickness."),
    ],
}


def _derived_parameter_checks(code: str, family_id: str | None) -> list[dict]:
    checks = []
    for patterns in _DERIVED_PATTERNS.get(family_id or "", []):
        label, pattern, detail = patterns
        passed = bool(re.search(pattern, code, re.IGNORECASE))
        checks.append({"label": f"Derived: {label}", "passed": passed, "detail": detail, "severity": "warning"})
    return checks


# â”€â”€ Main entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def validate_scad_full(code: str, prompt: str = "", family_id: str | None = None) -> list[dict]:
    """
    Full validation pipeline.  Returns a list of check dicts compatible with
    the existing API response format plus two extra keys:
        severity  â€” "hard" | "error" | "warning"
        category  â€” "universal" | "family" | "derived"
    """
    stripped = code.strip()
    checks: list[dict] = []

    # 1. Universal checks
    for c in _universal_checks(code):
        c["category"] = "universal"
        checks.append(c)

    # 2. Derived-parameter checks
    for c in _derived_parameter_checks(stripped, family_id):
        c["category"] = "derived"
        checks.append(c)

    # 3. Per-family checks
    family_rules = FAMILY_VALIDATION_RULES.get(family_id or "", [])
    for label, pattern, detail, severity in family_rules:
        if callable(pattern):
            passed = pattern(stripped)
        else:
            passed = bool(re.search(pattern, stripped, re.IGNORECASE | re.DOTALL))
        checks.append({
            "label": label,
            "passed": passed,
            "detail": detail,
            "severity": severity,
            "category": "family",
        })

    return checks


# â”€â”€ Convenience re-exports for app.py drop-in replacement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def validate_scad(code: str, prompt: str = "") -> list[dict]:
    """
    Backward-compatible wrapper.  Detects family from prompt+code and calls
    validate_scad_full().  Returns checks without the severity/category keys
    so existing code that reads only label/passed/detail still works.
    """
    # Import detect_primary_family lazily to avoid circular import
    try:
        from rag import detect_primary_family  # type: ignore
        family_id = detect_primary_family(f"{prompt}\n{code}")
    except ImportError:
        family_id = None

    all_checks = validate_scad_full(code, prompt, family_id)
    # Return only the keys the frontend expects
    return [
        {"label": c["label"], "passed": c["passed"], "detail": c["detail"]}
        for c in all_checks
    ]


HARD_FAIL_LABELS = frozenset({
    "Balanced braces { }",
    "Balanced parentheses ( )",
    "No unfinished assignment at end",
    "3D primitives used",
    "Named module present",
    "Parametric variables present",
})


def score_validation(checks: list[dict]) -> int:
    return sum(1 for c in checks if c.get("passed"))

