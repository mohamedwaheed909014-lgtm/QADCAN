"""
rag.py — Mechanical Parts RAG engine
=====================================
Key improvements over original:

1. SCORE CAPPING  — all final scores are clamped to [0.0, 1.0].  Raw cosine
   similarity is already in [0,1] for normalised embeddings.  Keyword bonuses
   are now *relative boosts* applied before normalisation, not additive offsets
   that could push totals above 1.

2. NEGATIVE-KEYWORD PENALTY — each JSON database may declare retrieval.
   negative_keywords.  A query containing those terms receives a penalty that
   prevents the wrong family database from being retrieved.

3. PER-FAMILY MIN_SCORE_THRESHOLD — each JSON database may declare
   retrieval.min_score_threshold.  Documents whose adjusted score is below
   that threshold are dropped from results even if they ranked in top_k.

4. FAMILY ISOLATION — once a primary family is detected via keyword matching,
   documents from *other unrelated families* receive a small isolation penalty,
   reducing noise from unrelated reference files.

5. SUPPORT DOCUMENT GATING — support docs (grabcad reference, true pillow
   block reference) are only included when their family is in the detected set.

6. SCORE SEMANTICS  — scores returned to callers are cosine similarities in
   [0.0, 1.0].  A score of 1.0 means perfect match; 0.0 means no match.
   Callers must NOT interpret scores > 1.0 as "extra relevant".
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import requests
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as SentenceTransformerModel

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:
    _SentenceTransformer = None

log = logging.getLogger("openscad-copilot.rag")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)

ACTIVE_PART_FAMILY_ID    = os.getenv("ACTIVE_PART_FAMILY_ID", "bearing_housing_reference").strip() or "bearing_housing_reference"
ACTIVE_PART_FAMILY_LABEL = os.getenv("ACTIVE_PART_FAMILY_LABEL", "Bearing Housing / Pillow Block").strip() or "Bearing Housing / Pillow Block"
GENERAL_RAG_ID           = "general_mechanical_parts"
GENERAL_RAG_LABEL        = "General Mechanical Parts"
EMBEDDING_BACKEND        = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()
SENTENCE_MODEL_NAME      = os.getenv("SENTENCE_MODEL_NAME", "all-MiniLM-L6-v2")
OLLAMA_BASE_URL          = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL       = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
NOMIC_USE_PREFIX         = os.getenv("NOMIC_USE_PREFIX", "false").lower() == "true"

# ── Score constants ────────────────────────────────────────────────────────────
# All scores are cosine similarities in [0.0, 1.0].
# Bonuses and penalties are fractional weights applied *before* the final
# clamp(0, 1), so no document can ever score above 1.0.

# Fraction of the gap-to-1.0 that a matched primary family document gains.
# e.g. raw_score=0.7 → gap=0.3 → bonus=0.3*0.6=0.18 → adjusted=0.88
PRIMARY_FAMILY_BONUS_FRACTION    = 0.60
SECONDARY_FAMILY_BONUS_FRACTION  = 0.45
PART_RECORD_BONUS_FRACTION       = 0.55
SUPPORT_DOC_BASE_BONUS_FRACTION  = 0.35

# Penalty subtracted from unrelated family documents when a primary family
# is already detected.  Keeps irrelevant reference files out of top results.
UNRELATED_FAMILY_PENALTY         = 0.12

# Penalty per negative-keyword hit (from retrieval.negative_keywords in JSON).
NEGATIVE_KEYWORD_PENALTY_PER_HIT = 0.08

# Default minimum score threshold for a document to appear in results.
# Individual JSON databases can override this in metadata.retrieval.min_score_threshold.
DEFAULT_MIN_SCORE_THRESHOLD      = 0.10


# ── Data models ───────────────────────────────────────────────────────────────
class Document(BaseModel):
    id: str
    title: str
    text: str
    source: str = "local"
    file_type: str = "txt"
    family: str | None = None
    record: dict | None = None
    min_score_threshold: float = DEFAULT_MIN_SCORE_THRESHOLD
    negative_keywords: list[str] = []


# ── Family / keyword tables ───────────────────────────────────────────────────
FAMILY_ID_TO_SHORT_KEY: dict[str, str] = {
    "shaft_reference":                          "shaft",
    "bearing_housing_reference":                "bearing",
    "bracket_and_motor_mount_reference":        "bracket",
    "cam_crank_linkage_reference":              "linkage",
    "clamp_jig_fixture_reference":              "fixture",
    "cooling_fan_mount_reference":              "fan",
    "enclosure_box_reference":                  "enclosure",
    "fastener_nut_trap_reference":              "fastener",
    "flange_pipe_fitting_reference":            "flange",
    "gear_reference":                           "gear",
    "gearbox_housing_reference":                "gearbox",
    "hinge_joint_snapfit_reference":            "hinge",
    "lead_screw_actuator_reference":            "actuator",
    "linear_rail_carriage_reference":           "linear_rail",
    "pulley_belt_drive_reference":              "pulley",
    "robotics_servo_reference":                 "servo",
    "shaft_coupler_reference":                  "coupler",
    "sprocket_chain_reference":                 "sprocket",
    "structural_profiles_reference":            "structural",
    "common_trusses_reference":                 "truss",
    "threads_knobs_shaft_features_reference":   "threaded",
}

FAMILY_ID_TO_LABEL: dict[str, str] = {
    fid: fid.replace("_reference", "").replace("_", " ").title()
    for fid in FAMILY_ID_TO_SHORT_KEY
}
FAMILY_ID_TO_LABEL["bearing_housing_reference"] = "Bearing Housing / Pillow Block"
FAMILY_ID_TO_LABEL[GENERAL_RAG_ID]              = GENERAL_RAG_LABEL

PULLEY_SUBTYPE_KEYWORDS = {
    "smooth_bearing_idler": [
        "smooth idler",
        "smooth idler pulley",
        "bearing idler",
        "bearing sleeve",
        "idler sleeve",
        "free spinning pulley",
        "passive pulley",
        "belt tensioner",
        "belt tensioner idler",
        "623zz",
        "624zz",
        "625zz",
        "608zz",
        "shoulder bolt idler",
        "m3 shoulder bolt",
        "m4 shoulder bolt",
        "m5 shoulder bolt"
    ],

    "toothed_timing_pulley": [
        "timing pulley",
        "drive pulley",
        "motor pulley",
        "toothed pulley",
        "grub screw pulley",
        "captive nut pulley",
        "pulley3dp",
        "pulleycad",
        "pulleyteeth",
        "pulleyretainer",
        "pulleyidler",
        "toothwidthtweak",
        "teeth",
        "tooth count"
    ]
}
PRIMARY_FAMILY_KEYWORDS: dict[str, list[str]] = {
    
    "shaft_reference": [
        "shaft", "axle", "spindle", "rotating", "cylindrical",
        "keyway", "d shaft", "splined shaft", "lead screw", "threaded shaft", "motor shaft",
    ],
    "bearing_housing_reference": [
        "bearing housing", "pillow block", "bearing block", "6204", "bearing seat",
        "plummer block", "pedestal bearing", "ucp", "ucf", "ucfl", "uct",
    ],
    "bracket_and_motor_mount_reference": [
        "bracket", "motor mount", "nema", "mounting bracket", "angle bracket",
        "l-bracket", "gusset", "motor flange",
    ],
    "cam_crank_linkage_reference": [
        "cam", "crank", "linkage", "slider crank", "four bar", "scotch yoke",
        "eccentric cam", "geneva mechanism",
    ],
    "clamp_jig_fixture_reference": [
        "clamp", "jig", "fixture", "v-block", "drill jig", "toggle clamp",
        "saddle clamp", "strap clamp", "workholding",
    ],
    "cooling_fan_mount_reference": [
        "fan mount", "cooling fan", "fan bracket", "40mm fan", "fan guard",
        "fan duct", "heatsink", "airflow opening",
    ],
    "enclosure_box_reference": [
        "enclosure", "electronics box", "project box", "lid", "screw posts",
        "din rail", "snap fit box", "raspberry pi case", "arduino enclosure",
    ],
    "fastener_nut_trap_reference": [
        "nut trap", "heat insert", "m3", "m4", "fastener", "through hole",
        "counterbore", "countersink", "hex nut pocket",
    ],
    "flange_pipe_fitting_reference": [
        "flange", "pipe flange", "pipe fitting", "bolt circle", "central bore",
        "blind flange", "weld neck", "hose barb", "dn50",
    ],
    "gear_reference": [
        "gear", "spur gear", "helical gear", "bevel gear", "worm gear", "worm wheel",
        "spiral gear", "spiral bevel gear", "herringbone gear", "double helical gear",
        "rack", "pinion", "planetary gear", "epicyclic gear", "ring gear", "annulus gear", "sun gear",
        "pitch diameter", "involute", "tooth count", "gear ratio",
        "keyway", "final drive gear", "gear assembly", "crown gear",
        "cone distance", "helix angle", "spiral angle", "pressure angle",
    ],
    "gearbox_housing_reference": [
        "gearbox", "gear housing", "gearbox housing", "two-piece gearbox",
        "gearbox cover", "gearbox half",
    ],
    "hinge_joint_snapfit_reference": [
        "hinge", "joint", "snap fit", "snap-fit", "mounting leaves",
        "living hinge", "print-in-place hinge",
    ],
    "lead_screw_actuator_reference": [
        "lead screw", "actuator", "nut carriage", "t8", "carriage block",
        "ballscrew", "acme screw", "linear actuator",
    ],
    "linear_rail_carriage_reference": [
        "linear rail", "carriage", "guide rail", "carriage block", "mgn12",
        "linear guide", "v-slot", "linear bearing",
    ],
    "pulley_belt_drive_reference": [
        "pulley", "belt drive", "timing pulley", "belt pulley",
        "toothed pulley", "flanged pulley", "idler pulley", "tensioner pulley",
        "smooth idler", "smooth idler pulley", "bearing idler", "bearing sleeve",
        "belt tensioner", "free spinning pulley", "passive pulley", "idler sleeve",
        "623zz", "624zz", "625zz", "608zz",
        "shoulder bolt idler", "m4 shoulder bolt", "m3 shoulder bolt", "m5 shoulder bolt",
        "gt2", "gt2 2mm", "gt2 3mm", "gt2 5mm",
        "htd", "htd 3mm", "htd 5mm", "htd 8mm", "htd3", "htd5", "htd8",
        "t2.5", "t5", "t10", "at5", "mxl", "xl timing", "40dp",
        "grub screw pulley", "captive nut pulley", "pulley-generator.scad",
        "belt width", "toothWidthTweak",
    ],
    "robotics_servo_reference": [
        "servo", "mg996r", "servo bracket", "robotics", "servo mount",
        "servo horn", "servo arm",
    ],
    "shaft_coupler_reference": [
        "shaft coupler", "coupler", "clamping slit", "split clamp coupler",
        "jaw coupler", "oldham coupler", "flexible coupler",
    ],
    "sprocket_chain_reference": [
        "sprocket", "chain drive", "roller chain", "chain sprocket",
        "chain wheel", "ansi chain", "#25 chain", "#35 chain",
    ],
    "structural_profiles_reference": [
        "i-beam", "i beam", "beam", "structural profile", "extrusion",
        "2020 extrusion", "i section", "standard i-beam", "standard i beam",
        "classic i-beam", "s-beam", "s beam", "tapered flange",
        "narrow flange", "narrow i-beam", "narrow i beam", "ipe beam",
        "h-beam", "h beam", "wide flange", "w-beam", "w beam", "hea beam", "heb beam",
        "truss", "roof truss", "bridge truss", "king post", "king-post truss",
        "king post truss", "queen post", "queen-post truss", "queen post truss",
        "pratt truss", "howe truss", "warren truss", "fink truss",
        "k truss", "k-truss", "bowstring truss", "scissor truss",
        "gusset plate", "top chord", "bottom chord", "web member",
        "diagonal member", "vertical member", "straining beam",
    ],
    "common_trusses_reference": [
        "truss", "trusses", "roof truss", "bridge truss", "steel truss",
        "timber truss", "aluminum truss", "aluminium truss",
        "king post", "king-post truss", "king post truss",
        "queen post", "queen-post truss", "queen post truss",
        "pratt truss", "howe truss", "warren truss", "fink truss",
        "k truss", "k-truss", "bowstring truss", "scissor truss",
        "top chord", "bottom chord", "web member", "truss web",
        "diagonal member", "vertical member", "panel point", "truss node",
        "gusset plate", "node plate", "bearing plate", "support pad",
        "truss parameters", "truss formula", "truss openscad",
    ],
    "threads_knobs_shaft_features_reference": [
        "knob", "thumb knob", "thread", "standoff", "shaft feature",
        "spacer", "washer", "threaded insert",
    ],
}


SUPPORT_DOC_KEYWORDS: dict[str, list[str]] = {
    "gear_examples_primary_reference": [
        "gear", "gears.scad", "gears-master", "spur gear", "helical gear",
        "spiral gear", "herringbone gear", "ring gear", "annulus",
        "rack", "pinion", "rack and pinion", "bevel gear", "bevel gear pair",
        "spiral bevel gear", "planetary gear", "epicyclic gear",
        "worm", "worm gear", "worm drive",
        "tooth_number", "helix_angle", "pressure_angle", "lead_angle",
    ],
    "bearing_housing_grabcad_real_examples_reference": [
        "grabcad", "real example", "ucp", "ucp 208", "ucp plummer",
        "ucf", "ucf 204", "ucf 205", "ucfl", "uct", "take-up",
        "pillow block", "plummer block", "pedestal bearing",
        "igus", "xiros", "igubal", "schaeffler",
    ],
    "bearing_housing_true_pillow_block_reference": [
        "pillow block", "plummer block", "pedestal bearing",
        "bearing housing", "bearing block", "6204", "608",
        "shaft support", "cnc leadscrew", "conveyor", "robot axle",
        "saddle", "pedestal", "cap split",
    ],
    "research_paper_modeling_gap_corrections_reference": [
        "i-beam", "i section", "h-beam", "queen post", "queen-post truss",
        "king post", "king-post truss", "pratt truss", "howe truss",
        "warren truss", "fink truss", "k truss", "truss",
        "spatial reasoning",
    ],
    "research_paper_few_shot_examples_reference": [
        "example", "few shot", "few-shot", "reference design",
        "baseline",
    ],
    "sprocket": [
        "sprocket", "Sprockets.scad", "roller chain sprocket",
        "ansi sprocket", "#25 sprocket", "#35 sprocket", "#40 sprocket",
        "#41 sprocket", "#50 sprocket", "#60 sprocket", "#80 sprocket",
        "chain wheel", "chain sprocket", "keyway sprocket",
        "set screw sprocket", "roller pockets", "pitch radius",
        "circular flank tooth", "sprocket_plate",
    ],
}

PRIMARY_SUPPORT_DOC_BY_FAMILY: dict[str, str] = {
    "gear_reference": "gear_examples_primary_reference",
    "sprocket_chain_reference": "sprocket",
}

# Populated at document load time from JSON retrieval.negative_keywords
PART_DATABASE_KEYWORDS:   dict[str, list[str]] = {}
NEGATIVE_KEYWORDS_BY_DOC: dict[str, list[str]] = {}
MIN_SCORE_BY_DOC:         dict[str, float]     = {}
PART_DATABASE_DOC_FAMILY: dict[str, str]       = {}

# Legacy gear references contained simplified cube/peg tooth examples that
# produce blocky, unrealistic gears. Keep the family router, but do not load
# those documents into RAG now that gears-master is the authoritative source.
DISABLED_RAG_DOC_STEMS: set[str] = {
    "gear_reference",
    "sprocket_chain_reference",
    "sprocket_exact_openscad_reference",
}


# ── JSON helpers ─────────────────────────────────────────────────────────────
def _json_to_rag_text(data: dict, stem: str) -> str:
    meta = data.get("__metadata__", {})
    if meta.get("format") == "assimp2json":
        lines: list[str] = [
            f"Document: {stem.replace('_', ' ')}",
            "Format: assimp2json mesh export",
            f"Format version: {meta.get('version', 'unknown')}",
            "",
        ]
        meshes = data.get("meshes", [])
        total_vertices = 0
        total_faces = 0
        all_coords: list[float] = []
        for index, mesh in enumerate(meshes):
            vertices   = mesh.get("vertices", [])
            faces      = mesh.get("faces", [])
            name       = mesh.get("name") or f"mesh_{index}"
            vcount     = len(vertices) // 3
            fcount     = len(faces)
            total_vertices += vcount
            total_faces    += fcount
            all_coords.extend(vertices)
            lines.append(
                f"Mesh '{name}': {vcount} vertices, {fcount} triangles"
                + (", normals present" if "normals" in mesh else "")
            )
        if all_coords:
            xs = all_coords[0::3]; ys = all_coords[1::3]; zs = all_coords[2::3]
            lines += [
                "", f"Total vertices: {total_vertices}",
                f"Total triangles: {total_faces}", "",
                "Bounding box (mm):",
                f"  X: {min(xs):.3f} to {max(xs):.3f} (width {max(xs)-min(xs):.3f})",
                f"  Y: {min(ys):.3f} to {max(ys):.3f} (depth {max(ys)-min(ys):.3f})",
                f"  Z: {min(zs):.3f} to {max(zs):.3f} (height {max(zs)-min(zs):.3f})",
            ]
        return "\n".join(lines)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _part_record_to_rag_text(record: dict, database_meta: dict) -> str:
    standards         = record.get("standards") or []
    standards_used    = record.get("standards_used") or database_meta.get("standard_notes") or []
    output_policy     = record.get("output_policy") or database_meta.get("output_policy") or {}
    source_refs       = record.get("source_references") or database_meta.get("source_references") or []
    dimensions        = record.get("standard_dimensions_mm") or {}
    fit_guidance      = record.get("fit_guidance_mm") or {}
    material_sug      = record.get("material_suggestions") or database_meta.get("material_suggestions") or []
    main_params       = record.get("main_parameters") or []
    secondary_params  = record.get("secondary_parameters") or []
    derived_params    = record.get("derived_parameters") or {}
    feature_library   = record.get("feature_library") or {}
    constraint_checks = record.get("constraint_checks") or []
    external_examples = record.get("external_examples") or []

    lines: list[str] = [
        f"Mechanical part database record: {record.get('name', record.get('id', 'unknown part'))}",
        f"Part family: {record.get('family', 'unknown')}",
        f"Database version: {database_meta.get('version', 'unknown')}",
        f"Units: {database_meta.get('units', 'millimeters')}",
        f"Intended manufacturing: {', '.join(record.get('manufacturing_processes', [])) or 'general mechanical CAD'}",
        f"Standards and catalog basis: {', '.join(standards) if standards else 'curated engineering defaults'}",
        "",
        "Output policy:",
        f"- OpenSCAD code: {output_policy.get('openscad_code', 'geometry only') if isinstance(output_policy, dict) else 'geometry only'}",
        "- Material suggestions and standard explanations are chat-after-code only; do not put them in OpenSCAD comments.",
        "- Material suggestions apply to the generated part body or housing, not purchased bearings, fasteners, balls, races, or inserts.",
        "",
        "Search aliases:",
        ", ".join(record.get("aliases", [])),
        "",
        "Standard dimensions in millimeters:",
    ]

    if dimensions:
        for key, value in dimensions.items():
            if not key.startswith("_"):
                lines.append(f"- {key}: {value}")
    else:
        lines.append("- No fixed dimensions; use parametric defaults.")

    if fit_guidance:
        lines += ["", "Fit and clearance guidance in millimeters:"]
        for key, value in fit_guidance.items():
            if not key.startswith("_"):
                lines.append(f"- {key}: {value}")

    if standards_used:
        lines += ["", "Structured standards used:"]
        for item in standards_used:
            if isinstance(item, dict):
                line = f"- {item.get('name', 'standard')}: {item.get('applies_to', '')}"
                note = item.get("note") or item.get("short_explanation")
                if note:
                    line += f". {note}"
                lines.append(line.rstrip())
            else:
                lines.append(f"- {item}")

    if material_sug:
        lines += ["", "Generated body or housing material suggestions (chat-after-code only):"]
        for mat in material_sug:
            if isinstance(mat, dict):
                lines.append(
                    f"- {mat.get('material', 'unknown')}: {mat.get('use_case', '')} "
                    f"{mat.get('notes', '')}".strip()
                )
            else:
                lines.append(f"- {mat}")

    if main_params:
        lines += ["", "Structured main parameters:"]
        for item in main_params:
            if isinstance(item, dict):
                lines.append(f"- {item.get('name', 'parameter')}: {json.dumps(item, ensure_ascii=False)}")
            else:
                lines.append(f"- {item}")

    if secondary_params:
        lines += ["", "Structured secondary parameters that may be assumed:"]
        for item in secondary_params:
            if isinstance(item, dict):
                lines.append(f"- {item.get('name', 'parameter')}: {json.dumps(item, ensure_ascii=False)}")
            else:
                lines.append(f"- {item}")

    if derived_params:
        lines += ["", "Derived parameters and formulas:"]
        for key, value in derived_params.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict):
                lines.append(f"- {key}: {value.get('formula', '')}. {value.get('note', '')}".rstrip())
            else:
                lines.append(f"- {key}: {value}")

    if feature_library:
        lines += ["", "Structured feature library with OpenSCAD implementation snippets:"]
        for fname, spec in feature_library.items():
            if fname.startswith("_"):
                continue
            if not isinstance(spec, dict):
                lines.append(f"- {fname}: {spec}")
                continue
            lines.append(f"- {fname}: {spec.get('purpose', '')}".rstrip())
            for label, key in (("main parameters", "main_parameters"), ("derived from", "derived_from"), ("constraints", "constraints")):
                vals = spec.get(key)
                if vals:
                    lines.append(f"  {label}: {', '.join(str(v) for v in vals)}")
            snippet = spec.get("openscad_snippet")
            if snippet:
                lines.append(f"  OpenSCAD snippet: {snippet}")

    if constraint_checks:
        lines += ["", "Machine-readable constraint checks:"]
        for chk in constraint_checks:
            if isinstance(chk, dict):
                lines.append(
                    f"- {chk.get('id', 'check')}: {chk.get('severity', 'review')} | "
                    f"{chk.get('expression', '')} | {chk.get('message', '')}".rstrip()
                )
            else:
                lines.append(f"- {chk}")

    for section_name, heading in (
        ("main_features",       "Main functional features"),
        ("required_features",   "Required functional features"),
        ("constraint_parameters","Main constraint parameters"),
        ("design_rules",        "Design rules"),
        ("validation_criteria", "Validation criteria"),
        ("failure_modes",       "Common failure modes to avoid"),
    ):
        values = record.get(section_name) or []
        if isinstance(values, dict):
            lines += ["", f"{heading}:"]
            for key, value in values.items():
                if not key.startswith("_"):
                    lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value}")
        elif values:
            lines += ["", f"{heading}:"]
            lines.extend(f"- {item}" for item in values)

    scad = (record.get("openscad_module") or "").strip()
    if scad:
        lines += ["", "Reusable vanilla OpenSCAD module for this real mechanical part:", scad]

    if source_refs:
        lines += ["", "Source references encoded in this record:"]
        for src in source_refs:
            if isinstance(src, dict):
                title = src.get("title", "unnamed source")
                url   = src.get("url")
                note  = src.get("note")
                line  = f"- {title}"
                if url:
                    line += f" ({url})"
                if note:
                    line += f": {note}"
                lines.append(line)
            else:
                lines.append(f"- {src}")

    if external_examples:
        lines += ["", "External GrabCAD preview/reference examples:"]
        for ex in external_examples:
            lines.append(f"- {ex.get('title', 'GrabCAD example')}: {ex.get('url', '')}")

    return "\n".join(lines)


def _load_mechanical_parts_database(data: dict, path: Path) -> list[Document]:
    meta      = data.get("metadata", {})
    retrieval = meta.get("retrieval", {})
    neg_kw    = [str(k).lower() for k in retrieval.get("negative_keywords", [])]
    threshold = float(retrieval.get("min_score_threshold", DEFAULT_MIN_SCORE_THRESHOLD))
    documents: list[Document] = []

    for record in data.get("parts", []):
        record_id = str(record.get("id", "")).strip()
        family    = str(record.get("family") or meta.get("family") or "").strip()
        if not record_id:
            continue

        doc_id   = f"partdb_{record_id}"
        aliases  = record.get("aliases") or []
        features = record.get("main_features") or record.get("required_features") or []

        PART_DATABASE_KEYWORDS[doc_id] = [
            str(item).lower()
            for item in [record_id.replace("_", " "), record.get("name", ""), *aliases, *features]
            if str(item).strip()
        ]
        NEGATIVE_KEYWORDS_BY_DOC[doc_id] = neg_kw
        MIN_SCORE_BY_DOC[doc_id]         = threshold
        PART_DATABASE_DOC_FAMILY[doc_id] = family or GENERAL_RAG_ID

        documents.append(Document(
            id=doc_id,
            title=f"Part Database: {record.get('name', record_id).strip()}",
            text=_part_record_to_rag_text(record, meta),
            source="mechanical-parts-database",
            file_type=path.suffix.lstrip(".") or "json",
            family=family or GENERAL_RAG_ID,
            record=record,
            min_score_threshold=threshold,
            negative_keywords=neg_kw,
        ))

    return documents


# ── Document loading ──────────────────────────────────────────────────────────
def load_documents() -> list[Document]:
    seen: set[str] = set()
    documents: list[Document] = []
    PART_DATABASE_KEYWORDS.clear()
    NEGATIVE_KEYWORDS_BY_DOC.clear()
    MIN_SCORE_BY_DOC.clear()
    PART_DATABASE_DOC_FAMILY.clear()

    frontend_json = list(FRONTEND_DIR.glob("*.json")) if FRONTEND_DIR.exists() else []
    paths = sorted(list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.json")) + list(DOCS_DIR.glob("*.scad")) + frontend_json)

    for path in paths:
        if path.stem in DISABLED_RAG_DOC_STEMS:
            continue
        if path.stem in seen:
            continue
        seen.add(path.stem)

        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed JSON %s: %s", path.name, exc)
                continue
            if isinstance(data, dict) and data.get("type") == "mechanical_parts_database":
                documents.extend(_load_mechanical_parts_database(data, path))
                continue
            text      = _json_to_rag_text(data, path.stem)
            file_type = "json"
        elif path.suffix == ".scad":
            text      = path.read_text(encoding="utf-8").strip()
            file_type = "scad"
        else:
            text      = path.read_text(encoding="utf-8").strip()
            file_type = "txt"

        support_family = next(
            (fid for fid, doc_id in PRIMARY_SUPPORT_DOC_BY_FAMILY.items() if doc_id == path.stem),
            None,
        )
        family = path.stem if path.stem in PRIMARY_FAMILY_KEYWORDS else support_family
        documents.append(Document(
            id=path.stem,
            title=path.stem.replace("_", " ").title(),
            text=text,
            source="local",
            file_type=file_type,
            family=family,
        ))

    return documents


# ── Keyword helpers ───────────────────────────────────────────────────────────
def _keyword_hits(query: str, keywords: list[str]) -> int:
    lowered = query.lower()
    hits = 0
    for kw in keywords:
        term = kw.lower().strip()
        if not term:
            continue
        if term[0].isalnum() and term[-1].isalnum():
            pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
            if re.search(pattern, lowered):
                hits += 1
        elif term in lowered:
            hits += 1
    return hits


def detect_primary_families(query: str) -> list[str]:
    lowered = query.lower()
    if re.search(r"\b(gt2|htd|mxl|40dp|t2\.5|t5|t10|at5|timing\s+pulley|belt\s+pulley|pulley-generator\.scad)\b", lowered):
        remaining = [
            doc_id for doc_id in _detect_primary_families_by_keyword(query)
            if doc_id != "pulley_belt_drive_reference"
        ]
        return ["pulley_belt_drive_reference", *remaining]
    if re.search(r"\b(sprocket|roller chain|chain sprocket|chain wheel|ansi\s*#?\s*(25|35|40|41|50|60|80)|#\s*(25|35|40|41|50|60|80)\s*(chain|sprocket))\b", lowered):
        remaining = [
            doc_id for doc_id in _detect_primary_families_by_keyword(query)
            if doc_id != "sprocket_chain_reference"
        ]
        return ["sprocket_chain_reference", *remaining]
    return _detect_primary_families_by_keyword(query)


def _detect_primary_families_by_keyword(query: str) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for doc_id, keywords in PRIMARY_FAMILY_KEYWORDS.items():
        matches = _keyword_hits(query, keywords)
        if matches > 0:
            ranked.append((matches, doc_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in ranked]


def detect_primary_family(text: str) -> str | None:
    families = detect_primary_families(text)
    return families[0] if families else None


def detect_pulley_subtype(query: str) -> str | None:
    q = query.lower()

    for subtype, keywords in PULLEY_SUBTYPE_KEYWORDS.items():
        if _keyword_hits(q, keywords) > 0:
            return subtype

    return None


def _is_family_database_doc(doc_id: str, family_id: str) -> bool:
    family = PART_DATABASE_DOC_FAMILY.get(doc_id)
    if family:
        return family == family_id
    short_key = FAMILY_ID_TO_SHORT_KEY.get(family_id, "")
    return bool(short_key and doc_id.startswith("partdb_") and f"_{short_key}_" in f"{doc_id}_")


# ── Embedding back-ends ───────────────────────────────────────────────────────
def embedding_model_name() -> str:
    if EMBEDDING_BACKEND == "ollama":
        return OLLAMA_EMBED_MODEL
    if EMBEDDING_BACKEND == "tfidf":
        return "tfidf"
    return SENTENCE_MODEL_NAME


@lru_cache(maxsize=1)
def _sentence_model() -> SentenceTransformerModel:
    if _SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    return _SentenceTransformer(SENTENCE_MODEL_NAME, local_files_only=True)


def _embed_texts_local(texts: list[str]) -> np.ndarray:
    model      = _sentence_model()
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(embeddings, dtype=np.float32)


def _embed_texts_ollama(texts: list[str], is_query: bool = False) -> np.ndarray:
    if NOMIC_USE_PREFIX:
        prefix = "search_query: " if is_query else "search_document: "
        texts  = [prefix + t for t in texts]
    vectors = []
    for text in texts:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": text},
            timeout=60,
        )
        if response.status_code == 404:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=60,
            )
        response.raise_for_status()
        payload = response.json()
        vector  = payload.get("embeddings") or payload.get("embedding")
        if isinstance(vector, list) and vector and isinstance(vector[0], list):
            vector = vector[0]
        if not vector:
            raise RuntimeError("Ollama returned an empty embedding.")
        vectors.append(vector)
    return np.asarray(vectors, dtype=np.float32)


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    if not texts:
        return np.array([])
    if EMBEDDING_BACKEND == "ollama":
        return _embed_texts_ollama(texts, is_query=is_query)
    return _embed_texts_local(texts)


# ── Score adjustment helpers ──────────────────────────────────────────────────
def _adjust_score(
    raw: float,
    doc: Document,
    family_ids: list[str],
    query: str,
) -> float:
    """
    Apply bonuses and penalties to a raw cosine similarity score.

    Rules:
    - All arithmetic keeps the value in [0.0, 1.0].
    - Bonuses are applied as a fraction of the *remaining gap to 1.0*, so they
      can never push a score over 1.0.
    - Penalties are simple subtractions, floored at 0.0.
    - The returned value is always clamped to [0.0, 1.0].
    """
    score = float(raw)

    # ── Negative keyword penalty ──────────────────────────────────────────────
    neg_hits = _keyword_hits(query, NEGATIVE_KEYWORDS_BY_DOC.get(doc.id, []))
    if neg_hits:
        score = max(0.0, score - neg_hits * NEGATIVE_KEYWORD_PENALTY_PER_HIT)

    # ── Primary family boost ──────────────────────────────────────────────────
    if doc.id in family_ids:
        rank  = family_ids.index(doc.id)
        frac  = PRIMARY_FAMILY_BONUS_FRACTION - rank * 0.05
        score = score + (1.0 - score) * max(0.0, frac)

    # ── Part-database record boost (family match) ─────────────────────────────
    elif any(_is_family_database_doc(doc.id, fid) for fid in family_ids):
        score = score + (1.0 - score) * SECONDARY_FAMILY_BONUS_FRACTION

    # ── Support document bonus (keyword-gated) ────────────────────────────────
    elif doc.id in SUPPORT_DOC_KEYWORDS:
        hits = _keyword_hits(query, SUPPORT_DOC_KEYWORDS.get(doc.id, []))
        if hits > 0:
            frac  = SUPPORT_DOC_BASE_BONUS_FRACTION + 0.04 * hits
            score = score + (1.0 - score) * min(frac, 0.65)
        else:
            # No matching keywords — lightly penalise to avoid noise
            score = max(0.0, score - 0.06)

    # ── Unrelated family penalty ──────────────────────────────────────────────
    elif family_ids and doc.family and doc.family not in family_ids:
        score = max(0.0, score - UNRELATED_FAMILY_PENALTY)

    # ── Exact part-record keyword boost ──────────────────────────────────────
    exact_hits = _keyword_hits(query, PART_DATABASE_KEYWORDS.get(doc.id, []))
    if exact_hits:
        frac  = PART_RECORD_BONUS_FRACTION + 0.08 * (exact_hits - 1)
        score = score + (1.0 - score) * min(frac, 0.80)

    return min(1.0, max(0.0, score))


# ── Knowledge base ────────────────────────────────────────────────────────────
class KnowledgeBase:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._selected_tfidf_cache:      dict[tuple[str, ...], tuple[TfidfVectorizer, object, list[Document]]] = {}
        self._selected_embedding_cache:  dict[tuple[str, ...], tuple[np.ndarray, list[Document]]]              = {}
        self.runtime_backend = EMBEDDING_BACKEND
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._documents                = load_documents()
            self._embeddings               = None
            self._tfidf_vectorizer         = None
            self._tfidf_matrix             = None
            self._selected_tfidf_cache     = {}
            self._selected_embedding_cache = {}
            self.runtime_backend           = EMBEDDING_BACKEND

    @property
    def doc_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return len(self._documents)

    @property
    def index_built(self) -> bool:
        return self._embeddings is not None or self._tfidf_matrix is not None

    def documents(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id":      doc.id,
                    "title":   doc.title,
                    "source":  doc.source,
                    "file_type": doc.file_type,
                    "family":  doc.family,
                    "chars":   len(doc.text),
                    "excerpt": doc.text[:260],
                }
                for doc in self._documents
            ]

    def part_records(self) -> list[dict]:
        with self._lock:
            return [
                {"id": doc.id, "title": doc.title, "family": doc.family, "record": doc.record}
                for doc in self._documents
                if doc.source == "mechanical-parts-database" and doc.record
            ]

    def family_schema(self, family_id: str | None = None) -> dict:
        records    = [i for i in self.part_records() if family_id is None or i.get("family") == family_id]
        features:   list[str]       = []
        constraints: dict[str, dict] = {}
        derived:    dict[str, dict]  = {}
        materials:  list[dict]       = []
        examples:   list[dict]       = []
        validation: list[str]        = []
        standards:  list[dict]       = []
        feature_lib: dict[str, dict] = {}
        chk_list:   list[dict]       = []
        out_policy: dict             = {}

        for item in records:
            rec = item["record"] or {}
            if not out_policy and isinstance(rec.get("output_policy"), dict):
                out_policy = rec["output_policy"]
            for f in rec.get("main_features", []) or rec.get("required_features", []):
                if f not in features:
                    features.append(f)
            constraints.update(rec.get("constraint_parameters") or {})
            derived.update(rec.get("derived_parameters") or {})
            for s in rec.get("standards_used", []):
                if s not in standards:
                    standards.append(s)
            for name, spec in (rec.get("feature_library") or {}).items():
                feature_lib.setdefault(name, spec)
            for chk in rec.get("constraint_checks", []):
                if chk not in chk_list:
                    chk_list.append(chk)
            for mat in rec.get("material_suggestions", []):
                if mat not in materials:
                    materials.append(mat)
            for ex in rec.get("external_examples", []):
                if ex not in examples:
                    examples.append(ex)
            for v in rec.get("validation_criteria", []):
                if v not in validation:
                    validation.append(v)

        return {
            "id":                  family_id or GENERAL_RAG_ID,
            "label":               FAMILY_ID_TO_LABEL.get(family_id or GENERAL_RAG_ID, GENERAL_RAG_LABEL),
            "record_count":        len(records),
            "main_features":       features,
            "constraint_parameters": constraints,
            "derived_parameters":  derived,
            "standards_used":      standards,
            "feature_library":     feature_lib,
            "constraint_checks":   chk_list,
            "output_policy":       out_policy,
            "material_suggestions": materials,
            "external_examples":   examples,
            "validation_criteria": validation,
        }

    # ── Index management ──────────────────────────────────────────────────────
    def _ensure_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self._embeddings = embed_texts([doc.text for doc in self._documents], is_query=False)
        return self._embeddings

    def _ensure_tfidf(self):
        if self._tfidf_matrix is None:
            self._tfidf_vectorizer = TfidfVectorizer(
                stop_words="english", max_features=20000, ngram_range=(1, 2)
            )
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(
                [doc.text for doc in self._documents]
            )
        return self._tfidf_matrix

    def warmup(self) -> None:
        with self._lock:
            if not self._documents:
                return
            if EMBEDDING_BACKEND == "tfidf":
                self._ensure_tfidf()
                self.runtime_backend = "tfidf-warmed"
                return
            try:
                self._ensure_embeddings()
                self.runtime_backend = (
                    "ollama-warmed" if EMBEDDING_BACKEND == "ollama"
                    else "sentence-transformers-warmed"
                )
            except Exception as exc:
                log.warning("Dense RAG warmup failed, prebuilding TF-IDF fallback (%s).", exc)
                self._ensure_tfidf()
                self.runtime_backend = "tfidf-warmed-fallback"

    # ── Hit construction ──────────────────────────────────────────────────────
    @staticmethod
    def _make_hit(doc: Document, score: float) -> dict:
        # Score is always in [0.0, 1.0] — enforced by _adjust_score.
        return {
            "id":      doc.id,
            "title":   doc.title,
            "source":  doc.source,
            "score":   round(min(1.0, max(0.0, float(score))), 4),
            "excerpt": doc.text[:420],
            "text":    doc.text,
        }

    # ── Core ranking ──────────────────────────────────────────────────────────
    def _rank_documents(
        self,
        query: str,
        docs: list[Document],
        scores: np.ndarray,
        top_k: int,
    ) -> list[dict]:
        family_ids = detect_primary_families(query)
        pulley_subtype = detect_pulley_subtype(query)

        scored: list[tuple[float, Document]] = []

        for doc, raw in zip(docs, scores):
            if (
                "gear_reference" in family_ids
                and doc.id != PRIMARY_SUPPORT_DOC_BY_FAMILY.get("gear_reference")
                and doc.family != "gear_reference"
            ):
                continue

            if (
                family_ids
                and doc.id in SUPPORT_DOC_KEYWORDS
                and doc.family
                and doc.family not in family_ids
            ):
                continue

            adj = _adjust_score(raw, doc, family_ids, query)

            # ── Pulley subtype correction ────────────────────────────────────
            doc_text = doc.text.lower()
            doc_id = doc.id.lower()

            if pulley_subtype == "smooth_bearing_idler":
                # Prefer standalone smooth bearing-idler records.
                if (
                    "smooth_bearing_idler" in doc_id
                    or "smooth bearing idler" in doc_text
                    or "standalone" in doc_text
                    or "no include required" in doc_text
                    or "does not use pulley-generator.scad" in doc_text
                ):
                    adj = min(1.0, adj + 0.35)

                # Penalize timing pulley/library examples.
                if (
                    "pulley3dp" in doc_text
                    or "pulleycad" in doc_text
                    or "include <pulley-generator.scad>" in doc_text
                    or "teethcount" in doc_text
                ):
                    adj = max(0.0, adj - 0.45)

            elif pulley_subtype == "toothed_timing_pulley":
                # Penalize smooth idler records for toothed pulley prompts.
                if (
                    "smooth bearing idler" in doc_text
                    or "standalone" in doc_text
                    or "does not use pulley-generator.scad" in doc_text
                    or "no include required" in doc_text
                ):
                    adj = max(0.0, adj - 0.30)

                # Prefer pulley-generator timing pulley examples.
                if (
                    "pulley3dp" in doc_text
                    or "pulleycad" in doc_text
                    or "include <pulley-generator.scad>" in doc_text
                ):
                    adj = min(1.0, adj + 0.20)

            # Drop documents below their declared or default threshold.
            threshold = MIN_SCORE_BY_DOC.get(doc.id, doc.min_score_threshold)
            if adj >= threshold:
                scored.append((adj, doc))

        scored.sort(key=lambda item: item[0], reverse=True)

        hits: list[dict] = []
        used_ids: set[str] = set()

        # 1. Surface any family-specific primary support reference first.
        for fid in family_ids:
            primary_support_id = PRIMARY_SUPPORT_DOC_BY_FAMILY.get(fid)
            if not primary_support_id:
                continue

            match = next(
                (item for item in scored if item[1].id == primary_support_id),
                None,
            )
            if match and match[1].id not in used_ids:
                hits.append(self._make_hit(match[1], match[0]))
                used_ids.add(match[1].id)
                if len(hits) >= top_k:
                    return hits

        # 2. Always surface the best database match for each detected family.
        for fid in family_ids:
            match = next(
                (
                    item
                    for item in scored
                    if item[1].id == fid or _is_family_database_doc(item[1].id, fid)
                ),
                None,
            )
            if match and match[1].id not in used_ids:
                hits.append(self._make_hit(match[1], match[0]))
                used_ids.add(match[1].id)
                if len(hits) >= top_k:
                    return hits

        # 3. For bearing queries: also pull in the true-pillow-block support doc.
        if "bearing_housing_reference" in family_ids and _keyword_hits(
            query,
            SUPPORT_DOC_KEYWORDS.get("bearing_housing_true_pillow_block_reference", []),
        ):
            tpb = next(
                (
                    item
                    for item in scored
                    if item[1].id == "bearing_housing_true_pillow_block_reference"
                ),
                None,
            )
            if tpb and tpb[1].id not in used_ids and len(hits) < top_k:
                hits.append(self._make_hit(tpb[1], tpb[0]))
                used_ids.add(tpb[1].id)

        # 4. Fill remaining slots from the sorted list, capping support docs to 1.
        support_added = 0
        for score, doc in scored:
            if doc.id in used_ids:
                continue

            if doc.id in SUPPORT_DOC_KEYWORDS:
                if support_added >= 1 and family_ids:
                    continue
                support_added += 1

            hits.append(self._make_hit(doc, score))
            used_ids.add(doc.id)

            if len(hits) >= top_k:
                break

        return hits

    # ── Selected-document retrieval ───────────────────────────────────────────
    def _selected_retrieve(
        self,
        query: str,
        docs: list[Document],
        top_k: int,
    ) -> list[dict]:
        texts     = [doc.text for doc in docs]
        cache_key = tuple(doc.id for doc in docs)

        if EMBEDDING_BACKEND == "tfidf":
            cached = self._selected_tfidf_cache.get(cache_key)
            if cached is None:
                vect   = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                matrix = vect.fit_transform(texts)
                cached = (vect, matrix, docs)
                self._selected_tfidf_cache[cache_key] = cached
            vect, matrix, _ = cached
            q_vec  = vect.transform([query])
            scores = cosine_similarity(q_vec, matrix).ravel()
            self.runtime_backend = f"selected-records-tfidf-cached:{len(docs)}"
        else:
            try:
                cached_emb = self._selected_embedding_cache.get(cache_key)
                if cached_emb is None:
                    cached_emb = (embed_texts(texts, is_query=False), docs)
                    self._selected_embedding_cache[cache_key] = cached_emb
                doc_emb, _ = cached_emb
                q_vec  = embed_texts([query], is_query=True)
                scores = cosine_similarity(q_vec, doc_emb)[0]
                self.runtime_backend = f"selected-records-ranked-cached:{len(docs)}"
            except Exception:
                cached = self._selected_tfidf_cache.get(cache_key)
                if cached is None:
                    vect   = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                    matrix = vect.fit_transform(texts)
                    cached = (vect, matrix, docs)
                    self._selected_tfidf_cache[cache_key] = cached
                vect, matrix, _ = cached
                q_vec  = vect.transform([query])
                scores = cosine_similarity(q_vec, matrix).ravel()
                self.runtime_backend = f"selected-records-tfidf-cached:{len(docs)}"

        return self._rank_documents(query, docs, scores, top_k)

    # ── Public retrieve ───────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        selected_doc_ids: list[str] | None = None,
    ) -> list[dict]:
        with self._lock:
            if not self._documents:
                return []

            selected = [did for did in (selected_doc_ids or []) if did]
            if selected:
                by_id         = {doc.id: doc for doc in self._documents}
                selected_docs = [by_id[did] for did in selected if did in by_id]
                if selected_docs:
                    return self._selected_retrieve(query, selected_docs, max(1, top_k))

            if EMBEDDING_BACKEND == "tfidf":
                matrix = self._ensure_tfidf()
                q_vec  = self._tfidf_vectorizer.transform([query])
                scores = cosine_similarity(q_vec, matrix)[0]
                self.runtime_backend = "tfidf-fallback"
                return self._rank_documents(query, self._documents, scores, top_k)

            try:
                embeddings = self._ensure_embeddings()
                q_vec      = embed_texts([query], is_query=True)
                scores     = cosine_similarity(q_vec, embeddings)[0]
                self.runtime_backend = (
                    "ollama" if EMBEDDING_BACKEND == "ollama" else "sentence-transformers"
                )
                return self._rank_documents(query, self._documents, scores, top_k)
            except Exception as exc:
                log.warning("Dense retrieval failed, using TF-IDF fallback (%s).", exc)
                matrix = self._ensure_tfidf()
                q_vec  = self._tfidf_vectorizer.transform([query])
                scores = cosine_similarity(q_vec, matrix)[0]
                self.runtime_backend = "tfidf-fallback"
                return self._rank_documents(query, self._documents, scores, top_k)