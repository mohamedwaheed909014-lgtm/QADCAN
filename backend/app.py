"""
Mechanical OpenSCAD Copilot backend.

This module keeps the HTTP layer, prompt construction, generation gateway,
history storage, and validation rules. Retrieval-augmented generation lives in
backend/rag.py so ingestion and retrieval can evolve independently.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from validation_and_logging import (
    validate_scad_full,
    log_generation_event,
    HARD_FAIL_LABELS as _VAL_HARD_FAIL_LABELS,
    score_validation as _val_score,
)

from auth import (
    init_db,
    create_user,
    authenticate_user,
    update_user,
    make_token,
    get_current_user,
)

from mechanical_intent import analyze_intent, format_intent_summary
from constraint_solver import solve_constraints, format_constraint_summary
from tolerance_engine import generate_tolerance_block, get_tolerance_table_html
from physics_engine import analyze_physics, format_physics_summary
from design_failure_detector import detect_failures, format_failure_report
from manufacturing_rules import format_mfg_context, check_dfm

from rag import (
    DOCS_DIR,
    EMBEDDING_BACKEND,
    FAMILY_ID_TO_LABEL,
    GENERAL_RAG_ID,
    GENERAL_RAG_LABEL,
    NOMIC_USE_PREFIX,
    OLLAMA_EMBED_MODEL,

    PART_DATABASE_KEYWORDS,
    PRIMARY_FAMILY_KEYWORDS,
    SUPPORT_DOC_KEYWORDS,
    KnowledgeBase,
    detect_primary_family,
    embed_texts,
    embedding_model_name,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("openscad-copilot")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
ACCEPTED_HISTORY_PATH = BASE_DIR / "accepted_history.jsonl"


def _load_windows_user_env(names: list[str]) -> None:
    if os.name != "nt":
        return
    try:
        import winreg
    except ImportError:
        return
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for name in names:
                if os.getenv(name):
                    continue
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    continue
                if value:
                    os.environ[name] = str(value)
    except OSError:
        return


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or os.getenv(name):
            continue
        os.environ[name] = value.strip().strip('"').strip("'")


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")
_load_windows_user_env(
    [
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACE_API_KEY",
    ]
)

OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MODELS = [
    item.strip()
    for item in os.getenv("OPENAI_MODELS", "gpt-4.1-mini").split(",")
    if item.strip()
]
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "Mechanical OpenSCAD Copilot")
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
OPENROUTER_MODELS = [
    item.strip()
    for item in os.getenv(
        "OPENROUTER_MODELS",
        "openai/gpt-oss-120b:free,google/gemma-4-31b-it:free,qwen/qwen3-coder:free",
    ).split(",")
    if item.strip()
]
OPENROUTER_PAID_MODELS = [
    item.strip()
    for item in os.getenv(
        "OPENROUTER_PAID_MODELS",
        "deepseek/deepseek-chat-v3-0324,deepseek/deepseek-chat,deepseek/deepseek-v3.2",
    ).split(",")
    if item.strip()
]
OPENROUTER_ALLOW_PAID = os.getenv("OPENROUTER_ALLOW_PAID", "false").strip().lower() in {"1", "true", "yes", "on"}
OPENROUTER_TIMEOUT_SEC = float(os.getenv("OPENROUTER_TIMEOUT_SEC", "120"))
HUGGINGFACE_BASE_URL = os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1")
HUGGINGFACE_DEFAULT_MODEL = os.getenv("HUGGINGFACE_MODEL", "openai/gpt-oss-120b:fastest")
HUGGINGFACE_MODELS = [
    item.strip()
    for item in os.getenv(
        "HUGGINGFACE_MODELS",
        "openai/gpt-oss-120b:fastest,deepseek-ai/DeepSeek-R1:fastest,Qwen/Qwen3-Coder-480B-A35B-Instruct:fastest",
    ).split(",")
    if item.strip()
]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_GENERATE_TIMEOUT_SEC = int(os.getenv("OLLAMA_GENERATE_TIMEOUT_SEC", "900"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "2400"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
MAX_REPAIR_ATTEMPTS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "2"))

OLLAMA_DISABLED_MODELS = {
    item.strip()
    for item in os.getenv("OLLAMA_DISABLED_MODELS", "qwen3:8b").split(",")
    if item.strip()
}
OLLAMA_CHAT_MODELS = [
    item.strip()
    for item in os.getenv(
        "OLLAMA_CHAT_MODELS",
        "qwen2.5-coder:7b,qwen3:8b,codellama:latest,mistral:7b-instruct",
    ).split(",")
    if item.strip() and item.strip() not in OLLAMA_DISABLED_MODELS
]
PREFERRED_OLLAMA_MODELS = ["qwen2.5-coder:7b", "mistral:7b-instruct", "codellama:latest"]

MAX_KNOWLEDGE_HITS_IN_PROMPT = 3
MAX_ACCEPTED_HITS_IN_PROMPT = 1
MAX_KNOWLEDGE_CHARS_OPENAI = 26000
MAX_KNOWLEDGE_CHARS_OLLAMA = 4200
MAX_ACCEPTED_CHARS_OPENAI = 3500
MAX_ACCEPTED_CHARS_OLLAMA = 1600
MAX_HISTORY_CHARS_OPENAI = 12000
MAX_HISTORY_CHARS_OLLAMA = 2500

OPENSCAD_CHEATSHEET = """
OFFICIAL OPENSCAD CHEATSHEET v2021.01
Source: https://openscad.org/cheatsheet/
Use these constructs directly, and use OpenSCAD libraries with include<> or use<> when they are appropriate for the requested part.

SYNTAX
var = value;
var = condition ? value_if_true : value_if_false;
var = function(x) x + x;
module name(...) { ... }  name();
function name(...) = ...; name();
include <file.scad>
use <file.scad>

CONSTANTS
undef
PI

OPERATORS
n + m
n - m
n * m
n / m
n % m
n ^ m
n < m
n <= m
b == c
b != c
n >= m
n > m
b && c
b || c
!b

SPECIAL VARIABLES
$fa
$fs
$fn
$t
$vpr
$vpt
$vpd
$vpf
$children
$preview

MODIFIER CHARACTERS
*
!
#
%

2D PRIMITIVES
circle(r)
circle(d=diameter)
square(size, center=false)
square([width, height], center=false)
polygon(points)
polygon(points, paths)
text(text, size, font, direction, language, script, halign, valign, spacing)
import("file.dxf", convexity=...)
import("file.svg", convexity=...)
projection(cut=false)

3D PRIMITIVES
sphere(r)
sphere(d=diameter)
cube(size, center=false)
cube([width, depth, height], center=false)
cylinder(h, r, center=false)
cylinder(h, d=diameter, center=false)
cylinder(h, r1, r2, center=false)
cylinder(h, d1=..., d2=..., center=false)
polyhedron(points, faces, convexity=...)
import("file.stl", convexity=...)
import("file.off", convexity=...)
import("file.amf", convexity=...)
import("file.3mf", convexity=...)
linear_extrude(height, center=false, convexity=..., twist=..., slices=...)
rotate_extrude(angle=360, convexity=...)
surface(file="file.dat", center=false, convexity=...)
surface(file="file.png", center=false, convexity=...)

TRANSFORMATIONS
translate([x, y, z])
rotate([x, y, z])
rotate(a, [x, y, z])
scale([x, y, z])
resize([x, y, z], auto=..., convexity=...)
mirror([x, y, z])
multmatrix(m)
color("name", alpha)
color("#rgb")
color("#rgba")
color("#rrggbb")
color("#rrggbbaa")
color([r, g, b, a])
offset(r=..., chamfer=false)
offset(delta=..., chamfer=false)
hull()
minkowski(convexity=...)

LISTS
list = [a, b, c];
value = list[2];
// WARNING: Slice notation l[1:], l[0:n], l[a:b] is NOT valid OpenSCAD — it will cause a syntax error.
// For recursive list summation use index-based recursion:
//   function list_sum(l, n) = (n <= 0) ? 0 : l[n-1] + list_sum(l, n-1);
//   Call full sum: list_sum(my_list, len(my_list))
//   Call partial sum of first i elements: list_sum(my_list, i)

BOOLEAN OPERATIONS
union()
difference()
intersection()

LIST COMPREHENSIONS
[for (i = range_or_list) i]
[for (init; condition; next) i]
[each i]
[for (i = ...) if (condition(i)) i]
[for (i = ...) if (condition(i)) x else y]
[for (i = ...) let(assignments) a]

FLOW CONTROL
for (i = [start:end]) { ... }
for (i = [start:step:end]) { ... }
for (i = [..., ..., ...]) { ... }
for (i = ..., j = ..., ...) { ... }
intersection_for(i = [start:end]) { ... }
intersection_for(i = [start:step:end]) { ... }
intersection_for(i = [..., ..., ...]) { ... }
if (...) { ... }
let(...) { ... }

TYPE TEST FUNCTIONS
is_undef()
is_bool()
is_num()
is_string()
is_list()
is_function()

OTHER
echo(...)
render(convexity=...)
children([idx])
assert(condition, message)

FUNCTIONS
concat()
lookup()
str()
chr()
ord()
search()
version()
version_num()
parent_module(idx)

MATHEMATICAL FUNCTIONS
abs()
sign()
sin()
cos()
tan()
acos()
asin()
atan()
atan2()
floor()
round()
ceil()
ln()
len()
log()
pow()
sqrt()
exp()
rands()
min()
max()
norm()
cross()

OUTPUT RULES FOR THIS APP
- Put editable numeric parameters at the top of the file.
- OpenSCAD libraries are allowed. Use include<> or use<> for available library modules when they improve correctness.
- Use PI, not pi.
- Set $fn = 96 or higher for circular features.
- Holes, bores, slots, and cutouts must be subtractive with difference().
- End the file by calling the main module exactly once.
- Use millimeters throughout.
- Keep coaxial bores aligned to the main axis.
- Repeated holes should come from for-loops or explicit symmetric placement.
- CRITICAL — OpenSCAD modules produce geometry but do NOT return values and CANNOT be assigned to variables.
  The following patterns are ALL INVALID and silently produce nothing:
    outer   = minkowski() { cube(...); sphere(...); };
    cavity  = hull()       { ... };
    body    = some_module(...);
    shafts  = union()      { ... };
    bolts   = difference() { ... };
  Write every CSG operation (union, difference, intersection, hull, minkowski) DIRECTLY
  nested inside a module body or at the top level — never pre-computed into a named variable.
  This applies equally to built-in operations AND user-defined module calls.
- OpenSCAD does NOT support list slicing. l[1:], l[0:n], and l[a:b] are SYNTAX ERRORS that crash the parser. For cumulative list summation always use index-based recursion: function list_sum(l, n) = (n <= 0) ? 0 : l[n-1] + list_sum(l, n-1); and call it as list_sum(arr, i) for the first i elements, or list_sum(arr, len(arr)) for the full sum.
- Bearing seats, shaft bores, flange bores, and mounting holes are functional features, not decoration.
- Preserve user dimensions and design intent over visual detail.
- For pillow blocks/plummer blocks: do not create only a circular tube or hollow ring on a flat plate.
- A true pillow block needs a footed base plus pedestal/saddle support, integrated bearing boss, paired mounting holes, and either cap/split-line, clamp, ribs, or side-web detail.
- Prefer a compact UCP-style cast housing silhouette; do not create tall external arch ribs or bridge frames around the bearing.
""".strip()

SYSTEM_PROMPT = "\n\n".join(
    [
        "You are  a responsive senior mechanical design copilot specialized in parametric OpenSCAD code generation for mechanical components. You have deep expertise in standard mechanical design practices and design intent. Your goal is to generate correct, editable, and standards-aligned OpenSCAD code based on user prompts describing mechanical design needs. Generate directly; when parameters are missing, choose reasonable mechanical defaults and make them editable named parameters. Always prioritize functional design features and user dimensions over visual detail.",
        "Generate one complete .scad file with no markdown, no prose, and no language labels.",
        "The .scad file must contain geometry parameters only. Do not create string metadata variables such as usage, standard_used, material, or notes.",
        "Avoid explanatory section comments about user parameters, catalog defaults, standards, or materials inside the code.",
        "Before generation, do not stop for questions; infer missing main parameters from the active mechanical family when possible and use conservative editable defaults.",
        "When main parameters are present, assume only missing secondary dimensions from relevant mechanical design practice and clearly encode them as named OpenSCAD parameters. Never override or replace secondary parameters explicitly stated by the user.",
        "Keep material suggestions and standard explanations out of the OpenSCAD file; the backend chat response will report them after code generation.",
        
        OPENSCAD_CHEATSHEET
    ]
)

ENGINEERING_FAMILY_PROFILES: dict[str, dict] = {
   
    "bearing_housing_reference": {
        "label": "Bearing Housing",
        "criticality": "high",
        "review_required": True,
        "manufacturability": [
            "Keep bearing seat and shaft bore subtractive.",
            "Preserve wall thickness around the seat and mounting holes.",
            "Use repeated mounting holes instead of a single center hole.",
        ],
        "sources": [{"name": "SKF catalogues", "type": "manufacturer"}, {"name": "MISUMI bearing dimensions", "type": "manufacturer"}],
    },
    "flange_pipe_fitting_reference": {
        "label": "Flange / Pipe Fitting",
        "criticality": "medium",
        "review_required": True,
        "manufacturability": [
            "Drive repeated bolt holes from a bolt-circle parameter.",
            "Maintain enough edge distance between bolt holes and outer diameter.",
            "Keep the center bore independent from the bolt pattern.",
        ],
        "sources": [{"name": "ASME flange practices", "type": "industry"}],
    },
    "gear_reference": {
        "label": "Gear",
        "criticality": "high",
        "review_required": True,
        "manufacturability": [
            "Keep tooth-count logic separate from the bore.",
            "Treat hub, bore, and keyway as functional subfeatures.",
            "Use review before fabrication for loaded drivetrains.",
        ],
        "sources": [{"name": "Machinery design references", "type": "textbook"}],
    },
    "shaft_coupler_reference": {
        "label": "Shaft Coupler",
        "criticality": "high",
        "review_required": True,
        "manufacturability": [
            "Model two bores explicitly and keep them coaxial.",
            "Use a real clamp slit or set-screw logic for retention.",
            "Leave enough wall thickness around bore and screw features.",
        ],
        "sources": [{"name": "Ruland coupler patterns", "type": "manufacturer"}, {"name": "MISUMI shaft coupling references", "type": "manufacturer"}],
    },
    "structural_profiles_reference": {
        "label": "Structural Profile",
        "criticality": "medium",
        "review_required": True,
        "manufacturability": [
            "Represent I-beams with flanges and a web, not a solid bar.",
            "Use profile thickness parameters rather than hardcoded dimensions.",
            "Keep extrusion cross-sections symmetric around the main axis when appropriate.",
        ],
        "sources": [{"name": "Steel section handbooks", "type": "industry"}],
    },
    "pulley_belt_drive_reference": {
        "label": "Pulley / Belt Drive",
        "criticality": "medium",
        "review_required": True,
        "manufacturability": [
           "INCLUDE POLICY: Use include <pulley-generator.scad> only for toothed/timing pulleys that call pulley3DP(), pulley(), pulleyCAD(), pulleyTeeth(), pulleyRetainer(), pulleyIdler(), pulleyBase(), or captiveGrubAndNut().",
    "NO-INCLUDE POLICY: Smooth bearing idlers using 623ZZ/624ZZ/625ZZ/608ZZ are standalone bearing sleeves and must NOT use include <pulley-generator.scad>.",
    "NO-INCLUDE POLICY: Smooth bearing idlers must NOT use pulley3DP(), pulley(), pulleyCAD(), pulleyTeeth(), pulleyRetainer(), pulleyIdler(), pulleyBase(), grub screws, captive nuts, or motor hubs.",
    "Smooth bearing idlers must be generated using standalone OpenSCAD primitives only: cylinder(), union(), difference(), and translate().",
    "If the prompt contains both GT2 and smooth/idler/bearing/624ZZ/623ZZ/625ZZ/608ZZ, classify it as smooth_bearing_idler, not toothed_timing_pulley.",
    "Use pulley-generator.scad for timing pulley tooth profiles instead of hand-made triangular teeth.",
    "Use exact case-sensitive belt model strings such as HTD 5mm and GT2 2mm.",
    "Keep the shaft bore coaxial on Z; grub/set screws and nut pockets are radial side features, not coaxial bores.",
    "Use toothWidthTweak around 0.2 mm for FDM 3D printing clearance.",
            "Use exact case-sensitive belt model strings such as HTD 5mm and GT2 2mm.",
            "Keep the shaft bore coaxial on Z; grub/set screws and nut pockets are radial side features, not coaxial bores.",
            "Use toothWidthTweak around 0.2 mm for FDM 3D printing clearance.",
        ],
        "include_policy": {
    "toothed_timing_pulley": {
        "include_required": True,
        "required_include": "include <pulley-generator.scad>",
        "allowed_modules": [
            "pulley3DP",
            "pulley",
            "pulleyCAD",
            "pulleyTeeth",
            "pulleyRetainer",
            "pulleyIdler",
            "pulleyBase",
            "captiveGrubAndNut"
        ]
    },
    "smooth_bearing_idler": {
        "include_required": False,
        "forbidden_include": "include <pulley-generator.scad>",
        "forbidden_modules": [
            "pulley3DP",
            "pulley",
            "pulleyCAD",
            "pulleyTeeth",
            "pulleyRetainer",
            "pulleyIdler",
            "pulleyBase",
            "captiveGrubAndNut"
        ],
        "required_method": "standalone OpenSCAD primitives only"
    }
},
        "sources": [{"name": "pulley-generator.scad reference", "type": "library"}],
    },
    
}

DEFAULT_ENGINEERING_PROFILE = {
    "label": "Mechanical Part",
    "criticality": "medium",
    "review_required": False,
    "manufacturability": [
        "Use parametric dimensions and realistic wall thickness.",
        "Model holes and bores with subtractive geometry.",
        "Prefer symmetric placement for repeated features.",
    ],
    "sources": [{"name": "Curated internal RAG documents", "type": "project"}],
}

app = FastAPI(title="Mechanical OpenSCAD Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled backend error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    prompt: str
    provider: Literal["ollama", "openai", "openrouter", "huggingface"] = "ollama"
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=1.2)
    top_k: int = Field(default=4, ge=1, le=8)
    selected_doc_ids: list[str] = Field(default_factory=list)
    disable_rag: bool = False
    allow_fallback: bool = True
    history: list[ChatMessage] = Field(default_factory=list)
    manufacturing: str = "fdm"   # fdm | resin | cnc | laser


class AcceptRequest(BaseModel):
    prompt: str
    code: str
    provider: str | None = None
    model: str | None = None
    selected_doc_ids: list[str] = Field(default_factory=list)


def detect_family(prompt: str, code: str = "") -> str | None:
    return detect_primary_family(f"{prompt}\n{code}")


def _extract_scad_modules(code: str) -> list[str]:
    return sorted(set(re.findall(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)))


def _extract_scad_parameters(code: str, limit: int = 24) -> list[str]:
    params: list[str] = []
    for match in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);", code, re.MULTILINE):
        name = match.group(1)
        if name.startswith("$"):
            continue
        value = " ".join(match.group(2).strip().split())
        params.append(f"{name}={value}")
        if len(params) >= limit:
            break
    return params


def _extract_design_features(prompt: str, code: str) -> list[str]:
    text = f"{prompt}\n{code}".lower()
    feature_patterns = [
        ("two perpendicular plates", r"\b(l[-_ ]?bracket|angle bracket|perpendicular plates?)\b"),
        ("horizontal base plate", r"\bhorizontal (leg|plate)|base plate\b"),
        ("vertical upright plate", r"\bvertical (leg|plate)|upright\b"),
        ("triangular gusset", r"\bgusset|triangular web|reinforc"),
        ("side gussets", r"\bside gussets?|two gussets?\b"),
        ("mounting holes", r"\bmounting holes?|hole_d|hole_diameter|clearance holes?\b"),
        ("slotted holes", r"\bslot|slotted|hull\s*\("),
        ("counterbore", r"\bcounterbore"),
        ("countersink", r"\bcountersink"),
        ("motor shaft clearance", r"\bshaft clearance|shaft_clearance|boss"),
        ("NEMA bolt pattern", r"\bnema|bolt circle|bolt_circle"),
        ("split clamp", r"\bsplit clamp|clamp screw|saddle"),
        ("coaxial bore", r"\bcoaxial|central bore|shaft bore"),
        ("D-flat shaft", r"\bd[-_ ]?flat|flat shaft"),
        ("set screw", r"\bset screw|grub screw"),
        ("hex pocket", r"\bhex|nut trap|captive nut"),
    ]
    return [label for label, pattern in feature_patterns if re.search(pattern, text)]


def build_accepted_design_summary(prompt: str, code: str) -> dict:
    family_id = detect_family(prompt, code)
    modules = _extract_scad_modules(code)
    parameters = _extract_scad_parameters(code)
    features = _extract_design_features(prompt, code)
    primary_shape = (
        modules[0].replace("_", " ") if modules else
        FAMILY_ID_TO_LABEL.get(family_id or GENERAL_RAG_ID, "mechanical part")
    )
    title_parts = [primary_shape]
    if features:
        title_parts.append(", ".join(features[:3]))
    return {
        "family_id": family_id,
        "family_label": FAMILY_ID_TO_LABEL.get(family_id or GENERAL_RAG_ID, GENERAL_RAG_LABEL),
        "primary_shape": primary_shape,
        "modules": modules,
        "parameters": parameters,
        "features": features,
        "title": " - ".join(title_parts),
    }


def build_engineering_profile(prompt: str, family_id: str | None) -> dict:
    profile = dict(DEFAULT_ENGINEERING_PROFILE)
    
    if family_id and family_id in ENGINEERING_FAMILY_PROFILES:
        profile.update(ENGINEERING_FAMILY_PROFILES[family_id])

    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ("load", "torque", "safety", "pressure", "critical", "production")):
        profile["review_required"] = True
        if profile.get("criticality") == "medium":
            profile["criticality"] = "high"

    profile["family_id"] = family_id
    profile["active_database_family"] = FAMILY_ID_TO_LABEL.get(family_id or GENERAL_RAG_ID, GENERAL_RAG_LABEL)
    profile["material_suggestions"] = suggest_materials(prompt, family_id)
    profile["usage_required"] = True
  
    profile["market_fit"] = [
        "Parametric CAD acceleration",
        "Mechanical prototyping and design iteration",
        "Editable OpenSCAD baselines for engineering teams",
    ]
    profile.pop("materials", None)
    return profile


def conversation_user_memory(history: list[ChatMessage], current_prompt: str) -> str:
    user_turns = [
        message.content.strip()
        for message in history[-8:]
        if message.role == "user" and message.content.strip()
    ]
    user_turns.append(current_prompt.strip())
    return "\n".join(user_turns)


def _is_generation_intent(prompt: str) -> bool:
    return bool(re.search(
        r"\b(generate|create|design|make|model|build|draw|give me code|openscad|scad|cad)\b",
        prompt,
        re.IGNORECASE,
    ))


def detect_pulley_subtype(prompt: str) -> str | None:
    p = prompt.lower()

    smooth_keywords = [
        "smooth idler",
        "bearing idler",
        "624zz",
        "623zz",
        "625zz",
        "608zz",
        "idler sleeve",
        "free spinning pulley",
        "belt tensioner",
        "passive pulley",
        "bearing sleeve"
    ]

    toothed_keywords = [
        "tooth",
        "teeth",
        "timing pulley",
        "drive pulley",
        "motor pulley",
        "grub screw",
        "set screw",
        "captive nut"
    ]

    if any(k in p for k in smooth_keywords):
        return "smooth_bearing_idler"

    if any(k in p for k in toothed_keywords):
        return "timing_pulley"

    return None
def _material_scope_for_family(family_id: str | None) -> str:
    if family_id == "bearing_housing_reference":
        return "housing body, pedestal, and mounting feet; not the bearing internals"
    return "generated part body; not purchased bearings, fasteners, or inserts"


def suggest_materials(prompt: str, family_id: str | None = None) -> list[dict]:
    lowered = prompt.lower()
    scope = _material_scope_for_family(family_id)

    def item(material: str, reason: str, standard: str, duty: str = "general") -> dict:
        return {
            "material": material,
            "applies_to": scope,
            "reason": reason,
            "standard_basis": standard,
            "duty": duty,
        }

    printed = bool(re.search(r"\b(print|printed|3d|prototype|pla|petg|nylon|pa-cf)\b", lowered))
    heavy = bool(re.search(r"\b(industrial|heavy|conveyor|production|shock|load-bearing|walkway|bridge)\b", lowered))
    light = bool(re.search(r"\b(cnc|robot|lightweight|aluminum|aluminium|fixture|display|rig)\b", lowered))
    timber = bool(re.search(r"\b(timber|wood|roof|residential)\b", lowered))

    family_materials: dict[str, list[dict]] = {
        "structural_profiles_reference": [
            item("S275JR / S355JR structural steel", "standard welded/bolted structural member material for frames and trusses", "EN 10025; Eurocode 3 design basis", "structural"),
            item("ASTM A36 or ASTM A500 Grade B/C steel", "common North American structural plate, angle, channel, and tube baseline", "ASTM A36 / ASTM A500; AISC steel design basis", "structural"),
            item("C24 timber or GL24h glulam", "appropriate baseline for timber roof truss concepts", "EN 338 / EN 14080; Eurocode 5 design basis", "timber roof"),
        ],
        "common_trusses_reference": [
            item("C24 timber or GL24h glulam", "standard timber roof-truss baseline for residential Fink, king-post, and queen-post concepts", "EN 338 / EN 14080; Eurocode 5 design basis", "timber roof"),
            item("S275JR / S355JR structural steel", "standard welded or bolted truss member material for bridge/walkway and industrial frames", "EN 10025; Eurocode 3 design basis", "structural"),
            item("ASTM A36 or ASTM A500 Grade B/C steel", "common North American steel baseline for truss plates, angles, and tube members", "ASTM A36 / ASTM A500; AISC steel design basis", "structural"),
            item("6061-T6 / 6082-T6 aluminum tube", "lightweight truss material for display rigs, staging prototypes, and low-load architectural models", "ASTM B221 / EN AW-6082 T6", "lightweight"),
        ],
        "shaft_reference": [
            item("AISI 1045 / C45 medium-carbon steel", "standard machinable shaft material with better strength than mild steel", "AISI 1045 / EN C45; ISO shaft fit practice", "rotating shaft"),
            item("AISI 4140 / 42CrMo4 alloy steel", "higher strength shaft material for torque, fatigue, and threaded ends", "AISI 4140 / EN 42CrMo4", "high load"),
            item("AISI 304 stainless steel", "corrosion-resistant shaft option where strength demands are moderate", "ASTM A276 Type 304", "corrosion resistant"),
        ],
        "shaft_coupler_reference": [
            item("6061-T6 aluminum", "common clamp coupler body material for light torque and low inertia", "ASTM B221 / EN AW-6061 T6", "light duty"),
            item("AISI 1215 or 1045 steel", "stronger coupler body for higher clamp force and torque transfer", "AISI 1215 / AISI 1045", "medium duty"),
            item("AISI 303/304 stainless steel", "corrosion-resistant coupler body for exposed machinery", "ASTM A582 / ASTM A276", "corrosion resistant"),
        ],
        "gear_reference": [
            item("POM/acetal", "low-friction plastic gear material for quiet light-duty prototypes", "ISO 1874 material designation practice", "light duty"),
            item("C45 / AISI 1045 steel", "machinable steel gear blank for moderate loaded gears", "EN C45 / AISI 1045; ISO 6336 design checks", "medium duty"),
            item("16MnCr5 / 8620 case-hardening steel", "case-hardening gear material for wear-resistant power transmission", "EN 10084 / AISI 8620; ISO 6336 design checks", "high duty"),
        ],
        "sprocket_chain_reference": [
            item("AISI 1045 steel", "common machined sprocket material for ANSI roller chain drives", "ANSI B29.1 chain geometry; AISI 1045 material practice", "chain drive"),
            item("4140 steel, hardened teeth", "better wear resistance for high-load or high-cycle sprockets", "ANSI B29.1; AISI 4140 heat-treated", "heavy duty"),
            item("7075-T6 aluminum", "lightweight sprocket option for low-load robotics or prototypes", "ASTM B221 / EN AW-7075 T6", "lightweight"),
        ],
        "pulley_belt_drive_reference": [
            item("6061-T6 aluminum", "machinable timing pulley or idler body with good dimensional stability", "ASTM B221 / EN AW-6061 T6; ISO metric fit practice", "general"),
            item("POM/acetal", "low-friction pulley/idler material for quiet light belt drives", "ISO 1874 material designation practice", "light duty"),
            item("PA-CF nylon", "functional printed pulley/idler material with better heat and creep resistance than PLA", "ISO 16396 material designation practice", "printed prototype"),
        ],
        "bracket_and_motor_mount_reference": [
            item("6061-T6 aluminum plate", "good default for machined motor brackets and lightweight mounts", "ASTM B209/B221 / EN AW-6061 T6", "machined"),
            item("S275 / ASTM A36 steel plate", "robust welded or bent bracket material for higher loads", "EN 10025 / ASTM A36", "heavy duty"),
            item("PA-CF nylon", "functional printed bracket material for prototypes with improved stiffness", "ISO 16396 material designation practice", "printed prototype"),
        ],
        "robotics_servo_reference": [
            item("6061-T6 aluminum", "stiff servo bracket material with good tapped-hole strength", "ASTM B209/B221 / EN AW-6061 T6", "robotics"),
            item("PA-CF nylon", "good printed servo bracket option when layer direction and screw bosses are designed carefully", "ISO 16396 material designation practice", "printed prototype"),
            item("PETG", "acceptable fit-check material for light servo loads", "ISO 19063 material designation practice", "fit check"),
        ],
        "flange_pipe_fitting_reference": [
            item("ASTM A105 forged carbon steel", "standard forged flange material for many pressure-piping services", "ASME B16.5 geometry; ASTM A105 material", "pressure flange"),
            item("ASTM A182 F304/F316 stainless steel", "corrosion-resistant flange material", "ASME B16.5 geometry; ASTM A182", "corrosion resistant"),
            item("EN 1.0038/S235JR or S275JR", "general fabricated pipe adapter or non-pressure flange material", "EN 10025", "fabricated"),
        ],
        "enclosure_box_reference": [
            item("ABS", "common injection-molded electronics enclosure material with impact resistance", "ISO 2580 material designation practice", "enclosure"),
            item("PC/ABS", "higher impact and heat resistance than ABS for electronics housings", "ISO polymer designation practice", "rugged enclosure"),
            item("PETG or PA-CF", "practical 3D printed enclosure choices depending on heat/stiffness needs", "ISO 19063 / ISO 16396", "printed prototype"),
        ],
        "cooling_fan_mount_reference": [
            item("ABS or PC/ABS", "heat-tolerant plastic for fan shrouds and duct mounts", "ISO 2580 / ISO polymer designation practice", "duct"),
            item("PETG", "printable fan duct material with better heat resistance than PLA", "ISO 19063 material designation practice", "printed prototype"),
            item("6061-T6 aluminum", "stiff machined fan mount plate material", "ASTM B209/B221", "machined plate"),
        ],
        "fastener_nut_trap_reference": [
            item("PA-CF nylon", "best functional printed insert-boss/nut-trap body material for heat and creep resistance", "ISO 16396 material designation practice", "printed functional"),
            item("PETG", "good light-duty printed boss material for fit and assembly tests", "ISO 19063 material designation practice", "prototype"),
            item("Brass heat-set inserts", "standard insert material for thermoplastic screw bosses", "DIN/ISO metric screw compatibility", "insert"),
        ],
    }

    suggestions = family_materials.get(family_id or "")
    if suggestions:
        if printed:
            printed_first = [s for s in suggestions if "print" in s.get("duty", "") or "PA-CF" in s.get("material", "") or "PETG" in s.get("material", "")]
            suggestions = printed_first + [s for s in suggestions if s not in printed_first]
        elif family_id == "common_trusses_reference" and light:
            lightweight_first = [s for s in suggestions if "lightweight" in s.get("duty", "") or "aluminum" in s.get("material", "").lower()]
            suggestions = lightweight_first + [s for s in suggestions if s not in lightweight_first]
        elif heavy:
            heavy_first = [s for s in suggestions if any(word in s.get("duty", "") for word in ("heavy", "structural", "high", "pressure")) or "steel" in s.get("material", "").lower()]
            suggestions = heavy_first + [s for s in suggestions if s not in heavy_first]
        elif light:
            light_first = [s for s in suggestions if any(word in s.get("material", "").lower() for word in ("6061", "7075", "aluminum", "pom"))]
            suggestions = light_first + [s for s in suggestions if s not in light_first]
        elif timber and family_id == "structural_profiles_reference":
            timber_first = [s for s in suggestions if "timber" in s.get("duty", "")]
            suggestions = timber_first + [s for s in suggestions if s not in timber_first]
        return suggestions[:3]

    if any(word in lowered for word in ("industrial", "heavy", "conveyor", "production", "shock", "steel frame")):
        return [
            item("ASTM A36 / S275 structural steel", "robust weldable material for load-bearing mechanical parts", "ASTM A36 / EN 10025", "heavy duty"),
            item("AISI 1045 steel", "stronger machined material for shafts, hubs, and loaded mechanical parts", "AISI 1045 / EN C45", "heavy duty"),
        ]
    if any(word in lowered for word in ("cnc", "robot", "lightweight", "aluminum", "fixture")):
        return [
            item("6061-T6 aluminum", "light, machinable body material for CNC, robotics, or fixture duty", "ASTM B221 / EN AW-6061 T6", "light duty"),
            item("7075-T6 aluminum", "higher strength aluminum where weight matters and cost is acceptable", "ASTM B221 / EN AW-7075 T6", "high-strength lightweight"),
        ]
    if any(word in lowered for word in ("print", "printed", "3d", "prototype", "pla", "petg", "nylon")):
        return [
            item("PA-CF / nylon carbon fiber", "best functional printed body choice for heat and creep resistance", "ISO 16396 material designation practice", "printed functional"),
            item("PETG", "acceptable body material for light prototypes and fit checks", "ISO 19063 material designation practice", "prototype"),
        ]
    return [
        item("6061-T6 aluminum", "default body material for light machined mechanical parts", "ASTM B221 / EN AW-6061 T6", "general"),
        item("Low-carbon steel", "default robust body material for load-bearing mechanical parts", "ASTM A36 / EN S275", "general"),
        item("PA-CF", "default body material for functional printed prototypes", "ISO 16396 material designation practice", "prototype"),
    ]


def build_design_notes(prompt: str, engineering_profile: dict, family_schema: dict) -> dict:
    """Chat-only notes shown after code generation."""
    lowered = prompt.lower()
    standards = family_schema.get("standards_used") or []
    selected: list[dict] = []

    def add_if(predicate) -> None:
        for item in standards:
            if not isinstance(item, dict):
                continue
            searchable = " ".join(str(item.get(key, "")) for key in ("name", "applies_to", "note", "short_explanation")).lower()
            if predicate(searchable) and item not in selected:
                selected.append(item)

    if any(word in lowered for word in ("flange", "ucf", "wall mount", "four bolt", "4 bolt")):
        add_if(lambda text: "ucf" in text or "square-flange" in text or "four" in text)
    elif any(word in lowered for word in ("take-up", "take up", "slotted", "belt", "chain", "tension")):
        add_if(lambda text: "uct" in text or "take-up" in text or "slot" in text)
    else:
        add_if(lambda text: "ucp" in text or "plummer" in text or "pillow" in text or "6204" in text)

    add_if(lambda text: "clearance" in text or "iso metric" in text)
    if not selected:
        selected = [item for item in standards if isinstance(item, dict)][:3]

    brief_standards = []
    for item in selected[:3]:
        note = item.get("note") or item.get("short_explanation") or item.get("applies_to", "")
        brief_standards.append(
            {
                "name": item.get("name", "Mounted bearing housing practice"),
                "summary": note,
            }
        )

    return {
        "standards_summary": brief_standards,
        "material_suggestions": engineering_profile.get("material_suggestions", []),
        "material_scope": _material_scope_for_family(engineering_profile.get("family_id")),
        "material_output_policy": "chat_after_code_only",
    }


class AcceptedHistory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._items: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        with self._lock:
            self._items = []
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("prompt") and item.get("code"):
                    item.setdefault("summary", build_accepted_design_summary(item.get("prompt", ""), item.get("code", "")))
                    self._items.append(item)
            self._embeddings = None

    @property
    def count(self) -> int:
        return len(self._items)

    def add(self, request: AcceptRequest) -> dict:
        item = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt": request.prompt.strip(),
            "code": request.code.strip(),
            "provider": request.provider,
            "model": request.model,
            "selected_doc_ids": request.selected_doc_ids,
        }
        if not item["prompt"] or not item["code"]:
            raise ValueError("Accepted history needs both prompt and code.")
        item["summary"] = build_accepted_design_summary(item["prompt"], item["code"])

        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._items.append(item)
            self._embeddings = None
        return item

    def _history_text(self, item: dict) -> str:
        summary = item.get("summary") or build_accepted_design_summary(item.get("prompt", ""), item.get("code", ""))
        lines = [
            f"Family: {summary.get('family_label', GENERAL_RAG_LABEL)} ({summary.get('family_id') or GENERAL_RAG_ID})",
            f"Primary shape: {summary.get('primary_shape', '')}",
            "Modules: " + ", ".join(summary.get("modules") or []),
            "Features: " + ", ".join(summary.get("features") or []),
            "Parameters: " + ", ".join(summary.get("parameters") or []),
            "Original request: " + item.get("prompt", ""),
            "Code structure:",
            item.get("code", "")[:1800],
        ]
        return "\n".join(line for line in lines if line.strip())

    def _ensure_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            texts = [self._history_text(item) for item in self._items]
            self._embeddings = embed_texts(texts, is_query=False) if texts else np.array([])
        return self._embeddings

    def retrieve(self, query: str, top_k: int = 2, family_id: str | None = None) -> list[dict]:
        with self._lock:
            if not self._items:
                return []
            try:
                candidate_items = [
                    item for item in self._items
                    if not family_id or (item.get("summary") or {}).get("family_id") == family_id
                ]
                if not candidate_items:
                    return []
                if EMBEDDING_BACKEND == "tfidf":
                    texts = [self._history_text(item) for item in candidate_items]
                    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                    matrix = vectorizer.fit_transform(texts + [query])
                    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
                else:
                    if len(candidate_items) == len(self._items):
                        embeddings = self._ensure_embeddings()
                    else:
                        embeddings = embed_texts([self._history_text(item) for item in candidate_items], is_query=False)
                    if embeddings.size == 0:
                        return []
                    query_vector = embed_texts([query], is_query=True)
                    scores = cosine_similarity(query_vector, embeddings)[0]
                order = np.argsort(scores)[::-1]
                minimum_score = 0.05 if EMBEDDING_BACKEND == "tfidf" else 0.35
                hits = []
                for index in order:
                    if len(hits) >= top_k or scores[index] < minimum_score:
                        break
                    item = candidate_items[index]
                    code = item.get("code", "")
                    summary = item.get("summary") or build_accepted_design_summary(item.get("prompt", ""), code)
                    title = summary.get("title") or item.get("prompt", "")[:70]
                    hits.append(
                        {
                            "id": f"accepted::{item['id']}",
                            "title": f"Accepted Design: {title[:90]}",
                            "source": "accepted-history",
                            "score": round(float(scores[index]), 4),
                            "family": summary.get("family_id"),
                            "excerpt": (
                                f"Shape: {summary.get('primary_shape', '')}\n"
                                f"Features: {', '.join(summary.get('features') or [])}\n"
                                f"Parameters: {', '.join((summary.get('parameters') or [])[:10])}\n"
                                f"Original request: {item.get('prompt', '')}"
                            ),
                            "text": (
                                "Accepted user-approved design. Prefer this only when the family, shape, features, and parameters are similar.\n"
                                f"Family: {summary.get('family_label', GENERAL_RAG_LABEL)} ({summary.get('family_id') or GENERAL_RAG_ID})\n"
                                f"Primary shape: {summary.get('primary_shape', '')}\n"
                                f"Modules: {', '.join(summary.get('modules') or [])}\n"
                                f"Features: {', '.join(summary.get('features') or [])}\n"
                                f"Parameters: {', '.join(summary.get('parameters') or [])}\n\n"
                                f"Original prompt:\n{item.get('prompt', '')}\n\n"
                                f"Accepted OpenSCAD code:\n{code[:3000]}"
                            ),
                        }
                    )
                return hits
            except Exception as exc:  # pragma: no cover - retrieval fallback
                log.warning("Accepted-history retrieval failed (%s).", exc)
                return []

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [
                {key: item.get(key) for key in ("id", "created_at", "prompt", "provider", "model", "summary")}
                for item in self._items[-limit:][::-1]
            ]


class ModelGateway:
    def __init__(self) -> None:
        self._openai_client: OpenAI | None = None
        self._openrouter_client: OpenAI | None = None
        self._huggingface_client: OpenAI | None = None
        

    def ollama_available(self) -> bool:
        try:
            requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).raise_for_status()
            return True
        except requests.RequestException:
            return False

    def openai_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def openrouter_available(self) -> bool:
        return bool(os.getenv("OPENROUTER_API_KEY"))

    def huggingface_available(self) -> bool:
        return bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY"))

   
    def providers(self) -> list[dict]:
        providers: list[dict] = []
        if OLLAMA_CHAT_MODELS:
            providers.append(
                {
                    "id": "ollama",
                    "label": "Ollama Local",
                    "models": OLLAMA_CHAT_MODELS,
                    "available": self.ollama_available(),
                    "default_model": self.default_model("ollama"),
                }
            )
        providers.append(
            {
                "id": "openai",
                "label": "OpenAI API",
                "models": OPENAI_MODELS or [OPENAI_DEFAULT_MODEL],
                "available": self.openai_available(),
                "default_model": OPENAI_DEFAULT_MODEL,
            }
        )
        providers.append(
            {
                "id": "openrouter",
                "label": "OpenRouter",
                "models": (OPENROUTER_MODELS + OPENROUTER_PAID_MODELS) if OPENROUTER_ALLOW_PAID else (OPENROUTER_MODELS or [OPENROUTER_DEFAULT_MODEL]),
                "available": self.openrouter_available(),
                "default_model": OPENROUTER_DEFAULT_MODEL,
            }
        )
        providers.append(
            {
                "id": "huggingface",
                "label": "Hugging Face Router",
                "models": HUGGINGFACE_MODELS or [HUGGINGFACE_DEFAULT_MODEL],
                "available": self.huggingface_available(),
                "default_model": HUGGINGFACE_DEFAULT_MODEL,
            }
        )
      
        return providers

    def default_model(self, provider: str) -> str | None:
        if provider == "ollama":
            for model in PREFERRED_OLLAMA_MODELS:
                if model in OLLAMA_CHAT_MODELS:
                    return model
            return OLLAMA_CHAT_MODELS[0] if OLLAMA_CHAT_MODELS else None
        if provider == "openai":
            return OPENAI_DEFAULT_MODEL
        if provider == "openrouter":
            return OPENROUTER_DEFAULT_MODEL
        if provider == "huggingface":
            return HUGGINGFACE_DEFAULT_MODEL
      
           
        return None

    def _get_openai(self) -> OpenAI:
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not configured.")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _get_openrouter(self) -> OpenAI:
        if self._openrouter_client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not configured.")
            self._openrouter_client = OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                timeout=OPENROUTER_TIMEOUT_SEC,
                max_retries=0,
                default_headers={
                    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000"),
                    "X-Title": OPENROUTER_APP_TITLE,
                },
            )
        return self._openrouter_client

    def _get_huggingface(self) -> OpenAI:
        if self._huggingface_client is None:
            api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                raise RuntimeError("HF_TOKEN or HUGGINGFACE_API_KEY is not configured.")
            self._huggingface_client = OpenAI(api_key=api_key, base_url=HUGGINGFACE_BASE_URL)
        return self._huggingface_client

   

    def generate(self, provider: str, model: str, messages: list[dict], temperature: float) -> str:
        if provider == "ollama":
            return self._generate_ollama(model, messages, temperature)
        if provider == "openai":
            return self._generate_openai(model, messages, temperature)
        if provider == "openrouter":
            return self._generate_openrouter(model, messages, temperature)
        if provider == "huggingface":
            return self._generate_huggingface(model, messages, temperature)
       
        raise RuntimeError(f"Unsupported provider '{provider}'.")

    def _generate_openai(self, model: str, messages: list[dict], temperature: float) -> str:
        client = self._get_openai()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=4200,
        )
        text = response.choices[0].message.content if response.choices else ""
        if not text:
            raise RuntimeError("OpenAI returned an empty response.")
        return text

    def _generate_openrouter(self, model: str, messages: list[dict], temperature: float) -> str:
        client = self._get_openrouter()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4200,
        )
        text = response.choices[0].message.content if response.choices else ""
        if not text:
            raise RuntimeError("OpenRouter returned an empty response.")
        return text

    def _generate_huggingface(self, model: str, messages: list[dict], temperature: float) -> str:
        client = self._get_huggingface()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4200,
        )
        text = response.choices[0].message.content if response.choices else ""
        if not text:
            raise RuntimeError("Hugging Face returned an empty response.")
        return text

    

    def _generate_ollama(self, model: str, messages: list[dict], temperature: float) -> str:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_predict": OLLAMA_MAX_TOKENS,
                },
            },
            timeout=OLLAMA_GENERATE_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        return text


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32].rstrip() + "\n\n[truncated for context budget]"


def build_messages(
    user_prompt: str,
    history: list[ChatMessage],
    context_hits: list[dict],
    provider: str = "openai",
    family_id: str | None = None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    family_schema = knowledge_base.family_schema(family_id)
    family_lines = [
        f"DETECTED MECHANICAL FAMILY: {family_schema['label']} ({family_schema['id']}).",
        "Use the retrieved family references and part records that best match the user's requested mechanical part.",
        "Ask for missing main parameters only when they are essential; assume secondary dimensions from mechanical practice only when the user did not state them. Preserve every stated secondary parameter exactly unless it is physically impossible.",
    ]
    main_features = family_schema.get("main_features", [])[:12]
    if main_features:
        family_lines.append("Main features: " + ", ".join(main_features))
        family_lines.append("Use structured feature_library snippets, derived parameters, constraint checks, and validation criteria from the family database.")
    if family_id == "bearing_housing_reference":
        family_lines.append(
            "For pillow blocks, the RAG rulebook named bearing_housing_true_pillow_block_reference is authoritative. "
            "Reject the ring-on-plate failure mode and tall arch/bridge-frame shapes. "
            "Use a compact UCP-style body with low feet, solid pedestal/saddle, rounded integrated boss, subtle cap split, and paired foot slots or holes."
        )
    if family_id == "shaft_reference":
        family_lines.append(
            "For transmission shaft, countershaft, gearbox shaft, or gear shaft requests with gears, use the shaft_transmission record and the project gears.scad library. "
            "Call spur_gear(..., helix_angle=...) for gear sections; never hand-build gear teeth from cubes or rectangular blocks, and never reference undeclared gear_teeth variables. "
            "Use the fixed numeric positions from the RAG example; do not create shaft_pos with concat() or reassign arrays inside for-loops because OpenSCAD variables are immutable."
        )
    if family_id == "gear_reference":
        family_lines.append(
            "For gear requests, the gears-master reference is authoritative. "
            "Prefer include <D:/Downloads/openscad_copilot (1)/openscad_copilot/gears.scad> and call the matching gears.scad module. "
            "Do not ask for missing gear parameters in the generated OpenSCAD. "
            "For generic requests like 'generate bevel gear' or 'generate worm gear', use the default example values from the gears-master reference and output a complete module call. "
            "Never output a placeholder parameter-request module or echo('Please specify ...'). "
            "For a plain spur gear use spur_gear(..., helix_angle=0, optimized=true). "
            "Use nonzero helix_angle only when the user asks for helical, spiral, or herringbone gears. "
            "Do not hand-build gears from rectangular/cube teeth or lumped cylinders when a gears.scad module exists. "
            "\n"
            "BEVEL / SPIRAL BEVEL GEAR PAIR — mandatory rules: "
            "1. gear_teeth is ALWAYS the larger wheel (more teeth); pinion_teeth is ALWAYS the smaller pinion (fewer teeth). "
            "   If the user states a ratio like 16:32, set pinion_teeth=16 and gear_teeth=32. "
            "2. For a spiral bevel pair use bevel_gear_pair(..., helix_angle=30) — NOT a wrapper module with difference(). "
            "   The library internally applies -helix_angle to the pinion for correct hand matching. "
            "3. gear_bore and pinion_bore are subtracted INSIDE the library. "
            "   Pass the real bore diameter values directly. "
            "   NEVER pass gear_bore=0 / pinion_bore=0 and then subtract cylinders manually. "
            "4. OpenSCAD modules do not return values. "
            "   NEVER write: body = bevel_gear_pair(...) — that is invalid OpenSCAD and silently does nothing. "
            "   Call the module directly at the top level or inside another module. "
            "\n"
            "PLANETARY GEAR — mandatory rules: "
            "Call planetary_gear(modul, sun_teeth, planet_teeth, number_planets, width, rim_width, bore, "
            "together_built=true, optimized=true) ONCE. "
            "This single call renders ALL THREE components: sun gear (centre), planet gears (orbiting), "
            "and ring/annulus gear (outer). "
            "NEVER simulate a planetary gear with a for-loop of spur_gear() or herringbone_gear() "
            "calls positioned around a circle — that outputs only loose planet shapes with no sun "
            "and no ring gear, which is always wrong. "
            "ring_teeth is computed inside the module as sun_teeth + 2*planet_teeth — do not pass it. "
            "The module always uses herringbone teeth internally."
        )
    if family_id == "sprocket_chain_reference":
        family_lines.append(
            "For sprocket requests, the project-local sprocket.scad library is authoritative. "
            "Prefer returning a tiny OpenSCAD file that uses the library with: use <D:/downloads/openscad_copilot (1)/openscad_copilot/backend/docs/sprocket.scad>. "
            "Call sprocket(size, teeth, bore, hub_diameter, hub_height, keyway, setscrew) with inch inputs for bore and hub dimensions. "
            "Do not reimplement sprocket geometry from scratch, do not use gear module/addendum/dedendum logic, and never output a plain cylinder or smooth disk."
        )
    if family_id == "bracket_and_motor_mount_reference":
        family_lines.append(
            "For L-brackets and angle brackets, use the canonical coordinate model from the bracket RAG: "
            "base plate in XY with Z thickness; upright plate in YZ at one base edge with X thickness; "
            "base holes pass through Z; upright holes pass through X using rotate([0,90,0]); "
            "upright hole Z positions are positive within the upright height; "
            "plain L-bracket is the default, so do not add gussets unless the user explicitly asks for gusset, rib, reinforced, heavy-duty, load-bearing, or shelf support; "
            "when requested, gussets/ribs are triangular prisms using polygon()+linear_extrude() or polyhedron(), not flat rectangular blocks. "
            "When four holes are requested on a leg, use a 2x2 pattern within that leg's own width/length."
        )
    if family_id == "pulley_belt_drive_reference":
        family_lines.append(
            "For timing pulley and belt-drive requests, pulley-generator.scad is authoritative. "
            "Always begin with include <pulley-generator.scad>. "
            "Use the exact model string from the database, for example HTD 5mm or GT2 2mm, including spaces and case. "
            "CRITICAL — set teethCount, beltWidth, and shaftDiameter to the exact numeric values stated in the user prompt. "
            "Never substitute the database default (20 teeth) when the user specifies a different tooth count. "
            "MODULE SELECTION: use pulley3DP() for any simple request that does not explicitly ask for "
            "captive nuts, grub screws, retainerInfo, idlerInfo, or baseInfo customization. "
            "Use pulley() only when the user explicitly requests custom nut/screw/retainer configuration. "
            "Do not invent triangular teeth, do not approximate HTD/GT2 grooves with polygons, and do not create smooth cylinders as timing pulleys. "
            "For FDM 3D printing use toothWidthTweak around 0.2. "
            "autoFlip=true is already hardcoded inside pulley3DP — do NOT pass it as a parameter to pulley3DP. "
            "Pass autoFlip=true explicitly only when using the full pulley() module. "
            "SHAFT BORE for FDM: shaftDiameter = shaft_nominal + 0.2 mm for clearance (e.g. NEMA17 5mm shaft → shaftDiameter = 5.2). "
            "BELT WIDTH — use only standard widths per profile: "
            "MXL: 3.175, 6.35, 9.525 mm — do NOT use 6 mm for MXL (nearest standard is 6.35 mm). "
            "GT2 2mm: 6, 9, 15 mm. HTD 5mm: 9, 15, 25 mm. T5: 6, 10, 16, 25 mm."
        )
    if family_id == "common_trusses_reference":
        family_lines.append(
            "For ALL truss requests, the retrieved king_post_roof_truss / queen_post_roof_truss / "
            "pratt_howe_bridge_truss / warren_fink_k_truss_patterns records are authoritative. "
            "Follow every design_rule and validation_criterion from those records exactly. "
            "\n"
            "KING-POST TRUSS — mandatory rules (cannot be overridden by training-data patterns): "
            "1. Define SIX nodes: A=[-span/2,0,0], B=[span/2,0,0], C=[0,0,rise], D=[0,0,0], "
            "   E=[-span/4,0,rise/2], F=[span/4,0,rise/2]. "
            "   E and F are strut attachment nodes at the midpoints of the two rafters. "
            "2. Split each rafter at its strut node — left rafter: member(A,E)+member(E,C); "
            "   right rafter: member(C,F)+member(F,B). "
            "   NEVER use a single member(A,C) or member(C,B) — that is a listed failure mode. "
            "3. Add diagonal struts member(D,E,web_d) and member(D,F,web_d). "
            "4. Total member calls = 9: A-D, D-B, A-E, E-C, C-F, F-B, C-D, D-E, D-F. "
            "\n"
            "MEMBER MODULE — mandatory for ALL truss types: "
            "module member(p1,p2,d=4) must contain ONLY: rod(p1,p2,d); node(p1); node(p2); "
            "Do NOT wrap rod() or member() in minkowski() under any circumstances. "
            "Do NOT define or use a fillet_r variable in truss code. "
            "minkowski on a cylinder produces capsule geometry, not a rod, and makes render 100x slower."
        )
    if family_id == "shaft_coupler_reference":
        family_lines.append(
            "For shaft coupler requests, the helical_beam_shaft_coupling record is authoritative. "
            "\n"
            "HELICAL BEAM COUPLER — mandatory rules: "
            "1. Helical cuts use linear_extrude(height=length+2, twist=-(360*turns), slices=300, center=true) "
            "   with the 2D slot profile translate([slot_r,0]) square([wall_depth+2, slot_w], center=true). "
            "   NEVER use rotate_extrude for helical cuts — it creates a torus at fixed radius, not a helix. "
            "2. The outer body CYLINDER is the POSITIVE volume. Bore and slots are subtracted FROM it. "
            "   NEVER add the bore cylinder in union() and then subtract the same bore — they cancel out, "
            "   producing an empty difference with no body. "
            "3. For 2-start helical coupler: for(i=[0:1]) rotate([0,0,i*180]) linear_extrude(twist=...). "
            "4. Do NOT generate: helix_point() function, manufacturing='fdm' string variable, "
            "   circular_grooves() (horizontal rings), or rotate_extrude for helical geometry."
        )
    if family_id == "hinge_joint_snapfit_reference":
        family_lines.append(
            "For print-in-place hinge requests, the print_in_place_hinge_5knuckle record is authoritative. "
            "\n"
            "5-KNUCKLE HINGE — mandatory rules: "
            "1. knuckle_l = leaf_length / knuckle_count (= 8 mm for 40mm/5). "
            "   NEVER use leaf_length/(2*knuckle_count-1) — that formula gives 9 knuckle slots, not 5. "
            "2. Leaf A loop: for(i=[0:(knuckle_count+1)/2-1]) → 3 knuckles. "
            "   Leaf B loop: for(i=[0:(knuckle_count-1)/2-1]) → 2 knuckles. "
            "   NEVER use [0:knuckle_count-1] for A (=5) and [0:knuckle_count-2] for B (=4) — that gives 9 total. "
            "3. Pin axis is along X. ALL knuckle cylinders use rotate([0,90,0]). "
            "   NEVER use rotate([90,0,0]) — that points cylinders along Y, every knuckle on a different axis. "
            "4. ALL knuckle centres share the SAME position: y=0, z = leaf_thick + knuckle_OD/2. "
            "   Leaf A x-centres: (2*i)*knuckle_l + knuckle_l/2. "
            "   Leaf B x-centres: (2*i+1)*knuckle_l + knuckle_l/2. "
            "5. BOTH leaf A and leaf B knuckles use difference(). "
            "   Leaf A bore = pin_d + clearance; leaf B bore = pin_d + 2*clearance. "
            "   NEVER add leaf A cylinders without difference() — pin will fuse to barrel. "
            "6. Leaf A plate: translate([0, clearance, 0]) cube([leaf_length, leaf_width, leaf_thick]). "
            "   Leaf B plate: translate([0, -(leaf_width+clearance), 0]) cube([...]). "
            "   NEVER put leaf B at z=leaf_thick+clearance — that offsets the barrel off the pin axis."
        )
    if family_id == "linear_rail_carriage_reference":
        family_lines.append(
            "For LM8UU rod carriage and linear rail carriage requests, the dual_rod_carriage_lm8uu record is authoritative. "
            "\n"
            "BOTTOM RELIEF — mandatory formula: "
            "translate([0,0,-car_h/2 + bottom_wall + relief_depth/2]) cube([...,relief_depth], center=true). "
            "The relief bottom is at -car_h/2+bottom_wall so the floor stays solid. "
            "NEVER use -car_h/2 + wall_thickness/2 — that centers the cube on the bottom face and removes the entire bottom wall. "
            "\n"
            "MOUNTING HOLES — mandatory: center the cylinder at z=0 for full through-holes: "
            "translate([x,y,0]) cylinder(h=car_h+2, d=d, center=true). "
            "NEVER translate to carriage_height/2+0.1 with center=true — that creates half-depth holes only. "
            "\n"
            "HEX NUT TRAP — mandatory: nut pocket and bolt hole must match the SAME bolt family. "
            "M3: hex_nut_af=5.5mm → nut_d=hex_nut_af/cos(30)+0.2=6.55mm, bolt_d=3.4mm. "
            "M4: hex_nut_af=7.0mm, bolt_d=4.5mm. "
            "M6: hex_nut_af=10mm, bolt_d=6.5mm. "
            "NEVER pair bolt_d=3.4mm (M3) with hex_nut_d=10mm (M6) — bolt will rattle loose in nut."
        )
    if family_id == "lead_screw_actuator_reference":
        family_lines.append(
            "For T8 nut carriage and lead screw actuator requests, the t8_nut_carriage record is authoritative. "
            "\n"
            "T8 NUT CARRIAGE STRUCTURE — mandatory: "
            "1. LM8UU guide rod bores: HORIZONTAL along X — rotate([0,90,0]) at y=±rod_spacing/2, z=0. "
            "2. T8 screw clearance bore: VERTICAL (Z) — plain unrotated cylinder, full height, centered. "
            "3. T8 nut pocket: VERTICAL from top — "
            "   translate([0,0,carriage_height/2-t8_nut_h/2]) cylinder(h=t8_nut_h+0.2, d=nut_pocket_d). "
            "   nut_pocket_d must be LARGER than screw_clear (22.6 mm vs 9.5 mm). "
            "4. Nut capture bolt holes: VERTICAL (Z) on bolt circle — "
            "   for(a=[0,90,180,270]) rotate([0,0,a]) translate([nut_bolt_circle/2, 0, nut_z]) cylinder(d=3.4). "
            "5. Nut bolt d = nut_bolt_d = 3.4 mm (M3 FDM clearance). "
            "   NEVER compute as thread_clear + 1.0 = 1.4 mm — that is too small for any bolt. "
            "6. Optional bottom relief: "
            "   translate([0,0,-carriage_height/2+5]) cube([carriage_length-8, carriage_width-8, 8], center=true). "
            "Do NOT use minkowski() anywhere — no sphere convolution, no fillet_r expansion."
        )
    if family_id == "gearbox_housing_reference":
        family_lines.append(
            "For gearbox housing requests, the gearbox_housing_reference record is authoritative. "
            "\n"
            "GEOMETRY VARIABLE ASSIGNMENT — strictly forbidden: "
            "OpenSCAD modules do not return values. NEVER write: "
            "outer = minkowski(){...};  cavity = hull(){...};  shafts = union(){...};  "
            "body = some_module(...); — these are all invalid and silently produce nothing. "
            "NEVER then reference those names inside difference() { outer; cavity; shafts; } — "
            "those identifiers hold no geometry. "
            "Write every CSG operation (union, difference, hull, minkowski, intersection) "
            "DIRECTLY nested inline inside the module body. "
            "\n"
            "GEARBOX HOUSING STRUCTURE — mandatory: "
            "module lower_half() { difference() { "
            "  /* outer solid inline */ minkowski(){...} or cube(...); "
            "  /* gear cavity inline */ hull(){translate(c1)cylinder(...); translate(c2)cylinder(...);} "
            "  /* shaft bores inline */ translate(c1)cylinder(d=shaft_bore,...); translate(c2)cylinder(...); "
            "  /* bearing seats inline */ translate(c1)cylinder(d=bearing_od,...); ... "
            "  /* bolt holes inline */ for(dx=...){for(dy=...){translate([dx,dy,0])cylinder(...);}} "
            "} } "
            "All subtractions go inside ONE difference() call — no pre-computed geometry variables."
        )
    if family_id == "robotics_servo_reference":
        family_lines.append(
            "For MG996R U-bracket requests, the mg996r_servo_bracket record is authoritative. "
            "Wing plates must be centered at x = +/-tab_span/2 (not y-direction walls). "
            "Tab holes must drill along X using rotate([0,90,0]) cylinder at z = bracket_t + tab_hole_z. "
            "Do NOT place tab holes as vertical Z cylinders through the base plate. "
            "Do NOT use minkowski() or fillet_r on bracket walls. "
            "No servo body pocket cutout is needed — servo slides into the open top of the U."
        )
    family_lines.append("Do not include material recommendations or standard explanations in OpenSCAD comments; those are chat-after-code notes only.")
    messages.append({"role": "system", "content": "\n".join(family_lines)})

    knowledge_hits = [hit for hit in context_hits if hit.get("source") != "accepted-history"][:MAX_KNOWLEDGE_HITS_IN_PROMPT]
    accepted_hits = [hit for hit in context_hits if hit.get("source") == "accepted-history"][:MAX_ACCEPTED_HITS_IN_PROMPT]
    knowledge_budget = MAX_KNOWLEDGE_CHARS_OLLAMA if provider == "ollama" else MAX_KNOWLEDGE_CHARS_OPENAI
    accepted_budget = MAX_ACCEPTED_CHARS_OLLAMA if provider == "ollama" else MAX_ACCEPTED_CHARS_OPENAI
    history_budget = MAX_HISTORY_CHARS_OLLAMA if provider == "ollama" else MAX_HISTORY_CHARS_OPENAI

    if knowledge_hits:
        rag_lines = [
            "RETRIEVED MECHANICAL KNOWLEDGE",
            "Map the design description into explicit OpenSCAD operations and preserve the requested dimensions.",
        ]
        remaining = knowledge_budget
        for hit in knowledge_hits:
            if remaining <= 400:
                break
            context_text = _trim_text(hit["text"], min(len(hit["text"]), remaining))
            remaining -= len(context_text)
            rag_lines.append(f"\n[{hit['title']} | score={hit['score']}]\n{context_text}")
        messages.append({"role": "system", "content": "\n".join(rag_lines)})

    if accepted_hits:
        accepted_lines = [
            "ACCEPTED USER-APPROVED DESIGNS",
            "Reuse these only when the new request is clearly similar.",
        ]
        remaining = accepted_budget
        for hit in accepted_hits:
            if remaining <= 300:
                break
            snippet = _trim_text(hit["text"], min(2200, remaining))
            remaining -= len(snippet)
            accepted_lines.append(f"\n[{hit['title']} | score={hit['score']}]\n{snippet}")
        messages.append({"role": "system", "content": "\n".join(accepted_lines)})

    history_limit = 4 if provider == "ollama" else 10
    recent_history: list[ChatMessage] = []
    used_chars = 0
    for message in reversed(history[-history_limit:]):
        if message.role not in ("user", "assistant") or not message.content.strip():
            continue
        if used_chars >= history_budget:
            break
        remaining = history_budget - used_chars
        content = _trim_text(message.content.strip(), min(len(message.content.strip()), remaining))
        used_chars += len(content)
        recent_history.append(ChatMessage(role=message.role, content=content))

    for message in reversed(recent_history):
        messages.append({"role": message.role, "content": message.content})

    messages.append({"role": "user", "content": user_prompt.strip()})
    return messages


def extract_scad(raw: str) -> str:
    blocks = re.findall(r"```(?:openscad|scad|scss)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if blocks:
        cleaned = max(blocks, key=len).strip()
    else:
        cleaned = raw.strip().replace("```", "")

    cleaned = re.sub(r"^\s*(openscad|scad|scss)\s*\n+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Here is the final OpenSCAD code for.*?:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = sanitize_scad_metadata(cleaned.strip())
    cleaned = fix_openscad_list_slices(cleaned)   # repair l[1:] / l[0:n] slice syntax
    return normalize_library_includes(cleaned)


CHAT_METADATA_ASSIGNMENT_RE = re.compile(
    r'^\s*(usage|use_case|bearing_series|standard_used|standards_used|material|materials|material_suggestion|notes?)\s*=\s*"[^"]*"\s*;.*$',
    re.IGNORECASE,
)
CHAT_METADATA_COMMENT_RE = re.compile(
    r"\b(main user parameters|bearing catalog defaults|standard used|standards?|catalog|material|usage|informational only|nominal bore|mounted-bearing)\b",
    re.IGNORECASE,
)


def sanitize_scad_metadata(code: str) -> str:
    """Remove chat-only metadata that models sometimes leak into OpenSCAD."""
    cleaned_lines: list[str] = []
    for line in code.splitlines():
        if CHAT_METADATA_ASSIGNMENT_RE.match(line):
            continue
        stripped = line.strip()
        if stripped.startswith("//") and CHAT_METADATA_COMMENT_RE.search(stripped):
            continue
        if "//" in line and CHAT_METADATA_COMMENT_RE.search(line.split("//", 1)[1]):
            line = line.split("//", 1)[0].rstrip()
        cleaned_lines.append(line)

    compacted: list[str] = []
    blank_count = 0
    for line in cleaned_lines:
        if line.strip():
            blank_count = 0
            compacted.append(line)
            continue
        blank_count += 1
        if blank_count <= 1:
            compacted.append(line)
    return "\n".join(compacted).strip()


# ── List-slice auto-repair ──────────────────────────────────────────────────
# OpenSCAD has no slice operator.  LLMs sometimes generate l[1:] or arr[0:n]
# which crash the parser.  This function patches the two most common patterns
# produced by code-generation models before the code reaches the validator.

_SLICE_PRESENT_RE = re.compile(r"\w\s*\[\s*\d*\s*:\s*[\w\d]*\s*\]")

# Pattern: function list_sum(l) = (len(l) == 0) ? 0 : l[0] + list_sum(l[1:]);
_RECURSIVE_SLICE_FUNC_RE = re.compile(
    r"function\s+(\w+)\s*\(\s*(\w+)\s*\)\s*=[^;]*\b\2\s*\[\s*1\s*:\s*\]\s*\)\s*;",
    re.DOTALL,
)

# Pattern: any_func(array[0:var]) — slice passed as argument
_SLICE_ARG_RE = re.compile(
    r"\b(list_sum|sum_list|cumsum|cum_sum|prefix_sum)\s*\(\s*(\w+)\s*\[\s*0\s*:\s*(\w+|\d+)\s*\]\s*\)"
)

# Pattern: same func called with full array and no second arg — list_sum(arr)
_BARE_CALL_RE = re.compile(
    r"\b(list_sum|sum_list|cumsum|cum_sum|prefix_sum)\s*\(\s*(\w+)\s*\)(?!\s*,\s*[\d\w])"
)


def fix_openscad_list_slices(code: str) -> str:
    """
    Detect and repair Python-style list-slice syntax that OpenSCAD does not support.

    Handles the two patterns LLMs most commonly generate:
      1. function list_sum(l) = ... l[0] + list_sum(l[1:])
         → function list_sum(l, n) = (n <= 0) ? 0 : l[n-1] + list_sum(l, n-1);
      2. list_sum(step_lengths[0:i])
         → list_sum(step_lengths, i)
      3. list_sum(step_lengths)   (bare call after the function signature changed)
         → list_sum(step_lengths, len(step_lengths))
    """
    if not _SLICE_PRESENT_RE.search(code):
        return code  # fast path — no slice found

    # 1. Replace recursive-slice function definition
    def _replace_func(m: re.Match) -> str:
        fname = m.group(1)
        return (
            f"function {fname}(l, n) = (n <= 0) ? 0 : l[n-1] + {fname}(l, n-1);"
        )
    code = _RECURSIVE_SLICE_FUNC_RE.sub(_replace_func, code)

    # 2. Fix slice-as-argument calls: func(arr[0:i]) → func(arr, i)
    code = _SLICE_ARG_RE.sub(r"\1(\2, \3)", code)

    # 3. Fix bare full-array calls: func(arr) → func(arr, len(arr))
    #    Only when the function was just rewritten to take two arguments.
    code = _BARE_CALL_RE.sub(r"\1(\2, len(\2))", code)

    return code


GEARS_SCAD_PATH = str((BASE_DIR.parent / "gears.scad").resolve()).replace("\\", "/")
SPROCKET_SCAD_PATH = str((BASE_DIR / "docs" / "sprocket.scad").resolve()).replace("\\", "/")
GEARS_INCLUDE_RE = re.compile(
    r"^\s*(include|use)\s*<[^>\n]*gears\.scad>\s*;?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
SPROCKET_INCLUDE_RE = re.compile(
    r"^\s*(include|use)\s*<[^>\n]*sprocket\.scad>\s*;?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_library_includes(code: str) -> str:
    """Make generated gears.scad imports resolvable from browser downloads and temp exports."""
    if "sprocket.scad" in code.lower():
        code = SPROCKET_INCLUDE_RE.sub(f"use <{SPROCKET_SCAD_PATH}>", code)
    if "gears.scad" not in code.lower():
        return code
    return GEARS_INCLUDE_RE.sub(f"include <{GEARS_SCAD_PATH}>", code)


def deterministic_sprocket_library_call(prompt: str) -> str:
    size = 40
    size_match = re.search(r"#\s*(25|35|40|41|50|60|80)\b|ansi\s*#?\s*(25|35|40|41|50|60|80)\b", prompt, re.IGNORECASE)
    if size_match:
        size = int(next(group for group in size_match.groups() if group))

    teeth = 17
    teeth_match = re.search(r"\b(\d+)\s*(?:t|tooth|teeth)\b", prompt, re.IGNORECASE)
    if teeth_match:
        teeth = max(6, int(teeth_match.group(1)))

    def number(pattern: str) -> float | None:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return None

    bore_mm = (
        number(r"(\d+(?:\.\d+)?)\s*mm\s+(?:bore|shaft)")
        or number(r"(?:bore|shaft)(?:_d| diameter)?\s*(?:=|of)?\s*(\d+(?:\.\d+)?)\s*mm?")
        or 10
    )
    hub_od_mm = (
        number(r"(\d+(?:\.\d+)?)\s*mm\s+hub\s*(?:od|outer|diameter)")
        or number(r"hub(?:_|\s*)od\s*(?:=|of)?\s*(\d+(?:\.\d+)?)")
        or max(bore_mm + 12, 22)
    )
    hub_h_mm = (
        number(r"(\d+(?:\.\d+)?)\s*mm\s+hub\s*(?:height|h)")
        or number(r"hub(?:_|\s*)(?:h|height)\s*(?:=|of)?\s*(\d+(?:\.\d+)?)")
        or 14
    )
    keyway = 1 if re.search(r"\bkeyway|key\s*slot\b", prompt, re.IGNORECASE) else 0
    setscrew = 1 if re.search(r"\bset\s*-?\s*screw|setscrew\b", prompt, re.IGNORECASE) else 0

    return f"""use <{SPROCKET_SCAD_PATH}>
$fn = 180;
sprocket(size={size}, teeth={teeth}, bore={bore_mm:g}/25.4, hub_diameter={hub_od_mm:g}/25.4, hub_height={hub_h_mm:g}/25.4, keyway={keyway}, setscrew={setscrew});"""


def deterministic_pulley_library_call(prompt: str) -> str:
    """Return a correct pulley-generator.scad call derived entirely from the prompt.

    Key fixes:
    - Teeth regex handles hyphens: "30-tooth" and "30 tooth" both match.
    - Belt width regex handles "6 mm belt" (no 'width' word needed).
    - Module selection: use pulley3DP() for simple requests (no structured
      parameter keywords); only use pulley() when grub/nut/captive config
      is explicitly requested.
    - All values come from the prompt; defaults only fill genuine gaps.
    """
    lowered = prompt.lower()

    # ── Belt profile detection ──────────────────────────────────────────
    supported_models = [
        "GT2 2mm", "GT2 3mm", "GT2 5mm",
        "HTD 3mm", "HTD 5mm", "HTD 8mm",
        "T2.5", "T5", "T10", "AT5", "MXL", "40DP", "XL", "H",
    ]
    model = "GT2 2mm"
    for candidate in supported_models:
        if candidate.lower() in lowered:
            model = candidate
            break
    if model == "GT2 2mm":
        if re.search(r"\bhtd\s*5\s*mm|\bhtd5\b|\b5m\s*htd\b", lowered):
            model = "HTD 5mm"
        elif re.search(r"\bhtd\s*3\s*mm|\bhtd3\b|\b3m\s*htd\b", lowered):
            model = "HTD 3mm"
        elif re.search(r"\bhtd\s*8\s*mm|\bhtd8\b|\b8m\s*htd\b", lowered):
            model = "HTD 8mm"

    # ── Value extractor ─────────────────────────────────────────────────
    def number(patterns: list[str], default: float) -> float:
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if not match:
                continue
            for group in match.groups():
                if group is None:
                    continue
                try:
                    return float(group)
                except ValueError:
                    continue
        return default

    # ── Tooth count — handles "30-tooth", "30 tooth", "30t", "30 teeth" ─
    teeth = int(number([
        r"\b(\d+)\s*[-–]?\s*(?:tooth|teeth)\b",   # "30-tooth", "30 tooth"
        r"\b(\d+)\s*[-–]?\s*t\b",                  # "30t"
        r"\bteeth(?:Count| count)?\s*(?:=|:)?\s*(\d+)\b",
    ], 20))

    # ── Belt width — handles "6 mm belt", "6mm belt", "belt width 6mm" ──
    default_belt_width = 15 if model == "HTD 5mm" else 9 if model in {"HTD 3mm", "GT2 3mm"} else 6
    belt_width = number([
        r"\b(\d+(?:\.\d+)?)\s*mm\s*belt(?:\s*width)?\b",      # "6 mm belt" or "6 mm belt width"
        r"\bbelt(?:\s*width)?\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*mm?\b",  # "belt width 6mm"
        r"\bfor\s+a\s+(\d+(?:\.\d+)?)\s*mm\s*belt\b",         # "for a 6mm belt"
    ], default_belt_width)

    # ── Shaft bore ──────────────────────────────────────────────────────
    default_shaft = 8.0 if model == "HTD 5mm" else 5.2 if ("3d" in lowered or "print" in lowered) else 5.0
    shaft_d = number([
        r"\b(\d+(?:\.\d+)?)\s*mm\s*(?:shaft|bore)\b",          # "5 mm bore" / "5 mm shaft"
        r"\b(?:shaft|bore)(?:\s*diameter|_d)?\s*(?:=|:|of)?\s*(\d+(?:\.\d+)?)\s*mm?\b",
        r"\bbore\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*mm?\b",
    ], default_shaft)

    # ── Module selection ────────────────────────────────────────────────
    # Use pulley() only when the prompt explicitly requests custom nut/grub config.
    # Use pulley3DP() for all simple generation requests (the common case).
    use_full_pulley = bool(re.search(
        r"\b(captive\s*nut|grub\s*screw|set\s*screw|retainer\s*info|idler\s*info"
        r"|base\s*info|nut\s*profile|pulley\s*\(\)|dual\s*nut|two\s*nut"
        r"|structured\s*param|full\s*param)\b",
        lowered,
    ))

    if not use_full_pulley:
        # ── Simple path: pulley3DP() ────────────────────────────────────
        tooth_tweak = 0.25 if "petg" in lowered else 0.2
        return f"""include <pulley-generator.scad>

// ── MAIN PARAMETERS ──
model         = "{model}";
teethCount    = {teeth};
beltWidth     = {belt_width:g};
shaftDiameter = {shaft_d:g};

// ── TWEAKS ──
toothWidthTweak = {tooth_tweak:g};
toothDepthTweak = 0.0;

pulley3DP(
    model           = model,
    teethCount      = teethCount,
    beltWidth       = beltWidth,
    shaftDiameter   = shaftDiameter,
    toothWidthTweak = toothWidthTweak,
    toothDepthTweak = toothDepthTweak
);"""

    # ── Full path: pulley() with structured params ──────────────────────
    grub_match = re.search(r"\bM\s*(3|4|5|6)\b", prompt, re.IGNORECASE)
    grub_size = int(grub_match.group(1)) if grub_match else 3
    nut_count = 2 if re.search(r"\bdual\b|\btwo\b|\b2\s*(nuts?|screws?)\b", lowered) else 1
    nut_angle = 90 if nut_count <= 2 else 120
    screw_clearance = {3: 3.2, 4: 4.2, 5: 5.3, 6: 6.4}.get(grub_size, grub_size + 0.2)
    nut_flat  = {3: 5.7, 4: 7.0, 5: 8.0, 6: 10.0}.get(grub_size, grub_size * 1.7)
    nut_thick = {3: 2.7, 4: 3.2, 5: 4.0, 6: 5.0}.get(grub_size, grub_size * 0.8)
    base_d = max(shaft_d + 2 * (nut_thick + 3) + 2, 20 if shaft_d <= 8 else shaft_d + 16)
    base_h = max(nut_flat + 2, 8)
    tooth_tweak = 0.25 if "petg" in lowered else 0.2

    return f"""include <pulley-generator.scad>

// ── MAIN PARAMETERS ──
model         = "{model}";
teethCount    = {teeth};
beltWidth     = {belt_width:g};
shaftDiameter = {shaft_d:g};

// ── STRUCTURED PARAMETERS ──
retainerInfo = [2, 1, 1];
idlerInfo    = [2, 1, 1];
baseInfo     = [{base_d:g}, {base_h:g}];
nutProfile   = ["hex", {screw_clearance:g}, {nut_flat:g}, {nut_thick:g}];
captiveNuts  = [{nut_count}, {nut_angle}, 1.5];

// ── TWEAKS ──
toothWidthTweak = {tooth_tweak:g};
toothDepthTweak = 0.0;

pulley(
    model           = model,
    teethCount      = teethCount,
    beltWidth       = beltWidth,
    shaftDiameter   = shaftDiameter,
    retainerInfo    = retainerInfo,
    idlerInfo       = idlerInfo,
    baseInfo        = baseInfo,
    nutProfile      = nutProfile,
    captiveNuts     = captiveNuts,
    toothWidthTweak = toothWidthTweak,
    toothDepthTweak = toothDepthTweak,
    autoFlip        = true
);"""


def enforce_mechanical_library_output(prompt: str, code: str, family_id: str | None) -> str:
    if family_id == "pulley_belt_drive_reference":
        return deterministic_pulley_library_call(prompt)
    return enforce_gears_master_output(prompt, code, family_id)


def enforce_gears_master_output(prompt: str, code: str, family_id: str | None) -> str:
    """Keep model output intact, but replace non-geometry gear placeholders."""
    if family_id == "sprocket_chain_reference":
        return deterministic_sprocket_library_call(prompt)
    if family_id == "gear_reference" and re.search(r"Please specify|Required user parameters|echo\s*\(", code, re.IGNORECASE):
        p = prompt.lower()
        if "bevel" in p:
            return f"""include <{GEARS_SCAD_PATH}>

bevel_gear(
  modul=1,
  tooth_number=30,
  partial_cone_angle=45,
  tooth_width=5,
  bore=4,
  pressure_angle=20,
  helix_angle=0
);"""
        if "worm" in p:
            return f"""include <{GEARS_SCAD_PATH}>

worm_gear(
  modul=1,
  tooth_number=30,
  thread_starts=2,
  width=8,
  length=20,
  worm_bore=4,
  gear_bore=4,
  pressure_angle=20,
  lead_angle=10,
  optimized=true,
  together_built=true,
  show_spur=1,
  show_worm=1
);"""
        if "planet" in p:
            return f"""include <{GEARS_SCAD_PATH}>

planetary_gear(
  modul=1,
  sun_teeth=16,
  planet_teeth=9,
  number_planets=5,
  width=5,
  rim_width=3,
  bore=4,
  pressure_angle=20,
  helix_angle=30,
  together_built=true,
  optimized=true
);"""
        if "rack" in p and "pinion" in p:
            return f"""include <{GEARS_SCAD_PATH}>

rack_and_pinion(
  modul=1,
  rack_length=50,
  gear_teeth=30,
  rack_height=4,
  gear_bore=4,
  width=5,
  pressure_angle=20,
  helix_angle=0,
  together_built=true,
  optimized=true
);"""
        return f"""include <{GEARS_SCAD_PATH}>

spur_gear(
  modul=1,
  tooth_number=30,
  width=5,
  bore=4,
  pressure_angle=20,
  helix_angle=0,
  optimized=true
);"""
    return normalize_library_includes(code)


def _looks_truncated(code: str) -> bool:
    stripped = code.strip()
    if not stripped:
        return True
    last_line = stripped.splitlines()[-1].strip()
    return bool(
        stripped.endswith(("=", ",", "(", "[", "{"))
        or re.search(r"=\s*[^;]+$", last_line)
        or last_line in {"difference()", "union()", "intersection()"}
    )


def validate_scad(code: str, prompt: str = "") -> list[dict]:
    family_id = detect_primary_family(f"{prompt}\n{code}")
    return validate_scad_full(code, prompt, family_id)


def evaluate_generation(code: str, prompt: str, checks: list[dict], family_schema: dict) -> dict:
    validity_labels = {
        "No markdown fences",
        "No stray language label",
        "Balanced braces { }",
        "Balanced parentheses ( )",
        "3D primitives used",
        "Named module present",
        "Module called at end",
        "No unfinished assignment at end",
        "External library use allowed",
    }
    parametric_labels = {
        "Parametric variables present",
        "$fn resolution set",
        "Boolean subtraction for cut features",
        "difference() wraps union() correctly",
    }

    buckets = {
        "validity": [],
        "parametricity": [],
        "usability": [],
    }
    for check in checks:
        label = check.get("label", "")
        category = check.get("category", "")
        if label in validity_labels:
            buckets["validity"].append(check)
        elif label in parametric_labels or category == "derived" or label.lower().startswith("derived:"):
            buckets["parametricity"].append(check)
        else:
            buckets["usability"].append(check)

    def score(items: list[dict]) -> dict:
        total = len(items)
        passed = sum(1 for item in items if item.get("passed"))
        blocking = sum(
            1
            for item in items
            if not item.get("passed") and item.get("severity") in {"hard", "error"}
        )
        return {
            "passed": passed,
            "total": total,
            "score": round((passed / total) * 100) if total else 100,
            "blocking": blocking,
        }

    parameters = _extract_scad_parameters(code, limit=60)
    modules = _extract_scad_modules(code)
    features = _extract_design_features(prompt, code)
    material_count = len(family_schema.get("material_suggestions") or [])

    return {
        "validity": score(buckets["validity"]),
        "parametricity": score(buckets["parametricity"]),
        "usability": score(buckets["usability"]),
        "metrics": {
            "parameters": len(parameters),
            "modules": len(modules),
            "detected_features": len(features),
            "rag_family": family_schema.get("label") or family_schema.get("family_label"),
            "material_options": material_count,
        },
        "notes": [
            "Validity checks focus on parseable OpenSCAD structure and model completeness.",
            "Parametricity checks focus on editable dimensions, derived formulas, and reusable feature logic.",
            "Usability checks focus on family-specific mechanical intent and fabrication review signals.",
        ],
    }


def _validate_scad_original(code: str, prompt: str = "") -> list[dict]:
    """Original inline implementation kept as reference."""
    stripped = code.strip()
    last_line = stripped.splitlines()[-1] if stripped else ""
    family_id = detect_family(prompt) or detect_family(code)
    hole_words = bool(re.search(r"\b(bore|hole|slot|cutout|clearance|set[_ -]?screw|counterbore)\b", stripped, re.IGNORECASE))
    bearing_housing = bool(re.search(r"\b(pillow|bearing[_ ]?housing|bearing_od|bearing seat)\b", stripped, re.IGNORECASE))
    shaft_param = bool(re.search(r"\bshaft_diameter\s*=", stripped))
    bearing_param = bool(re.search(r"\bbearing_od\s*=", stripped))
    first_difference = stripped.find("difference()")
    first_union = stripped.find("union()")
    subtracts_after_union = first_difference >= 0 and first_union >= 0 and first_difference < first_union and stripped.rfind("cylinder(") > first_union
    has_bearing_cut_comment = bool(re.search(r"//\s*(bearing seat|shaft|mount).*?\n\s*(translate|rotate|cylinder)", stripped, re.IGNORECASE))

    checks = [
        {"label": "No markdown fences", "passed": "```" not in code, "detail": "Output should be raw OpenSCAD, not markdown-fenced."},
        {"label": "No stray language label", "passed": not bool(re.match(r"^\s*(openscad|scad|scss)\b", stripped, re.IGNORECASE)), "detail": "Output must start with OpenSCAD code, not a language label."},
        {"label": "No chat metadata string variables", "passed": not bool(CHAT_METADATA_ASSIGNMENT_RE.search(stripped)), "detail": "Usage, bearing series, material, standard, and note text belong in chat notes, not string variables inside OpenSCAD."},
        {"label": "No material or standard prose comments", "passed": not any(line.strip().startswith("//") and CHAT_METADATA_COMMENT_RE.search(line) for line in stripped.splitlines()), "detail": "Standards and materials should be explained after the code, not as explanatory comments in the .scad file."},
        {"label": "Balanced braces { }", "passed": code.count("{") == code.count("}"), "detail": "Unbalanced braces cause OpenSCAD parse failures."},
        {"label": "Balanced parentheses ( )", "passed": code.count("(") == code.count(")"), "detail": "Unbalanced parentheses indicate broken primitive calls."},
        {"label": "Uses PI constant, not pi", "passed": not bool(re.search(r"\bpi\b", code)), "detail": "OpenSCAD uses PI, not pi."},
        {"label": "Parametric variables", "passed": bool(re.search(r"^\s*[a-zA-Z_]\w*\s*=\s*[\d\.]", code, re.MULTILINE)), "detail": "At least one named numeric parameter should be at the top."},
        {"label": "3D primitives used", "passed": any(token in code for token in ("cube(", "cylinder(", "sphere(", "polyhedron(", "linear_extrude(", "rotate_extrude(")), "detail": "Code should contain at least one 3D primitive or extrusion."},
        {"label": "Named module present", "passed": bool(re.search(r"\bmodule\s+\w+", code)), "detail": "Wrapping geometry in a named module enables reuse."},
        {"label": "Module called at end", "passed": bool(re.match(r"^\s*\w+\s*\([^;]*\);?\s*$", last_line)), "detail": "The main module should be instantiated once at the end."},
        {"label": "External library use allowed", "passed": True, "detail": "OpenSCAD include<> and use<> statements are allowed when the referenced library is available to OpenSCAD."},
        {"label": "$fn resolution set", "passed": "$fn" in code, "detail": "Set $fn=96 or higher for smooth circular features."},
        {"label": "No unfinished assignment at end", "passed": not _looks_truncated(code), "detail": "The file appears truncated or ends with an incomplete assignment or block."},
        {"label": "Boolean subtraction for cut features", "passed": ("difference()" in code) if hole_words else True, "detail": "Bores, holes, slots, and cutouts should be made with difference()."},
        {"label": "Bearing seat uses bearing_od", "passed": (not bearing_housing or not bearing_param) or bool(re.search(r"d\s*=\s*bearing_od|d\s*=\s*bearing_outer_diameter|bearing_od\s*[+)]", code)), "detail": "Bearing housing must subtract a bearing seat using bearing_od."},
        {"label": "Shaft bore uses shaft diameter", "passed": (not bearing_housing or not shaft_param) or bool(re.search(r"d\s*=\s*shaft_diameter|shaft_diameter\s*[+)]", code)), "detail": "Shaft through-bore should use shaft_diameter plus clearance."},
        {"label": "Pillow block has multiple mounting holes", "passed": (not bool(re.search(r"\bpillow\b", stripped, re.IGNORECASE))) or ("for" in code and bool(re.search(r"mount|hole_spacing|mount_hole", code, re.IGNORECASE))), "detail": "A pillow block should include repeated base mounting holes."},
        {"label": "Bearing housing cuts are subtractive", "passed": (not bearing_housing) or (subtracts_after_union and not has_bearing_cut_comment), "detail": "Bearing seat, shaft clearance, and mounting holes must be subtracted after the main union body."},
    ]
    if family_id == "flange_pipe_fitting_reference":
        checks.extend(
            [
                {"label": "Flange has central bore", "passed": bool(re.search(r"\b(bore|inner_d|pipe_d|central bore)\b", stripped, re.IGNORECASE)), "detail": "A flange should include a central bore or pipe opening."},
                {"label": "Flange has bolt circle logic", "passed": bool(re.search(r"\b(bolt_circle|hole_count|bolt_count|for\s*\()", stripped, re.IGNORECASE)), "detail": "A flange should place repeated holes on a bolt circle."},
            ]
        )
    elif family_id == "shaft_coupler_reference":
        checks.extend(
            [
                {"label": "Coupler defines two shaft bores", "passed": bool(re.search(r"\bshaft_a|shaft_b|bore_a|bore_b\b", stripped, re.IGNORECASE)), "detail": "A shaft coupler should model both shaft interfaces."},
                {"label": "Coupler includes clamp feature", "passed": bool(re.search(r"\b(clamp_gap|slit|clamp|set_screw|screw)\b", stripped, re.IGNORECASE)), "detail": "A clamp-style coupler should include a slit or screw feature."},
            ]
        )
    elif family_id == "gear_reference":
        checks.extend(
            [
                {"label": "Gear has tooth-count logic", "passed": bool(re.search(r"\b(teeth|tooth_count)\b", stripped, re.IGNORECASE)), "detail": "A gear should define tooth count or repeated tooth logic."},
                {"label": "Gear separates bore from gear body", "passed": bool(re.search(r"\b(bore_diameter|bore_d|bore)\b", stripped, re.IGNORECASE)), "detail": "A gear should model a bore separately from the pitch or body dimensions."},
            ]
        )
    elif family_id == "enclosure_box_reference":
        checks.extend(
            [
                {"label": "Enclosure has wall thickness", "passed": bool(re.search(r"\bwall(_thickness)?\b", stripped, re.IGNORECASE)), "detail": "An enclosure should define wall thickness parametrically."},
                {"label": "Enclosure includes lid or posts", "passed": bool(re.search(r"\b(lid|post|screw_post|standoff)\b", stripped, re.IGNORECASE)), "detail": "An enclosure usually needs a lid strategy or internal mounting posts."},
            ]
        )
    elif family_id == "hinge_joint_snapfit_reference":
        checks.extend(
            [
                {"label": "Hinge has pin logic", "passed": bool(re.search(r"\b(pin|pin_diameter|knuckle)\b", stripped, re.IGNORECASE)), "detail": "A hinge should include a pin or knuckle diameter definition."},
                {"label": "Hinge has two leaves or bodies", "passed": stripped.lower().count("leaf") >= 1 or stripped.lower().count("translate(") >= 2, "detail": "A hinge should represent at least two leaves around a pin axis."},
            ]
        )
    elif family_id == "structural_profiles_reference" and bool(
        re.search(r"\b(i[- ]?beam|i section|h[- ]?beam|ipe|hea|flange|web)\b", f"{prompt}\n{stripped}", re.IGNORECASE)
    ):
        checks.append(
            {
                "label": "I-beam has flange and web logic",
                "passed": bool(re.search(r"\bflange(_|\b)", stripped, re.IGNORECASE))
                and bool(re.search(r"\bweb(_|\b)", stripped, re.IGNORECASE)),
                "detail": "An I-beam should define flanges and a web, not a single rectangular bar.",
            }
        )
    elif family_id == "bearing_housing_reference":
        checks.append(
            {
                "label": "Bearing housing includes seat and base",
                "passed": bool(re.search(r"\b(seat|pocket|bearing_outer_diameter|bearing_od)\b", stripped, re.IGNORECASE))
                and bool(re.search(r"\bbase(_length|_width|_thickness)?\b", stripped, re.IGNORECASE)),
                "detail": "A bearing housing should include both bearing-seat logic and a supporting base or body.",
            }
        )
        if re.search(r"\b(pillow|plummer)\b", f"{prompt}\n{stripped}", re.IGNORECASE):
            checks.extend(
                [
                    {
                        "label": "Pillow block has pedestal support",
                        "passed": bool(re.search(r"\b(pedestal|saddle|cap|split|rib|web|foot|feet)\b", stripped, re.IGNORECASE)),
                        "detail": "A real pillow block should not be only a hollow ring on a plate; include pedestal/saddle/cap/rib/foot geometry.",
                    },
                    {
                        "label": "Pillow block has paired foot mounting holes",
                        "passed": bool(re.search(r"\bfor\s*\(|mount_hole_spacing|mounting_hole_spacing|foot", stripped, re.IGNORECASE))
                        and bool(re.search(r"\bmount_hole|mounting_hole|bolt_hole", stripped, re.IGNORECASE)),
                        "detail": "A pillow block needs two symmetric base/foot mounting holes.",
                    },
                    {
                        "label": "Pillow block avoids arch-frame shape",
                        "passed": not bool(re.search(r"\b(arch|bridge|support_rib|rib_offset|side_rib|triangular_rib)\b", stripped, re.IGNORECASE)),
                        "detail": "Use a compact UCP-style cast body, not tall arch ribs or bridge-frame supports around the boss.",
                    },
                    {
                        "label": "Pillow block avoids solid top slab",
                        "passed": not bool(re.search(r"\b(top_slab|top_block|cap_block|cap_land|roof|rectangular\s+cap)\b", stripped, re.IGNORECASE)),
                        "detail": "Use only a thin subtractive split-line groove; do not add a massive solid cap slab or roof.",
                    },
                ]
            )
    return checks


def build_repair_prompt(original_user_prompt: str, bad_code: str, failed_checks: list[dict], attempt: int) -> str:
    failed_lines = "\n".join(f"- {item['label']}: {item['detail']}" for item in failed_checks)
    severity_note = (
        "The previous output appears truncated or structurally invalid. Regenerate the full file from scratch."
        if any(item["label"] in {"Balanced braces { }", "Balanced parentheses ( )", "No unfinished assignment at end", "Module called at end"} and not item["passed"] for item in failed_checks)
        else "Repair the previous file while preserving valid dimensions and intent."
    )
    return f"""
Repair the following OpenSCAD output so it becomes one complete, valid, parametric OpenSCAD file.

Original design request:
{original_user_prompt}

Problems found:
{failed_lines}

Repair strategy:
{severity_note}

Broken OpenSCAD output:
{bad_code}

Rules:
- Return raw OpenSCAD only
- No markdown fences
- No labels like openscad or scss
- Use PI, not pi
- Do not assign geometry primitives to variables as if they were values
- Build final geometry using modules plus union() and/or difference()
- For bearing housings and pillow blocks: union() must contain only solid material; bearing seat, shaft bore, and mounting holes must be subtracted after the union body
- Do not add material recommendations or standard explanations as OpenSCAD comments
- Do not add string metadata variables such as usage, bearing_series, material, standard_used, or notes
- Store bearing choice as numeric geometry values only: shaft_diameter, bearing_od, and bearing_width
- Keep parameter names and valid dimensions when possible
- End every assignment with a semicolon
- Instantiate the primary module exactly once at the end
- If attempt {attempt} still cannot be repaired safely, regenerate the full file from scratch
""".strip()


BEARING_SERIES_DEFAULTS: dict[str, tuple[float, float, float]] = {
    "608": (8, 22, 7),
    "6001": (12, 28, 8),
    "6200": (10, 30, 9),
    "6201": (12, 32, 10),
    "6202": (15, 35, 11),
    "6203": (17, 40, 12),
    "6204": (20, 47, 14),
    "6205": (25, 52, 15),
    "6206": (30, 62, 16),
    "6207": (35, 72, 17),
    "6208": (40, 80, 18),
    "6305": (25, 62, 17),
}


def _number_for_pattern(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def deterministic_sprocket_scad(prompt: str) -> str:
    lowered = prompt.lower()

    size = 40
    size_match = re.search(r"#\s*(25|35|40|41|50|60|80)\b|ansi\s*#?\s*(25|35|40|41|50|60|80)\b", prompt, re.IGNORECASE)
    if size_match:
        size = int(next(group for group in size_match.groups() if group))

    teeth = 17
    teeth_match = re.search(r"\b(\d+)\s*(?:t|tooth|teeth)\b", prompt, re.IGNORECASE)
    if teeth_match:
        teeth = max(6, int(teeth_match.group(1)))

    bore_d = (
        _number_for_pattern(prompt, r"(\d+(?:\.\d+)?)\s*mm\s+(?:bore|shaft)")
        or _number_for_pattern(prompt, r"(?:bore|shaft)(?:_d| diameter)?\s*(?:=|of)?\s*(\d+(?:\.\d+)?)\s*mm?")
        or 10
    )
    hub_od = (
        _number_for_pattern(prompt, r"(\d+(?:\.\d+)?)\s*mm\s+hub\s*(?:od|outer)")
        or _number_for_pattern(prompt, r"hub(?:_|\s*)od\s*(?:=|of)?\s*(\d+(?:\.\d+)?)")
        or max(bore_d + 12, 22)
    )
    hub_h = (
        _number_for_pattern(prompt, r"(\d+(?:\.\d+)?)\s*mm\s+hub\s*(?:height|h)")
        or _number_for_pattern(prompt, r"hub(?:_|\s*)(?:h|height)\s*(?:=|of)?\s*(\d+(?:\.\d+)?)")
        or 14
    )
    set_screw_d = 4.5 if re.search(r"\b(set\s*-?\s*screw|setscrew)\b", lowered) else 0
    keyway_requested = bool(re.search(r"\bkeyway|key\s*slot\b", lowered))

    return f"""$fn = 180;

FUDGE_BORE = 0.25;
FUDGE_ROLLER = 0.15;
FUDGE_TEETH = 1.0;
FUDGE_KEYWAY = 0.2;

size = {size};
teeth = {teeth};
bore_d = {bore_d:g};
hub_od = {hub_od:g};
hub_h = {hub_h:g};
make_keyway = {1 if keyway_requested else 0};
set_screw_d = {set_screw_d:g};

function inches2mm(inches) = inches * 25.4;
function get_pitch(size) =
  size == 25 ? 1/4 :
  size == 35 ? 3/8 :
  size == 40 ? 1/2 :
  size == 41 ? 1/2 :
  size == 50 ? 5/8 :
  size == 60 ? 3/4 :
  size == 80 ? 1 : 0;

function get_roller_diameter(size) =
  size == 25 ? 0.130 :
  size == 35 ? 0.200 :
  size == 40 ? 5/16 :
  size == 41 ? 0.306 :
  size == 50 ? 0.400 :
  size == 60 ? 15/32 :
  size == 80 ? 5/8 : 0;

function get_thickness(size) =
  size == 25 ? 0.110 :
  size == 35 ? 0.168 :
  size == 40 ? 0.284 :
  size == 41 ? 0.227 :
  size == 50 ? 0.343 :
  size == 60 ? 0.459 :
  size == 80 ? 0.575 : 0;

function get_keyway_width(bore_in) =
  bore_in <= 0.375 ? 0 :
  bore_in <= 0.5625 ? 0.125 :
  bore_in <= 0.875 ? 0.1875 :
  bore_in <= 1.25 ? 0.250 :
  bore_in <= 1.375 ? 0.3125 :
  bore_in <= 1.75 ? 0.375 :
  bore_in <= 2.25 ? 0.5 : 0;

module sprocket_plate(size, teeth) {{
  angle = 360 / teeth;
  pitch = inches2mm(get_pitch(size));
  roller = inches2mm(get_roller_diameter(size) / 2);
  thickness = inches2mm(get_thickness(size));
  pitch_radius = inches2mm(get_pitch(size) / sin(180 / teeth)) / 2;
  middle_radius = sqrt(pow(pitch_radius, 2) - pow(pitch / 2, 2));
  fudge_teeth_x = FUDGE_TEETH * cos(angle / 2);
  fudge_teeth_y = FUDGE_TEETH * sin(angle / 2);

  difference() {{
    intersection() {{
      cylinder(r = pitch_radius - roller + pitch / 2, h = thickness);
      union() {{
        for (i = [0 : teeth - 1])
          rotate([0, 0, angle * i])
            intersection() {{
              translate([-fudge_teeth_x, pitch_radius - fudge_teeth_y, 0])
                cylinder(r = pitch - roller - FUDGE_ROLLER - FUDGE_TEETH, h = thickness);
              rotate([0, 0, angle])
                translate([fudge_teeth_x, pitch_radius - fudge_teeth_y, 0])
                  cylinder(r = pitch - roller - FUDGE_ROLLER - FUDGE_TEETH, h = thickness);
            }}
        for (i = [0 : teeth - 1])
          rotate([0, 0, angle * i - angle / 2])
            translate([-pitch / 2, -0.01, 0])
              cube([pitch, middle_radius + 0.01, thickness]);
      }}
    }}
    for (i = [0 : teeth - 1])
      rotate([0, 0, angle * i])
        translate([0, pitch_radius, -1])
          cylinder(r = roller + FUDGE_ROLLER, h = thickness + 2);
  }}
}}

module roller_chain_sprocket() {{
  plate_h = inches2mm(get_thickness(size));
  bore_r = bore_d / 2 + FUDGE_BORE;
  key_w = inches2mm(get_keyway_width(bore_d / 25.4));

  difference() {{
    union() {{
      sprocket_plate(size, teeth);
      translate([0, 0, (plate_h - hub_h) / 2])
        cylinder(h = hub_h, d = hub_od);
    }}

    translate([0, 0, -hub_h])
      cylinder(h = hub_h * 3, r = bore_r);

    if (make_keyway && key_w > 0)
      translate([-(bore_r + key_w / 2), -key_w / 2, -hub_h])
        cube([key_w + FUDGE_KEYWAY, key_w + FUDGE_KEYWAY, hub_h * 3]);

    if (set_screw_d > 0)
      translate([0, 0, plate_h / 2])
        rotate([90, 0, 0])
          cylinder(h = hub_od + 4, d = set_screw_d, center = true);
  }}
}}

roller_chain_sprocket();"""


def deterministic_pillow_block_scad(prompt: str) -> str:
    return """// No-flicker compact pillow block
$fn = 96;
eps = 0.05;

// Size
base_len = 80;
base_width = 22;
base_thick = 7;

center_height = 24;
housing_od = 42;
housing_width = 20;

// Bearing
bearing_od = 32;
bearing_id = 15;
bearing_width = 12;
clearance = 0.35;

// Bolts
bolt_dia = 7;
bolt_spacing = 64;
counterbore_dia = 13;
counterbore_depth = 3;

module body() {
    union() {
        // Base
        hull() {
            translate([-base_len/2 + 8, 0, base_thick/2])
                cylinder(d=16, h=base_thick, center=true);

            translate([ base_len/2 - 8, 0, base_thick/2])
                cylinder(d=16, h=base_thick, center=true);

            translate([0, 0, base_thick/2])
                cube([base_len-16, base_width, base_thick], center=true);
        }

        // Main upright
        hull() {
            translate([0, 0, center_height])
                rotate([90, 0, 0])
                    cylinder(d=housing_od, h=housing_width, center=true);

            translate([0, 0, base_thick + 4])
                cube([housing_od * 0.85, housing_width, 8], center=true);
        }

        // Front raised rim
        translate([0, -housing_width/2 - 1.25, center_height])
            rotate([90, 0, 0])
                cylinder(d=38, h=2.5, center=true);

        // Small label pad, lifted off base front face
        translate([30, -base_width/2 - 1.1, 3.8])
            cube([18, 1.8, 6], center=true);
    }
}

module pillow_block_bearing_housing() {
    difference() {
        body();

        // Bearing pocket
        translate([0, 0, center_height])
            rotate([90, 0, 0])
                cylinder(
                    d=bearing_od + clearance,
                    h=bearing_width + 0.8,
                    center=true
                );

        // Shaft hole
        translate([0, 0, center_height])
            rotate([90, 0, 0])
                cylinder(
                    d=bearing_id + 1,
                    h=housing_width + 8,
                    center=true
                );

        // Mounting holes
        for (x = [-bolt_spacing/2, bolt_spacing/2]) {
            translate([x, 0, -eps])
                cylinder(d=bolt_dia, h=base_thick + 2*eps);

            translate([x, 0, base_thick - counterbore_depth])
                cylinder(d=counterbore_dia, h=counterbore_depth + eps);
        }
    }

    // Text placed slightly in front, not coplanar
    translate([30, -base_width/2 - 2.05, 3.8])
        rotate([90, 0, 0])
            linear_extrude(0.6)
                text("P002", size=4.2, halign="center", valign="center");
}

pillow_block_bearing_housing();"""


def deterministic_l_bracket_scad(prompt: str) -> str:
    lowered = prompt.lower()
    leg_a = (
        _number_for_pattern(prompt, r"leg[_\s-]*a(?:_length|\s+length)?\s*(?:=|of)?\s*(\d+(?:\.\d+)?)")
        or _number_for_pattern(prompt, r"(\d+(?:\.\d+)?)\s*mm\s+(?:base|horizontal)")
        or 50
    )
    leg_b = (
        _number_for_pattern(prompt, r"leg[_\s-]*b(?:_length|\s+length)?\s*(?:=|of)?\s*(\d+(?:\.\d+)?)")
        or _number_for_pattern(prompt, r"(\d+(?:\.\d+)?)\s*mm\s+(?:upright|vertical)")
        or leg_a
    )
    bracket_width = (
        _number_for_pattern(prompt, r"bracket[_\s-]*width\s*(?:=|of)?\s*(\d+(?:\.\d+)?)")
        or _number_for_pattern(prompt, r"\bwidth\s*(?:=|of)?\s*(\d+(?:\.\d+)?)")
        or 30
    )
    plate_thickness = (
        _number_for_pattern(prompt, r"plate[_\s-]*thickness\s*(?:=|of)?\s*(\d+(?:\.\d+)?)")
        or _number_for_pattern(prompt, r"\bthickness\s*(?:=|of)?\s*(\d+(?:\.\d+)?)")
        or 4
    )
    hole_d = 5.5
    if re.search(r"\bm3\b", lowered):
        hole_d = 3.4
    elif re.search(r"\bm4\b", lowered):
        hole_d = 4.5
    elif re.search(r"\bm6\b", lowered):
        hole_d = 6.6
    elif re.search(r"\bm8\b", lowered):
        hole_d = 8.8
    edge_margin = (
        _number_for_pattern(prompt, r"edge[_\s-]*margin\s*(?:=|of)?\s*(\d+(?:\.\d+)?)")
        or max(12, 1.5 * hole_d)
    )
    wants_gusset = bool(re.search(r"\b(gussets?|ribs?|reinforced|heavy[- ]?duty|load[- ]?bearing|shelf support)\b", lowered))
    gusset_block = ""
    if wants_gusset:
        gusset_block = """
      for (y = [-bracket_width/2 + gusset_depth/2, bracket_width/2 - gusset_depth/2])
        translate([-leg_a_length/2 + plate_thickness, y, plate_thickness])
          rotate([90, 0, 0])
            linear_extrude(height = gusset_depth, center = true)
              polygon(points = [[0, 0], [gusset_run, 0], [0, gusset_run]]);
"""

    return f"""$fn = 96;

leg_a_length    = {leg_a:g};
leg_b_length    = {leg_b:g};
bracket_width   = {bracket_width:g};
plate_thickness = {plate_thickness:g};
hole_d          = {hole_d:g};
edge_margin     = {edge_margin:g};
gusset_depth    = 6;
gusset_run      = min(leg_a_length, leg_b_length) * 0.6;

module l_bracket() {{
  difference() {{
    union() {{
      translate([0, 0, plate_thickness/2])
        cube([leg_a_length, bracket_width, plate_thickness], center = true);

      translate([-leg_a_length/2 + plate_thickness/2, 0, leg_b_length/2])
        cube([plate_thickness, bracket_width, leg_b_length], center = true);
{gusset_block}    }}

    for (x = [-leg_a_length/2 + edge_margin, leg_a_length/2 - edge_margin])
      for (y = [-bracket_width/2 + edge_margin, bracket_width/2 - edge_margin])
        translate([x, y, plate_thickness/2])
          cylinder(h = plate_thickness + 2, d = hole_d, center = true, $fn = 40);

    for (z = [edge_margin, leg_b_length - edge_margin])
      for (y = [-bracket_width/2 + edge_margin, bracket_width/2 - edge_margin])
        translate([-leg_a_length/2 + plate_thickness/2, y, z])
          rotate([0, 90, 0])
            cylinder(h = plate_thickness + 2, d = hole_d, center = true, $fn = 40);
  }}
}}

l_bracket();"""


def score_validation(checks: list[dict]) -> int:
    return _val_score(checks)


# HARD_FAIL_LABELS now imported from validation_and_logging
HARD_FAIL_LABELS = _VAL_HARD_FAIL_LABELS


def _hard_failures(validation: list[dict]) -> list[dict]:
    return [item for item in validation if item["label"] in HARD_FAIL_LABELS and not item["passed"]]


def _should_fallback(provider: str, exc: Exception, allow: bool, gateway: ModelGateway) -> bool:
    if not allow or provider != "ollama" or not (gateway.openrouter_available() or gateway.openai_available()):
        return False
    message = str(exc).lower()
    return isinstance(exc, (requests.Timeout, requests.ConnectionError, requests.RequestException)) or any(
        token in message for token in ("timed out", "timeout", "connection", "refused")
    )


def _is_openrouter_free_route_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "429",
            "503",
            "502",
            "rate-limited",
            "rate limited",
            "no endpoints found",
            "no healthy upstream",
            "provider returned error",
            "temporarily",
            "unavailable",
        )
    )


def _unique_models(models: list[str | None]) -> list[str]:
    unique: list[str] = []
    for model in models:
        if model and model not in unique:
            unique.append(model)
    return unique


def _normalize_openrouter_model(model: str | None) -> str | None:
    if not model:
        return model
    normalized = model.replace(" ", "")
    if normalized == "deepseek/deepseek-chat-v3-0324:free" and OPENROUTER_ALLOW_PAID:
        return "deepseek/deepseek-chat-v3-0324"
    return normalized


def _openrouter_route_candidates(first_model: str | None) -> list[str]:
    first_model = _normalize_openrouter_model(first_model)
    if first_model and ":free" not in first_model:
        return _unique_models([first_model])
    return _unique_models([first_model, *OPENROUTER_MODELS])


def _openrouter_free_alternates(first_model: str | None) -> list[str]:
    first_model = _normalize_openrouter_model(first_model)
    models = []
    if first_model:
        models.append(first_model)
    models.extend(OPENROUTER_MODELS)

    return [model for model in _unique_models(models) if ":free" in model or model == "openrouter/free"]


def generate_with_fallback(request: ChatRequest, messages: list[dict]) -> tuple[str, str, str, bool]:
    provider = request.provider
    model = request.model or model_gateway.default_model(provider)
    if provider == "openrouter":
        last_error: Exception | None = None
        for candidate in _openrouter_route_candidates(model):
            try:
                raw = model_gateway.generate(
                    provider="openrouter",
                    model=candidate,
                    messages=messages,
                    temperature=request.temperature,
                )
                return raw, "openrouter", candidate, candidate != model
            except Exception as exc:
                if ":free" not in candidate or not _is_openrouter_free_route_error(exc):
                    raise
                last_error = exc
                log.warning("OpenRouter free model %s unavailable: %s", candidate, exc)
        if request.allow_fallback and model_gateway.openai_available():
            openai_model = model_gateway.default_model("openai")
            if openai_model:
                log.warning("All OpenRouter free routes failed; falling back to OpenAI model %s.", openai_model)
                raw = model_gateway.generate(
                    provider="openai",
                    model=openai_model,
                    messages=messages,
                    temperature=request.temperature,
                )
                return raw, "openai", openai_model, True
        if request.allow_fallback and model_gateway.ollama_available():
            ollama_model = model_gateway.default_model("ollama")
            if ollama_model:
                log.warning("All OpenRouter free routes failed; falling back to Ollama model %s.", ollama_model)
                raw = model_gateway.generate(
                    provider="ollama",
                    model=ollama_model,
                    messages=messages,
                    temperature=request.temperature,
                )
                return raw, "ollama", ollama_model, True
        if last_error:
            raise RuntimeError(
                "All configured free OpenRouter routes are currently unavailable or rate-limited upstream. "
                f"Last error: {last_error}"
            ) from last_error

    try:
        raw = model_gateway.generate(provider=provider, model=model, messages=messages, temperature=request.temperature)
        return raw, provider, model, False
    except Exception as exc:
        if not _should_fallback(provider, exc, request.allow_fallback, model_gateway):
            raise
        log.warning("Ollama failed (%s); falling back to cloud model.", exc)

    fallback_provider = "openrouter" if model_gateway.openrouter_available() else "openai"
    fallback_model = model_gateway.default_model(fallback_provider)
    if fallback_provider == "openrouter":
        fallback_request = request.model_copy(update={"provider": "openrouter", "model": fallback_model})
        raw, _, fallback_model, _ = generate_with_fallback(fallback_request, messages)
    else:
        raw = model_gateway.generate(provider=fallback_provider, model=fallback_model, messages=messages, temperature=request.temperature)
    return raw, fallback_provider, fallback_model, True


def _provider_error_response(exc: Exception, provider: str = "") -> HTTPException | None:
    message = str(exc)
    lowered = message.lower()
    is_rate_limit = "429" in message or "rate-limited" in lowered or "rate limited" in lowered
    if is_rate_limit:
        if provider == "openai" or "openai.com" in lowered:
            return HTTPException(
                status_code=429,
                detail=(
                    "OpenAI returned a rate-limit error (429). "
                    "Your quota may be exhausted or the API key may be invalid. "
                    "Check your usage at platform.openai.com/usage."
                ),
            )
        return HTTPException(
            status_code=429,
            detail=(
                "All configured free OpenRouter routes are temporarily rate-limited upstream. "
                "Wait and retry, choose a paid/non-free OpenRouter model if you have credits, "
                "or use Ollama/OpenAI."
            ),
        )
    if "no endpoints found" in lowered or "no healthy upstream" in lowered:
        return HTTPException(
            status_code=503,
            detail=(
                "No active endpoint available for this model right now. "
                "Pick another model, or use OpenAI/Ollama."
            ),
        )
    if "401" in message or "invalid api key" in lowered or "incorrect api key" in lowered:
        return HTTPException(
            status_code=401,
            detail=f"Invalid or missing API key for provider '{provider}'. Check your .env file.",
        )
    return None


def auto_repair_result(
    request: ChatRequest,
    original_prompt: str,
    retrieved: list[dict],
    initial_result: str,
    initial_validation: list[dict],
) -> tuple[str, list[dict], str | None, str | None, bool]:
    best_result = initial_result
    best_validation = initial_validation
    repair_provider: str | None = None
    repair_model: str | None = None
    repair_used_fallback = False

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        failed = [item for item in best_validation if not item["passed"]]
        if not failed:
            break

        repair_prompt = build_repair_prompt(original_prompt, best_result, failed, attempt)
        repair_family = detect_family(original_prompt, best_result)
        repair_messages = build_messages(repair_prompt, request.history, retrieved, provider=request.provider, family_id=repair_family)
        repaired_raw, repair_provider, repair_model, repair_used_fallback = generate_with_fallback(request, repair_messages)
        repaired_result = extract_scad(repaired_raw)
        repaired_validation = validate_scad(repaired_result, original_prompt)

        if score_validation(repaired_validation) >= score_validation(best_validation):
            best_result = repaired_result
            best_validation = repaired_validation

        if not [item for item in best_validation if not item["passed"]]:
            break

    return best_result, best_validation, repair_provider, repair_model, repair_used_fallback


knowledge_base = KnowledgeBase()
accepted_history = AcceptedHistory(ACCEPTED_HISTORY_PATH)
model_gateway = ModelGateway()


@app.on_event("startup")
def startup_event() -> None:
    init_db()
    knowledge_base.warmup()
    log.info(
        "RAG ready: %d records, backend=%s",
        knowledge_base.doc_count,
        knowledge_base.runtime_backend,
    )


@app.get("/api/status")
def status() -> dict:
    family_schema = knowledge_base.family_schema()
    return {
        "status": "ok",
        "active_family_id": GENERAL_RAG_ID,
        "active_family_label": GENERAL_RAG_LABEL,

        # Frontend summary counts.
        "families": len(PRIMARY_FAMILY_KEYWORDS),
        "support_docs": len(SUPPORT_DOC_KEYWORDS),
        "partdb_records": family_schema["record_count"],
        "main_features": len(family_schema["main_features"]),
        "constraint_parameters": len(family_schema["constraint_parameters"]),
        "derived_parameters": len(family_schema["derived_parameters"]),
        "materials": len(family_schema["material_suggestions"]),
        "external_examples": len(family_schema["external_examples"]),

        # Keep useful info.
        "rag_mode": "multi-family mechanical RAG",
        "embedding_model": embedding_model_name(),
        "retrieval_backend": "hybrid (keyword + embedding)",

        # optional
        "ollama_available": model_gateway.ollama_available(),
        "openai_available": model_gateway.openai_available(),
        "openrouter_available": model_gateway.openrouter_available(),
        "huggingface_available": model_gateway.huggingface_available(),
    }

@app.get("/api/history")
def history() -> dict:
    return {"items": accepted_history.recent(), "count": accepted_history.count}


@app.post("/api/accept")
def accept(request: AcceptRequest) -> dict:
    try:
        item = accepted_history.add(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "saved", "id": item["id"], "count": accepted_history.count}


@app.get("/api/knowledge")
def list_knowledge() -> dict:
    return {
        "mode": "multi-family structured mechanical RAG",
        "active_family": knowledge_base.family_schema(),
        "documents": knowledge_base.documents(),
    }


@app.get("/api/part-family")
def part_family() -> dict:
    return {
        "family": knowledge_base.family_schema(),
        "records": knowledge_base.part_records(),
    }


@app.get("/api/models")
def list_models() -> dict:
    default_provider = (
        "ollama"
        if model_gateway.ollama_available()
        else (
            "openrouter"
            if model_gateway.openrouter_available()
            else ("huggingface" if model_gateway.huggingface_available() else "openai")
        )
    )
    return {
        "providers": model_gateway.providers(),
        "defaults": {
            "provider": default_provider,
            "ollama_model": model_gateway.default_model("ollama"),
            "openai_model": OPENAI_DEFAULT_MODEL,
            "openrouter_model": OPENROUTER_DEFAULT_MODEL,
            "huggingface_model": HUGGINGFACE_DEFAULT_MODEL,
            
        },
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    text = request.prompt.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    provider = request.provider
    model = request.model or model_gateway.default_model(provider)
    if not model:
        raise HTTPException(status_code=400, detail=f"No model configured for '{provider}'.")
    if provider == "ollama" and model in OLLAMA_DISABLED_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is disabled in this app because it fails in the current Ollama app path.")

    try:
        retrieval_k = min(max(request.top_k, 4), 6) if provider == "ollama" else request.top_k
        memory_text = conversation_user_memory(request.history, text)
        detected_family = detect_family(memory_text)
        engineering_profile = build_engineering_profile(memory_text, detected_family)

        # ── Enhancement modules: pre-generation analysis ──────────────────
        mfg = request.manufacturing
        intent_result    = analyze_intent(text)
        constraint_result = solve_constraints(intent_result.extracted_params | intent_result.defaults_applied, detected_family)
        physics_result   = analyze_physics(intent_result.extracted_params | intent_result.defaults_applied)
        mfg_context      = format_mfg_context(mfg, detected_family)

        # Build intent + constraint context to inject into prompt
        intent_ctx = format_intent_summary(intent_result)
        constraint_ctx = format_constraint_summary(constraint_result)
        physics_ctx = format_physics_summary(physics_result)
        tol_block = generate_tolerance_block(
            intent_result.extracted_params | intent_result.defaults_applied, mfg
        )

        # Augment the prompt text with engineering context
        extra_ctx_parts = []
        if intent_ctx:    extra_ctx_parts.append(intent_ctx)
        if constraint_ctx and not constraint_result.passed:
            extra_ctx_parts.append(constraint_ctx)
        if physics_ctx:   extra_ctx_parts.append(physics_ctx)
        if mfg_context:   extra_ctx_parts.append(mfg_context)
        if tol_block:
            extra_ctx_parts.append(f"[TOLERANCE BLOCK — include at top of code]\n{tol_block}")
        extra_ctx_str = "\n\n".join(extra_ctx_parts)
        augmented_text = f"{text}\n\n{extra_ctx_str}" if extra_ctx_str else text
        # ── End pre-generation ────────────────────────────────────────────

        retrieval_query = memory_text
        if request.disable_rag:
            retrieved = []
            history_hits = []
        else:
            retrieved = knowledge_base.retrieve(retrieval_query, top_k=retrieval_k, selected_doc_ids=request.selected_doc_ids)
            history_hits = accepted_history.retrieve(retrieval_query, top_k=1, family_id=detected_family) if detected_family else []
        combined_context = retrieved + history_hits

        messages = build_messages(augmented_text, request.history, combined_context, provider=provider, family_id=detected_family)
        raw, actual_provider, actual_model, used_fallback = generate_with_fallback(request, messages)
        result = extract_scad(raw)
        result = enforce_mechanical_library_output(text, result, detected_family)
        initial_validation = validate_scad(result, text)

        if actual_provider == "ollama" and not used_fallback:
            validity = initial_validation
            repair_provider = actual_provider
            repair_model = actual_model
            repair_used_fallback = False
        else:
            result, validity, repair_provider, repair_model, repair_used_fallback = auto_repair_result(
                request, text, combined_context, result, initial_validation
            )
            result = enforce_mechanical_library_output(text, result, detected_family)
            validity = validate_scad(result, text)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Service error: {exc}") from exc
    except Exception as exc:
        if (
            locals().get("detected_family") == "bracket_and_motor_mount_reference"
            and re.search(r"\b(l[-_ ]?bracket|angle bracket)\b", locals().get("memory_text", text), re.IGNORECASE)
        ):
            result = sanitize_scad_metadata(deterministic_l_bracket_scad(memory_text))
            validity = validate_scad(result, memory_text)
            family_schema = knowledge_base.family_schema(detected_family)
            design_notes = build_design_notes(memory_text, engineering_profile, family_schema)
            evaluation = evaluate_generation(result, memory_text, validity, family_schema)
            return {
                "result": result,
                "provider": "deterministic",
                "model": "l-bracket-fallback",
                "requested_provider": provider,
                "requested_model": model,
                "rag_disabled": request.disable_rag,
                "used_fallback": True,
                "auto_repaired": False,
                "repair_provider": "deterministic",
                "repair_model": "l-bracket-fallback",
                "retrieved": [{key: value for key, value in hit.items() if key != "text"} for hit in locals().get("combined_context", [])],
                "history_hits": [{key: value for key, value in hit.items() if key != "text"} for hit in locals().get("history_hits", [])],
                "engineering": engineering_profile,
                "part_family": family_schema,
                "design_notes": design_notes,
                "evaluation": evaluation,
                "initial_validation": validity,
                "validation": validity,
            }
        provider_error = _provider_error_response(exc)
        if provider_error:
            raise provider_error from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not result.strip():
        if detected_family == "bearing_housing_reference":
            result = sanitize_scad_metadata(deterministic_pillow_block_scad(memory_text))
            validity = validate_scad(result, memory_text)
            repair_provider = "deterministic"
            repair_model = "bearing-housing-fallback"
            repair_used_fallback = True
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Model {actual_provider}:{actual_model} returned no OpenSCAD code. Try a more specific prompt or a more capable model.",
            )

    hard_fails = _hard_failures(validity)
    if hard_fails:
        if detected_family == "bearing_housing_reference":
            fallback_result = sanitize_scad_metadata(deterministic_pillow_block_scad(memory_text))
            fallback_validity = validate_scad(fallback_result, memory_text)
            fallback_hard_fails = _hard_failures(fallback_validity)
            if not fallback_hard_fails:
                result = fallback_result
                validity = fallback_validity
                repair_provider = "deterministic"
                repair_model = "bearing-housing-fallback"
                repair_used_fallback = True
                hard_fails = []
        if hard_fails:
            labels = ", ".join(item["label"] for item in hard_fails)
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Model {actual_provider}:{actual_model} produced incomplete code after {MAX_REPAIR_ATTEMPTS} repair attempt(s). "
                    f"Failed: {labels}. Try a more capable model or simplify your request."
                ),
            )

    family_schema = knowledge_base.family_schema(detected_family)
    design_notes = build_design_notes(memory_text, engineering_profile, family_schema)
    evaluation = evaluate_generation(result, memory_text, validity, family_schema)

    # ── Enhancement modules: post-generation analysis ─────────────────────
    failure_report  = detect_failures(result, intent_result.extracted_params, detected_family)
    dfm_warnings    = check_dfm(result, mfg)
    tol_table       = get_tolerance_table_html(mfg)

    log_generation_event({
        "request_id": str(uuid.uuid4()),
        "prompt": text[:200],
        "family_id": detected_family,
        "rag_disabled": request.disable_rag,
        "provider": actual_provider,
        "model": actual_model,
        "code_chars": len(result),
        "passed": score_validation(validity),
        "failed": sum(1 for c in validity if not c.get("passed")),
        "total": len(validity),
        "hard_fails": [c["label"] for c in validity if not c.get("passed") and c.get("label") in HARD_FAIL_LABELS],
        "repair_used": bool(repair_used_fallback),
        "error": None,
    })

    return {
        "result": result,
        "provider": actual_provider,
        "model": actual_model,
        "requested_provider": provider,
        "requested_model": model,
        "rag_disabled": request.disable_rag,
        "used_fallback": used_fallback or repair_used_fallback,
        "auto_repaired": score_validation(validity) > score_validation(initial_validation),
        "repair_provider": repair_provider,
        "repair_model": repair_model,
        "retrieved": [{key: value for key, value in hit.items() if key != "text"} for hit in combined_context],
        "history_hits": [{key: value for key, value in hit.items() if key != "text"} for hit in history_hits],
        "engineering": engineering_profile,
        "part_family": family_schema,
        "design_notes": design_notes,
        "evaluation": evaluation,
        "initial_validation": initial_validation,
        "validation": validity,
        # ── Enhancement module outputs ──────────────────────────────────
        "intent": {
            "part_family":     intent_result.part_family,
            "intent_class":    intent_result.intent_class,
            "extracted_params": intent_result.extracted_params,
            "missing_params":  intent_result.missing_params,
            "defaults_applied": intent_result.defaults_applied,
            "is_assembly":     intent_result.is_assembly,
            "confidence":      intent_result.confidence,
            "warnings":        intent_result.warnings,
        },
        "constraints": {
            "passed":  constraint_result.passed,
            "issues":  [
                {"severity": i.severity, "rule_id": i.rule_id, "message": i.message, "fix": i.fix}
                for i in constraint_result.issues
            ],
        },
        "physics": {
            "calculations":    physics_result.calculations,
            "warnings":        physics_result.warnings,
            "recommendations": physics_result.recommendations,
        },
        "mechanical_quality": {
            "score":          failure_report.score,
            "grade":          failure_report.grade,
            "issues":         [
                {"severity": i.severity, "rule_id": i.rule_id, "message": i.message, "suggestion": i.suggestion}
                for i in failure_report.issues
            ],
            "passed_checks":  failure_report.passed_checks,
        },
        "manufacturing": {
            "process":    mfg,
            "dfm_warnings": dfm_warnings,
            "tolerance_table": tol_table,
        },
    }


class ExportStlRequest(BaseModel):
    code: str
    filename: str = "generated_part"


OPENSCAD_TIMEOUT_SEC = int(os.getenv("OPENSCAD_EXPORT_TIMEOUT_SEC", "120"))


@app.post("/api/export-stl")
def export_stl(request: ExportStlRequest) -> FileResponse:
    """
    Render the supplied OpenSCAD source to an STL file using the local
    OpenSCAD CLI and return the binary for download.

    Requires `openscad` to be on PATH (or set OPENSCAD_BIN env-var).
    """
    openscad_bin = os.getenv("OPENSCAD_BIN", "openscad")
    if not shutil.which(openscad_bin):
        raise HTTPException(
            status_code=503,
            detail=(
                f"OpenSCAD CLI not found ('{openscad_bin}'). "
                "Install OpenSCAD and make sure it is on PATH, "
                "or set the OPENSCAD_BIN environment variable."
            ),
        )

    code = request.code.strip()
    if not code:
        raise HTTPException(status_code=400, detail="No OpenSCAD code supplied.")

    safe_stem = re.sub(r"[^\w\-]", "_", request.filename or "generated_part")

    tmp_dir = tempfile.mkdtemp(prefix="openscad_export_")
    try:
        scad_path = Path(tmp_dir) / f"{safe_stem}.scad"
        stl_path  = Path(tmp_dir) / f"{safe_stem}.stl"

        scad_path.write_text(code, encoding="utf-8")

        cmd = [openscad_bin, "--export-format", "stl", "-o", str(stl_path), str(scad_path)]
        log.info("Running: %s", " ".join(cmd))

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=OPENSCAD_TIMEOUT_SEC,
        )

        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
            log.error("OpenSCAD export failed: %s", stderr)
            raise HTTPException(
                status_code=422,
                detail=f"OpenSCAD export failed: {stderr}",
            )

        if not stl_path.exists() or stl_path.stat().st_size == 0:
            raise HTTPException(
                status_code=422,
                detail="OpenSCAD ran but produced an empty or missing STL file.",
            )

        return FileResponse(
            path=str(stl_path),
            media_type="model/stl",
            filename=f"{safe_stem}.stl",
            background=None,   # keep tmp_dir alive until response is sent
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail=f"OpenSCAD export timed out after {OPENSCAD_TIMEOUT_SEC}s.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Unexpected error during STL export")
        raise HTTPException(status_code=500, detail=str(exc)) from exc



@app.get("/api/generation-log")
def generation_log(limit: int = 20) -> dict:
    from validation_and_logging import _EVENTS_LOG
    events = []
    if _EVENTS_LOG.exists():
        lines = _EVENTS_LOG.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines[-400:]):
            try:
                ev = json.loads(line)
                ev.pop("validation", None)
                events.append(ev)
                if len(events) >= limit:
                    break
            except json.JSONDecodeError:
                continue
    return {"events": events, "count": len(events)}


# ── Auth routes ───────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str
    role: str = "student"
    company: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class UpdateProfileRequest(BaseModel):
    name: str
    role: str
    company: str


def _safe_user(user: dict) -> dict:
    return {k: user[k] for k in ("id", "email", "name", "role", "company", "created_at") if k in user}


@app.post("/api/auth/register")
def register(req: RegisterRequest) -> dict:
    if len(req.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters")
    user = create_user(req.email, req.name, req.password, req.role, req.company)
    token = make_token(user["id"])
    return {"token": token, "user": _safe_user(user)}


@app.post("/api/auth/login")
def login(req: LoginRequest) -> dict:
    user = authenticate_user(req.email, req.password)
    token = make_token(user["id"])
    return {"token": token, "user": _safe_user(user)}


@app.get("/api/auth/me")
def me(current_user: dict = Depends(get_current_user)) -> dict:
    return _safe_user(current_user)


@app.put("/api/auth/profile")
def update_profile(req: UpdateProfileRequest, current_user: dict = Depends(get_current_user)) -> dict:
    updated = update_user(current_user["id"], req.name, req.role, req.company)
    return _safe_user(updated)


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
