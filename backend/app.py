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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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


_load_windows_user_env(
    [
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACE_API_KEY",
    ]
)

OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
OPENAI_MODELS = [
    item.strip()
    for item in os.getenv("OPENAI_MODELS", "gpt-5,gpt-5.2,gpt-4.1-mini,gpt-5-mini").split(",")
    if item.strip()
]
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "Mechanical OpenSCAD Copilot")
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
OPENROUTER_MODELS = [
    item.strip()
    for item in os.getenv(
        "OPENROUTER_MODELS",
        "google/gemma-4-31b-it:free,openai/gpt-oss-120b:free",
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
CLARIFICATION_ENABLED = os.getenv("ENABLE_CLARIFICATION", "false").strip().lower() in {"1", "true", "yes", "on"}

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
- Bearing seats, shaft bores, flange bores, and mounting holes are functional features, not decoration.
- Preserve user dimensions and design intent over visual detail.
- For pillow blocks/plummer blocks: do not create only a circular tube or hollow ring on a flat plate.
- A true pillow block needs a footed base plus pedestal/saddle support, integrated bearing boss, paired mounting holes, and either cap/split-line, clamp, ribs, or side-web detail.
- Prefer a compact UCP-style cast housing silhouette; do not create tall external arch ribs or bridge frames around the bearing.
""".strip()

SYSTEM_PROMPT = "\n\n".join(
    [
        "You are  a responsive senior mechanical design copilot specialized in parametric OpenSCAD code generation for mechanical components. You have deep expertise in standard mechanical design practices and design intent. Your goal is to generate correct, editable, and standards-aligned OpenSCAD code based on user prompts describing mechanical design needs. You should ask for clarification when essential parameters are missing, but you should not ask for secondary details that can be reasonably assumed from standards or typical engineering practice. Always prioritize functional design features and user dimensions over visual detail.",
        "Generate one complete .scad file with no markdown, no prose, and no language labels.",
        "The .scad file must contain geometry parameters only. Do not create string metadata variables such as usage, standard_used, material, or notes.",
        "Avoid explanatory section comments about user parameters, catalog defaults, standards, or materials inside the code.",
        "Before generation, ask for clarification when a truly essential main parameter is missing for the requested mechanical family.",
        "When main parameters are present, assume secondary dimensions from relevant mechanical design practice and clearly encode them as named OpenSCAD parameters.",
        "Keep material suggestions and standard explanations out of the OpenSCAD file; the backend chat response will report them after code generation.",
        
        OPENSCAD_CHEATSHEET,
    ]
)

MAIN_PARAMETER_QUESTIONS: dict[str, list[tuple[str, str, list[str]]]] = {
   
    "bearing_housing_reference": [
        ("usage", "What will this bearing housing be used for? Example: 3D printer shaft support, CNC leadscrew, conveyor roller, robot axle, or belt tensioner.", [r"\b(3d printer|printer|cnc|leadscrew|lead screw|conveyor|roller|robot|axle|belt|chain|tension|industrial|fixture|machine|motor|fan)\b"]),
        ("shaft diameter", "Please specify the shaft diameter in mm.", [r"\bshaft\s*(diameter|d)?\s*=?\s*\d+(\.\d+)?\s*mm\b", r"\b\d+(\.\d+)?\s*mm\s+shaft\b"]),
        ("bearing size", "Please specify the bearing series, or bearing outside diameter and width. Example: 6204, 20 mm shaft with 47 mm OD and 14 mm width.", [r"\b(608|6001|6200|6201|6202|6203|6204|6205|6206|6207|6208|6305|ucf\s*\d+|ucfl\s*\d+)\b", r"\bbearing[_ -]?(od|outer|width)\b", r"\b\d+(\.\d+)?\s*mm\s*od\b"]),
        ("housing type", "What housing style do you need: pillow block, square flange, two-bolt flange, take-up/slotted, or simple support block?", [r"\b(pillow|plummer|flange|flanged|ucf|ucfl|take[- ]?up|slotted|support block|bearing block)\b"]),
    ],
    "flange_pipe_fitting_reference": [
        ("central bore", "Please specify the pipe or bore diameter in mm.", [r"\b(bore|pipe|inner)\b", r"\b\d+(\.\d+)?\s*mm\b"]),
        ("bolt pattern", "Please specify the bolt count and bolt-circle diameter.", [r"\bbolt[_ -]?circle\b", r"\bbolt[_ -]?count\b", r"\b\d+\s*(holes|bolts)\b"]),
        ("clearance preference", "Do you want close-fit bolt holes or extra assembly clearance?", [r"\b(clearance|fit)\b"]),
    ],
    "shaft_coupler_reference": [
        ("shaft A diameter", "Please specify the first shaft diameter in mm.", [r"\b\d+(\.\d+)?\s*mm\b.*\bshaft\b", r"\bshaft[_ -]?a\b"]),
        ("shaft B diameter", "Please specify the second shaft diameter in mm.", [r"\bto\b.*\b\d+(\.\d+)?\s*mm\b", r"\bshaft[_ -]?b\b"]),
        ("clearance preference", "Do you want a standard slip-fit bore or a tighter bore clearance?", [r"\b(clearance|slip fit|tight fit|fit)\b"]),
    ],
    "gear_reference": [
        ("gear size", "Please specify module and tooth count, or give pitch diameter and tooth count.", [r"\bmodule\b", r"\bteeth\b", r"\btooth[_ -]?count\b"]),
        ("bore", "Please specify the bore diameter in mm.", [r"\bbore\b", r"\bshaft\b"]),
        ("clearance preference", "Do you want standard backlash or tighter meshing?", [r"\b(backlash|clearance|mesh)\b"]),
    ],
    "enclosure_box_reference": [
        ("overall size", "Please specify enclosure length, width, and height in mm.", [r"\b(length|width|height)\b", r"\b\d+(\.\d+)?\s*x\s*\d+(\.\d+)?\s*x\s*\d+(\.\d+)?\b"]),
        ("wall thickness", "Please specify wall thickness in mm, or ask me to choose a default.", [r"\bwall[_ -]?thickness\b"]),
        ("clearance preference", "Do you want extra internal clearance for electronics and cables?", [r"\b(clearance|cable|electronics|pcb)\b"]),
    ],
}

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
    allow_fallback: bool = True
    history: list[ChatMessage] = Field(default_factory=list)
    enable_clarification: bool = False


class AcceptRequest(BaseModel):
    prompt: str
    code: str
    provider: str | None = None
    model: str | None = None
    selected_doc_ids: list[str] = Field(default_factory=list)


def detect_family(prompt: str, code: str = "") -> str | None:
    return detect_primary_family(f"{prompt}\n{code}")


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


def build_clarification_request(prompt: str, family_id: str | None) -> dict | None:
    if not CLARIFICATION_ENABLED:
        return None
    if not family_id or family_id not in MAIN_PARAMETER_QUESTIONS:
        return None

    questions = []
    if family_id == "bearing_housing_reference":
        lowered = prompt.lower()
        has_usage = bool(re.search(r"\b(3d printer|printer|cnc|leadscrew|lead screw|conveyor|roller|robot|axle|belt|chain|tension|industrial|fixture|machine|motor|fan)\b", lowered))
        has_bearing = bool(re.search(r"\b(608|6001|6200|6201|6202|6203|6204|6205|6206|6207|6208|6305|ucf\s*\d+|ucfl\s*\d+)\b", lowered)) or bool(re.search(r"\bbearing[_ -]?(od|outer|width)\b|\b\d+(\.\d+)?\s*mm\s*od\b", lowered))
        has_shaft = bool(re.search(r"\bshaft\s*(diameter|d)?\s*=?\s*\d+(\.\d+)?\s*mm\b|\b\d+(\.\d+)?\s*mm\s+shaft\b", lowered))
        has_type = bool(re.search(r"\b(pillow|plummer|flange|flanged|ucf|ucfl|take[- ]?up|slotted|support block|bearing block)\b", lowered))

        if not has_usage:
            questions.append({"field": "usage", "question": MAIN_PARAMETER_QUESTIONS[family_id][0][1]})
        elif not has_bearing and not has_shaft:
            questions.append({"field": "bearing size", "question": MAIN_PARAMETER_QUESTIONS[family_id][2][1]})
        elif has_shaft and not has_bearing:
            questions.append({"field": "bearing size", "question": "Which bearing series should I use, or should I choose a standard bearing from the shaft diameter?"})
        elif not has_type:
            questions.append({"field": "housing type", "question": MAIN_PARAMETER_QUESTIONS[family_id][3][1]})
    else:
        for field_name, question, patterns in MAIN_PARAMETER_QUESTIONS.get(family_id, []):
            if not any(re.search(pattern, prompt, re.IGNORECASE) for pattern in patterns):
                questions.append({"field": field_name, "question": question})

    if not questions:
        return None

    return {
        "family_id": family_id,
        "family_label": FAMILY_ID_TO_LABEL.get(family_id, "Mechanical Part"),
        "message": "I need one main mechanical parameter before generating a correct design. I will assume secondary values from mechanical standards after the essentials are known.",
        "questions": questions[:1],
        "assumptions_after_clarification": [
            "reasonable wall thickness from load/use case",
            "hole, bore, and fit clearances from standard mechanical practice",
            "symmetric feature placement where appropriate",
            "material suggestion from usage and duty level",
        ],
    }


def _extract_json_object(raw: str) -> dict | None:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def generate_llm_clarification(
    request: ChatRequest,
    user_prompt: str,
    conversation_context: str,
    rule_clarification: dict,
    engineering_profile: dict,
) -> tuple[dict, str | None, str | None, bool]:
    family_schema = knowledge_base.family_schema(rule_clarification.get("family_id"))
    required_fields = [item.get("field", "unknown") for item in rule_clarification.get("questions", [])]
    material_options = engineering_profile.get("material_suggestions", [])

    messages = [
        {
            "role": "system",
            "content": (
                "You are  a senior mechanical design engineer specialized in parametric OpenSCAD code generation for mechanical components. "
                "Generate clarification questions before CAD generation. Be concise, practical, and reasonable. "
                "Ask exactly one question: the single most important missing main parameter. "
                "Do not ask for secondary parameters that can be assumed from standards. "
                "Use the conversation context as memory, so do not ask for anything already answered. "
                "Return JSON only with keys: message, questions, assumptions_after_clarification, material_suggestions. "
                "questions must be an array containing exactly one object with field and question."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "latest_user_message": user_prompt,
                    "conversation_context_user_messages_only": conversation_context,
                    "active_family": family_schema.get("label"),
                    "missing_main_fields": required_fields,
                    "available_database_features": family_schema.get("main_features", [])[:12],
                    "material_options_from_rules": material_options,
                    "rule_based_questions_to_improve": rule_clarification.get("questions", []),
                    "instruction": (
                        "Rewrite the one rule-based question into a more natural engineering clarification. "
                        "Mention what you can infer or assume after the user answers. "
                        "Ask one question only. If usage is missing, ask for it because material and wall thickness depend on it."
                    ),
                },
                ensure_ascii=False,
            ),
        },
    ]

    raw, actual_provider, actual_model, used_fallback = generate_with_fallback(request, messages)
    parsed = _extract_json_object(raw) or {}
    questions = parsed.get("questions")
    if not isinstance(questions, list) or not questions:
        parsed["questions"] = rule_clarification.get("questions", [])
    else:
        allowed_field = (rule_clarification.get("questions") or [{}])[0].get("field")
        first_question = questions[0] if isinstance(questions[0], dict) else {}
        if allowed_field:
            first_question["field"] = allowed_field
        if not first_question.get("question"):
            first_question["question"] = (rule_clarification.get("questions") or [{}])[0].get("question", "")
        parsed["questions"] = [first_question]
    parsed.setdefault("message", rule_clarification.get("message", "I need a few main parameters before generating this mechanical part."))
    parsed.setdefault(
        "assumptions_after_clarification",
        rule_clarification.get("assumptions_after_clarification", []),
    )
    parsed.setdefault("material_suggestions", material_options)
    parsed["family_id"] = rule_clarification.get("family_id")
    parsed["family_label"] = rule_clarification.get("family_label")
    parsed["generated_by"] = "llm"
    return parsed, actual_provider, actual_model, used_fallback


def _material_scope_for_family(family_id: str | None) -> str:
    if family_id == "bearing_housing_reference":
        return "housing body, pedestal, and mounting feet; not the bearing internals"
    return "generated part body; not purchased bearings, fasteners, or inserts"


def suggest_materials(prompt: str, family_id: str | None = None) -> list[dict]:
    lowered = prompt.lower()
    scope = _material_scope_for_family(family_id)
    bearing_housing = family_id == "bearing_housing_reference"

    def item(material: str, reason: str) -> dict:
        return {
            "material": material,
            "applies_to": scope,
            "reason": reason,
        }

    if any(word in lowered for word in ("industrial", "heavy", "conveyor", "production", "shock", "steel frame")):
        cast_reason = "rigid housing body with good damping for mounted bearing supports" if bearing_housing else "rigid body material with good damping for heavy mechanical parts"
        return [
            item("Cast iron", cast_reason),
            item("Cast steel or low-carbon steel", "better toughness for shock-loaded or welded assemblies"),
        ]
    if any(word in lowered for word in ("cnc", "robot", "lightweight", "aluminum", "fixture")):
        aluminum_reason = "light, machinable housing body for light CNC, robotics, or fixture duty" if bearing_housing else "light, machinable body material for CNC, robotics, or fixture duty"
        return [
            item("6061-T6 aluminum", aluminum_reason),
            item("7075-T6 aluminum", "higher strength body material if weight matters and cost is acceptable"),
        ]
    if any(word in lowered for word in ("print", "printed", "3d", "prototype", "pla", "petg", "nylon")):
        return [
            item("PA-CF / nylon carbon fiber", "best functional printed body choice for heat and creep resistance"),
            item("PETG", "acceptable body material for light prototypes and fit checks"),
        ]
    return [
        item("6061-T6 aluminum", "default body material for light machined mechanical parts"),
        item("Cast iron" if bearing_housing else "Low-carbon steel", "default robust body material for load-bearing mechanical parts"),
        item("PA-CF", "default body material for functional printed prototypes"),
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

        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._items.append(item)
            self._embeddings = None
        return item

    def _history_text(self, item: dict) -> str:
        return f"{item.get('prompt', '')}\n{item.get('code', '')[:1800]}"

    def _ensure_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            texts = [self._history_text(item) for item in self._items]
            self._embeddings = embed_texts(texts, is_query=False) if texts else np.array([])
        return self._embeddings

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        with self._lock:
            if not self._items:
                return []
            try:
                if EMBEDDING_BACKEND == "tfidf":
                    texts = [self._history_text(item) for item in self._items]
                    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                    matrix = vectorizer.fit_transform(texts + [query])
                    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
                else:
                    embeddings = self._ensure_embeddings()
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
                    item = self._items[index]
                    code = item.get("code", "")
                    hits.append(
                        {
                            "id": f"accepted::{item['id']}",
                            "title": f"Accepted Design: {item.get('prompt', '')[:70]}",
                            "source": "accepted-history",
                            "score": round(float(scores[index]), 4),
                            "excerpt": f"Prompt: {item.get('prompt', '')}\nCode:\n{code[:420]}",
                            "text": (
                                "Accepted user-approved design. Prefer this when the new request is similar.\n"
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
                {key: item.get(key) for key in ("id", "created_at", "prompt", "provider", "model")}
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
        "Ask for missing main parameters only when they are essential; assume secondary dimensions from mechanical practice.",
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
    if family_id == "gear_reference":
        family_lines.append(
            "For gear requests, the gears-master reference is authoritative. "
            "Prefer include <D:/Downloads/openscad_copilot (1)/openscad_copilot/gears.scad> and call the matching gears.scad module. "
            "Do not ask for missing gear parameters in the generated OpenSCAD. "
            "For generic requests like 'generate bevel gear' or 'generate worm gear', use the default example values from the gears-master reference and output a complete module call. "
            "Never output a placeholder parameter-request module or echo('Please specify ...'). "
            "For a plain spur gear use spur_gear(..., helix_angle=0, optimized=true). "
            "Use nonzero helix_angle only when the user asks for helical, spiral, or herringbone gears. "
            "Do not hand-build gears from rectangular/cube teeth or lumped cylinders when a gears.scad module exists."
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
    return normalize_library_includes(sanitize_scad_metadata(cleaned.strip()))


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


GEARS_SCAD_PATH = str((BASE_DIR.parent / "gears.scad").resolve()).replace("\\", "/")
GEARS_INCLUDE_RE = re.compile(
    r"^\s*(include|use)\s*<[^>\n]*gears\.scad>\s*;?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_library_includes(code: str) -> str:
    """Make generated gears.scad imports resolvable from browser downloads and temp exports."""
    if "gears.scad" not in code.lower():
        return code
    return GEARS_INCLUDE_RE.sub(f"include <{GEARS_SCAD_PATH}>", code)


def enforce_gears_master_output(prompt: str, code: str, family_id: str | None) -> str:
    """Keep model output intact, but replace non-geometry gear placeholders."""
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
    return [{
        "label": "Validation disabled",
        "passed": True,
        "detail": "Backend validation is disabled; OpenSCAD will be the source of truth.",
        "severity": "warning",
        "category": "validation",
    }]


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


def deterministic_pillow_block_scad(prompt: str) -> str:
    lowered = prompt.lower()
    bearing_key = next((key for key in BEARING_SERIES_DEFAULTS if re.search(rf"\b{re.escape(key)}\b", lowered)), "6204")
    shaft_diameter, bearing_od, bearing_width = BEARING_SERIES_DEFAULTS[bearing_key]

    explicit_shaft = _number_for_pattern(prompt, r"(\d+(?:\.\d+)?)\s*mm\s+shaft")
    if explicit_shaft:
        shaft_diameter = explicit_shaft

    wall_thickness = _number_for_pattern(prompt, r"wall(?:_|\s*)thickness\s*(?:=|of)?\s*(\d+(?:\.\d+)?)") or 8
    mount_hole_diameter = 6.6 if re.search(r"\bm6\b", lowered) else 5.5
    if re.search(r"\bm4\b", lowered):
        mount_hole_diameter = 4.5
    if re.search(r"\bm8\b", lowered):
        mount_hole_diameter = 8.8

    base_thickness = max(7, wall_thickness * 0.9)
    boss_diameter = bearing_od + 2 * wall_thickness
    boss_width = bearing_width + 2 * wall_thickness
    base_length = max(boss_width + 54, bearing_od + 42)
    base_width = boss_diameter + 18
    mount_hole_spacing = base_length - max(22, 4 * mount_hole_diameter)
    foot_pad_length = max(26, mount_hole_diameter * 5)
    foot_pad_width = max(28, mount_hole_diameter * 5)
    foot_pad_thickness = 4
    slot_length = max(16, mount_hole_diameter * 3)
    boss_center_z = base_thickness + boss_diameter / 2 - 4
    pedestal_width = boss_width + 14
    pedestal_depth = base_width - 18
    pedestal_height = boss_center_z - base_thickness + 4

    return f"""$fn = 128;
// Deterministic fallback: compact UCP-style pillow-block bearing housing.
// Generated because the model returned incomplete OpenSCAD.
bearing_series = "{bearing_key}";
shaft_diameter = {shaft_diameter:g};
bearing_od = {bearing_od:g};
bearing_width = {bearing_width:g};
wall_thickness = {wall_thickness:g};
base_thickness = {base_thickness:g};
base_length = {base_length:g};
base_width = {base_width:g};
foot_pad_length = {foot_pad_length:g};
foot_pad_width = {foot_pad_width:g};
foot_pad_thickness = {foot_pad_thickness:g};
mount_hole_diameter = {mount_hole_diameter:g};
mount_hole_spacing = {mount_hole_spacing:g};
slot_length = {slot_length:g};
shaft_clearance = 0.4;
bearing_fit_clearance = 0.15;
boss_diameter = {boss_diameter:g};
boss_width = {boss_width:g};
boss_center_z = {boss_center_z:g};
pedestal_width = {pedestal_width:g};
pedestal_depth = {pedestal_depth:g};
pedestal_height = {pedestal_height:g};
cap_groove_height = 1.2;
cap_groove_z = boss_center_z + boss_diameter * 0.20;

module rounded_slot(length, diameter, height) {{
  hull() {{
    translate([-length / 2, 0, 0])
      cylinder(h = height, d = diameter, center = true);
    translate([ length / 2, 0, 0])
      cylinder(h = height, d = diameter, center = true);
  }}
}}

module pillow_block_bearing_housing() {{
  difference() {{
    union() {{
      // Low footed base with raised pads.
      translate([0, 0, base_thickness / 2])
        cube([base_length, base_width, base_thickness], center = true);
      for (x = [-mount_hole_spacing / 2, mount_hole_spacing / 2])
        translate([x, 0, base_thickness + foot_pad_thickness / 2])
          cube([foot_pad_length, foot_pad_width, foot_pad_thickness], center = true);

      // Solid central pedestal and rounded body, like a compact cast UCP housing.
      translate([0, 0, base_thickness + pedestal_height / 2])
        cube([pedestal_width, pedestal_depth, pedestal_height], center = true);
      hull() {{
        translate([0, 0, base_thickness + 8])
          cube([pedestal_width + 8, pedestal_depth, 10], center = true);
        translate([0, 0, boss_center_z])
          rotate([90, 0, 0])
            cylinder(h = boss_width, d = boss_diameter, center = true);
      }}

      // Integrated horizontal bearing boss blended into the pedestal.
      translate([0, 0, boss_center_z])
        rotate([90, 0, 0])
          cylinder(h = boss_width, d = boss_diameter, center = true);
    }}

    // Bearing pocket and shaft bore through the same horizontal axis.
    translate([0, 0, boss_center_z])
      rotate([90, 0, 0])
        cylinder(h = boss_width + 4, d = bearing_od + bearing_fit_clearance, center = true);
    translate([0, 0, boss_center_z])
      rotate([90, 0, 0])
        cylinder(h = base_width + 8, d = shaft_diameter + shaft_clearance, center = true);

    // Thin subtractive cap split-line groove only; do not add a solid top slab.
    translate([0, 0, cap_groove_z])
      cube([pedestal_width + 10, boss_width + 4, cap_groove_height], center = true);

    // Two foot mounting holes.
    for (x = [-mount_hole_spacing / 2, mount_hole_spacing / 2])
      translate([x, 0, base_thickness / 2])
        rounded_slot(slot_length, mount_hole_diameter, base_thickness + foot_pad_thickness + 8);
  }}
}}

pillow_block_bearing_housing();"""


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
            "rate-limited",
            "rate limited",
            "no endpoints found",
            "temporarily",
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


def _provider_error_response(exc: Exception) -> HTTPException | None:
    message = str(exc)
    lowered = message.lower()
    if "429" in message or "rate-limited" in lowered or "rate limited" in lowered:
        return HTTPException(
            status_code=429,
            detail=(
                "All configured free OpenRouter routes are temporarily rate-limited upstream. "
                "Wait and retry, choose a paid/non-free OpenRouter model if you have credits, "
                "or use Ollama/OpenAI."
            ),
        )
    if "no endpoints found" in lowered:
        return HTTPException(
            status_code=503,
            detail=(
                "OpenRouter has no active endpoint for this model right now. "
                "Pick another OpenRouter model, such as qwen/qwen3-coder:free, or use Ollama/OpenAI."
            ),
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
        retrieval_k = min(request.top_k, 2) if provider == "ollama" else request.top_k
        memory_text = conversation_user_memory(request.history, text)
        detected_family = detect_family(memory_text)
        clarification = build_clarification_request(memory_text, detected_family) if request.enable_clarification else None
        engineering_profile = build_engineering_profile(memory_text, detected_family)
        if clarification:
            clarification_provider = provider
            clarification_model = model
            clarification_used_fallback = False
            try:
                clarification, clarification_provider, clarification_model, clarification_used_fallback = generate_llm_clarification(
                    request, text, memory_text, clarification, engineering_profile
                )
            except Exception as exc:
                log.warning("LLM clarification failed; using deterministic fallback (%s).", exc)
                clarification["generated_by"] = "rules-fallback"
            return {
                "result": "",
                "provider": clarification_provider,
                "model": clarification_model,
                "requested_provider": provider,
                "requested_model": model,
                "used_fallback": clarification_used_fallback,
                "auto_repaired": False,
                "retrieved": [],
                "history_hits": [],
                "engineering": engineering_profile,
                "part_family": knowledge_base.family_schema(detected_family),
                "initial_validation": [],
                "validation": [],
                "needs_clarification": True,
                "clarification": clarification,
            }

        retrieval_query = memory_text
        retrieved = knowledge_base.retrieve(retrieval_query, top_k=retrieval_k, selected_doc_ids=request.selected_doc_ids)
        history_hits = accepted_history.retrieve(retrieval_query, top_k=2)
        combined_context = history_hits + retrieved

        messages = build_messages(text, request.history, combined_context, provider=provider, family_id=detected_family)
        raw, actual_provider, actual_model, used_fallback = generate_with_fallback(request, messages)
        result = extract_scad(raw)
        result = enforce_gears_master_output(text, result, detected_family)
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
            result = enforce_gears_master_output(text, result, detected_family)
            validity = validate_scad(result, text)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Service error: {exc}") from exc
    except Exception as exc:
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

    log_generation_event({
        "request_id": str(uuid.uuid4()),
        "prompt": text[:200],
        "family_id": detected_family,
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
        "used_fallback": used_fallback or repair_used_fallback,
        "auto_repaired": score_validation(validity) > score_validation(initial_validation),
        "repair_provider": repair_provider,
        "repair_model": repair_model,
        "retrieved": [{key: value for key, value in hit.items() if key != "text"} for hit in combined_context],
        "history_hits": [{key: value for key, value in hit.items() if key != "text"} for hit in history_hits],
        "engineering": engineering_profile,
        "part_family": family_schema,
        "design_notes": design_notes,
        "initial_validation": initial_validation,
        "validation": validity,
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


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
