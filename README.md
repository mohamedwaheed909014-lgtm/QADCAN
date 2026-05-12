# OpenSCAD Mechanical Copilot

A RAG-assisted web application that generates parametric OpenSCAD code for
mechanical parts from natural-language prompts. It validates every output,
auto-repairs structural failures, and logs every generation event for inspection.

---

## Project structure

```
openscad_copilot/
│
├── start.sh                  ← one-command startup (installs deps, runs server)
├── requirements.txt          ← Python dependencies
├── .env.example              ← copy to .env and fill in API keys
│
├── backend/                  ← Python server (FastAPI)
│   ├── app.py                ← HTTP routes, prompt assembly, repair loop
│   ├── rag.py                ← document loading, embedding, retrieval engine
│   ├── validation_and_logging.py  ← all validation checks + generation log
│   │
│   ├── docs/                 ← knowledge base (loaded at startup)
│   │   ├── *.json            ← structured family databases (21 files)
│   │   └── *.txt             ← supplementary reference documents (24 files)
│   │
│   ├── accepted_history.jsonl  ← user-accepted designs (auto-created)
│   └── logs/
│       └── generation_events.jsonl  ← structured generation log (auto-created)
│
└── frontend/                 ← static HTML/CSS/JS (served by FastAPI)
    ├── index.html            ← single-page app shell
    ├── styles.css            ← all styles including validation panel
    └── main.js               ← chat logic, rendering, 3D preview
```

---

## How to run

```bash
# 1. Copy and fill in the environment file
cp .env.example .env
# At minimum: set OPENROUTER_API_KEY or OPENAI_API_KEY, or have Ollama running.

# 2. Start the server
bash start.sh

# 3. Open in browser
open http://localhost:8000
```

The server serves the frontend from `/frontend` and exposes all API routes under
`/api/*`. Changing any `.py` file in `backend/` auto-reloads the server (uvicorn
`--reload` is on by default in `start.sh`).

---

## How the system works

### Request lifecycle

```
Browser prompt
      │
      ▼
POST /api/chat  (app.py)
      │
      ├─ 1. Family detection      detect_primary_family(prompt)
      │                           → matches PRIMARY_FAMILY_KEYWORDS → "gear_reference"
      │
      ├─ 2. RAG retrieval         knowledge_base.retrieve(prompt, top_k=4)
      │       rag.py              → cosine similarity (TF-IDF or sentence-transformers)
      │                           → keyword boost / negative penalty / threshold filter
      │                           → returns top docs with text + score
      │
      ├─ 3. Prompt assembly       build_messages(prompt, history, context_hits)
      │       app.py              → system prompt + OpenSCAD cheatsheet
      │                           → family context block (main features, constraints)
      │                           → retrieved knowledge docs (trimmed to token budget)
      │                           → accepted history hits (similar past designs)
      │                           → conversation history
      │                           → user message
      │
      ├─ 4. LLM generation        generate_with_fallback(request, messages)
      │       app.py              → tries primary provider/model
      │                           → falls back to OpenRouter free → OpenAI → Ollama
      │                           → extracts and sanitises OpenSCAD from response
      │
      ├─ 5. Validation            validate_scad_full(code, prompt, family_id)
      │       validation_and_logging.py
      │                           → 16 universal checks (braces, $fn, module call…)
      │                           → derived-formula checks (pitch_d = module×teeth…)
      │                           → 69 family-specific checks (varies by family)
      │                           → returns list[{label, passed, detail, severity, category}]
      │
      ├─ 6. Auto-repair loop      auto_repair_result(…)
      │       app.py              → if any check failed: build repair prompt listing failures
      │                           → re-generate; keep if score improved
      │                           → up to MAX_REPAIR_ATTEMPTS (default 2)
      │
      ├─ 7. Structured log        log_generation_event(event)
      │       validation_and_logging.py
      │                           → appends JSON line to backend/logs/generation_events.jsonl
      │                           → fields: prompt, family_id, model, passed, failed,
      │                                      hard_fails, repair_used, error
      │
      └─ 8. Response JSON → Browser
              {result, validation, retrieved, design_notes, engineering, …}
```

### RAG — how retrieval works

Every file in `backend/docs/` is loaded on startup.

**JSON databases** (`.json` with `"type": "mechanical_parts_database"`) are parsed
into structured `Document` objects. Each part record contributes:
- retrieval keywords (aliases, feature names) → `PART_DATABASE_KEYWORDS`
- negative keywords → penalise this document when unrelated terms appear
- minimum score threshold → documents scoring below this are dropped

**Text files** (`.txt`) are loaded as plain documents and matched by the
`PRIMARY_FAMILY_KEYWORDS` table.

Retrieval pipeline per query:

1. **Negative keyword penalty** — subtract `0.08 × hits` from score for each
   negative keyword found in the query
2. **Cosine similarity** — TF-IDF (fast, always available) or sentence-transformers
   (better recall, requires `pip install sentence-transformers`)
3. **Gap-fraction bonus** — primary family match: `score += (1-score) × 0.60`;
   scores are capped at `1.0` (no score above 1 is possible)
4. **Threshold filter** — hits below `min_score_threshold` are dropped
5. **Family isolation penalty** — documents from unrelated families get `-0.12`

### Validation — three layers

Every generated file passes through three independent check sets:

| Layer | Count | What it checks |
|---|---|---|
| Universal | 16 | Syntax correctness: balanced braces, `$fn` set, no markdown, module present, `PI` not `pi`, `difference()` wraps `union()`… |
| Derived formula | varies | Key mechanical formulas appear: `pitch_d = teeth × pitch / PI`, `boss_diameter = bearing_od + 2*wall_thickness`… |
| Family-specific | 69 total | Mandatory geometry patterns for the detected family (see table below) |

**Severity levels:**

| Severity | Meaning | Triggers repair? |
|---|---|---|
| `hard` | File cannot be valid OpenSCAD | Yes — repair loop fires |
| `error` | Geometry is mechanically wrong | Yes — repair loop fires |
| `warning` | Sub-optimal but not breaking | No — shown to user only |

**Family checks (examples):**

| Family | Key checks |
|---|---|
| Gear | `pitch_d = module × tooth_count`, bore separate from pitch, tooth for-loop, hub present |
| I-beam | Three cube() calls (flanges + web), no single solid bar spanning full height |
| GT2 Pulley | `pitch_d = teeth × belt_pitch / PI`, flanges present, bore through hub |
| Sprocket | `pitch_d = chain_pitch / sin(180/N)`, roller pockets in for-loop |
| Pillow block | Bearing seat uses `bearing_od`, shaft bore uses `shaft_diameter`, pedestal present, no arch-frame |
| Shaft coupler | Two distinct bores (shaft_a, shaft_b), clamp slit defined, screws cross slit |
| Enclosure | Hollow shell (outer minus inner cube), standoffs, lid/lip present |
| NEMA motor mount | 4-bolt square pattern, shaft clearance hole |

### Generation log — how to inspect errors

Every `/api/chat` call writes one line to `backend/logs/generation_events.jsonl`:

```json
{
  "timestamp": "2026-05-11T14:32:11Z",
  "prompt": "design a 20-tooth GT2 pulley for a 5 mm shaft",
  "family_id": "pulley_belt_drive_reference",
  "provider": "openrouter",
  "model": "google/gemma-4-31b-it:free",
  "code_chars": 1842,
  "passed": 19,
  "failed": 2,
  "total": 21,
  "hard_fails": [],
  "repair_used": false,
  "error": null
}
```

**Live tail:** `tail -f backend/logs/generation_events.jsonl | python3 -m json.tool`

**In the UI:** click **Refresh** in the Generation Log panel (bottom of the page)
to load the last 20 events from `GET /api/generation-log`.

**Common failure patterns to look for:**

| `hard_fails` value | Cause | Fix |
|---|---|---|
| `"Balanced braces { }"` | LLM output was truncated | Increase `OLLAMA_MAX_TOKENS` or use a larger model |
| `"Module called at end"` | LLM added explanatory prose after the code | Lower temperature, or the sanitiser missed it |
| `"3D primitives used"` | LLM returned only comments | Prompt is ambiguous; add more context |
| (empty, `failed > 0`) | Soft warnings only | Review warnings in UI Validity Signals panel |

---

## Customising knowledge

### Add a new part family

1. Copy `FAMILY_DATABASE_TEMPLATE.json` from the repo root into `backend/docs/`
2. Rename it to `<family_id>.json` matching the `FAMILY_ID_TO_SHORT_KEY` entry in `rag.py`
3. Fill in `metadata.retrieval.primary_keywords`, `negative_keywords`, and `parts[]`
4. Add the family ID to `PRIMARY_FAMILY_KEYWORDS` and `FAMILY_ID_TO_SHORT_KEY` in `rag.py`
5. Add family-specific validation rules to `FAMILY_VALIDATION_RULES` in `validation_and_logging.py`
6. Restart the server — the document loads automatically

### Improve retrieval for an existing family

- **Too many false positives** → add terms to `metadata.retrieval.negative_keywords`
- **Missing true matches** → add terms to `metadata.retrieval.primary_keywords`
- **Irrelevant docs appearing** → raise `metadata.retrieval.min_score_threshold` (try `0.18–0.25`)

### Accept a good design to history

Click **Accept** after a generation. The design is saved to `accepted_history.jsonl`
and re-surfaced automatically for similar future prompts via semantic search.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Main generation endpoint |
| `GET`  | `/api/status` | Server health, family/doc counts |
| `GET`  | `/api/models` | Available providers and models |
| `GET`  | `/api/knowledge` | All loaded documents with excerpts |
| `POST` | `/api/accept` | Save an accepted design to history |
| `GET`  | `/api/history` | Recent accepted designs |
| `POST` | `/api/export-stl` | Render OpenSCAD to STL (requires OpenSCAD CLI) |
| `GET`  | `/api/generation-log` | Last N generation events from the log file |
