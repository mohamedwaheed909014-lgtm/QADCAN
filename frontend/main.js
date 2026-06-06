/* ═══════════════════════════════════════════════════════════════════
   OpenSCAD Mechanical Copilot — main.js
   Chat-native refinement · OpenSCAD-only enforcement · RAG status
═══════════════════════════════════════════════════════════════════ */

// ── DOM ─────────────────────────────────────────────────────────────
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

const promptInput    = document.querySelector('#prompt');
const providerSelect = document.querySelector('#provider');
const modelSelect    = document.querySelector('#model');
const temperatureInput = document.querySelector('#temperature');
const allowFallback  = document.querySelector('#allow-fallback');
const temperatureVal = document.querySelector('#temperature-value');
const chatWindow     = document.querySelector('#chat-window');
const statusDot      = document.querySelector('#status-dot');
const statusText     = document.querySelector('#status-text');
const codeOutput     = document.querySelector('#code-output');
const codeFilename   = document.querySelector('#code-filename');
const referencesEl   = document.querySelector('#references');
const validationEl   = document.querySelector('#validation');
const materialsEl    = document.querySelector('#materials');
const submitBtn      = document.querySelector('#submit-button');
const acceptBtn      = document.querySelector('#accept-button');
const copyBtn        = document.querySelector('#copy-button');
const downloadBtn    = document.querySelector('#download-button');
const exportStlBtn   = document.querySelector('#export-stl-button');
const previewViewport = document.querySelector('#preview-viewport');
const previewCanvas  = document.querySelector('#preview-canvas');
const previewStatus  = document.querySelector('#preview-status');
const previewEmpty   = document.querySelector('#preview-empty');
const renderPreviewBtn = document.querySelector('#render-preview-button');
const resetPreviewBtn = document.querySelector('#reset-preview-button');
const autoPreviewInput = document.querySelector('#auto-preview');
const docCountEl     = document.querySelector('#doc-count');
const ragModeEl      = document.querySelector('#rag-mode');
const knowledgeFilesEl = document.querySelector('#knowledge-files');
const knowledgeSelectedEl = document.querySelector('#knowledge-selected');
const selectAllKnowledgeBtn = document.querySelector('#select-all-knowledge');
const clearKnowledgeBtn = document.querySelector('#clear-knowledge');
const disableRagInput = document.querySelector('#disable-rag');
const quickPrompts   = document.querySelector('#quick-prompts');
const clearBtn       = document.querySelector('#clear-button');
const promptTabs     = document.querySelector('#prompt-tabs');
const recentPromptsEl = document.querySelector('#recent-prompts');
const promptScoreEl  = document.querySelector('#prompt-score');
const promptChecksEl = document.querySelector('#prompt-checks');
const promptTools    = document.querySelector('#prompt-tools');
const resultSummary  = document.querySelector('#result-summary');
const codeStat       = document.querySelector('#code-stat');

// ── State ────────────────────────────────────────────────────────────
let history      = [];   // [{role, content}] full conversation
let lastCode     = '';
let lastPrompt   = '';
let lastGeneration = null;
let providerMap  = new Map();
let defaultModels = {};
let knowledgeFiles = [];
let selectedKnowledge = new Set();
let previewScene = null;
let previewCamera = null;
let previewRenderer = null;
let previewControls = null;
let previewMesh = null;
let previewRenderToken = 0;
let lastPreviewGeometry = null;
const PROVIDER_STORAGE_KEY = 'cadies-provider-v3';
const MODEL_STORAGE_KEY = 'cadies-model-v3';
const RECENT_PROMPTS_KEY = 'cadies-recent-prompts-v1';
const DISABLE_RAG_STORAGE_KEY = 'cadies-disable-rag-v1';

const PROMPT_GROUPS = {
  truss: [
    'standard Fink roof truss span 4200 mm rise 900 mm',
    'Pratt bridge truss span 6000 mm height 900 mm 8 bays',
    'Warren truss span 3000 mm height 500 mm 8 bays',
    'Queen-post roof truss span 4800 mm rise 1200 mm',
  ],
  motion: [
    'GT2 20-tooth timing pulley with 5 mm bore and set screw',
    'clamp-style shaft coupler for two 8 mm shafts',
    'stepped motor shaft with bearing seats, keyway, and threaded end',
    'print-in-place hinge with 5 knuckles and clearance gaps',
  ],
  fixture: [
    'MG996R U-bracket with M3 holes and cable relief',
    'NEMA23 L-bracket with slotted base holes',
    'drill jig with two bushings and workpiece stop',
    '60 mm fan duct mount with rounded airflow opening and screw bosses',
  ],
};

// ── Helpers ──────────────────────────────────────────────────────────
function escHtml(s) {
  return s.replaceAll('&','&amp;').replaceAll('<','&lt;')
          .replaceAll('>','&gt;').replaceAll('"','&quot;');
}

function setStatus(msg, tone = 'neutral') {
  statusText.textContent    = msg;
  statusText.dataset.tone   = tone;
  statusDot.dataset.tone    = tone;
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function getRecentPrompts() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_PROMPTS_KEY) || '[]').filter(Boolean).slice(0, 6);
  } catch {
    return [];
  }
}

function saveRecentPrompt(text) {
  const clean = text.trim();
  if (!clean) return;
  const next = [clean, ...getRecentPrompts().filter(item => item !== clean)].slice(0, 6);
  localStorage.setItem(RECENT_PROMPTS_KEY, JSON.stringify(next));
  renderRecentPrompts();
}

function renderRecentPrompts() {
  const recent = getRecentPrompts();
  if (!recent.length) {
    recentPromptsEl.className = 'recent-prompts empty-knowledge';
    recentPromptsEl.textContent = 'No recent prompts yet.';
    return;
  }
  recentPromptsEl.className = 'recent-prompts';
  recentPromptsEl.innerHTML = recent.map(item => `
    <button type="button" class="recent-prompt" title="${escHtml(item)}">${escHtml(item)}</button>
  `).join('');
}

function renderPromptGroup(group = 'truss') {
  const prompts = PROMPT_GROUPS[group] || PROMPT_GROUPS.truss;
  quickPrompts.innerHTML = prompts.map(text => (
    `<button type="button" class="chip">${escHtml(text)}</button>`
  )).join('');
  promptTabs?.querySelectorAll('.prompt-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.group === group);
  });
}

function usePrompt(text, decorate = true) {
  promptInput.value = decorate
    ? `Design a ${text.trim()} in parametric OpenSCAD with clearly named editable parameters, material suggestions, assumptions, and review limits.`
    : text.trim();
  autoResize(promptInput);
  updatePromptCoach();
  promptInput.focus();
}

function analyzePrompt(text) {
  const lower = text.toLowerCase();
  return [
    { label: 'part family', ok: /(truss|shaft|coupler|pulley|sprocket|gear|bracket|mount|enclosure|flange|jig|fixture|hinge|knob|insert|carriage|adapter|clamp|fan)/.test(lower) },
    { label: 'main dimensions', ok: /\b\d+(\.\d+)?\s*(mm|cm|m|inch|inches|in|ft|feet|tooth|teeth|bays|panels)\b/.test(lower) },
    { label: 'usage or duty', ok: /(roof|bridge|walkway|display|prototype|motor|servo|cnc|pipe|fan|bearing|load|light|heavy|timber|steel|aluminum)/.test(lower) },
    { label: 'features', ok: /(hole|bore|slot|keyway|gusset|web|flange|hub|boss|thread|knurl|standoff|lid|post|bearing|set screw|chamfer|groove)/.test(lower) },
    { label: 'documentation', ok: /(material|bom|assumption|derivation|standard|iso|ansi|review|limit|notes?)/.test(lower) },
  ];
}

function updatePromptCoach() {
  const checks = analyzePrompt(promptInput.value.trim());
  const score = checks.filter(item => item.ok).length;
  promptScoreEl.textContent = `${score}/${checks.length}`;
  promptScoreEl.dataset.tone = score >= 4 ? 'success' : score >= 2 ? 'working' : 'neutral';
  promptChecksEl.innerHTML = checks.map(item => `
    <span class="coach-pill ${item.ok ? 'ok' : ''}">${item.ok ? '✓' : '+'} ${item.label}</span>
  `).join('');
}

function codeMetrics(code) {
  const lines = code ? code.split(/\r?\n/).length : 0;
  const modules = [...code.matchAll(/\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/g)].map(m => m[1]);
  const params = [...code.matchAll(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^=]/gm)].map(m => m[1]);
  return { lines, modules: [...new Set(modules)], params: [...new Set(params)] };
}

function renderResultSummary(code, data = {}) {
  if (!code) {
    resultSummary.hidden = true;
    resultSummary.innerHTML = '';
    codeStat.textContent = '0 lines';
    return;
  }
  const metrics = codeMetrics(code);
  const validation = data.validation || [];
  const passed = validation.filter(item => item.passed).length;
  const total = validation.length;
  const family = data.engineering?.active_database_family || data.part_family?.label || data.part_family?.family_label || 'mechanical part';
  codeStat.textContent = `${metrics.lines} lines`;
  resultSummary.hidden = false;
  resultSummary.innerHTML = `
    <div class="result-item"><strong>${escHtml(String(metrics.lines))}</strong><span>code lines</span></div>
    <div class="result-item"><strong>${escHtml(String(metrics.modules.length))}</strong><span>modules</span></div>
    <div class="result-item"><strong>${escHtml(String(metrics.params.length))}</strong><span>parameters</span></div>
    <div class="result-item"><strong>${total ? `${passed}/${total}` : '-'}</strong><span>checks</span></div>
    <div class="result-family">${escHtml(family)}</div>
  `;
}

function stripFences(code) {
  return code.replace(/^```[\w]*\n?/gm, '').replace(/^```\s*$/gm, '').trim();
}

function formatDesignNotes(notes) {
  if (!notes) return '';

  const lines = [];
  const standards = notes.standards_summary || [];
  if (standards.length) {
    lines.push('Standard used:');
    standards.slice(0, 3).forEach(item => {
      lines.push(`- ${item.name || 'Mechanical design practice'}: ${item.summary || 'relevant proportions, clearances, and feature constraints'}`);
    });
  }

  return lines.join('\n');
}

// 3D preview
function initPreview() {
  if (previewRenderer) return;

  previewScene = new THREE.Scene();
  previewScene.background = new THREE.Color(0x182232);

  previewCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
  previewCamera.position.set(120, 95, 140);

  previewRenderer = new THREE.WebGLRenderer({
    canvas: previewCanvas,
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: true,
  });
  previewRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  previewRenderer.outputColorSpace = THREE.SRGBColorSpace;

  previewControls = new OrbitControls(previewCamera, previewCanvas);
  previewControls.enableDamping = true;
  previewControls.dampingFactor = 0.08;
  previewControls.screenSpacePanning = true;
  previewControls.target.set(0, 0, 0);

  previewScene.add(new THREE.HemisphereLight(0xf8efe4, 0x192536, 2.4));

  const key = new THREE.DirectionalLight(0xffffff, 3.2);
  key.position.set(100, 120, 80);
  previewScene.add(key);

  const fill = new THREE.DirectionalLight(0x75d6c9, 1.1);
  fill.position.set(-100, -60, 90);
  previewScene.add(fill);

  const grid = new THREE.GridHelper(240, 24, 0x5f7892, 0x34485c);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.5;
  previewScene.add(grid);

  const resizePreview = () => {
    const rect = previewViewport.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    previewRenderer.setSize(width, height, false);
    previewCamera.aspect = width / height;
    previewCamera.updateProjectionMatrix();
  };

  new ResizeObserver(resizePreview).observe(previewViewport);
  resizePreview();

  function animatePreview() {
    requestAnimationFrame(animatePreview);
    previewControls.update();
    previewRenderer.render(previewScene, previewCamera);
  }
  animatePreview();
}

function setPreviewStatus(text, tone = 'neutral') {
  previewStatus.textContent = text;
  previewStatus.dataset.tone = tone;
}

function clearPreviewMesh() {
  if (!previewMesh) return;
  previewScene?.remove(previewMesh);
  previewMesh.geometry.dispose();
  previewMesh.material.dispose();
  previewMesh = null;
  lastPreviewGeometry = null;
}

function resetPreviewCamera() {
  if (!lastPreviewGeometry || !previewCamera || !previewControls) return;

  lastPreviewGeometry.computeBoundingSphere();
  const sphere = lastPreviewGeometry.boundingSphere;
  const radius = Math.max(sphere?.radius || 40, 20);
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(previewCamera.fov * 0.5));

  previewCamera.near = Math.max(0.1, distance / 1000);
  previewCamera.far = distance * 10;
  previewCamera.position.set(distance * 0.72, distance * 0.58, distance * 0.86);
  previewCamera.updateProjectionMatrix();
  previewControls.target.set(0, 0, 0);
  previewControls.update();
}

function preparePreviewGeometry(geometry) {
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();

  const center = new THREE.Vector3();
  geometry.boundingBox.getCenter(center);
  geometry.translate(-center.x, -center.y, -center.z);

  geometry.rotateX(-Math.PI / 2);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

async function renderPreview() {
  if (!lastCode) return;

  const token = ++previewRenderToken;
  initPreview();
  renderPreviewBtn.disabled = true;
  resetPreviewBtn.disabled = true;
  previewEmpty.hidden = false;
  previewEmpty.classList.add('is-loading');
  previewEmpty.innerHTML = '<strong>Rendering STL</strong><span>OpenSCAD is producing the mesh.</span>';
  setPreviewStatus('rendering via OpenSCAD...', 'working');

  try {
    const res = await fetch('/api/export-stl', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: lastCode, filename: 'preview_part' }),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `Server error ${res.status}`);
    }

    const loader = new STLLoader();
    const geometry = preparePreviewGeometry(loader.parse(await res.arrayBuffer()));
    if (token !== previewRenderToken) return;

    const material = new THREE.MeshStandardMaterial({
      color: 0xd57a35,
      metalness: 0.18,
      roughness: 0.42,
      side: THREE.DoubleSide,
    });

    clearPreviewMesh();
    previewMesh = new THREE.Mesh(geometry, material);
    previewScene.add(previewMesh);
    lastPreviewGeometry = geometry;
    resetPreviewCamera();

    previewEmpty.hidden = true;
    setPreviewStatus('live STL mesh', 'success');
    resetPreviewBtn.disabled = false;
  } catch (err) {
    clearPreviewMesh();
    previewEmpty.hidden = false;
    previewEmpty.innerHTML = `<strong>Preview unavailable</strong><span>${escHtml(err.message)}</span>`;
    setPreviewStatus('preview failed', 'error');
  } finally {
    if (token === previewRenderToken) {
      renderPreviewBtn.disabled = !lastCode;
      resetPreviewBtn.disabled = !previewMesh;
      previewEmpty.classList.remove('is-loading');
    }
  }
}

function resetPreviewState(message = 'waiting for generated geometry') {
  previewRenderToken++;
  clearPreviewMesh();
  renderPreviewBtn.disabled = !lastCode;
  resetPreviewBtn.disabled = true;
  previewEmpty.hidden = false;
  previewEmpty.classList.remove('is-loading');
  previewEmpty.innerHTML = '<strong>No model rendered yet</strong><span>Generate OpenSCAD code to inspect the mesh here.</span>';
  setPreviewStatus(message, 'neutral');
}

// ── Chat window ──────────────────────────────────────────────────────
function removeWelcome() {
  document.querySelector('.welcome-state')?.remove();
}

function addMessage(role, content = '', meta = '') {
  removeWelcome();
  const art    = document.createElement('article');
  art.className = `message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  if (role === 'assistant') {
    const pre = document.createElement('pre');
    pre.innerHTML = escHtml(content);
    bubble.appendChild(pre);
  } else {
    bubble.textContent = content;
  }
  art.appendChild(bubble);
  if (meta) {
    const m = document.createElement('div');
    m.className   = 'message-meta';
    m.textContent = meta;
    art.appendChild(m);
  }
  chatWindow.appendChild(art);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return art;
}

function addTyping() {
  removeWelcome();
  const art = document.createElement('article');
  art.className = 'message assistant loading';
  art.innerHTML = `<div class="message-bubble">
    <div class="typing-dots"><span></span><span></span><span></span></div>
    <span>Generating…</span>
  </div>`;
  chatWindow.appendChild(art);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return art;
}

// ── Error message bubble ───────────────────────────────────────────────────────
function addErrorMessage(detail) {
  removeWelcome();
  const art = document.createElement('article');
  art.className = 'message assistant error-message';
  art.innerHTML = `
    <div class="message-bubble error-bubble">
      <div class="error-icon">⚠</div>
      <div class="error-body">
        <strong>Generation failed</strong>
        <p>${escHtml(detail)}</p>
      </div>
    </div>`;
  chatWindow.appendChild(art);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ── Code panel ───────────────────────────────────────────────────────
function setCode(raw) {
  lastCode = stripFences(raw || '');
  codeOutput.textContent   = lastCode || '// No code generated.';
  codeFilename.textContent = lastCode ? 'generated_part.scad' : 'no file yet';
  renderResultSummary(lastCode);
  acceptBtn.disabled       = !lastCode;
  copyBtn.disabled         = !lastCode;
  downloadBtn.disabled     = !lastCode;
  exportStlBtn.disabled    = !lastCode;
  renderPreviewBtn.disabled = !lastCode;

  if (!lastCode) {
    resetPreviewState('waiting for generated geometry');
  } else if (autoPreviewInput.checked) {
    renderPreview();
  } else {
    resetPreviewState('ready to render');
  }
}

// ── RAG panels ───────────────────────────────────────────────────────
function renderReferences(items) {
  if (!items?.length) {
    referencesEl.className   = 'empty-state';
    referencesEl.textContent = 'No retrieval results returned.';
    return;
  }
  referencesEl.className = 'stack-list';
  referencesEl.innerHTML = items.map(i => `
    <article class="stack-card">
      <div class="stack-card-top">
        <strong>${escHtml(i.title)}</strong>
        <span>score ${i.score}</span>
      </div>
      <div class="stack-card-source">${escHtml(i.source || 'local knowledge')}</div>
      <p>${escHtml(i.excerpt || '')}</p>
    </article>`).join('');
}

const SEV_META = {
  hard:    { icon: '\u2717', label: 'hard fail',  cls: 'sev-hard'    },
  error:   { icon: '\u2717', label: 'error',      cls: 'sev-error'   },
  warning: { icon: '\u26a0', label: 'warning',    cls: 'sev-warning' },
};
const CAT_LABELS = { universal: 'Universal', family: 'Family-specific', derived: 'Derived formula' };

function renderEvaluationSummary(evaluation) {
  if (!evaluation) return '';
  const cards = ['validity', 'parametricity', 'usability'].map(key => {
    const item = evaluation[key] || {};
    const label = key === 'parametricity' ? 'Parametricity' : key[0].toUpperCase() + key.slice(1);
    const score = Number.isFinite(item.score) ? item.score : 0;
    const detail = `${item.passed ?? 0}/${item.total ?? 0} checks${item.blocking ? `, ${item.blocking} blocking` : ''}`;
    return `<article class="evaluation-card">
      <span>${label}</span>
      <strong>${score}%</strong>
      <small>${detail}</small>
    </article>`;
  }).join('');
  return `<div class="evaluation-grid">${cards}</div>`;
}

function renderValidation(items, evaluation = null) {
  if (!items?.length) {
    validationEl.className   = 'empty-state';
    validationEl.textContent = 'Validation checks will appear after generation.';
    return;
  }

  const passed   = items.filter(i => i.passed);
  const failed   = items.filter(i => !i.passed);
  const hardF    = failed.filter(i => i.severity === 'hard' || i.severity === 'error');
  const summCls  = hardF.length ? 'val-summary fail' : failed.length ? 'val-summary warn' : 'val-summary pass';

  const summary = `<div class="${summCls}">
    <span class="val-score">${passed.length}\u202f/\u202f${items.length} passed</span>
    ${hardF.length  ? `<span class="val-badge">${hardF.length} blocking</span>` : ''}
    ${failed.length && !hardF.length ? `<span class="val-badge">${failed.length} warning</span>` : ''}
  </div>`;

  const groups = {};
  for (const item of items) {
    const cat = item.category || 'universal';
    (groups[cat] = groups[cat] || []).push(item);
  }

  const groupsHtml = Object.entries(groups).map(([cat, catItems]) => {
    const catPassed = catItems.filter(i => i.passed).length;
    const rows = catItems.map(item => {
      const sev     = item.severity || (item.passed ? 'info' : 'warning');
      const meta    = SEV_META[sev] || SEV_META.warning;
      const rowCls  = item.passed ? 'chk-pass' : `chk-fail ${meta.cls}`;
      return `<article class="chk-row ${rowCls}">
        <span class="chk-icon">${item.passed ? '\u2713' : meta.icon}</span>
        <div class="chk-body">
          <strong>${escHtml(item.label)}</strong>
          <p>${escHtml(item.detail || '')}</p>
        </div>
        <span class="chk-status">${item.passed ? 'pass' : meta.label}</span>
      </article>`;
    }).join('');
    return `<div class="val-group">
      <div class="val-grp-hd">
        <span>${escHtml(CAT_LABELS[cat] || cat)}</span>
        <span class="val-grp-sc">${catPassed}/${catItems.length}</span>
      </div>
      ${rows}
    </div>`;
  }).join('');

  validationEl.className = 'val-panel';
  validationEl.innerHTML = renderEvaluationSummary(evaluation) + summary + groupsHtml;
}

function renderMaterials(notes) {
  const materials = notes?.material_suggestions || [];
  if (!materials.length) {
    materialsEl.className = 'empty-state';
    materialsEl.textContent = 'No material suggestions returned.';
    return;
  }

  const materialScope = notes.material_scope || 'generated part body, not purchased bearings or fasteners';
  materialsEl.className = 'materials-panel';
  materialsEl.innerHTML = `
    <div class="material-scope">${escHtml(materialScope)}</div>
    <div class="material-list">
      ${materials.slice(0, 5).map((item, index) => {
        const material = item.material || `Material ${index + 1}`;
        const reason = item.reason || item.notes || 'Chosen from usage and duty level.';
        const appliesTo = item.applies_to ? `<span>${escHtml(item.applies_to)}</span>` : '';
        const standard = item.standard_basis ? `<small>${escHtml(item.standard_basis)}</small>` : '';
        return `<article class="material-card">
          <div class="material-card-top">
            <strong>${escHtml(material)}</strong>
            ${appliesTo}
          </div>
          <p>${escHtml(reason)}</p>
          ${standard}
        </article>`;
      }).join('')}
    </div>`;
}

// ── Enhancement panel renderers ──────────────────────────────────────────────
const intentPanelEl    = document.querySelector('#intent-panel');
const constraintPanelEl= document.querySelector('#constraint-panel');
const physicsPanelEl   = document.querySelector('#physics-panel');
const mfgPanelEl       = document.querySelector('#mfg-panel');
const qualityPanelEl   = document.querySelector('#quality-panel');
const mfgSelect        = document.querySelector('#mfg-select');

function renderIntentPanel(intent) {
  if (!intent || !intentPanelEl) return;
  const family = (intent.part_family || 'unknown').replace(/_reference$/, '').replace(/_/g, ' ');
  const params = Object.entries(intent.extracted_params || {});
  const defaults = Object.entries(intent.defaults_applied || {});
  const missing = intent.missing_params || [];

  intentPanelEl.className = 'intent-panel';
  intentPanelEl.innerHTML = `
    <div class="intent-row">
      <span class="intent-badge">${escHtml(family)}</span>
      <span class="intent-class">${escHtml(intent.intent_class || 'functional')}</span>
      ${intent.is_assembly ? '<span class="intent-badge assembly">assembly</span>' : ''}
      <span class="intent-conf">confidence ${Math.round((intent.confidence || 0) * 100)}%</span>
    </div>
    ${params.length ? `<div class="intent-params">
      <strong>Extracted:</strong>
      ${params.map(([k,v]) => `<span class="intent-param">${escHtml(k)} = ${escHtml(String(v))}</span>`).join('')}
    </div>` : ''}
    ${defaults.length ? `<div class="intent-params defaults">
      <strong>Defaults applied:</strong>
      ${defaults.map(([k,v]) => `<span class="intent-param default">${escHtml(k)} = ${escHtml(String(v))}</span>`).join('')}
    </div>` : ''}
    ${missing.length ? `<div class="intent-missing">⚠️ Missing: ${missing.map(escHtml).join(', ')}</div>` : ''}
    ${(intent.warnings||[]).map(w => `<div class="intent-warning">${escHtml(w)}</div>`).join('')}
  `;
}

function renderConstraintPanel(constraints) {
  if (!constraints || !constraintPanelEl) return;
  const issues = constraints.issues || [];
  if (!issues.length) {
    constraintPanelEl.className = 'constraint-panel pass';
    constraintPanelEl.innerHTML = '<div class="constraint-pass">✅ All mechanical constraints passed.</div>';
    return;
  }
  constraintPanelEl.className = 'constraint-panel';
  constraintPanelEl.innerHTML = issues.map(issue => {
    const icon = issue.severity === 'error' ? '❌' : issue.severity === 'warning' ? '⚠️' : 'ℹ️';
    return `<div class="constraint-issue sev-${issue.severity}">
      <div class="constraint-header">${icon} <span class="rule-id">[${escHtml(issue.rule_id)}]</span> ${escHtml(issue.message)}</div>
      ${issue.fix ? `<div class="constraint-fix">→ Fix: ${escHtml(issue.fix)}</div>` : ''}
    </div>`;
  }).join('');
}

function renderPhysicsPanel(physics) {
  if (!physics || !physicsPanelEl) return;
  const calcs = Object.entries(physics.calculations || {});
  if (!calcs.length && !(physics.warnings||[]).length) {
    physicsPanelEl.className = 'empty-state';
    physicsPanelEl.textContent = 'No physics calculations for this part.';
    return;
  }
  physicsPanelEl.className = 'physics-panel';
  physicsPanelEl.innerHTML = `
    ${calcs.length ? `<div class="physics-calcs">
      ${calcs.map(([k,v]) => `<div class="physics-row">
        <span class="physics-key">${escHtml(k.replace(/_/g,' '))}</span>
        <span class="physics-val">${escHtml(String(v))}</span>
      </div>`).join('')}
    </div>` : ''}
    ${(physics.warnings||[]).map(w => `<div class="physics-warning">⚠️ ${escHtml(w)}</div>`).join('')}
    ${(physics.recommendations||[]).map(r => `<div class="physics-rec">→ ${escHtml(r)}</div>`).join('')}
  `;
}

function renderMfgPanel(manufacturing) {
  if (!manufacturing || !mfgPanelEl) return;
  const warnings = manufacturing.dfm_warnings || [];
  const tol = manufacturing.tolerance_table || [];
  mfgPanelEl.className = 'mfg-panel';
  mfgPanelEl.innerHTML = `
    <div class="mfg-process">Process: <strong>${escHtml(manufacturing.process || 'fdm').toUpperCase()}</strong></div>
    ${warnings.length ? `<div class="mfg-warnings">
      ${warnings.map(w => `<div class="mfg-warning">⚠️ ${escHtml(w)}</div>`).join('')}
    </div>` : '<div class="mfg-ok">✅ No DFM issues detected.</div>'}
    ${tol.length ? `<table class="tol-table">
      <tr><th>Fit type</th><th>Offset (mm)</th><th>Description</th></tr>
      ${tol.map(t => `<tr>
        <td>${escHtml(t.fit_type)}</td>
        <td class="tol-val">${t.offset_mm >= 0 ? '+' : ''}${t.offset_mm}</td>
        <td>${escHtml(t.description)}</td>
      </tr>`).join('')}
    </table>` : ''}
  `;
}

function renderQualityPanel(quality) {
  if (!quality || !qualityPanelEl) return;
  const score = quality.score ?? 0;
  const grade = quality.grade || 'F';
  const gradeColor = grade === 'A' ? 'success' : grade === 'B' ? 'working' : grade === 'C' ? 'warning' : 'error';
  const issues = quality.issues || [];
  qualityPanelEl.className = 'quality-panel';
  qualityPanelEl.innerHTML = `
    <div class="quality-header">
      <div class="quality-score-circle" data-tone="${gradeColor}">${grade}</div>
      <div class="quality-score-num">${score}/100</div>
      <div class="quality-score-label">Mechanical Quality</div>
    </div>
    ${issues.map(i => {
      const icon = i.severity === 'error' ? '❌' : i.severity === 'warning' ? '⚠️' : 'ℹ️';
      return `<div class="quality-issue sev-${i.severity}">
        ${icon} <span class="rule-id">[${escHtml(i.rule_id)}]</span> ${escHtml(i.message)}
        ${i.suggestion ? `<div class="quality-fix">→ ${escHtml(i.suggestion)}</div>` : ''}
      </div>`;
    }).join('')}
    ${(quality.passed_checks||[]).length ? `<div class="quality-passed">✅ ${(quality.passed_checks||[]).join(' · ')}</div>` : ''}
  `;
}

async function loadGenerationLog(limit = 10) {
  const logEl = document.querySelector('#generation-log');
  if (!logEl) return;
  logEl.innerHTML = '<p class="log-loading">Loading\u2026</p>';
  try {
    const res   = await fetch(`/api/generation-log?limit=${limit}`);
    const data  = await res.json();
    const evts  = data.events || [];
    if (!evts.length) { logEl.innerHTML = '<p class="empty-state">No events yet.</p>'; return; }
    logEl.innerHTML = evts.map(ev => {
      const ts    = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '\u2013';
      const hf    = (ev.hard_fails || []).map(escHtml).join(', ');
      const rep   = ev.repair_used ? ' \u00b7 repaired' : '';
      const cls   = ev.error ? 'log-err' : (ev.failed > 0 ? 'log-warn' : 'log-ok');
      return `<article class="log-evt ${cls}">
        <div class="log-hd">
          <span class="log-ts">${ts}</span>
          <span class="log-fam">${escHtml(ev.family_id || 'unknown')}</span>
          <span class="log-mdl">${escHtml(ev.model || '\u2013')}${rep}</span>
          <span class="log-sc">${ev.passed ?? '?'}/${ev.total ?? '?'}</span>
        </div>
        <div class="log-prompt">${escHtml((ev.prompt || '').slice(0,120))}</div>
        ${ev.error  ? `<div class="log-err-msg">Error: ${escHtml(ev.error)}</div>` : ''}
        ${hf        ? `<div class="log-hf">Blocking: ${hf}</div>` : ''}
      </article>`;
    }).join('');
  } catch(e) {
    logEl.innerHTML = `<p class="empty-state">Could not load log: ${escHtml(e.message)}</p>`;
  }
}

// ── Provider / model selects ─────────────────────────────────────────
function renderKnowledgeFiles() {
  const ragDisabled = !!disableRagInput?.checked;
  selectAllKnowledgeBtn.disabled = ragDisabled;
  clearKnowledgeBtn.disabled = ragDisabled;
  if (!knowledgeFiles.length) {
    knowledgeFilesEl.className = 'knowledge-files empty-knowledge';
    knowledgeFilesEl.textContent = 'No knowledge records found.';
    knowledgeSelectedEl.textContent = ragDisabled ? 'RAG off' : '0 selected';
    return;
  }

  knowledgeFilesEl.className = 'knowledge-files';
  knowledgeFilesEl.innerHTML = knowledgeFiles.map(doc => {
    const badge = doc.file_type === 'json'
      ? '<span class="file-type-badge file-type-json">JSON</span>'
      : '<span class="file-type-badge file-type-txt">TXT</span>';
    const kchars = Math.round((doc.chars || 0) / 100) / 10;
    const tooltip = doc.excerpt ? escHtml(doc.excerpt.slice(0, 120)) : '';
    return `
    <label class="knowledge-file ${ragDisabled ? 'is-disabled' : ''}" title="${tooltip}">
      <input type="checkbox" value="${escHtml(doc.id)}" ${selectedKnowledge.has(doc.id) ? 'checked' : ''} ${ragDisabled ? 'disabled' : ''} />
      <span>
        <span class="knowledge-file-title">
          ${badge}<strong>${escHtml(doc.title)}</strong>
        </span>
        <small>${kchars}k chars</small>
      </span>
    </label>`;
  }).join('');
  knowledgeSelectedEl.textContent = ragDisabled ? 'RAG off' : `${selectedKnowledge.size} selected`;
}

function updateModels(providerId, preferred = '') {
  const prov   = providerMap.get(providerId);
  const models = prov?.models || [];
  modelSelect.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
  const pick = preferred || prov?.default_model || models[0] || '';
  if (pick) modelSelect.value = pick;
  if (providerId) localStorage.setItem(PROVIDER_STORAGE_KEY, providerId);
  if (modelSelect.value) localStorage.setItem(MODEL_STORAGE_KEY, modelSelect.value);
}

function defaultModelForProvider(providerId) {
  if (providerId === 'openai') return defaultModels.openai_model;
  if (providerId === 'openrouter') return defaultModels.openrouter_model;
  if (providerId === 'huggingface') return defaultModels.huggingface_model;
  return defaultModels.ollama_model;
}

// ── Config load ──────────────────────────────────────────────────────
async function loadConfig() {
  const [sr, mr, kr] = await Promise.all([fetch('/api/status'), fetch('/api/models'), fetch('/api/knowledge')]);
  if (!sr.ok || !mr.ok || !kr.ok) throw new Error('Could not load backend configuration.');

  const status = await sr.json();
  const config = await mr.json();
  const knowledge = await kr.json();
  defaultModels = config.defaults;

  const families = status.families ?? 0;
  const support = status.support_docs ?? 0;
  const parts = status.partdb_records ?? 0;
  const embed = status.embedding_model || status.embedding_backend;

  docCountEl.textContent = `${families} families / ${support} rulebooks / ${parts}+ parts`;
  ragModeEl.textContent = `Family-aware RAG / ${embed}`;
  knowledgeFiles = knowledge.documents || [];
  selectedKnowledge = new Set();
  disableRagInput.checked = localStorage.getItem(DISABLE_RAG_STORAGE_KEY) === 'true';
  renderKnowledgeFiles();
  if (disableRagInput.checked) ragModeEl.textContent = 'RAG disabled / ablation mode';

  providerMap = new Map(config.providers.map(p => [p.id, p]));
  const available = config.providers.filter(p => p.available);
  if (!available.length) throw new Error('No providers available. Start Ollama or set OPENROUTER_API_KEY / OPENAI_API_KEY.');

  providerSelect.innerHTML = available.map(p =>
    `<option value="${p.id}">${p.label}</option>`
  ).join('');
  const savedProvider = localStorage.getItem(PROVIDER_STORAGE_KEY);
  providerSelect.value = available.some(p => p.id === savedProvider)
    ? savedProvider
    : (config.defaults.provider || available[0].id);

  const savedModel = localStorage.getItem(MODEL_STORAGE_KEY);
  const providerModels = providerMap.get(providerSelect.value)?.models || [];
  const defModel = providerModels.includes(savedModel)
    ? savedModel
    : defaultModelForProvider(providerSelect.value);
  updateModels(providerSelect.value, defModel);

  submitBtn.disabled = false;
  setStatus(`Ready / ${families} families / hybrid retrieval / ${embed}`, 'success');
}

// Core send
async function send(userText) {
  const ragDisabled = !!disableRagInput?.checked;
  const payload = {
    prompt:        userText,
    provider:      providerSelect.value,
    model:         modelSelect.value,
    temperature:   Number(temperatureInput.value),
    allow_fallback: allowFallback.checked,
    disable_rag:   ragDisabled,
    selected_doc_ids: ragDisabled ? [] : Array.from(selectedKnowledge),
    manufacturing: mfgSelect?.value || 'fdm',
    history,
  };

  addMessage('user', userText);
  const typing = addTyping();
  submitBtn.disabled = true;
  setStatus(`Generating · ${payload.provider}:${payload.model}`, 'working');

  try {
    const res = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const raw = await res.text();
    let data = {};
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch {
      data = { detail: raw || `HTTP ${res.status}` };
    }
    if (!res.ok) throw new Error(data.detail || 'API error.');

    typing.remove();

    const code = stripFences(data.result || '');

    const meta = data.used_fallback
      ? `${data.provider} · ${data.model} · fallback used`
      : `${data.provider} · ${data.model}`;
    addMessage('assistant', code || '// No code returned.', meta);
    const designNotes = formatDesignNotes(data.design_notes);
    if (designNotes) addMessage('assistant', designNotes, 'Design notes');

    setCode(code);
    lastPrompt = userText;
    lastGeneration = {
      provider: data.provider || payload.provider,
      model: data.model || payload.model,
      disable_rag: payload.disable_rag,
      selected_doc_ids: payload.selected_doc_ids,
    };
    renderReferences(payload.disable_rag ? [] : (data.retrieved || []));
    renderValidation(data.validation || [], data.evaluation || null);
    renderMaterials(data.design_notes);
    renderResultSummary(code, data);
    // Enhancement module panels
    renderIntentPanel(data.intent);
    renderConstraintPanel(data.constraints);
    renderPhysicsPanel(data.physics);
    renderMfgPanel(data.manufacturing);
    renderQualityPanel(data.mechanical_quality);
    saveRecentPrompt(userText);

    history.push({ role: 'user',      content: userText });
    history.push({ role: 'assistant', content: code });
    if (designNotes) history.push({ role: 'assistant', content: designNotes });

    setStatus(
      data.used_fallback
        ? `Ready · fallback: ${data.provider}:${data.model}`
        : `Ready · ${data.model}`,
      'success'
    );
  } catch (err) {
    typing.remove();
    addMessage('assistant', `// Error: ${err.message}`);
    setStatus(err.message, 'error');
  } finally {
    submitBtn.disabled = false;
  }
}

async function acceptCurrentModel() {
  if (!lastCode || !lastPrompt || acceptBtn.disabled) return;
  acceptBtn.disabled = true;
  const original = acceptBtn.textContent;
  acceptBtn.textContent = 'Saving...';
  setStatus('Saving accepted model...', 'working');

  try {
    const payload = {
      prompt: lastPrompt,
      code: lastCode,
      provider: lastGeneration?.provider || providerSelect.value,
      model: lastGeneration?.model || modelSelect.value,
      selected_doc_ids: lastGeneration?.selected_doc_ids || Array.from(selectedKnowledge),
    };
    const res = await fetch('/api/accept', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not accept model.');
    acceptBtn.textContent = 'Accepted';
    setStatus(`Accepted model saved to history (${data.count}).`, 'success');
    setTimeout(() => {
      acceptBtn.textContent = original;
      acceptBtn.disabled = !lastCode;
    }, 1800);
  } catch (err) {
    acceptBtn.textContent = original;
    acceptBtn.disabled = !lastCode;
    setStatus(err.message, 'error');
  }
}

// ── Events ────────────────────────────────────────────────────────────
temperatureInput.addEventListener('input', () => {
  temperatureVal.textContent = Number(temperatureInput.value).toFixed(2);
});

providerSelect.addEventListener('change', () => {
  const pref = defaultModelForProvider(providerSelect.value);
  updateModels(providerSelect.value, pref);
});

modelSelect.addEventListener('change', () => {
  if (providerSelect.value) localStorage.setItem(PROVIDER_STORAGE_KEY, providerSelect.value);
  if (modelSelect.value) localStorage.setItem(MODEL_STORAGE_KEY, modelSelect.value);
});

knowledgeFilesEl.addEventListener('change', e => {
  if (disableRagInput?.checked) return;
  const box = e.target.closest('input[type="checkbox"]');
  if (!box) return;
  if (box.checked) {
    selectedKnowledge.add(box.value);
  } else {
    selectedKnowledge.delete(box.value);
  }
  knowledgeSelectedEl.textContent = `${selectedKnowledge.size} selected`;
});

selectAllKnowledgeBtn.addEventListener('click', () => {
  if (disableRagInput?.checked) return;
  selectedKnowledge = new Set(knowledgeFiles.map(doc => doc.id));
  renderKnowledgeFiles();
});

clearKnowledgeBtn.addEventListener('click', () => {
  if (disableRagInput?.checked) return;
  selectedKnowledge = new Set();
  renderKnowledgeFiles();
});

disableRagInput?.addEventListener('change', () => {
  localStorage.setItem(DISABLE_RAG_STORAGE_KEY, String(disableRagInput.checked));
  renderKnowledgeFiles();
  renderReferences([]);
  ragModeEl.textContent = disableRagInput.checked
    ? 'RAG disabled / ablation mode'
    : 'Family-aware RAG';
  setStatus(disableRagInput.checked ? 'Ablation mode: RAG disabled.' : 'RAG enabled.', disableRagInput.checked ? 'working' : 'success');
});

quickPrompts.addEventListener('click', e => {
  const chip = e.target.closest('.chip');
  if (!chip) return;
  usePrompt(chip.textContent);
});

promptInput.addEventListener('input', () => autoResize(promptInput));
promptInput.addEventListener('input', updatePromptCoach);

promptTabs?.addEventListener('click', e => {
  const tab = e.target.closest('.prompt-tab');
  if (!tab) return;
  renderPromptGroup(tab.dataset.group);
});

recentPromptsEl?.addEventListener('click', e => {
  const btn = e.target.closest('.recent-prompt');
  if (!btn) return;
  usePrompt(btn.textContent, false);
});

promptTools?.addEventListener('click', e => {
  const chip = e.target.closest('.tool-chip');
  if (!chip) return;
  const add = chip.dataset.add || '';
  if (!promptInput.value.includes(add.trim())) {
    promptInput.value = `${promptInput.value.trim()}${add}`.trim();
    autoResize(promptInput);
    updatePromptCoach();
  }
  promptInput.focus();
});

promptInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSubmit();
  }
});

submitBtn.addEventListener('click', handleSubmit);
acceptBtn.addEventListener('click', acceptCurrentModel);
renderPreviewBtn.addEventListener('click', renderPreview);
resetPreviewBtn.addEventListener('click', resetPreviewCamera);

function handleSubmit() {
  const text = promptInput.value.trim();
  if (!text || submitBtn.disabled) return;
  promptInput.value = '';
  autoResize(promptInput);
  send(text);
}

// Copy button
copyBtn.addEventListener('click', async () => {
  if (!lastCode) return;
  await navigator.clipboard.writeText(lastCode);
  const orig = copyBtn.textContent;
  copyBtn.textContent = 'Copied';
  setTimeout(() => { copyBtn.textContent = orig; }, 2000);
  setStatus('Copied to clipboard.', 'success');
});

// Download .scad button
downloadBtn.addEventListener('click', () => {
  if (!lastCode) return;
  const blob = new Blob([lastCode], { type: 'text/plain;charset=utf-8' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'generated_part.scad'; a.click();
  URL.revokeObjectURL(url);
  setStatus('Downloaded generated_part.scad', 'success');
});

// Export STL button
exportStlBtn.addEventListener('click', async () => {
  if (!lastCode || exportStlBtn.disabled) return;

  const orig = exportStlBtn.textContent;
  exportStlBtn.disabled = true;
  exportStlBtn.textContent = '⏳ Rendering…';
  setStatus('Exporting STL via OpenSCAD…', 'working');

  try {
    const res = await fetch('/api/export-stl', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ code: lastCode, filename: 'generated_part' }),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `Server error ${res.status}`);
    }

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'generated_part.stl'; a.click();
    URL.revokeObjectURL(url);
    setStatus('Downloaded generated_part.stl', 'success');
  } catch (err) {
    setStatus(`STL export failed: ${err.message}`, 'error');
    // Surface the error in chat so it's not silent
    addErrorMessage(`STL export failed: ${err.message}`);
  } finally {
    exportStlBtn.textContent = orig;
    exportStlBtn.disabled    = !lastCode;
  }
});

// Clear conversation
clearBtn.addEventListener('click', () => {
  history  = [];
  lastCode = '';
  lastPrompt = '';
  lastGeneration = null;
  chatWindow.innerHTML = `
    <div class="welcome-state">
      <div class="welcome-icon">⬡</div>
      <div class="welcome-title">What would you like to design?</div>
      <div class="welcome-body">
        Describe any mechanical part to get parametric OpenSCAD instantly.
        After generation just keep chatting to refine — change dimensions,
        add features, or ask follow-up questions exactly as you would
        in a normal conversation.
      </div>
      <div class="welcome-examples">
        <span class="example-tag">Try: "change bore to 25 mm"</span>
        <span class="example-tag">Try: "add chamfers to top edges"</span>
        <span class="example-tag">Try: "increase wall thickness to 10 mm"</span>
        <span class="example-tag">Try: "make the mounting holes M5"</span>
      </div>
    </div>`;
  codeOutput.textContent   = '// Parametric OpenSCAD code will appear here.\n// Describe a part to get started, then keep chatting to refine it.';
  codeFilename.textContent = 'no file yet';
  renderResultSummary('');
  acceptBtn.disabled = copyBtn.disabled = downloadBtn.disabled = exportStlBtn.disabled = true;
  resetPreviewState('waiting for generated geometry');
  referencesEl.className = validationEl.className = 'empty-state';
  referencesEl.textContent = 'No retrieval results yet.';
  validationEl.textContent = 'Checks appear after generation.';
  materialsEl.className = 'empty-state';
  materialsEl.textContent = 'Material suggestions appear after generation.';
  setStatus('New conversation started.', 'success');
});

// ── Boot ─────────────────────────────────────────────────────────────
submitBtn.disabled = true;
renderPromptGroup('truss');
renderRecentPrompts();
updatePromptCoach();
document.querySelector('#refresh-log-btn')?.addEventListener('click', () => loadGenerationLog(20));
loadConfig().catch(err => {
  setStatus(err.message, 'error');
  codeOutput.textContent = `// Could not load backend configuration.\n// ${err.message}`;
});
