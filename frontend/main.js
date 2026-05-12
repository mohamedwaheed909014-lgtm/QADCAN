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
const quickPrompts   = document.querySelector('#quick-prompts');
const clearBtn       = document.querySelector('#clear-button');

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

function renderValidation(items) {
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
  validationEl.innerHTML = summary + groupsHtml;
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
        return `<article class="material-card">
          <div class="material-card-top">
            <strong>${escHtml(material)}</strong>
            ${appliesTo}
          </div>
          <p>${escHtml(reason)}</p>
        </article>`;
      }).join('')}
    </div>`;
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
  if (!knowledgeFiles.length) {
    knowledgeFilesEl.className = 'knowledge-files empty-knowledge';
    knowledgeFilesEl.textContent = 'No knowledge records found.';
    knowledgeSelectedEl.textContent = '0 selected';
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
    <label class="knowledge-file" title="${tooltip}">
      <input type="checkbox" value="${escHtml(doc.id)}" ${selectedKnowledge.has(doc.id) ? 'checked' : ''} />
      <span>
        <span class="knowledge-file-title">
          ${badge}<strong>${escHtml(doc.title)}</strong>
        </span>
        <small>${kchars}k chars</small>
      </span>
    </label>`;
  }).join('');
  knowledgeSelectedEl.textContent = `${selectedKnowledge.size} selected`;
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

  // Update pills
  const families = status.families ?? 0;
const support  = status.support_docs ?? 0;
const parts    = status.partdb_records ?? 0;

docCountEl.textContent =
  `${families} families • ${support} rulebooks • ${parts}+ parts`;

ragModeEl.textContent =
  `Family-aware RAG • ${status.embedding_model}`;
  knowledgeFiles = knowledge.documents || [];
  selectedKnowledge = new Set();
  renderKnowledgeFiles();

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

  const embed = status.embedding_model || status.embedding_backend;
  const retrieval = status.retrieval_backend || status.embedding_backend;
 setStatus(
  `Ready • ${families} families • hybrid retrieval • ${embed}`,
  'success'
);}

// ── Core send ────────────────────────────────────────────────────────
async function send(userText) {
  const payload = {
    prompt:        userText,
    provider:      providerSelect.value,
    model:         modelSelect.value,
    temperature:   Number(temperatureInput.value),
    allow_fallback: allowFallback.checked,
    selected_doc_ids: Array.from(selectedKnowledge),
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
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'API error.');

    typing.remove();

    if (data.needs_clarification) {
      const clarification = data.clarification || {};
      const lines = [
        clarification.message || 'I need a few main parameters before generating this mechanical part.',
        '',
        ...(clarification.questions || []).map((item, index) => `${index + 1}. ${item.question}`),
        '',
        'After that I will assume secondary dimensions from mechanical standards and suggest suitable materials.'
      ];
      const messageText = lines.join('\n');
      addMessage('assistant', messageText, data.engineering?.active_database_family || 'Mechanical RAG');
      const examples = data.part_family?.external_examples || [];
      renderReferences(examples.map((item, index) => ({
        title: item.title || `External example ${index + 1}`,
        source: 'GrabCAD external preview',
        score: 'ref',
        excerpt: item.url || ''
      })));
      renderValidation([]);
      renderMaterials(null);
      setCode('');
      history.push({ role: 'user', content: userText });
      history.push({ role: 'assistant', content: messageText });
      setStatus('Waiting for main mechanical parameters.', 'working');
      return;
    }

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
      selected_doc_ids: payload.selected_doc_ids,
    };
    renderReferences(data.retrieved  || []);
    renderValidation(data.validation || []);
    renderMaterials(data.design_notes);

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
  selectedKnowledge = new Set(knowledgeFiles.map(doc => doc.id));
  renderKnowledgeFiles();
});

clearKnowledgeBtn.addEventListener('click', () => {
  selectedKnowledge = new Set();
  renderKnowledgeFiles();
});

quickPrompts.addEventListener('click', e => {
  const chip = e.target.closest('.chip');
  if (!chip) return;
  promptInput.value = `Design a ${chip.textContent.trim()} in parametric OpenSCAD with clearly named parameters and mechanically realistic proportions.`;
  autoResize(promptInput);
  promptInput.focus();
});

promptInput.addEventListener('input', () => autoResize(promptInput));

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
  copyBtn.textContent = '✓ Copied';
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
document.querySelector('#refresh-log-btn')?.addEventListener('click', () => loadGenerationLog(20));
loadConfig().catch(err => {
  setStatus(err.message, 'error');
  codeOutput.textContent = `// Could not load backend configuration.\n// ${err.message}`;
});
