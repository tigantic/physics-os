/**
 * HyperTensor Facial Plastics SPA — Client-side application.
 *
 * Pure vanilla JS (no framework dependencies).
 * Communicates with the Python backend via JSON API.
 *
 * Modes: G1 Case Library, G2 Twin Inspect, G3 Plan Author,
 *        G4 Consult, G5 Report, G6 3D Viz, G7 Timeline,
 *        G8 Compare.
 */

"use strict";

// ── State ────────────────────────────────────────────────────────

const State = {
  currentMode: "case-library",
  selectedCaseId: null,
  currentPlan: null,
  compileResult: null,
  contract: null,
  operators: {},
  templates: {},
  cases: [],
};

// ── API helpers ──────────────────────────────────────────────────

const API_BASE = "";

async function apiGet(path, params) {
  let url = `${API_BASE}/api${path}`;
  if (params) {
    const qs = new URLSearchParams(params).toString();
    if (qs) url += `?${qs}`;
  }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`GET ${path}: ${resp.status}`);
  return resp.json();
}

async function apiPost(path, body) {
  const resp = await fetch(`${API_BASE}/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`POST ${path}: ${resp.status}`);
  return resp.json();
}

// ── Toast notifications ──────────────────────────────────────────

function toast(message, type) {
  type = type || "info";
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(function() { el.remove(); }, 4000);
}

// ── Mode navigation ──────────────────────────────────────────────

function switchMode(mode) {
  State.currentMode = mode;
  document.querySelectorAll(".nav-btn").forEach(function(btn) {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  document.querySelectorAll(".mode-panel").forEach(function(panel) {
    panel.classList.toggle("active", panel.id === `mode-${mode}`);
  });
  onModeEnter(mode);
}

function onModeEnter(mode) {
  switch (mode) {
    case "case-library": loadCases(); break;
    case "twin-inspect": loadTwin(); break;
    case "plan-author": loadOperators(); break;
    case "consult": loadConsult(); break;
    case "report": break;
    case "visualization": loadVisualization(); break;
    case "timeline": loadTimeline(); break;
    case "compare": break;
  }
}

// ── G1: Case Library ─────────────────────────────────────────────

async function loadCases() {
  try {
    const procedure = document.getElementById("filter-procedure").value;
    const quality = document.getElementById("filter-quality").value;
    const params = {};
    if (procedure) params.procedure = procedure;
    if (quality) params.quality = quality;

    const data = await apiGet("/cases", params);
    State.cases = data.cases || [];
    renderCaseList(State.cases);
    document.getElementById("footer-stats").textContent =
      `${data.total || 0} cases`;
  } catch (e) {
    toast("Failed to load cases: " + e.message, "error");
  }
}

function renderCaseList(cases) {
  const grid = document.getElementById("case-list");
  if (!cases.length) {
    grid.innerHTML = '<p class="placeholder">No cases found. Create one to get started.</p>';
    return;
  }

  grid.innerHTML = cases.map(function(c) {
    const id = c.case_id || c.id || "unknown";
    const proc = c.procedure || c.metadata && c.metadata.procedure || "";
    const quality = c.quality || c.metadata && c.metadata.quality || "";
    const sel = id === State.selectedCaseId ? " selected" : "";
    return `
      <div class="case-card${sel}" data-case-id="${id}">
        <h4>${id.substring(0, 12)}...</h4>
        <p>${proc || "No procedure"}</p>
        ${quality ? `<span class="tag">${quality}</span>` : ""}
      </div>`;
  }).join("");

  grid.querySelectorAll(".case-card").forEach(function(card) {
    card.addEventListener("click", function() {
      selectCase(card.dataset.caseId);
    });
  });
}

function selectCase(caseId) {
  State.selectedCaseId = caseId;
  document.getElementById("case-indicator").textContent = caseId.substring(0, 12) + "...";
  document.querySelectorAll(".case-card").forEach(function(card) {
    card.classList.toggle("selected", card.dataset.caseId === caseId);
  });
  enablePlanButtons();
  toast("Case selected: " + caseId.substring(0, 12), "info");
}

function enablePlanButtons() {
  document.getElementById("btn-compile").disabled = !State.selectedCaseId || !State.currentPlan;
  document.getElementById("btn-generate-report").disabled = !State.compileResult;
  document.getElementById("btn-run-whatif").disabled = !State.currentPlan || !State.selectedCaseId;
  document.getElementById("btn-run-compare").disabled = !State.selectedCaseId;
}

async function createCase() {
  const proc = document.getElementById("new-case-procedure").value;
  const age = parseInt(document.getElementById("new-case-age").value, 10);
  const sex = document.getElementById("new-case-sex").value;
  const notes = document.getElementById("new-case-notes").value;

  try {
    const result = await apiPost("/cases", {
      procedure: proc,
      patient_age: age,
      patient_sex: sex,
      notes: notes,
    });
    if (result.error) { toast(result.error, "error"); return; }
    toast("Case created: " + result.case_id.substring(0, 12), "success");
    closeModal();
    loadCases();
  } catch (e) {
    toast("Create failed: " + e.message, "error");
  }
}

async function curateLibrary() {
  try {
    toast("Running library curation...", "info");
    const result = await apiPost("/curate", {});
    toast("Curation complete", "success");
    document.getElementById("case-list").innerHTML =
      `<div class="result-panel">${JSON.stringify(result, null, 2)}</div>`;
  } catch (e) {
    toast("Curation failed: " + e.message, "error");
  }
}

// ── G2: Twin Inspect ─────────────────────────────────────────────

async function loadTwin() {
  if (!State.selectedCaseId) {
    document.getElementById("twin-details").innerHTML =
      '<p class="placeholder">Select a case first.</p>';
    return;
  }
  try {
    const data = await apiGet(`/cases/${State.selectedCaseId}/twin`);
    if (data.error) {
      document.getElementById("twin-details").innerHTML =
        `<p class="placeholder">${data.error}</p>`;
      return;
    }
    document.getElementById("twin-details").innerHTML = `
      <h3>Digital Twin — ${State.selectedCaseId.substring(0, 12)}</h3>
      <div class="result-panel">${JSON.stringify(data, null, 2)}</div>`;
  } catch (e) {
    toast("Failed to load twin: " + e.message, "error");
  }
}

// ── G3: Plan Author ──────────────────────────────────────────────

async function loadOperators() {
  if (Object.keys(State.operators).length > 0) {
    renderOperators();
    return;
  }
  try {
    const data = await apiGet("/operators");
    State.operators = data.operators || {};
    renderOperators();

    const tmpl = await apiGet("/templates");
    State.templates = tmpl.templates || {};
    populateTemplateDropdowns();
  } catch (e) {
    toast("Failed to load operators: " + e.message, "error");
  }
}

function renderOperators() {
  const palette = document.getElementById("operator-palette");
  const ops = State.operators;
  const keys = Object.keys(ops).sort();

  palette.innerHTML = keys.map(function(name) {
    const op = ops[name];
    return `
      <div class="op-item" data-op="${name}">
        <span class="op-name">${name}</span>
        <span class="op-desc">${op.description || ""}</span>
      </div>`;
  }).join("");

  palette.querySelectorAll(".op-item").forEach(function(item) {
    item.addEventListener("click", function() {
      addPlanStep(item.dataset.op);
    });
  });
}

function populateTemplateDropdowns() {
  const catSelect = document.getElementById("template-category");
  const categories = Object.keys(State.templates);
  catSelect.innerHTML = '<option value="">Choose category...</option>' +
    categories.map(function(c) { return `<option value="${c}">${c}</option>`; }).join("");

  catSelect.addEventListener("change", function() {
    const nameSelect = document.getElementById("template-name");
    const methods = State.templates[catSelect.value] || [];
    nameSelect.innerHTML = '<option value="">Choose template...</option>' +
      methods.map(function(m) { return `<option value="${m}">${m}</option>`; }).join("");
  });
}

function addPlanStep(opName) {
  if (!State.currentPlan) {
    State.currentPlan = {
      name: "custom_plan",
      procedure: State.operators[opName] ? State.operators[opName].procedure : "rhinoplasty",
      steps: [],
    };
  }

  const op = State.operators[opName];
  if (!op) { toast("Unknown operator: " + opName, "error"); return; }

  State.currentPlan.steps.push({
    name: opName,
    operator: opName,
    params: Object.assign({}, op.params),
    description: op.description || "",
  });

  renderPlanSteps();
  enablePlanButtons();
}

function removePlanStep(index) {
  if (State.currentPlan) {
    State.currentPlan.steps.splice(index, 1);
    if (State.currentPlan.steps.length === 0) State.currentPlan = null;
    renderPlanSteps();
    enablePlanButtons();
  }
}

function renderPlanSteps() {
  const container = document.getElementById("plan-steps");
  if (!State.currentPlan || State.currentPlan.steps.length === 0) {
    container.innerHTML = '<p class="placeholder">Load a template or click operators to add steps.</p>';
    return;
  }

  container.innerHTML = State.currentPlan.steps.map(function(step, i) {
    return `
      <div class="plan-step">
        <span class="step-num">${i + 1}</span>
        <div class="step-info">
          <strong>${step.name}</strong>
          <div style="font-size:0.7rem;color:var(--c-muted)">${step.description || ""}</div>
        </div>
        <span class="step-remove" data-index="${i}">&times;</span>
      </div>`;
  }).join("");

  container.querySelectorAll(".step-remove").forEach(function(btn) {
    btn.addEventListener("click", function() {
      removePlanStep(parseInt(btn.dataset.index, 10));
    });
  });
}

function clearPlan() {
  State.currentPlan = null;
  State.compileResult = null;
  renderPlanSteps();
  document.getElementById("compile-result").textContent = "";
  enablePlanButtons();
}

async function loadTemplate() {
  const category = document.getElementById("template-category").value;
  const template = document.getElementById("template-name").value;
  if (!category || !template) { toast("Select a template", "warning"); return; }

  try {
    const data = await apiPost("/plan/template", { category: category, template: template });
    if (data.error) { toast(data.error, "error"); return; }
    State.currentPlan = data.plan;
    renderPlanSteps();
    enablePlanButtons();
    toast("Template loaded: " + template, "success");
  } catch (e) {
    toast("Template load failed: " + e.message, "error");
  }
}

async function compilePlan() {
  if (!State.selectedCaseId || !State.currentPlan) return;

  try {
    toast("Compiling plan...", "info");
    const data = await apiPost("/plan/compile", {
      case_id: State.selectedCaseId,
      plan: State.currentPlan,
    });
    State.compileResult = data;
    document.getElementById("compile-result").textContent = JSON.stringify(data, null, 2);
    enablePlanButtons();

    if (data.errors && data.errors.length > 0) {
      toast("Compilation has errors", "error");
    } else {
      toast("Plan compiled successfully", "success");
    }
  } catch (e) {
    toast("Compile failed: " + e.message, "error");
  }
}

// ── G4: Consult ──────────────────────────────────────────────────

function loadConsult() {
  const container = document.getElementById("whatif-params");
  if (!State.currentPlan) {
    container.innerHTML = '<p class="placeholder">Build a plan in Plan Author first.</p>';
    return;
  }

  // Build param editor from plan steps
  var html = "";
  State.currentPlan.steps.forEach(function(step) {
    html += `<h4 style="margin:0.5rem 0 0.25rem;font-size:0.85rem">${step.name}</h4>`;
    var params = step.params || {};
    Object.keys(params).forEach(function(key) {
      var val = params[key];
      html += `
        <div class="param-row">
          <label>${key}
            <input type="text" data-op="${step.name}" data-param="${key}" value="${val}">
          </label>
        </div>`;
    });
  });
  container.innerHTML = html || '<p class="placeholder">No parameters to modify.</p>';
  enablePlanButtons();
}

async function runWhatIf() {
  if (!State.selectedCaseId || !State.currentPlan) return;

  // Collect overrides
  var overrides = {};
  document.querySelectorAll("#whatif-params input[data-op]").forEach(function(input) {
    var op = input.dataset.op;
    var param = input.dataset.param;
    var val = input.value;
    // Try numeric
    var num = parseFloat(val);
    if (!isNaN(num) && val === String(num)) val = num;
    if (val === "true") val = true;
    if (val === "false") val = false;

    if (!overrides[op]) overrides[op] = {};
    overrides[op][param] = val;
  });

  try {
    toast("Running what-if...", "info");
    var data = await apiPost("/whatif", {
      case_id: State.selectedCaseId,
      plan: State.currentPlan,
      modified_params: overrides,
    });
    document.getElementById("whatif-result").textContent = JSON.stringify(data, null, 2);
    toast("What-if complete", "success");
  } catch (e) {
    toast("What-if failed: " + e.message, "error");
  }
}

// ── G5: Report ───────────────────────────────────────────────────

async function generateReport() {
  if (!State.selectedCaseId || !State.currentPlan) return;
  var fmt = document.getElementById("report-format").value;

  try {
    toast("Generating report...", "info");
    var data = await apiPost("/report", {
      case_id: State.selectedCaseId,
      plan: State.currentPlan,
      format: fmt,
    });

    var output = document.getElementById("report-output");
    if (fmt === "html") {
      output.innerHTML = `<iframe srcdoc="${escapeHtml(data.content || "")}"></iframe>`;
    } else {
      output.innerHTML = `<pre class="result-panel">${escapeHtml(data.content || JSON.stringify(data, null, 2))}</pre>`;
    }
    toast("Report generated", "success");
  } catch (e) {
    toast("Report failed: " + e.message, "error");
  }
}

// ── G6: 3D Visualization ─────────────────────────────────────────

async function loadVisualization() {
  if (!State.selectedCaseId) {
    document.getElementById("viz-placeholder").style.display = "block";
    return;
  }

  try {
    var data = await apiGet(`/cases/${State.selectedCaseId}/visualization`);
    if (data.error) {
      document.getElementById("viz-placeholder").textContent = data.error;
      return;
    }
    document.getElementById("viz-placeholder").style.display = "none";
    render3D(data);
  } catch (e) {
    document.getElementById("viz-placeholder").textContent = "Visualization error: " + e.message;
  }
}

function render3D(data) {
  // Minimal canvas-based wireframe renderer (no Three.js dependency)
  var canvas = document.getElementById("viz-canvas");
  var ctx = canvas.getContext("2d");
  if (!ctx) return;

  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;

  var mesh = data.mesh || {};
  var positions = mesh.positions || [];
  var indices = mesh.indices || [];

  if (positions.length === 0) {
    ctx.fillStyle = "#8b949e";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No mesh data available", canvas.width / 2, canvas.height / 2);
    return;
  }

  // Simple orthographic projection
  var cx = canvas.width / 2;
  var cy = canvas.height / 2;

  // Compute bounds
  var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (var i = 0; i < positions.length; i++) {
    var p = positions[i];
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[2] < minY) minY = p[2];
    if (p[2] > maxY) maxY = p[2];
  }
  var rangeX = maxX - minX || 1;
  var rangeY = maxY - minY || 1;
  var scale = Math.min(canvas.width * 0.8 / rangeX, canvas.height * 0.8 / rangeY);

  function project(p) {
    return [
      cx + (p[0] - (minX + maxX) / 2) * scale,
      cy - (p[2] - (minY + maxY) / 2) * scale
    ];
  }

  // Draw triangles
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(88, 166, 255, 0.3)";
  ctx.lineWidth = 0.5;
  ctx.beginPath();

  var maxTris = Math.min(indices.length, 50000);
  for (var t = 0; t < maxTris; t++) {
    var tri = indices[t];
    if (!tri || tri.length < 3) continue;
    var a = project(positions[tri[0]] || [0, 0, 0]);
    var b = project(positions[tri[1]] || [0, 0, 0]);
    var c = project(positions[tri[2]] || [0, 0, 0]);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.lineTo(c[0], c[1]);
    ctx.lineTo(a[0], a[1]);
  }
  ctx.stroke();

  // Draw landmarks
  var landmarks = (data.landmarks && data.landmarks.landmarks) || [];
  ctx.fillStyle = "#f85149";
  landmarks.forEach(function(lm) {
    var pos = lm.position;
    if (!pos) return;
    var p = project(pos);
    ctx.beginPath();
    ctx.arc(p[0], p[1], 4, 0, Math.PI * 2);
    ctx.fill();
  });

  // Legend
  ctx.fillStyle = "#8b949e";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(
    `${positions.length} vertices, ${indices.length} triangles`,
    10, canvas.height - 10
  );
}

// ── G7: Timeline ─────────────────────────────────────────────────

async function loadTimeline() {
  if (!State.selectedCaseId) {
    document.getElementById("timeline-events").innerHTML =
      '<p class="placeholder">Select a case first.</p>';
    return;
  }

  try {
    var data = await apiGet(`/cases/${State.selectedCaseId}/timeline`);
    var events = data.events || [];
    var track = document.getElementById("timeline-events");

    if (events.length === 0) {
      track.innerHTML = '<p class="placeholder">No events recorded for this case.</p>';
      return;
    }

    track.innerHTML = events.map(function(ev) {
      return `
        <div class="timeline-event">
          <div class="event-time">${ev.timestamp || ""}</div>
          <div>${ev.event_type || ev.action || JSON.stringify(ev)}</div>
        </div>`;
    }).join("");

    document.getElementById("timeline-frame").textContent =
      `${events.length} events`;
  } catch (e) {
    toast("Timeline failed: " + e.message, "error");
  }
}

// ── G8: Compare ──────────────────────────────────────────────────

async function runCompare() {
  if (!State.selectedCaseId) { toast("Select a case first", "warning"); return; }

  var type = document.getElementById("compare-type").value;

  if (type === "plans" && State.currentPlan) {
    // Compare current plan vs. a modified version
    try {
      var data = await apiPost("/compare/plans", {
        case_id: State.selectedCaseId,
        plan_a: State.currentPlan,
        plan_b: State.currentPlan, // same plan for demo; user would modify
      });
      document.getElementById("compare-result-a").textContent =
        JSON.stringify(data.plan_a, null, 2);
      document.getElementById("compare-result-b").textContent =
        JSON.stringify(data.plan_b, null, 2);
      document.getElementById("compare-diff").textContent =
        JSON.stringify(data.delta || {}, null, 2);
      toast("Comparison complete", "success");
    } catch (e) {
      toast("Compare failed: " + e.message, "error");
    }
  } else {
    toast("Load a plan or select two cases to compare", "warning");
  }
}

// ── Modal management ─────────────────────────────────────────────

function openModal() {
  document.getElementById("modal-overlay").classList.remove("hidden");
}

function closeModal() {
  document.getElementById("modal-overlay").classList.add("hidden");
}

// ── Utility ──────────────────────────────────────────────────────

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ── Initialization ───────────────────────────────────────────────

async function init() {
  // Mode navigation
  document.querySelectorAll(".nav-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      switchMode(btn.dataset.mode);
    });
  });

  // Case Library buttons
  document.getElementById("btn-refresh-cases").addEventListener("click", loadCases);
  document.getElementById("btn-create-case").addEventListener("click", openModal);
  document.getElementById("btn-curate").addEventListener("click", curateLibrary);
  document.getElementById("filter-procedure").addEventListener("change", loadCases);
  document.getElementById("filter-quality").addEventListener("change", loadCases);

  // Plan Author buttons
  document.getElementById("btn-load-template").addEventListener("click", loadTemplate);
  document.getElementById("btn-compile").addEventListener("click", compilePlan);
  document.getElementById("btn-clear-plan").addEventListener("click", clearPlan);

  // Consult
  document.getElementById("btn-run-whatif").addEventListener("click", runWhatIf);

  // Report
  document.getElementById("btn-generate-report").addEventListener("click", generateReport);

  // Compare
  document.getElementById("btn-run-compare").addEventListener("click", runCompare);

  // Modal
  document.getElementById("modal-cancel").addEventListener("click", closeModal);
  document.getElementById("modal-confirm").addEventListener("click", createCase);

  // Load contract to populate procedure dropdowns
  try {
    State.contract = await apiGet("/contract");
    var procedures = State.contract.procedures || [];

    var procOptions = procedures.map(function(p) {
      return `<option value="${p}">${p}</option>`;
    }).join("");

    document.getElementById("filter-procedure").innerHTML =
      '<option value="">All Procedures</option>' + procOptions;

    document.getElementById("new-case-procedure").innerHTML = procOptions;
  } catch (e) {
    console.warn("Contract load failed — running without metadata:", e);
  }

  // Initial load
  loadCases();

  toast("HyperTensor Facial Plastics UI ready", "success");
}

document.addEventListener("DOMContentLoaded", init);
