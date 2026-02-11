/**
 * HyperTensor Facial Plastics — G3 Plan Author Mode
 *
 * Operator palette, template loading, plan step list with
 * inline parameter editing, compile wiring.
 */

"use strict";

const PlanAuthor = (() => {
  let _el = null;

  function init() {
    _el = document.getElementById("mode-plan-author");
  }

  async function load() {
    if (!Store.get("operators.loaded")) {
      try {
        const data = await API.listOperators();
        Store.set("operators.registry", data.operators || {});
        Store.set("operators.loaded", true);
      } catch (err) { Toast.error("Failed to load operators: " + (err.message || err)); }
    }
    if (!Store.get("templates.loaded")) {
      try {
        const data = await API.listTemplates();
        Store.set("templates.registry", data.templates || {});
        Store.set("templates.loaded", true);
      } catch (err) { Toast.error("Failed to load templates: " + (err.message || err)); }
    }
  }

  function render() {
    if (!_el) return;
    const ops = Store.get("operators.registry") || {};
    const templates = Store.get("templates.registry") || {};
    const plan = Store.get("plan.current");
    const compiling = Store.get("plan.compiling");
    const caseId = Store.get("selectedCase.id");

    const categories = Object.keys(templates);

    _el.innerHTML = `
      <div class="panel-header">
        <h2>Plan Author</h2>
        <div class="panel-actions">
          <select id="template-category"><option value="">Template category...</option>${categories.map(c => `<option value="${_esc(c)}">${_esc(c)}</option>`).join("")}</select>
          <select id="template-name"><option value="">Template...</option></select>
          <button class="btn btn-secondary" id="btn-load-template">Load Template</button>
        </div>
      </div>
      <div class="split-view" style="min-height:400px">
        <div class="split-left" style="overflow-y:auto">
          <h3 style="font-size:var(--font-size-xs);color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:var(--space-2);">Operators</h3>
          <input type="search" id="op-search" placeholder="Filter operators..." style="width:100%;margin-bottom:var(--space-2)">
          <div id="operator-palette" class="operator-palette">${_renderPalette(ops, "")}</div>
        </div>
        <div class="split-right" style="overflow-y:auto">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:var(--space-2)">
            <h3 style="font-size:var(--font-size-xs);color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;">Plan Steps ${plan ? `(${plan.steps.length})` : ""}</h3>
            ${plan ? `<span class="tag tag-${_esc(plan.procedure)}">${_esc(plan.procedure)}</span>` : ""}
          </div>
          <div id="plan-steps" class="plan-steps">${plan ? _renderSteps(plan) : '<p class="placeholder">Load a template or add operators to build a plan.</p>'}</div>
          <div style="display:flex;gap:var(--space-2);margin-top:var(--space-3)">
            <button class="btn btn-primary" id="btn-compile" ${!plan || !caseId || compiling ? "disabled" : ""}>${compiling ? "Compiling..." : "Compile Plan"}</button>
            <button class="btn btn-danger" id="btn-clear-plan" ${!plan ? "disabled" : ""}>Clear</button>
          </div>
          ${Store.get("plan.compiled") ? _renderCompileResult(Store.get("plan.compiled")) : ""}
        </div>
      </div>
    `;

    _bind();
  }

  function _renderPalette(ops, filter) {
    const grouped = {};
    for (const [name, op] of Object.entries(ops)) {
      if (filter && !name.toLowerCase().includes(filter) && !(op.description || "").toLowerCase().includes(filter)) continue;
      const proc = op.procedure || "other";
      if (!grouped[proc]) grouped[proc] = [];
      grouped[proc].push({ name, op });
    }

    let html = "";
    for (const [proc, items] of Object.entries(grouped)) {
      html += `<div class="op-group-header" style="color:var(--procedure-${_esc(proc)}, var(--text-muted))">${_esc(proc)}</div>`;
      for (const { name, op } of items) {
        html += `<div class="op-item" data-op="${_esc(name)}" title="${_esc(op.description || "")}"><span class="op-add">+</span><span class="op-name">${_esc(name)}</span></div>`;
      }
    }
    return html || '<p class="placeholder">No operators</p>';
  }

  function _renderSteps(plan) {
    if (!plan || !plan.steps || plan.steps.length === 0) return '<p class="placeholder">No steps in plan.</p>';
    return plan.steps.map((step, i) => {
      const paramHtml = _renderParams(step, i);
      return `
        <div class="plan-step" draggable="true" data-step-idx="${i}">
          <div class="plan-step-header">
            <span class="step-drag-handle" title="Drag to reorder">⋮⋮</span>
            <span class="step-number">${i + 1}.</span>
            <span class="step-name">${_esc(step.name)}</span>
            <div class="step-actions">
              <button class="btn-icon btn-step-up" data-idx="${i}" title="Move up">↑</button>
              <button class="btn-icon btn-step-down" data-idx="${i}" title="Move down">↓</button>
              <button class="btn-icon btn-step-remove" data-idx="${i}" title="Remove" style="color:var(--c-error)">✗</button>
            </div>
          </div>
          <div class="param-group">${paramHtml}</div>
        </div>
      `;
    }).join("");
  }

  function _renderParams(step, stepIdx) {
    const params = step.params || {};
    const defs = step.param_defs || {};
    const entries = Object.entries(params);
    if (entries.length === 0) return '<span style="font-size:var(--font-size-xs);color:var(--text-muted)">No parameters</span>';

    return entries.map(([key, val]) => {
      const def = defs[key] || {};
      const ptype = def.param_type || "float";
      const unit = def.unit || "";
      let input = "";

      if (ptype === "bool" || typeof val === "boolean") {
        input = `<input type="checkbox" class="param-val" data-step="${stepIdx}" data-param="${_esc(key)}" ${val ? "checked" : ""}>`;
      } else if (def.enum_values && def.enum_values.length > 0) {
        input = `<select class="param-val" data-step="${stepIdx}" data-param="${_esc(key)}">${def.enum_values.map(e => `<option value="${_esc(e)}" ${e === val ? "selected" : ""}>${_esc(e)}</option>`).join("")}</select>`;
      } else {
        const min = def.min_value != null ? `min="${def.min_value}"` : "";
        const max = def.max_value != null ? `max="${def.max_value}"` : "";
        const step = ptype === "int" ? 'step="1"' : 'step="any"';
        input = `<input type="number" class="param-val" data-step="${stepIdx}" data-param="${_esc(key)}" value="${val}" ${min} ${max} ${step}>`;
      }

      return `<div class="param-row"><span class="param-label">${_esc(key)}</span><div class="param-input">${input}${unit ? `<span class="param-unit">${_esc(unit)}</span>` : ""}</div></div>`;
    }).join("");
  }

  function _renderCompileResult(result) {
    if (!result) return "";
    return `
      <div class="result-panel" style="margin-top:var(--space-3)">
        <div style="color:var(--c-success);font-weight:var(--font-weight-semibold);margin-bottom:var(--space-2)">✓ Compilation Result</div>
        <div class="param-row"><span class="param-label">Boundary Conditions</span><span class="param-baseline">${result.n_bcs || 0}</span></div>
        <div class="param-row"><span class="param-label">Material Mods</span><span class="param-baseline">${result.n_material_mods || 0}</span></div>
        <div class="param-row"><span class="param-label">Mesh Mods</span><span class="param-baseline">${result.n_mesh_mods || 0}</span></div>
        <div class="param-row"><span class="param-label">Content Hash</span><span class="param-baseline" style="font-family:var(--font-mono)">${_esc(_truncate(result.content_hash || "", 24))}</span></div>
      </div>
    `;
  }

  function _bind() {
    // Template category change
    const catSel = document.getElementById("template-category");
    const nameSel = document.getElementById("template-name");
    if (catSel && nameSel) {
      catSel.addEventListener("change", () => {
        const templates = Store.get("templates.registry") || {};
        const methods = templates[catSel.value] || [];
        nameSel.innerHTML = `<option value="">Template...</option>${methods.map(m => `<option value="${m}">${m}</option>`).join("")}`;
      });
    }

    _on("btn-load-template", "click", _loadTemplate);
    _on("btn-compile", "click", _compile);
    _on("btn-clear-plan", "click", () => { Store.set("plan.current", null); Store.set("plan.compiled", null); render(); });

    // Operator search
    const opSearch = document.getElementById("op-search");
    if (opSearch) {
      opSearch.addEventListener("input", () => {
        const palette = document.getElementById("operator-palette");
        if (palette) palette.innerHTML = _renderPalette(Store.get("operators.registry") || {}, opSearch.value.toLowerCase());
        _bindPaletteClicks();
      });
    }

    _bindPaletteClicks();

    // Step actions
    _el.querySelectorAll(".btn-step-remove").forEach(btn => {
      btn.addEventListener("click", () => _removeStep(parseInt(btn.dataset.idx)));
    });
    _el.querySelectorAll(".btn-step-up").forEach(btn => {
      btn.addEventListener("click", () => _moveStep(parseInt(btn.dataset.idx), -1));
    });
    _el.querySelectorAll(".btn-step-down").forEach(btn => {
      btn.addEventListener("click", () => _moveStep(parseInt(btn.dataset.idx), 1));
    });

    // Param changes
    _el.querySelectorAll(".param-val").forEach(input => {
      input.addEventListener("change", () => {
        const stepIdx = parseInt(input.dataset.step);
        const param = input.dataset.param;
        const plan = Store.get("plan.current");
        if (!plan || !plan.steps[stepIdx]) return;
        let val;
        if (input.type === "checkbox") val = input.checked;
        else if (input.type === "number") val = parseFloat(input.value);
        else val = input.value;
        plan.steps[stepIdx].params[param] = val;
        Store.set("plan.current", plan);
        Store.set("plan.dirty", true);
      });
    });

    // Drag-and-drop reordering
    _el.querySelectorAll(".plan-step[draggable]").forEach(step => {
      step.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("text/plain", step.dataset.stepIdx);
        step.classList.add("dragging");
      });
      step.addEventListener("dragend", () => step.classList.remove("dragging"));
      step.addEventListener("dragover", (e) => e.preventDefault());
      step.addEventListener("drop", (e) => {
        e.preventDefault();
        const from = parseInt(e.dataTransfer.getData("text/plain"));
        const to = parseInt(step.dataset.stepIdx);
        if (from !== to) _moveStep(from, to - from);
      });
    });
  }

  function _bindPaletteClicks() {
    if (!_el) return;
    _el.querySelectorAll(".op-item[data-op]").forEach(item => {
      item.addEventListener("click", () => _addOperator(item.dataset.op));
    });
  }

  function _addOperator(opName) {
    const ops = Store.get("operators.registry") || {};
    const opDef = ops[opName];
    if (!opDef) return;

    let plan = Store.get("plan.current");
    if (!plan) {
      plan = { name: "Custom Plan", procedure: opDef.procedure || "rhinoplasty", description: "", steps: [] };
    }

    const step = {
      name: opName,
      category: opDef.category,
      procedure: opDef.procedure,
      params: {},
      param_defs: opDef.param_defs || {},
      affected_structures: opDef.affected_structures || [],
      description: opDef.description || "",
    };

    // Set defaults from param_defs
    for (const [k, d] of Object.entries(step.param_defs)) {
      step.params[k] = d.default != null ? d.default : 0;
    }

    plan.steps.push(step);
    Store.set("plan.current", plan);
    Store.set("plan.dirty", true);
    render();
    Toast.info(`Added ${opName}`);
  }

  function _removeStep(idx) {
    const plan = Store.get("plan.current");
    if (!plan) return;
    plan.steps.splice(idx, 1);
    Store.set("plan.current", plan.steps.length > 0 ? plan : null);
    Store.set("plan.dirty", true);
    render();
  }

  function _moveStep(idx, delta) {
    const plan = Store.get("plan.current");
    if (!plan) return;
    const newIdx = idx + delta;
    if (newIdx < 0 || newIdx >= plan.steps.length) return;
    const [removed] = plan.steps.splice(idx, 1);
    plan.steps.splice(newIdx, 0, removed);
    Store.set("plan.current", plan);
    Store.set("plan.dirty", true);
    render();
  }

  async function _loadTemplate() {
    const cat = document.getElementById("template-category").value;
    const name = document.getElementById("template-name").value;
    if (!cat || !name) { Toast.warning("Select both category and template"); return; }

    try {
      const data = await API.loadTemplate(cat, name);
      if (data.plan) {
        Store.set("plan.current", data.plan);
        Store.set("plan.compiled", null);
        Store.set("plan.dirty", false);
        Toast.success(`Loaded template: ${cat}/${name}`);
        render();
      }
    } catch (err) {
      Toast.error("Template load failed: " + (err.message || err));
    }
  }

  async function _compile() {
    const caseId = Store.get("selectedCase.id");
    const plan = Store.get("plan.current");
    if (!caseId || !plan) { Toast.warning("Select a case and build a plan first"); return; }

    Store.set("plan.compiling", true);
    render();

    try {
      const result = await API.compilePlan(caseId, plan);
      Store.set("plan.compiled", result);
      Store.set("plan.dirty", false);
      Inspector.show("Compilation Result", result);
      Toast.success("Plan compiled");
    } catch (err) {
      Toast.error("Compilation failed: " + (err.message || err));
    } finally {
      Store.set("plan.compiling", false);
      render();
    }
  }

  function _on(id, evt, fn) { const el = document.getElementById(id); if (el) el.addEventListener(evt, fn); }
  function _esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }
  function _truncate(s, n) { return s && s.length > n ? s.slice(0, n) + "…" : s || ""; }

  return { init, load, render };
})();
