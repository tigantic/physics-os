/**
 * HyperTensor Facial Plastics — G4 What-If Explorer
 *
 * Parameter override editor with baseline labels.
 * Side-by-side diff of baseline vs modified results.
 */

"use strict";

const WhatIf = (() => {
  let _el = null;

  function init() {
    _el = document.getElementById("mode-consult");
  }

  function render() {
    if (!_el) return;
    const plan = Store.get("plan.current");
    const compiled = Store.get("plan.compiled");
    const caseId = Store.get("selectedCase.id");
    const result = Store.get("whatif.result");
    const running = Store.get("whatif.running");

    if (!plan || !compiled) {
      _el.innerHTML = `<div class="panel-header"><h2>Consult — What-If Explorer</h2></div><div class="placeholder">Compile a plan in Plan Author first.</div>`;
      return;
    }

    _el.innerHTML = `
      <div class="panel-header">
        <h2>Consult — What-If Explorer</h2>
        <div class="panel-actions">
          <button class="btn btn-primary" id="btn-run-whatif" ${!caseId || running ? "disabled" : ""}>${running ? "Running..." : "Run What-If"}</button>
          <button class="btn btn-secondary" id="btn-open-sweep">Parameter Sweep</button>
        </div>
      </div>
      <div class="split-view" style="min-height:400px">
        <div class="split-left" style="overflow-y:auto">
          <h3 style="font-size:var(--font-size-xs);color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:var(--space-2)">Parameter Overrides</h3>
          <p style="font-size:var(--font-size-xs);color:var(--text-muted);margin-bottom:var(--space-3)">Modify parameters below then run what-if to compare against baseline.</p>
          ${_renderOverrideEditor(plan)}
        </div>
        <div class="split-right" style="overflow-y:auto">
          <h3 style="font-size:var(--font-size-xs);color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:var(--space-2)">Results</h3>
          ${result ? _renderComparison(compiled, result) : '<p class="placeholder">Run a what-if scenario to see results.</p>'}
        </div>
      </div>
    `;

    _bind();
  }

  function _renderOverrideEditor(plan) {
    if (!plan || !plan.steps) return "";
    return plan.steps.map((step, i) => {
      const params = step.params || {};
      const defs = step.param_defs || {};
      const entries = Object.entries(params);
      if (entries.length === 0) return "";

      const paramRows = entries.map(([key, val]) => {
        const def = defs[key] || {};
        const unit = def.unit || "";
        const ptype = def.param_type || "float";
        let input;
        if (ptype === "bool" || typeof val === "boolean") {
          input = `<input type="checkbox" class="whatif-param" data-step="${step.name}" data-param="${_esc(key)}" ${val ? "checked" : ""}>`;
        } else if (def.enum_values && def.enum_values.length > 0) {
          input = `<select class="whatif-param" data-step="${step.name}" data-param="${_esc(key)}">${def.enum_values.map(e => `<option value="${e}" ${e === val ? "selected" : ""}>${e}</option>`).join("")}</select>`;
        } else {
          input = `<input type="number" class="whatif-param" data-step="${step.name}" data-param="${_esc(key)}" value="${val}" step="any">`;
        }
        return `<div class="param-row"><span class="param-label">${_esc(key)}</span><div class="param-input">${input}${unit ? `<span class="param-unit">${_esc(unit)}</span>` : ""}<span class="param-baseline">baseline: ${val}</span></div></div>`;
      }).join("");

      return `<div style="margin-bottom:var(--space-3)"><div style="font-weight:var(--font-weight-semibold);font-size:var(--font-size-sm);margin-bottom:var(--space-1)">${i + 1}. ${_esc(step.name)}</div><div class="param-group">${paramRows}</div></div>`;
    }).join("");
  }

  function _renderComparison(baseline, whatif) {
    const baseResult = baseline;
    const modResult = whatif.result || whatif;

    const fields = [
      { label: "Boundary Conditions", key: "n_bcs" },
      { label: "Material Mods", key: "n_material_mods" },
      { label: "Mesh Mods", key: "n_mesh_mods" },
    ];

    let html = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:var(--space-3);margin-bottom:var(--space-3)">
        <div class="stat-card"><div class="stat-label">Baseline</div>${fields.map(f => `<div class="param-row"><span class="param-label">${f.label}</span><span class="param-baseline">${baseResult[f.key] || 0}</span></div>`).join("")}</div>
        <div class="stat-card"><div class="stat-label">What-If</div>${fields.map(f => {
      const base = baseResult[f.key] || 0;
      const mod = modResult[f.key] || 0;
      const diff = mod - base;
      const cls = diff > 0 ? "positive" : diff < 0 ? "negative" : "zero";
      return `<div class="param-row"><span class="param-label">${f.label}</span><span class="param-baseline">${mod} <span class="delta-value ${cls}">(${diff >= 0 ? "+" : ""}${diff})</span></span></div>`;
    }).join("")}</div>
      </div>
    `;

    if (whatif.modified_operators) {
      html += `<div style="font-size:var(--font-size-xs);color:var(--text-muted)">Modified operators: ${whatif.modified_operators.join(", ")}</div>`;
    }

    return html;
  }

  function _bind() {
    _on("btn-run-whatif", "click", _runWhatIf);
    _on("btn-open-sweep", "click", () => {
      // Store does not have a dedicated sweep mode; render sweep inline
      Sweep.render();
    });
  }

  async function _runWhatIf() {
    const caseId = Store.get("selectedCase.id");
    const plan = Store.get("plan.current");
    if (!caseId || !plan) return;

    // Collect overrides from the form
    const overrides = {};
    const inputs = _el.querySelectorAll(".whatif-param");
    inputs.forEach(input => {
      const step = input.dataset.step;
      const param = input.dataset.param;
      let val;
      if (input.type === "checkbox") val = input.checked;
      else if (input.type === "number") val = parseFloat(input.value);
      else val = input.value;

      // Compare with baseline
      const planStep = plan.steps.find(s => s.name === step);
      if (planStep && planStep.params[param] !== val) {
        if (!overrides[step]) overrides[step] = {};
        overrides[step][param] = val;
      }
    });

    if (Object.keys(overrides).length === 0) {
      Toast.warning("No parameters modified — change values to run what-if");
      return;
    }

    Store.set("whatif.overrides", overrides);
    Store.set("whatif.running", true);
    render();

    try {
      const result = await API.runWhatIf(caseId, plan, overrides);
      Store.set("whatif.result", result);
      Inspector.show("What-If Result", result);
      Toast.success("What-if complete");
    } catch (err) {
      Toast.error("What-if failed: " + (err.message || err));
    } finally {
      Store.set("whatif.running", false);
      render();
    }
  }

  function _on(id, evt, fn) { const el = document.getElementById(id); if (el) el.addEventListener(evt, fn); }
  function _esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }

  return { init, render };
})();
