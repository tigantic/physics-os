/**
 * HyperTensor Facial Plastics — G8 Plan & Case Comparison
 *
 * Dual-slot plan comparison with delta highlights.
 * Case comparison with mesh-diff table.
 */

"use strict";

const Compare = (() => {
  let _el = null;

  function init() {
    _el = document.getElementById("mode-compare");
  }

  async function load() {
    if (!_el) return;
    render();
  }

  function render() {
    if (!_el) return;
    const tab = Store.get("compare.tab") || "plans";
    const result = Store.get("compare.result");
    const running = Store.get("compare.running");

    _el.innerHTML = `
      <div class="panel-header">
        <h2>Compare</h2>
        <div class="panel-actions">
          <button class="btn ${tab === "plans" ? "btn-primary" : "btn-secondary"} btn-sm" id="compare-tab-plans">Plans</button>
          <button class="btn ${tab === "cases" ? "btn-primary" : "btn-secondary"} btn-sm" id="compare-tab-cases">Cases</button>
        </div>
      </div>
      ${tab === "plans" ? _renderPlansTab(result, running) : _renderCasesTab(result, running)}
    `;

    _bind();
  }

  /* ── Plan comparison ─────────────────────────────────────── */

  function _renderPlansTab(result, running) {
    const plan = Store.get("plan.current");
    const caseId = Store.get("selectedCase.id");
    const planA = Store.get("compare.planA") || null;
    const planB = Store.get("compare.planB") || null;

    return `
      <div class="compare-header" style="margin-bottom:var(--space-3)">
        <p style="font-size:var(--font-size-xs);color:var(--text-muted)">Load two different plan configurations and compare compiled results side by side.</p>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:var(--space-3);margin-bottom:var(--space-3)">
        <div class="stat-card">
          <div class="stat-label">Plan A (current)</div>
          <div style="font-size:var(--font-size-xs);color:var(--text-secondary)">${plan ? (plan.steps || []).length + " steps" : "No plan loaded"}</div>
          <button class="btn btn-secondary btn-sm" id="btn-set-plan-a" style="margin-top:var(--space-2)" ${!plan ? "disabled" : ""}>Set as Plan A</button>
        </div>
        <div class="stat-card">
          <div class="stat-label">Plan B</div>
          <div style="font-size:var(--font-size-xs);color:var(--text-secondary)">${planB ? (planB.steps || []).length + " steps" : "Not set"}</div>
          <p style="font-size:var(--font-size-xs);color:var(--text-muted);margin-top:var(--space-1)">Modify plan in Plan Author, return here, and click "Set as Plan B".</p>
          <button class="btn btn-secondary btn-sm" id="btn-set-plan-b" style="margin-top:var(--space-2)" ${!plan ? "disabled" : ""}>Set as Plan B</button>
        </div>
      </div>
      <div style="margin-bottom:var(--space-3)">
        <button class="btn btn-primary" id="btn-compare-plans" ${!caseId || !planA || !planB || running ? "disabled" : ""}>${running ? "Comparing..." : "Compare Plans"}</button>
      </div>
      ${result && result.delta ? _renderPlanDelta(result) : ""}
    `;
  }

  function _renderPlanDelta(result) {
    const delta = result.delta || {};
    const planAResult = result.plan_a || result.baseline || {};
    const planBResult = result.plan_b || result.modified || {};

    const rows = Object.keys(delta).map(key => {
      const val = delta[key];
      const numVal = typeof val === "number" ? val : null;
      const cls = numVal > 0 ? "positive" : numVal < 0 ? "negative" : "zero";
      return `<tr><td>${_esc(key)}</td><td>${_esc(String(planAResult[key] ?? "—"))}</td><td>${_esc(String(planBResult[key] ?? "—"))}</td><td class="delta-value ${cls}">${numVal !== null ? (numVal >= 0 ? "+" : "") + numVal : _esc(String(val))}</td></tr>`;
    }).join("");

    return `
      <div class="compare-delta">
        <h3 style="font-size:var(--font-size-sm);margin-bottom:var(--space-2)">Delta Analysis</h3>
        <table class="data-table">
          <thead><tr><th>Metric</th><th>Plan A</th><th>Plan B</th><th>Delta</th></tr></thead>
          <tbody>${rows || '<tr><td colspan="4" class="placeholder">No differences detected</td></tr>'}</tbody>
        </table>
      </div>
    `;
  }

  /* ── Case comparison ─────────────────────────────────────── */

  function _renderCasesTab(result, running) {
    const cases = Store.get("cases") || [];
    const caseA = Store.get("compare.caseA") || "";
    const caseB = Store.get("compare.caseB") || "";

    const opts = cases.map(c => {
      const id = c.case_id || c.id || "";
      return `<option value="${_esc(id)}">${_esc(id)}</option>`;
    }).join("");

    return `
      <div style="display:grid;grid-template-columns:1fr 1fr auto;gap:var(--space-3);align-items:end;margin-bottom:var(--space-3)">
        <div>
          <label style="display:block;font-size:var(--font-size-xs);color:var(--text-muted);margin-bottom:var(--space-1)">Case A</label>
          <select class="input" id="compare-case-a"><option value="">Select…</option>${opts}</select>
        </div>
        <div>
          <label style="display:block;font-size:var(--font-size-xs);color:var(--text-muted);margin-bottom:var(--space-1)">Case B</label>
          <select class="input" id="compare-case-b"><option value="">Select…</option>${opts}</select>
        </div>
        <button class="btn btn-primary" id="btn-compare-cases" ${running ? "disabled" : ""}>${running ? "Comparing..." : "Compare"}</button>
      </div>
      ${result && result.mesh_diff ? _renderCaseDiff(result) : ""}
    `;
  }

  function _renderCaseDiff(result) {
    const diff = result.mesh_diff || {};
    const rows = Object.entries(diff).map(([key, val]) => {
      const numVal = typeof val === "number" ? val : null;
      const cls = numVal > 0 ? "positive" : numVal < 0 ? "negative" : "zero";
      return `<tr><td>${_esc(key)}</td><td class="delta-value ${cls}">${numVal !== null ? (numVal >= 0 ? "+" : "") + numVal : _esc(String(val))}</td></tr>`;
    }).join("");

    let summaryHtml = "";
    if (result.summary) {
      summaryHtml = `<p style="font-size:var(--font-size-xs);color:var(--text-muted);margin-bottom:var(--space-2)">${_esc(typeof result.summary === "string" ? result.summary : JSON.stringify(result.summary))}</p>`;
    }

    return `
      <div class="compare-delta">
        <h3 style="font-size:var(--font-size-sm);margin-bottom:var(--space-2)">Mesh Comparison</h3>
        ${summaryHtml}
        <table class="data-table">
          <thead><tr><th>Metric</th><th>Difference</th></tr></thead>
          <tbody>${rows || '<tr><td colspan="2" class="placeholder">No differences</td></tr>'}</tbody>
        </table>
      </div>
    `;
  }

  /* ── Bindings ────────────────────────────────────────────── */

  function _bind() {
    _on("compare-tab-plans", "click", () => { Store.set("compare.tab", "plans"); render(); });
    _on("compare-tab-cases", "click", () => { Store.set("compare.tab", "cases"); render(); });

    _on("btn-set-plan-a", "click", () => {
      const plan = Store.get("plan.current");
      if (plan) { Store.set("compare.planA", JSON.parse(JSON.stringify(plan))); Toast.info("Plan A captured"); render(); }
    });
    _on("btn-set-plan-b", "click", () => {
      const plan = Store.get("plan.current");
      if (plan) { Store.set("compare.planB", JSON.parse(JSON.stringify(plan))); Toast.info("Plan B captured"); render(); }
    });

    _on("btn-compare-plans", "click", _comparePlans);
    _on("btn-compare-cases", "click", _compareCases);
  }

  async function _comparePlans() {
    const caseId = Store.get("selectedCase.id");
    const planA = Store.get("compare.planA");
    const planB = Store.get("compare.planB");
    if (!caseId || !planA || !planB) return;

    Store.set("compare.running", true);
    Store.set("compare.result", null);
    render();

    try {
      const result = await API.comparePlans(caseId, planA, planB);
      Store.set("compare.result", result);
      Inspector.show("Plan Comparison", result);
      Toast.success("Plan comparison complete");
    } catch (err) {
      Toast.error("Comparison failed: " + (err.message || err));
    } finally {
      Store.set("compare.running", false);
      render();
    }
  }

  async function _compareCases() {
    const a = document.getElementById("compare-case-a");
    const b = document.getElementById("compare-case-b");
    const caseA = a ? a.value : "";
    const caseB = b ? b.value : "";
    if (!caseA || !caseB) { Toast.warning("Select two cases to compare"); return; }
    if (caseA === caseB) { Toast.warning("Select two different cases"); return; }

    Store.set("compare.caseA", caseA);
    Store.set("compare.caseB", caseB);
    Store.set("compare.running", true);
    Store.set("compare.result", null);
    render();

    try {
      const result = await API.compareCases(caseA, caseB);
      Store.set("compare.result", result);
      Inspector.show("Case Comparison", result);
      Toast.success("Case comparison complete");
    } catch (err) {
      Toast.error("Comparison failed: " + (err.message || err));
    } finally {
      Store.set("compare.running", false);
      render();
    }
  }

  function _on(id, evt, fn) { const el = document.getElementById(id); if (el) el.addEventListener(evt, fn); }
  function _esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }

  return { init, load, render };
})();
