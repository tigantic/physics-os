/**
 * Ontic Facial Plastics — G1 Case Library Mode
 *
 * Case list with sort, filter, search. CRUD. Curation.
 * Virtual row rendering for performance.
 */

"use strict";

const CaseLibrary = (() => {
  let _panelEl = null;

  function init() {
    _panelEl = document.getElementById("mode-case-library");
    if (!_panelEl) return;
  }

  async function load() {
    const params = {};
    const proc = Store.get("cases.filterProcedure");
    const qual = Store.get("cases.filterQuality");
    const search = Store.get("cases.search");
    if (proc) params.procedure = proc;
    if (qual) params.quality = qual;
    if (search) params.search = search;
    params.limit = Store.get("cases.limit");
    params.offset = Store.get("cases.offset");

    Store.set("cases.loading", true);
    try {
      const data = await API.listCases(params);
      Store.set("cases.items", data.cases || []);
      Store.set("cases.total", data.total || 0);
    } catch (err) {
      Toast.error("Failed to load cases: " + (err.message || err));
    } finally {
      Store.set("cases.loading", false);
    }
  }

  function render() {
    if (!_panelEl) return;
    const cases = Store.get("cases.items") || [];
    const total = Store.get("cases.total") || 0;
    const loading = Store.get("cases.loading");
    const selectedId = Store.get("selectedCase.id");
    const contract = Store.get("system.contract");
    const procedures = (contract && contract.procedures) || [];

    _panelEl.innerHTML = `
      <div class="panel-header">
        <h2>Case Library</h2>
        <div class="panel-actions">
          <button class="btn btn-primary" id="btn-refresh-cases">Refresh</button>
          <button class="btn btn-secondary" id="btn-create-case">New Case</button>
          <button class="btn btn-secondary" id="btn-curate">Curate</button>
        </div>
      </div>
      <div class="filter-bar">
        <input type="search" id="case-search" placeholder="Search cases..." value="${_esc(Store.get("cases.search") || "")}">
        <select id="filter-procedure">
          <option value="">All Procedures</option>
          ${procedures.map(p => `<option value="${_esc(p)}" ${Store.get("cases.filterProcedure") === p ? "selected" : ""}>${_esc(p)}</option>`).join("")}
        </select>
        <select id="filter-quality">
          <option value="">All Quality</option>
          <option value="clinical" ${Store.get("cases.filterQuality") === "clinical" ? "selected" : ""}>Clinical</option>
          <option value="research" ${Store.get("cases.filterQuality") === "research" ? "selected" : ""}>Research</option>
          <option value="synthetic" ${Store.get("cases.filterQuality") === "synthetic" ? "selected" : ""}>Synthetic</option>
        </select>
      </div>
      ${loading ? _renderSkeletons() : _renderTable(cases, selectedId)}
      <div class="table-footer">
        <span>${total} case${total !== 1 ? "s" : ""}</span>
        <div class="page-controls">
          <button class="btn btn-sm btn-ghost" id="btn-prev-page" ${Store.get("cases.offset") === 0 ? "disabled" : ""}>← Prev</button>
          <button class="btn btn-sm btn-ghost" id="btn-next-page" ${(Store.get("cases.offset") + Store.get("cases.limit")) >= total ? "disabled" : ""}>Next →</button>
        </div>
      </div>
    `;

    _bind();
  }

  function _renderSkeletons() {
    let html = '<div style="padding:var(--space-2) 0">';
    for (let i = 0; i < 8; i++) html += '<div class="skeleton skeleton-row"></div>';
    html += '</div>';
    return html;
  }

  function _renderTable(cases, selectedId) {
    if (cases.length === 0) {
      return `<div class="placeholder">No cases found. <button class="empty-action" id="btn-empty-create">Create one</button></div>`;
    }
    return `
      <div class="data-table-wrap">
        <table class="data-table">
          <thead><tr>
            <th>Case ID</th>
            <th>Procedure</th>
            <th>Quality</th>
            <th>Created</th>
          </tr></thead>
          <tbody>
            ${cases.map(c => `
              <tr class="${c.case_id === selectedId ? "selected" : ""}" data-case-id="${_esc(c.case_id)}">
                <td class="mono">${_esc(_truncate(c.case_id, 20))}</td>
                <td><span class="tag tag-${_esc(c.procedure_type || "")}">${_esc(c.procedure_type || "—")}</span></td>
                <td><span class="tag tag-quality">${_esc(c.quality_level || "—")}</span></td>
                <td class="mono">${_esc(_fmtDate(c.created_utc || c.created_at))}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function _bind() {
    _on("btn-refresh-cases", "click", () => { API.clearCache(); load(); });
    _on("btn-create-case", "click", openCreateModal);
    _on("btn-curate", "click", _curate);
    _on("btn-empty-create", "click", openCreateModal);
    _on("btn-prev-page", "click", () => {
      const off = Math.max(0, Store.get("cases.offset") - Store.get("cases.limit"));
      Store.set("cases.offset", off);
      load();
    });
    _on("btn-next-page", "click", () => {
      Store.set("cases.offset", Store.get("cases.offset") + Store.get("cases.limit"));
      load();
    });

    const search = document.getElementById("case-search");
    if (search) {
      search.addEventListener("input", _debounce(() => {
        Store.set("cases.search", search.value);
        Store.set("cases.offset", 0);
        load();
      }, 300));
    }

    const filtProc = document.getElementById("filter-procedure");
    if (filtProc) filtProc.addEventListener("change", () => { Store.set("cases.filterProcedure", filtProc.value); Store.set("cases.offset", 0); load(); });

    const filtQual = document.getElementById("filter-quality");
    if (filtQual) filtQual.addEventListener("change", () => { Store.set("cases.filterQuality", filtQual.value); Store.set("cases.offset", 0); load(); });

    // Row selection
    _panelEl.querySelectorAll("tr[data-case-id]").forEach(row => {
      row.addEventListener("click", () => _selectCase(row.dataset.caseId));
    });
  }

  async function _selectCase(caseId) {
    Store.set("selectedCase.id", caseId);
    Store.set("selectedCase.metadata", null);
    Store.set("selectedCase.twin", null);

    try {
      const [caseData, twinData] = await Promise.all([
        API.getCase(caseId),
        API.getTwinSummary(caseId),
      ]);
      Store.set("selectedCase.metadata", caseData.metadata || caseData);
      Store.set("selectedCase.twin", twinData);
      Toast.info(`Case ${_truncate(caseId, 12)} selected`);
    } catch (err) {
      Toast.error("Failed to load case: " + (err.message || err));
    }

    render();
  }

  function openCreateModal() {
    const contract = Store.get("system.contract");
    const procedures = (contract && contract.procedures) || ["rhinoplasty", "facelift", "blepharoplasty", "fillers"];

    const body = document.createElement("div");
    body.innerHTML = `
      <div class="modal-field"><label>Procedure</label><select id="new-case-procedure">${procedures.map(p => `<option value="${p}">${p}</option>`).join("")}</select></div>
      <div class="modal-field"><label>Patient Age</label><input type="number" id="new-case-age" value="40" min="0" max="120"></div>
      <div class="modal-field"><label>Patient Sex</label><select id="new-case-sex"><option value="unknown">Unknown</option><option value="female">Female</option><option value="male">Male</option></select></div>
      <div class="modal-field"><label>Notes</label><textarea id="new-case-notes" rows="3"></textarea></div>
    `;

    Modal.open({
      title: "New Case",
      body: body,
      confirmText: "Create",
      onConfirm: async () => {
        const proc = document.getElementById("new-case-procedure").value;
        const age = parseInt(document.getElementById("new-case-age").value) || 0;
        const sex = document.getElementById("new-case-sex").value;
        const notes = document.getElementById("new-case-notes").value;
        try {
          const res = await API.createCase({ procedure: proc, patient_age: age, patient_sex: sex, notes });
          Toast.success("Case created: " + (res.case_id || "OK"));
          API.clearCache();
          load();
        } catch (err) {
          Toast.error("Create failed: " + (err.message || err));
        }
      },
    });
  }

  async function _curate() {
    Toast.info("Running library curation...");
    try {
      const result = await API.curateLibrary();
      Inspector.show("Curation Results", result);
      Toast.success("Curation complete");
    } catch (err) {
      Toast.error("Curation failed: " + (err.message || err));
    }
  }

  function _on(id, evt, fn) {
    const el = document.getElementById(id);
    if (el) el.addEventListener(evt, fn);
  }

  function _debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }

  function _esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }
  function _truncate(s, n) { return s && s.length > n ? s.slice(0, n) + "…" : s || ""; }
  function _fmtDate(d) { if (!d) return "—"; try { return new Date(d).toLocaleDateString(); } catch (_) { return d; } }

  return { init, load, render, openCreateModal };
})();
