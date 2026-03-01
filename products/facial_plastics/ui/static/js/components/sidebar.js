/**
 * Ontic Facial Plastics — Context Sidebar
 *
 * Case metadata, twin status, quick actions.
 * Updates reactively when selectedCase changes.
 */

"use strict";

const Sidebar = (() => {
  let _el = null;

  function init() {
    _el = document.getElementById("sidebar");
    if (!_el) return;

    Store.subscribe("selectedCase", _render);
    Store.subscribe("ui.sidebarOpen", (open) => {
      document.getElementById("app")?.classList.toggle("sidebar-collapsed", !open);
    });

    // Apply initial sidebar state
    if (!Store.get("ui.sidebarOpen")) {
      document.getElementById("app")?.classList.add("sidebar-collapsed");
    }

    _render();
  }

  function _render() {
    if (!_el) return;
    const sc = Store.get("selectedCase");
    if (!sc || !sc.id) {
      _el.innerHTML = `<div class="sidebar-empty">Select a case from the Case Library to begin.</div>`;
      return;
    }

    const meta = sc.metadata || {};
    const twin = sc.twin || {};
    const mesh = twin.mesh;
    const landmarks = twin.landmarks || {};
    const seg = twin.segmentation;
    const lmCount = Object.keys(landmarks).length;
    // summary() returns "procedure" / "quality"; CaseIndex returns "procedure_type" / "quality_level"
    const procLabel = meta.procedure_type || meta.procedure || "";
    const qualLabel = meta.quality_level || meta.quality || "";

    _el.innerHTML = `
      <div class="sidebar-section">
        <h3>Selected Case</h3>
        <div class="sidebar-field"><span class="label">Case ID</span><span class="value" title="${_esc(sc.id)}">${_esc(_truncate(sc.id, 16))}</span></div>
        ${procLabel ? `<div class="sidebar-field"><span class="label">Procedure</span><span class="tag tag-${_esc(procLabel)}">${_esc(procLabel)}</span></div>` : ""}
        ${qualLabel ? `<div class="sidebar-field"><span class="label">Quality</span><span class="tag tag-quality">${_esc(qualLabel)}</span></div>` : ""}
      </div>
      <div class="sidebar-section">
        <h3>Twin Status</h3>
        <div class="sidebar-field"><span class="label">Mesh</span>${mesh ? `<span class="sidebar-badge available">✓ ${_fmtNum(mesh.n_nodes)} nodes</span>` : `<span class="sidebar-badge missing">✗ none</span>`}</div>
        ${mesh ? `<div class="sidebar-field"><span class="label">Elements</span><span class="sidebar-badge available">✓ ${_fmtNum(mesh.n_elements)} ${_esc(mesh.element_type || "")}</span></div>` : ""}
        <div class="sidebar-field"><span class="label">Landmarks</span>${lmCount > 0 ? `<span class="sidebar-badge available">✓ ${lmCount} points</span>` : `<span class="sidebar-badge missing">✗ none</span>`}</div>
        <div class="sidebar-field"><span class="label">Segmentation</span>${seg ? `<span class="sidebar-badge available">✓ ${seg.n_labels} labels</span>` : `<span class="sidebar-badge missing">✗ none</span>`}</div>
      </div>
      <div class="sidebar-section">
        <h3>Quick Actions</h3>
        <div class="sidebar-actions">
          <button class="sidebar-action-btn" data-action="twin-inspect">Inspect Twin</button>
          <button class="sidebar-action-btn" data-action="plan-author">Author Plan</button>
          <button class="sidebar-action-btn" data-action="viewer3d">3D View</button>
          <button class="sidebar-action-btn" data-action="timeline">Timeline</button>
        </div>
      </div>
    `;

    // Quick action handlers
    _el.querySelectorAll("[data-action]").forEach(btn => {
      btn.addEventListener("click", () => Router.navigate(btn.dataset.action));
    });
  }

  function _esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }
  function _truncate(s, n) { return s && s.length > n ? s.slice(0, n) + "…" : s; }
  function _fmtNum(n) { return typeof n === "number" ? n.toLocaleString() : String(n); }

  return { init };
})();
