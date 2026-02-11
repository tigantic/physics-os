/**
 * HyperTensor Facial Plastics — G2 Twin Inspector Mode
 *
 * Digital twin metadata, mesh statistics, landmarks table,
 * region breakdown, data quality indicators.
 */

"use strict";

const TwinInspect = (() => {
  let _el = null;

  function init() {
    _el = document.getElementById("mode-twin-inspect");
  }

  async function load() {
    const caseId = Store.get("selectedCase.id");
    if (!caseId) return;

    // Ensure twin data is loaded
    let twin = Store.get("selectedCase.twin");
    if (!twin || twin.case_id !== caseId) {
      try {
        twin = await API.getTwinSummary(caseId);
        Store.set("selectedCase.twin", twin);
      } catch (err) {
        Toast.error("Failed to load twin: " + (err.message || err));
        return;
      }
    }
  }

  function render() {
    if (!_el) return;
    const caseId = Store.get("selectedCase.id");
    if (!caseId) {
      _el.innerHTML = `<div class="panel-header"><h2>Twin Inspector</h2></div><div class="placeholder">Select a case from the Case Library to inspect its digital twin.</div>`;
      return;
    }

    const twin = Store.get("selectedCase.twin");
    if (!twin) {
      _el.innerHTML = `<div class="panel-header"><h2>Twin Inspector</h2></div><div class="placeholder">Loading twin data...</div>`;
      return;
    }

    const mesh = twin.mesh;
    const landmarks = twin.landmarks || {};
    const seg = twin.segmentation;
    const lmEntries = Object.entries(landmarks);

    _el.innerHTML = `
      <div class="panel-header">
        <h2>Twin Inspector</h2>
        <div class="panel-actions">
          <button class="btn btn-secondary" id="btn-open-3d">Open Full 3D</button>
        </div>
      </div>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-label">Nodes</div>
          <div class="stat-value">${mesh ? _fmtNum(mesh.n_nodes) : "—"}</div>
          <div class="stat-sub">${mesh ? "vertex positions" : "No mesh"}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Elements</div>
          <div class="stat-value">${mesh ? _fmtNum(mesh.n_elements) : "—"}</div>
          <div class="stat-sub">${mesh ? _esc(mesh.element_type || "") : "No mesh"}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Regions</div>
          <div class="stat-value">${mesh ? mesh.n_regions : "—"}</div>
          <div class="stat-sub">tissue groups</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Landmarks</div>
          <div class="stat-value">${lmEntries.length}</div>
          <div class="stat-sub">anatomical points</div>
        </div>
      </div>
      <div class="split-view" style="min-height:300px">
        <div class="split-left">
          ${mesh && mesh.regions ? _renderRegions(mesh.regions) : '<p class="placeholder">No region data</p>'}
          <h3 style="margin-top:var(--space-4);font-size:var(--font-size-sm);color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;">Landmarks</h3>
          ${lmEntries.length > 0 ? _renderLandmarks(lmEntries) : '<p class="placeholder">No landmarks</p>'}
        </div>
        <div class="split-right">
          <div id="twin-preview-container" class="viewer3d-wrap" style="min-height:350px">
            <canvas id="twin-preview-canvas"></canvas>
            <div class="viewer-info"><span>Preview</span></div>
          </div>
        </div>
      </div>
    `;

    _bind();
  }

  function _renderRegions(regions) {
    const entries = Object.entries(regions);
    if (entries.length === 0) return '<p class="placeholder">No regions</p>';
    let html = `
      <h3 style="font-size:var(--font-size-sm);color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:var(--space-2);">Regions</h3>
      <table class="data-table" style="font-size:var(--font-size-xs)">
        <thead><tr><th>ID</th><th>Structure</th><th>Material</th></tr></thead>
        <tbody>
    `;
    for (const [rid, info] of entries) {
      html += `<tr><td class="mono">${_esc(rid)}</td><td>${_esc(info.structure)}</td><td class="mono">${_esc(info.material)}</td></tr>`;
    }
    html += `</tbody></table>`;
    return html;
  }

  function _renderLandmarks(entries) {
    let html = '<div class="landmark-list">';
    for (const [name, pos] of entries) {
      const coords = Array.isArray(pos) ? pos.map(v => v.toFixed(2)).join(", ") : String(pos);
      html += `<div class="landmark-item"><span class="lm-name">${_esc(name)}</span><span class="lm-coords">[${coords}] mm</span></div>`;
    }
    html += '</div>';
    return html;
  }

  function _bind() {
    const btn3d = document.getElementById("btn-open-3d");
    if (btn3d) btn3d.addEventListener("click", () => Router.navigate("visualization"));

    // Try to render a preview in the canvas
    const canvas = document.getElementById("twin-preview-canvas");
    if (canvas) _renderPreview(canvas);
  }

  function _renderPreview(canvas) {
    // Use the lightweight Canvas2D preview since Three.js may not be loaded yet
    const caseId = Store.get("selectedCase.id");
    if (!caseId) return;

    API.getVisualization(caseId).then(data => {
      if (!data || !data.mesh) return;
      Viewer3D.renderPreview(canvas, data);
    }).catch(() => {});
  }

  function _esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }
  function _fmtNum(n) { return typeof n === "number" ? n.toLocaleString() : String(n || "0"); }

  return { init, load, render };
})();
