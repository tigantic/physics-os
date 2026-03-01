/**
 * Ontic Facial Plastics — G3.5 Simulation Console
 *
 * Structured compilation result viewer: boundary conditions,
 * material modifications, mesh modifications.
 */

"use strict";

const Simulation = (() => {
  // Simulation console renders inline within Plan Author's compile result
  // and in the Inspector drawer. This module provides structured formatting.

  function formatResult(result) {
    if (!result) return '<p class="placeholder">No compilation result available.</p>';

    const bcs = result.boundary_conditions || [];
    const matMods = result.material_modifications || [];
    const meshMods = result.mesh_modifications || [];

    let html = `
      <div style="margin-bottom:var(--space-3)">
        <div style="display:flex;align-items:center;gap:var(--space-2);margin-bottom:var(--space-2)">
          <span style="color:var(--c-success);font-weight:var(--font-weight-bold);font-size:var(--font-size-md)">✓</span>
          <span style="font-weight:var(--font-weight-semibold)">Compilation Successful</span>
          ${result.content_hash ? `<span style="font-family:var(--font-mono);font-size:var(--font-size-xs);color:var(--text-muted);margin-left:auto">${_esc(result.content_hash.slice(0, 16))}…</span>` : ""}
        </div>
      </div>
    `;

    // Boundary Conditions
    html += `<h4 style="font-size:var(--font-size-xs);color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin:var(--space-3) 0 var(--space-2)">Boundary Conditions (${result.n_bcs || bcs.length})</h4>`;
    if (bcs.length > 0) {
      html += `<table class="data-table" style="font-size:var(--font-size-xs)"><thead><tr><th>Type</th><th>Region</th><th>Value</th></tr></thead><tbody>`;
      for (const bc of bcs) {
        html += `<tr><td>${_esc(bc.type || bc.kind || "")}</td><td>${_esc(bc.region || "")}</td><td class="mono">${_esc(String(bc.value || ""))}</td></tr>`;
      }
      html += `</tbody></table>`;
    } else {
      html += `<div style="font-size:var(--font-size-sm);color:var(--text-muted)">Count: ${result.n_bcs || 0}</div>`;
    }

    // Material Modifications
    html += `<h4 style="font-size:var(--font-size-xs);color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin:var(--space-3) 0 var(--space-2)">Material Modifications (${result.n_material_mods || matMods.length})</h4>`;
    if (matMods.length > 0) {
      html += `<table class="data-table" style="font-size:var(--font-size-xs)"><thead><tr><th>Region</th><th>Property</th><th>Original</th><th>Modified</th></tr></thead><tbody>`;
      for (const m of matMods) {
        html += `<tr><td>${_esc(m.region || "")}</td><td>${_esc(m.property || "")}</td><td class="mono">${m.original || "—"}</td><td class="mono">${m.modified || "—"}</td></tr>`;
      }
      html += `</tbody></table>`;
    } else {
      html += `<div style="font-size:var(--font-size-sm);color:var(--text-muted)">Count: ${result.n_material_mods || 0}</div>`;
    }

    // Mesh Modifications
    html += `<h4 style="font-size:var(--font-size-xs);color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;margin:var(--space-3) 0 var(--space-2)">Mesh Modifications (${result.n_mesh_mods || meshMods.length})</h4>`;
    if (meshMods.length > 0) {
      for (const m of meshMods) {
        html += `<div style="font-size:var(--font-size-xs);color:var(--text-secondary);padding:var(--space-1) 0">${_esc(JSON.stringify(m))}</div>`;
      }
    } else {
      html += `<div style="font-size:var(--font-size-sm);color:var(--text-muted)">Count: ${result.n_mesh_mods || 0}</div>`;
    }

    return html;
  }

  function _esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }

  return { formatResult };
})();
