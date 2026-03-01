/**
 * Ontic Facial Plastics — G5 Report Generation
 *
 * Format selector (html/markdown/json), generate via API,
 * display in sandboxed iframe (HTML) or <pre> (markdown/json),
 * export / download.
 */

"use strict";

const Report = (() => {
  let _el = null;

  function init() {
    _el = document.getElementById("mode-report");
  }

  function render() {
    if (!_el) return;
    const caseId = Store.get("selectedCase.id");
    const plan = Store.get("plan.current");
    const compiled = Store.get("plan.compiled");
    const reportData = Store.get("report.data");
    const format = Store.get("report.format") || "html";
    const generating = Store.get("report.generating");

    _el.innerHTML = `
      <div class="panel-header">
        <h2>Report Generator</h2>
        <div class="panel-actions">
          <select class="input" id="report-format" style="width:auto">
            <option value="html" ${format === "html" ? "selected" : ""}>HTML Report</option>
            <option value="markdown" ${format === "markdown" ? "selected" : ""}>Markdown</option>
            <option value="json" ${format === "json" ? "selected" : ""}>JSON Data</option>
          </select>
          <button class="btn btn-primary" id="btn-generate-report" ${!caseId || !compiled || generating ? "disabled" : ""}>${generating ? "Generating..." : "Generate"}</button>
          ${reportData ? '<button class="btn btn-secondary" id="btn-export-report">Export</button>' : ""}
        </div>
      </div>
      <div class="report-output" id="report-output">
        ${reportData ? _renderOutput(reportData, format) : _emptyState(caseId, compiled)}
      </div>
    `;

    _bind();
  }

  function _emptyState(caseId, compiled) {
    if (!caseId) return '<div class="placeholder">Select a case from the Case Library.</div>';
    if (!compiled) return '<div class="placeholder">Compile a plan in Plan Author first.</div>';
    return '<div class="placeholder">Choose a format and click Generate.</div>';
  }

  function _renderOutput(data, format) {
    const content = data.content || data.report || data.html || JSON.stringify(data, null, 2);

    if (format === "html" && typeof content === "string" && content.includes("<")) {
      return `<iframe id="report-iframe" class="report-iframe" sandbox="allow-same-origin" srcdoc="${_escAttr(content)}" style="width:100%;min-height:500px;border:none;border-radius:var(--radius-md);background:#fff"></iframe>`;
    }

    if (format === "json") {
      const jsonStr = typeof data === "string" ? data : JSON.stringify(data, null, 2);
      return `<pre class="report-pre">${_esc(jsonStr)}</pre>`;
    }

    // Markdown or fallback
    return `<pre class="report-pre">${_esc(content)}</pre>`;
  }

  function _bind() {
    const formatSel = document.getElementById("report-format");
    if (formatSel) {
      formatSel.addEventListener("change", () => {
        Store.set("report.format", formatSel.value);
        // Re-render but keep existing data
        render();
      });
    }

    _on("btn-generate-report", "click", _generate);
    _on("btn-export-report", "click", _export);
  }

  async function _generate() {
    const caseId = Store.get("selectedCase.id");
    const plan = Store.get("plan.current");
    const compiled = Store.get("plan.compiled");
    const format = Store.get("report.format") || "html";

    if (!caseId || !compiled) return;

    Store.set("report.generating", true);
    Store.set("report.data", null);
    render();

    try {
      const result = await API.generateReport(caseId, plan, format);
      Store.set("report.data", result);
      Toast.success("Report generated");
    } catch (err) {
      Toast.error("Report generation failed: " + (err.message || err));
    } finally {
      Store.set("report.generating", false);
      render();
    }
  }

  function _export() {
    const data = Store.get("report.data");
    const format = Store.get("report.format") || "html";
    if (!data) return;

    const content = data.content || data.report || data.html || JSON.stringify(data, null, 2);
    const mimeMap = { html: "text/html", markdown: "text/markdown", json: "application/json" };
    const extMap = { html: "html", markdown: "md", json: "json" };

    const blob = new Blob([content], { type: mimeMap[format] || "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `surgical_report_${Store.get("selectedCase.id") || "unknown"}.${extMap[format] || "txt"}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    Toast.info("Report exported");
  }

  function _on(id, evt, fn) { const el = document.getElementById(id); if (el) el.addEventListener(evt, fn); }
  function _esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }
  function _escAttr(s) { return (s || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  return { init, render };
})();
