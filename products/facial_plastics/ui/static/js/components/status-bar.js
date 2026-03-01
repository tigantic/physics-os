/**
 * Ontic Facial Plastics — Status Bar
 *
 * Polls /health every 30s, /metrics every 60s.
 * Shows version, request/error counters, auth status.
 */

"use strict";

const StatusBar = (() => {
  let _el = null;
  let _healthTimer = null;
  let _metricsTimer = null;

  function init() {
    _el = document.getElementById("status-bar");
    if (!_el) return;
    _render();

    Store.subscribe("auth.connected", _render);
    Store.subscribe("system", _render);

    // Start polling
    _pollHealth();
    _pollMetrics();
    _healthTimer = setInterval(_pollHealth, 30000);
    _metricsTimer = setInterval(_pollMetrics, 60000);
  }

  async function _pollHealth() {
    try {
      const data = await API.getHealth();
      Store.set("system.health", data);
      Store.set("auth.connected", true);
    } catch (_) {
      Store.set("auth.connected", false);
    }
  }

  async function _pollMetrics() {
    try {
      const text = await API.getMetrics();
      const parsed = _parsePrometheus(text);
      Store.set("system.metrics", parsed);
    } catch (_) { /* silent */ }
  }

  function _parsePrometheus(text) {
    const metrics = {};
    for (const line of text.split("\n")) {
      if (line.startsWith("#") || !line.trim()) continue;
      const match = line.match(/^(\S+)\s+(\S+)/);
      if (match) metrics[match[1]] = parseFloat(match[2]);
    }
    return metrics;
  }

  function _render() {
    if (!_el) return;
    const connected = Store.get("auth.connected");
    const health = Store.get("system.health");
    const metrics = Store.get("system.metrics") || {};
    const version = Store.get("system.version");
    const apiKey = Store.get("auth.apiKey");

    const requests = metrics["fp_requests_total"] || 0;
    const errors = metrics["fp_errors_total"] || 0;

    _el.innerHTML = `
      <div class="status-group">
        <span class="status-item">The Physics OS FP v${_esc(version)}</span>
        <span class="status-item"><span class="status-dot ${connected ? "ok" : "err"}"></span> ${connected ? "Connected" : "Disconnected"}</span>
      </div>
      <div class="status-group">
        <span class="status-item">Requests: ${_fmtNum(requests)}</span>
        <span class="status-item">Errors: ${_fmtNum(errors)}</span>
        ${health && health.workers ? `<span class="status-item">Workers: ${health.workers}</span>` : ""}
        <span class="status-item status-auth-btn" id="status-auth-toggle" style="cursor:pointer;" title="Click to ${apiKey ? 'change' : 'set'} API key">${apiKey ? "🔑 Auth" : "⚠ No Key"}</span>
      </div>
    `;

    // Wire click on auth indicator
    const authBtn = document.getElementById("status-auth-toggle");
    if (authBtn) {
      authBtn.addEventListener("click", () => {
        if (typeof App !== "undefined" && App.showAuthPrompt) App.showAuthPrompt();
      });
    }
  }

  function _esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }
  function _fmtNum(n) { return typeof n === "number" ? n.toLocaleString() : "0"; }

  return { init };
})();
