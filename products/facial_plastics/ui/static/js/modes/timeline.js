/**
 * Ontic Facial Plastics — G7 Timeline / Audit Trail
 *
 * Vertical event timeline from API.getTimeline().
 * Expandable event details, simulation scrubber.
 */

"use strict";

const Timeline = (() => {
  let _el = null;

  function init() {
    _el = document.getElementById("mode-timeline");
  }

  async function load() {
    const caseId = Store.get("selectedCase.id");
    if (!caseId) { render(); return; }

    Store.set("timeline.loading", true);
    render();

    try {
      const data = await API.getTimeline(caseId);
      Store.set("timeline.events", data.events || data.timeline || []);
      Store.set("timeline.meta", data.meta || {});
    } catch (err) {
      Toast.error("Timeline load failed: " + (err.message || err));
      Store.set("timeline.events", []);
    } finally {
      Store.set("timeline.loading", false);
      render();
    }
  }

  function render() {
    if (!_el) return;
    const caseId = Store.get("selectedCase.id");
    const events = Store.get("timeline.events") || [];
    const loading = Store.get("timeline.loading");
    const expanded = Store.get("timeline.expanded") || {};

    if (!caseId) {
      _el.innerHTML = `<div class="panel-header"><h2>Timeline</h2></div><div class="placeholder">Select a case from the Case Library.</div>`;
      return;
    }

    _el.innerHTML = `
      <div class="panel-header">
        <h2>Timeline</h2>
        <div class="panel-actions">
          <button class="btn btn-secondary btn-sm" id="btn-timeline-refresh" ${loading ? "disabled" : ""}>Refresh</button>
          <button class="btn btn-secondary btn-sm" id="btn-timeline-expand-all">Expand All</button>
          <button class="btn btn-secondary btn-sm" id="btn-timeline-collapse-all">Collapse All</button>
        </div>
      </div>
      ${loading ? _skeleton() : _renderTimeline(events, expanded)}
    `;

    _bind(events);
  }

  function _renderTimeline(events, expanded) {
    if (events.length === 0) {
      return '<div class="placeholder">No timeline events for this case.</div>';
    }

    const items = events.map((evt, i) => {
      const isExpanded = !!expanded[i];
      const typeClass = _typeClass(evt.type || evt.event_type || "info");
      const ts = evt.timestamp || evt.time || "";
      const title = evt.title || evt.event_type || evt.type || "Event";
      const desc = evt.description || evt.summary || "";
      const detail = evt.detail || evt.data;

      return `
        <div class="timeline-event ${isExpanded ? "expanded" : ""}" data-idx="${i}">
          <div class="event-dot ${typeClass}"></div>
          <div class="event-body">
            <div class="event-header" data-toggle="${i}">
              <span class="event-type ${typeClass}">${_esc(String(evt.type || evt.event_type || ""))}</span>
              <span class="event-title">${_esc(title)}</span>
              <span class="event-time">${_formatTime(ts)}</span>
            </div>
            ${desc ? `<div class="event-desc">${_esc(desc)}</div>` : ""}
            ${isExpanded && detail ? `<div class="event-detail"><pre>${_esc(typeof detail === "string" ? detail : JSON.stringify(detail, null, 2))}</pre></div>` : ""}
          </div>
        </div>
      `;
    }).join("");

    return `<div class="timeline-track">${items}</div>`;
  }

  function _typeClass(type) {
    const t = (type || "").toLowerCase();
    if (t.includes("create") || t.includes("init")) return "type-create";
    if (t.includes("plan") || t.includes("compile")) return "type-plan";
    if (t.includes("sim") || t.includes("what")) return "type-sim";
    if (t.includes("report") || t.includes("export")) return "type-report";
    if (t.includes("error") || t.includes("fail")) return "type-error";
    return "type-info";
  }

  function _formatTime(ts) {
    if (!ts) return "";
    try {
      const d = new Date(ts);
      if (isNaN(d.getTime())) return String(ts);
      return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch { return String(ts); }
  }

  function _skeleton() {
    return `<div class="timeline-track">${Array.from({ length: 5 }, () => '<div class="skeleton skeleton-row" style="height:48px;margin-bottom:var(--space-2)"></div>').join("")}</div>`;
  }

  function _bind(events) {
    _on("btn-timeline-refresh", "click", load);

    _on("btn-timeline-expand-all", "click", () => {
      const ex = {};
      events.forEach((_, i) => { ex[i] = true; });
      Store.set("timeline.expanded", ex);
      render();
    });

    _on("btn-timeline-collapse-all", "click", () => {
      Store.set("timeline.expanded", {});
      render();
    });

    // Event header click to toggle
    if (_el) {
      _el.querySelectorAll("[data-toggle]").forEach(hdr => {
        hdr.addEventListener("click", () => {
          const idx = parseInt(hdr.dataset.toggle, 10);
          const ex = Store.get("timeline.expanded") || {};
          ex[idx] = !ex[idx];
          Store.set("timeline.expanded", ex);
          render();
        });
        hdr.style.cursor = "pointer";
      });
    }
  }

  function _on(id, evt, fn) { const el = document.getElementById(id); if (el) el.addEventListener(evt, fn); }
  function _esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }

  return { init, load, render };
})();
