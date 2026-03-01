/**
 * Ontic Facial Plastics — G4.5 Parameter Sweep
 *
 * Configure sweep: select operator, parameter, range.
 * Execute API.parameterSweep() and render sparkline chart on Canvas2D.
 */

"use strict";

const Sweep = (() => {
  let _el = null;

  function init() {
    _el = document.getElementById("mode-sweep");
  }

  function render() {
    if (!_el) return;
    const caseId = Store.get("selectedCase.id");
    const plan = Store.get("plan.current");
    const compiled = Store.get("plan.compiled");
    const sweepResult = Store.get("sweep.result");
    const running = Store.get("sweep.running");
    const config = Store.get("sweep.config") || { stepName: "", param: "", min: 0, max: 1, steps: 5 };

    if (!plan || !compiled) {
      _el.innerHTML = `<div class="panel-header"><h2>Parameter Sweep</h2></div><div class="placeholder">Compile a plan in Plan Author first.</div>`;
      return;
    }

    const stepsWithParams = (plan.steps || []).filter(s => s.params && Object.keys(s.params).length > 0);

    let paramOptions = "";
    const selectedStep = stepsWithParams.find(s => s.name === config.stepName);
    if (selectedStep) {
      paramOptions = Object.keys(selectedStep.params).map(p => `<option value="${_esc(p)}" ${p === config.param ? "selected" : ""}>${_esc(p)}</option>`).join("");
    }

    _el.innerHTML = `
      <div class="panel-header">
        <h2>Parameter Sweep</h2>
        <div class="panel-actions">
          <button class="btn btn-primary" id="btn-run-sweep" ${!caseId || running ? "disabled" : ""}>${running ? "Sweeping..." : "Run Sweep"}</button>
        </div>
      </div>
      <div class="sweep-config">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:var(--space-3);margin-bottom:var(--space-3)">
          <div>
            <label style="display:block;font-size:var(--font-size-xs);color:var(--text-muted);margin-bottom:var(--space-1)">Operator Step</label>
            <select class="input" id="sweep-step">${stepsWithParams.map(s => `<option value="${_esc(s.name)}" ${s.name === config.stepName ? "selected" : ""}>${_esc(s.name)}</option>`).join("")}</select>
          </div>
          <div>
            <label style="display:block;font-size:var(--font-size-xs);color:var(--text-muted);margin-bottom:var(--space-1)">Parameter</label>
            <select class="input" id="sweep-param">${paramOptions || '<option value="">—</option>'}</select>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:var(--space-3);margin-bottom:var(--space-3)">
          <div>
            <label style="display:block;font-size:var(--font-size-xs);color:var(--text-muted);margin-bottom:var(--space-1)">Min</label>
            <input type="number" class="input" id="sweep-min" value="${config.min}" step="any">
          </div>
          <div>
            <label style="display:block;font-size:var(--font-size-xs);color:var(--text-muted);margin-bottom:var(--space-1)">Max</label>
            <input type="number" class="input" id="sweep-max" value="${config.max}" step="any">
          </div>
          <div>
            <label style="display:block;font-size:var(--font-size-xs);color:var(--text-muted);margin-bottom:var(--space-1)">Steps</label>
            <input type="number" class="input" id="sweep-steps" value="${config.steps}" min="2" max="50" step="1">
          </div>
        </div>
      </div>
      <div class="sweep-chart" id="sweep-chart-area" style="margin-top:var(--space-3)">
        ${sweepResult ? "" : '<p class="placeholder">Configure and run a sweep to see results.</p>'}
        <canvas id="sweep-canvas" width="640" height="300" style="display:${sweepResult ? "block" : "none"};width:100%;height:auto;background:var(--surface-1);border-radius:var(--radius-md)"></canvas>
      </div>
    `;

    if (sweepResult) _drawChart(sweepResult);
    _bind(stepsWithParams);
  }

  function _bind(stepsWithParams) {
    const stepSel = document.getElementById("sweep-step");
    const paramSel = document.getElementById("sweep-param");

    if (stepSel) {
      stepSel.addEventListener("change", () => {
        const current = Store.get("sweep.config") || {};
        current.stepName = stepSel.value;
        current.param = "";
        Store.set("sweep.config", current);
        render();
      });
    }

    if (paramSel) {
      paramSel.addEventListener("change", () => {
        const current = Store.get("sweep.config") || {};
        current.param = paramSel.value;
        // Auto-fill min/max from param_defs if available
        const plan = Store.get("plan.current");
        if (plan) {
          const step = (plan.steps || []).find(s => s.name === current.stepName);
          if (step && step.param_defs && step.param_defs[current.param]) {
            const d = step.param_defs[current.param];
            if (d.min !== undefined) current.min = d.min;
            if (d.max !== undefined) current.max = d.max;
          }
        }
        Store.set("sweep.config", current);
        render();
      });
    }

    _on("btn-run-sweep", "click", _runSweep);
  }

  async function _runSweep() {
    const caseId = Store.get("selectedCase.id");
    const plan = Store.get("plan.current");
    if (!caseId || !plan) return;

    // Read current form values
    const config = {
      stepName: _val("sweep-step"),
      param: _val("sweep-param"),
      min: parseFloat(_val("sweep-min")) || 0,
      max: parseFloat(_val("sweep-max")) || 1,
      steps: parseInt(_val("sweep-steps"), 10) || 5,
    };

    if (!config.stepName || !config.param) {
      Toast.warning("Select an operator step and parameter");
      return;
    }

    Store.set("sweep.config", config);
    Store.set("sweep.running", true);
    render();

    // Generate values array from min/max/steps for the backend
    const values = [];
    const stepSize = config.steps > 1 ? (config.max - config.min) / (config.steps - 1) : 0;
    for (let i = 0; i < config.steps; i++) {
      values.push(config.min + stepSize * i);
    }

    try {
      const result = await API.parameterSweep(caseId, plan, config.stepName, config.param, values);
      Store.set("sweep.result", result);
      Inspector.show("Sweep Result", result);
      Toast.success("Sweep complete — " + (result.points ? result.points.length : 0) + " points");
    } catch (err) {
      Toast.error("Sweep failed: " + (err.message || err));
    } finally {
      Store.set("sweep.running", false);
      render();
    }
  }

  function _drawChart(result) {
    const canvas = document.getElementById("sweep-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const points = result.points || result.results || [];
    if (points.length === 0) return;

    const W = canvas.width;
    const H = canvas.height;
    const pad = { top: 30, right: 20, bottom: 40, left: 60 };
    const gw = W - pad.left - pad.right;
    const gh = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    // Extract x,y
    const xvals = points.map(p => p.value !== undefined ? p.value : p.x);
    const yvals = points.map(p => {
      if (p.metric !== undefined) return p.metric;
      if (p.n_bcs !== undefined) return p.n_bcs;
      if (p.y !== undefined) return p.y;
      return 0;
    });

    const xmin = Math.min(...xvals);
    const xmax = Math.max(...xvals);
    const ymin = Math.min(...yvals);
    const ymax = Math.max(...yvals);
    const xrange = xmax - xmin || 1;
    const yrange = ymax - ymin || 1;

    const toX = v => pad.left + ((v - xmin) / xrange) * gw;
    const toY = v => pad.top + gh - ((v - ymin) / yrange) * gh;

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (gh / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + gw, y); ctx.stroke();
    }

    // Axes labels
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "11px var(--font-mono, monospace)";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const val = ymin + (yrange / 4) * (4 - i);
      const y = pad.top + (gh / 4) * i;
      ctx.fillText(val.toFixed(2), pad.left - 6, y + 4);
    }
    ctx.textAlign = "center";
    for (let i = 0; i < xvals.length; i++) {
      ctx.fillText(xvals[i].toFixed(2), toX(xvals[i]), pad.top + gh + 18);
    }

    // Line
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent-blue").trim() || "#58a6ff";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < xvals.length; i++) {
      const x = toX(xvals[i]);
      const y = toY(yvals[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Points
    ctx.fillStyle = ctx.strokeStyle;
    for (let i = 0; i < xvals.length; i++) {
      const x = toX(xvals[i]);
      const y = toY(yvals[i]);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Title
    const cfg = Store.get("sweep.config") || {};
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "12px var(--font-sans, sans-serif)";
    ctx.textAlign = "center";
    ctx.fillText(`${cfg.param || "Parameter"} sweep: ${cfg.min}→${cfg.max} (${cfg.steps} steps)`, W / 2, 18);
  }

  function _on(id, evt, fn) { const el = document.getElementById(id); if (el) el.addEventListener(evt, fn); }
  function _val(id) { const el = document.getElementById(id); return el ? el.value : ""; }
  function _esc(s) { const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML; }

  return { init, render };
})();
