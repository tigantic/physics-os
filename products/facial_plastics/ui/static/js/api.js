/**
 * HyperTensor Facial Plastics — API Client
 *
 * All backend communication. Auth header injection, retry with
 * exponential backoff, error normalization, LRU response cache.
 */

"use strict";

const API = (() => {
  const MAX_RETRIES = 3;
  const RETRY_DELAYS = [500, 1000, 2000];
  const _cache = new Map();
  const CACHE_MAX = 50;

  function _headers() {
    const h = { "Content-Type": "application/json" };
    const key = Store.get("auth.apiKey");
    if (key) h["X-API-Key"] = key;
    return h;
  }

  function _cacheKey(method, path, body) {
    return `${method}:${path}:${body ? JSON.stringify(body) : ""}`;
  }

  function _cacheGet(key) {
    const entry = _cache.get(key);
    if (!entry) return null;
    if (Date.now() - entry.ts > 60000) { _cache.delete(key); return null; }
    return entry.data;
  }

  function _cacheSet(key, data) {
    if (_cache.size >= CACHE_MAX) {
      const oldest = _cache.keys().next().value;
      _cache.delete(oldest);
    }
    _cache.set(key, { data, ts: Date.now() });
  }

  function clearCache() { _cache.clear(); }

  async function _fetch(method, path, body, opts) {
    opts = opts || {};
    const url = path.startsWith("http") ? path : `/api${path}`;
    const fetchOpts = { method, headers: _headers() };
    if (body && method !== "GET") fetchOpts.body = JSON.stringify(body);

    // Query params for GET
    let finalUrl = url;
    if (method === "GET" && body) {
      const params = new URLSearchParams();
      for (const [k, v] of Object.entries(body)) {
        if (v !== "" && v !== null && v !== undefined) params.set(k, String(v));
      }
      const qs = params.toString();
      if (qs) finalUrl += `?${qs}`;
    }

    // Cache check for GET
    if (method === "GET" && !opts.noCache) {
      const ck = _cacheKey(method, finalUrl, null);
      const cached = _cacheGet(ck);
      if (cached) return cached;
    }

    let lastErr = null;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        const resp = await fetch(finalUrl, fetchOpts);
        if (resp.status === 401) {
          Store.set("auth.connected", false);
          throw { status: 401, message: "Authentication failed — check API key" };
        }
        if (!resp.ok) {
          const text = await resp.text().catch(() => "");
          throw { status: resp.status, message: `${method} ${path}: ${resp.status} ${text}` };
        }
        const data = await resp.json();
        if (data.error) {
          throw { status: 200, message: data.error };
        }

        // Cache successful GET
        if (method === "GET" && !opts.noCache) {
          _cacheSet(_cacheKey(method, finalUrl, null), data);
        }

        Store.set("auth.connected", true);
        return data;
      } catch (err) {
        lastErr = err;
        if (err.status === 401) throw err;
        if (attempt < MAX_RETRIES) {
          await new Promise(r => setTimeout(r, RETRY_DELAYS[attempt]));
        }
      }
    }
    throw lastErr;
  }

  // ── G1: Case Library ───────────────────────────────────────
  async function listCases(params) { return _fetch("GET", "/cases", params); }
  async function getCase(id) { return _fetch("GET", `/cases/${id}`); }
  async function createCase(data) { return _fetch("POST", "/cases", data); }
  async function deleteCase(id) { return _fetch("POST", `/cases/${id}/delete`); }
  async function curateLibrary() { return _fetch("POST", "/curate"); }

  // ── G2: Twin Inspect ───────────────────────────────────────
  async function getTwinSummary(id) { return _fetch("GET", `/cases/${id}/twin`); }
  async function getMeshData(id) { return _fetch("GET", `/cases/${id}/mesh`); }
  async function getLandmarks(id) { return _fetch("GET", `/cases/${id}/landmarks`); }

  // ── G3: Plan Author ────────────────────────────────────────
  async function listOperators(params) { return _fetch("GET", "/operators", params); }
  async function listTemplates() { return _fetch("GET", "/templates"); }
  async function loadTemplate(category, template, params) {
    return _fetch("POST", "/plan/template", { category, template, params });
  }
  async function createCustomPlan(name, procedure, steps) {
    return _fetch("POST", "/plan/custom", { name, procedure, steps });
  }
  async function compilePlan(caseId, plan) {
    return _fetch("POST", "/plan/compile", { case_id: caseId, plan });
  }

  // ── G4: Consult ────────────────────────────────────────────
  async function runWhatIf(caseId, plan, overrides) {
    return _fetch("POST", "/whatif", { case_id: caseId, plan, modified_params: overrides });
  }
  async function parameterSweep(caseId, plan, sweepOp, sweepParam, values) {
    return _fetch("POST", "/sweep", { case_id: caseId, plan, sweep_op: sweepOp, sweep_param: sweepParam, values });
  }

  // ── G5: Report ─────────────────────────────────────────────
  async function generateReport(caseId, plan, format) {
    return _fetch("POST", "/report", { case_id: caseId, plan, format });
  }

  // ── G6: Visualization ──────────────────────────────────────
  async function getVisualization(id) { return _fetch("GET", `/cases/${id}/visualization`); }

  // ── G7: Timeline ───────────────────────────────────────────
  async function getTimeline(id) { return _fetch("GET", `/cases/${id}/timeline`); }

  // ── G8: Compare ────────────────────────────────────────────
  async function comparePlans(caseId, planA, planB) {
    return _fetch("POST", "/compare/plans", { case_id: caseId, plan_a: planA, plan_b: planB });
  }
  async function compareCases(idA, idB) {
    return _fetch("POST", "/compare/cases", { case_id_a: idA, case_id_b: idB });
  }

  // ── System ─────────────────────────────────────────────────
  async function getContract() { return _fetch("GET", "/contract"); }
  async function getHealth() {
    const r = await fetch("/health");
    return r.json();
  }
  async function getMetrics() {
    const r = await fetch("/metrics");
    return r.text();
  }

  return {
    listCases, getCase, createCase, deleteCase, curateLibrary,
    getTwinSummary, getMeshData, getLandmarks,
    listOperators, listTemplates, loadTemplate, createCustomPlan, compilePlan,
    runWhatIf, parameterSweep,
    generateReport,
    getVisualization,
    getTimeline,
    comparePlans, compareCases,
    getContract, getHealth, getMetrics,
    clearCache,
  };
})();
