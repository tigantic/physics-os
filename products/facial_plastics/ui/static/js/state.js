/**
 * HyperTensor Facial Plastics — Centralized State Store
 *
 * Pure pub/sub event bus with path-based subscriptions.
 * No framework dependency. Immutable snapshots on change.
 */

"use strict";

const Store = (() => {
  const _initial = {
    auth: { apiKey: "", connected: false },
    ui: {
      mode: "case-library",
      sidebarOpen: true,
      inspectorOpen: false,
      inspectorTab: "json",
    },
    cases: { items: [], total: 0, loading: false, offset: 0, limit: 100, filterProcedure: "", filterQuality: "", search: "" },
    selectedCase: { id: null, metadata: null, twin: null, mesh: null, landmarks: null, visualization: null },
    plan: { current: null, compiled: null, dirty: false, compiling: false },
    operators: { registry: {}, loaded: false },
    templates: { registry: {}, loaded: false },
    whatif: { overrides: {}, result: null, running: false },
    sweep: { sweepOp: "", sweepParam: "", values: [], results: null, running: false },
    report: { format: "html", content: null, generating: false },
    compare: { type: "plans", resultA: null, resultB: null, delta: null, planB: null, caseIdB: null, running: false },
    timeline: { events: [], simFrames: [], currentFrame: 0, loading: false },
    viewer3d: { settings: { landmarks: true, regions: true, wireframe: false, opacity: 1.0 } },
    system: { health: null, metrics: null, version: "1.0.0", contract: null },
    inspector: { title: "", content: "" },
  };

  let _state = _deepClone(_initial);
  const _listeners = new Map();

  function _deepClone(obj) {
    if (obj === null || typeof obj !== "object") return obj;
    if (Array.isArray(obj)) return obj.map(_deepClone);
    const out = {};
    for (const k in obj) out[k] = _deepClone(obj[k]);
    return out;
  }

  function _getByPath(obj, path) {
    const parts = path.split(".");
    let cur = obj;
    for (const p of parts) {
      if (cur == null) return undefined;
      cur = cur[p];
    }
    return cur;
  }

  function _setByPath(obj, path, value) {
    const parts = path.split(".");
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i++) {
      if (cur[parts[i]] == null) cur[parts[i]] = {};
      cur = cur[parts[i]];
    }
    cur[parts[parts.length - 1]] = value;
  }

  function get(path) {
    if (!path) return _deepClone(_state);
    return _deepClone(_getByPath(_state, path));
  }

  function set(path, value) {
    _setByPath(_state, path, value);
    _notify(path);
  }

  function update(path, fn) {
    const current = _getByPath(_state, path);
    const next = fn(current);
    _setByPath(_state, path, next);
    _notify(path);
  }

  function _notify(changedPath) {
    for (const [watchPath, fns] of _listeners) {
      if (changedPath.startsWith(watchPath) || watchPath.startsWith(changedPath)) {
        const val = _getByPath(_state, watchPath);
        for (const fn of fns) {
          try { fn(val, watchPath); } catch (e) { console.error("[Store] listener error:", e); }
        }
      }
    }
  }

  function subscribe(path, fn) {
    if (!_listeners.has(path)) _listeners.set(path, new Set());
    _listeners.get(path).add(fn);
    return () => { _listeners.get(path).delete(fn); };
  }

  function snapshot() {
    return _deepClone(_state);
  }

  // Persist UI prefs to localStorage
  function _loadPrefs() {
    try {
      // Check both legacy key names for backwards compatibility
      const raw = localStorage.getItem("fp_prefs") || localStorage.getItem("ht_fp_prefs");
      if (raw) {
        const prefs = JSON.parse(raw);
        if (prefs.apiKey) _state.auth.apiKey = prefs.apiKey;
        if (prefs.sidebarOpen !== undefined) _state.ui.sidebarOpen = prefs.sidebarOpen;
      }
    } catch (_) { /* ignore */ }
  }

  function savePrefs() {
    try {
      localStorage.setItem("fp_prefs", JSON.stringify({
        apiKey: _state.auth.apiKey,
        sidebarOpen: _state.ui.sidebarOpen,
      }));
    } catch (_) { /* ignore */ }
  }

  _loadPrefs();

  return { get, set, update, subscribe, snapshot, savePrefs };
})();
