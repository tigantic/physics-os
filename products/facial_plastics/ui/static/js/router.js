/**
 * HyperTensor Facial Plastics — Router & Keyboard Shortcut Manager
 *
 * Hash-based mode routing. Keyboard shortcut registry.
 * Ctrl+1..8 for mode switch, / for command palette, Ctrl+J for inspector.
 */

"use strict";

const Router = (() => {
  const MODES = [
    { id: "case-library", label: "Case Library", shortcut: "1", key: "Ctrl+1" },
    { id: "twin-inspect", label: "Twin Inspect", shortcut: "2", key: "Ctrl+2" },
    { id: "plan-author", label: "Plan Author", shortcut: "3", key: "Ctrl+3" },
    { id: "consult", label: "Consult", shortcut: "4", key: "Ctrl+4" },
    { id: "sweep", label: "Sweep", shortcut: "5", key: "Ctrl+5" },
    { id: "report", label: "Report", shortcut: "6", key: "Ctrl+6" },
    { id: "viewer3d", label: "3D View", shortcut: "7", key: "Ctrl+7" },
    { id: "timeline", label: "Timeline", shortcut: "8", key: "Ctrl+8" },
    { id: "compare", label: "Compare", shortcut: "9", key: "Ctrl+9" },
  ];

  const _shortcuts = new Map();
  const _modeCallbacks = new Set();

  function init() {
    // Read initial hash
    const hash = window.location.hash.slice(1);
    if (hash && MODES.some(m => m.id === hash)) {
      Store.set("ui.mode", hash);
    }

    // Register mode shortcuts
    for (const m of MODES) {
      registerShortcut(`Ctrl+${m.shortcut}`, () => navigate(m.id), `Switch to ${m.label}`);
    }

    // Inspector toggle
    registerShortcut("Ctrl+j", () => {
      const open = Store.get("ui.inspectorOpen");
      Store.set("ui.inspectorOpen", !open);
    }, "Toggle Inspector");

    // Sidebar toggle
    registerShortcut("Ctrl+b", () => {
      const open = Store.get("ui.sidebarOpen");
      Store.set("ui.sidebarOpen", !open);
      Store.savePrefs();
    }, "Toggle Sidebar");

    // Command palette
    registerShortcut("/", () => {
      if (document.activeElement && (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA" || document.activeElement.tagName === "SELECT")) return;
      CommandBar.open();
    }, "Command Palette");

    registerShortcut("Escape", () => {
      CommandBar.close();
      Modal.close();
    }, "Close");

    // Global keydown listener
    document.addEventListener("keydown", _onKeyDown);

    // Hash change
    window.addEventListener("hashchange", () => {
      const h = window.location.hash.slice(1);
      if (h && MODES.some(m => m.id === h)) navigate(h);
    });
  }

  function navigate(modeId) {
    if (!MODES.some(m => m.id === modeId)) return;
    const prev = Store.get("ui.mode");
    if (prev === modeId) return;
    Store.set("ui.mode", modeId);
    window.location.hash = modeId;
    _notifyMode(modeId);
  }

  function onModeChange(fn) {
    _modeCallbacks.add(fn);
    return () => _modeCallbacks.delete(fn);
  }

  function _notifyMode(mode) {
    for (const fn of _modeCallbacks) {
      try { fn(mode); } catch (e) { console.error("[Router]", e); }
    }
  }

  function registerShortcut(combo, fn, description) {
    _shortcuts.set(combo.toLowerCase(), { fn, description, combo });
  }

  function getShortcuts() {
    return Array.from(_shortcuts.entries()).map(([combo, v]) => ({
      combo: v.combo,
      description: v.description,
    }));
  }

  function _onKeyDown(e) {
    const parts = [];
    if (e.ctrlKey || e.metaKey) parts.push("ctrl");
    if (e.shiftKey) parts.push("shift");
    if (e.altKey) parts.push("alt");

    let key = e.key.toLowerCase();
    if (key === " ") key = "space";

    // Try with modifiers
    if (parts.length > 0) {
      parts.push(key);
      const combo = parts.join("+");
      const entry = _shortcuts.get(combo);
      if (entry) {
        e.preventDefault();
        entry.fn();
        return;
      }
    }

    // Try key alone (for / and Escape)
    const entry = _shortcuts.get(key);
    if (entry) {
      // Don't intercept if typing in input (except Escape)
      if (key !== "escape" && document.activeElement && (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA" || document.activeElement.tagName === "SELECT")) return;
      if (key === "/") e.preventDefault();
      entry.fn();
    }
  }

  return { MODES, init, navigate, onModeChange, registerShortcut, getShortcuts };
})();
