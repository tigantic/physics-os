/**
 * HyperTensor Facial Plastics — Application Bootstrap
 *
 * Initializes all modules, loads contract/config, sets initial
 * mode, wires up mode transitions and global event handlers.
 */

"use strict";

const App = (() => {
  const MODES = {
    "case-library": { module: typeof CaseLibrary  !== "undefined" ? CaseLibrary  : null, hasLoad: true  },
    "twin-inspect": { module: typeof TwinInspect   !== "undefined" ? TwinInspect   : null, hasLoad: true  },
    "plan-author":  { module: typeof PlanAuthor    !== "undefined" ? PlanAuthor    : null, hasLoad: true  },
    "consult":      { module: typeof WhatIf        !== "undefined" ? WhatIf        : null, hasLoad: false },
    "sweep":        { module: typeof Sweep         !== "undefined" ? Sweep         : null, hasLoad: false },
    "report":       { module: typeof Report        !== "undefined" ? Report        : null, hasLoad: false },
    "viewer3d":     { module: typeof Viewer3D      !== "undefined" ? Viewer3D      : null, hasLoad: true  },
    "timeline":     { module: typeof Timeline      !== "undefined" ? Timeline      : null, hasLoad: true  },
    "compare":      { module: typeof Compare       !== "undefined" ? Compare       : null, hasLoad: true  },
  };

  let _currentMode = null;

  async function boot() {
    // 1. Restore persisted preferences (apiKey, sidebarOpen)
    _restorePrefs();

    // 2. Initialize all component modules
    if (typeof Toast      !== "undefined") Toast.init?.();
    if (typeof Modal      !== "undefined") Modal.init();
    if (typeof Sidebar    !== "undefined") Sidebar.init();
    if (typeof CommandBar  !== "undefined") CommandBar.init();
    if (typeof Inspector  !== "undefined") Inspector.init();
    if (typeof StatusBar  !== "undefined") StatusBar.init();

    // 3. Initialize all mode modules
    Object.values(MODES).forEach(m => { if (m.module && m.module.init) m.module.init(); });

    // 4. Wire router mode transitions
    Router.onModeChange(_onModeChange);

    // 5. Global sidebar toggle
    document.getElementById("btn-toggle-sidebar")?.addEventListener("click", _toggleSidebar);

    // 6. Apply initial sidebar state
    if (!Store.get("ui.sidebarOpen")) {
      document.getElementById("app-shell")?.classList.add("sidebar-collapsed");
    }

    // 7. Connect status indicator
    _updateConnectionDot();
    Store.subscribe("auth.connected", _updateConnectionDot);

    // 8. Auth gate — validate stored key or prompt for one
    const storedKey = Store.get("auth.apiKey");
    if (storedKey) {
      // Validate the stored key before trusting it
      try {
        const contract = await API.getContract();
        Store.set("auth.connected", true);
        Store.set("system.contract", contract);
        Store.set("system.version", contract.version || "unknown");
        await _loadCases();
        const hash = window.location.hash.replace("#", "");
        Router.navigate(MODES[hash] ? hash : "case-library");
      } catch {
        // Stored key is invalid — clear it and prompt
        Store.set("auth.apiKey", "");
        Store.set("auth.connected", false);
        Store.savePrefs();
        showAuthPrompt();
      }
    } else {
      showAuthPrompt();
    }

    console.info("[FP] Application boot complete");
  }

  async function _initData() {
    // Contract may already be loaded during auth validation;
    // reload cases to ensure fresh data and navigate to the default mode.
    await _loadCases();

    // Navigate to initial mode (from URL hash or default)
    const hash = window.location.hash.replace("#", "");
    const initialMode = MODES[hash] ? hash : "case-library";
    Router.navigate(initialMode);
  }

  function showAuthPrompt() {
    // Prevent duplicate prompts
    if (showAuthPrompt._open) return;
    showAuthPrompt._open = true;

    const body = document.createElement("div");
    body.innerHTML = `
      <p style="color:var(--text-secondary);font-size:var(--font-size-sm);margin-bottom:var(--space-3);">
        Enter your API key to connect to the HyperTensor Facial Plastics platform.
      </p>
      <div class="modal-field">
        <label for="auth-key-input">API Key</label>
        <input type="text" id="auth-key-input" placeholder="fp_..." autocomplete="off"
               value="${_escAttr(Store.get("auth.apiKey") || "")}" spellcheck="false">
      </div>
      <p id="auth-status-msg" style="font-size:var(--font-size-xs);color:var(--text-muted);min-height:1.2em;"></p>
    `;

    Modal.open({
      title: "Authentication Required",
      body: body,
      confirmText: "Connect",
      cancelText: "Cancel",
      onCancel: () => { showAuthPrompt._open = false; },
      onConfirm: async () => {
        const input = document.getElementById("auth-key-input");
        const statusEl = document.getElementById("auth-status-msg");
        const key = input ? input.value.trim() : "";
        if (!key) {
          if (statusEl) { statusEl.textContent = "Please enter an API key."; statusEl.style.color = "var(--accent-orange, #e6a700)"; }
          return false;  // Keep modal open
        }
        if (statusEl) { statusEl.textContent = "Connecting..."; statusEl.style.color = "var(--text-muted)"; }
        Store.set("auth.apiKey", key);
        Store.savePrefs();
        // Validate the key against the contract endpoint
        try {
          API.clearCache();
          const contract = await API.getContract();
          Store.set("auth.connected", true);
          Store.set("system.contract", contract);
          Store.set("system.version", contract.version || "unknown");
          Toast.success("Connected successfully");
          showAuthPrompt._open = false;
          // Always load data after successful auth
          _initData();
          // modal will close (return undefined)
        } catch (err) {
          Store.set("auth.connected", false);
          Store.set("auth.apiKey", "");
          if (statusEl) { statusEl.textContent = "Authentication failed — check your API key."; statusEl.style.color = "var(--accent-red, #f44)"; }
          return false;  // Keep modal open
        }
      },
    });

    // Allow Enter key to submit
    setTimeout(() => {
      const input = document.getElementById("auth-key-input");
      if (input) {
        input.addEventListener("keydown", (e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            const confirmBtn = document.getElementById("modal-confirm");
            if (confirmBtn) confirmBtn.click();
          }
        });
        input.focus();
        input.select();
      }
    }, 100);
  }
  showAuthPrompt._open = false;

  function _escAttr(s) {
    return (s || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
  }

  /* ── Mode switching ────────────────────────────────────── */

  async function _onModeChange(modeId) {
    // Deactivate previous
    if (_currentMode && MODES[_currentMode]?.module?.deactivate) {
      MODES[_currentMode].module.deactivate();
    }

    // Hide all panels, show target
    document.querySelectorAll(".mode-panel").forEach(el => { el.style.display = "none"; });
    const target = document.getElementById("mode-" + modeId);
    if (target) target.style.display = "block";

    // Highlight active nav item
    document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
    const navItem = document.querySelector(`.nav-item[data-mode="${modeId}"]`);
    if (navItem) navItem.classList.add("active");

    _currentMode = modeId;

    // Activate/load
    const entry = MODES[modeId];
    if (entry && entry.module) {
      if (entry.module.activate) entry.module.activate();
      if (entry.hasLoad && entry.module.load) {
        await entry.module.load();
      } else if (entry.module.render) {
        entry.module.render();
      }
    }
  }

  /* ── Contract & Cases ──────────────────────────────────── */

  async function _loadContract() {
    try {
      const contract = await API.getContract();
      Store.set("system.contract", contract);
      Store.set("system.version", contract.version || "unknown");
    } catch (err) {
      console.warn("[FP] Contract load failed:", err.message || err);
    }
  }

  async function _loadCases() {
    try {
      const data = await API.listCases();
      Store.set("cases.items", data.cases || []);
      Store.set("cases.total", data.total || 0);
    } catch (err) {
      console.warn("[FP] Cases load failed:", err.message || err);
    }
  }

  /* ── Preferences ───────────────────────────────────────── */

  function _restorePrefs() {
    try {
      const raw = localStorage.getItem("fp_prefs");
      if (raw) {
        const prefs = JSON.parse(raw);
        if (prefs.apiKey) Store.set("auth.apiKey", prefs.apiKey);
        if (prefs.sidebarOpen !== undefined) Store.set("ui.sidebarOpen", prefs.sidebarOpen);
      }
    } catch { /* ignore corrupt prefs */ }

    // Also check if API key was set in state defaults or via URL param
    const urlParams = new URLSearchParams(window.location.search);
    const keyParam = urlParams.get("api_key") || urlParams.get("apiKey");
    if (keyParam) Store.set("auth.apiKey", keyParam);
  }

  /* ── Sidebar ───────────────────────────────────────────── */

  function _toggleSidebar() {
    const shell = document.getElementById("app-shell");
    if (!shell) return;
    shell.classList.toggle("sidebar-collapsed");
    const open = !shell.classList.contains("sidebar-collapsed");
    Store.set("ui.sidebarOpen", open);
    Store.savePrefs();

    // Resize viewer if active
    if (_currentMode === "viewer3d" && MODES.viewer3d?.module?._resize) {
      setTimeout(() => MODES.viewer3d.module._resize?.(), 300);
    }
  }

  /* ── Connection indicator ──────────────────────────────── */

  function _updateConnectionDot() {
    const dot = document.getElementById("connection-dot");
    if (!dot) return;
    const connected = Store.get("auth.connected");
    dot.className = "connection-dot " + (connected ? "connected" : "disconnected");
    dot.title = connected ? "Connected" : "Disconnected";
  }

  return { boot, showAuthPrompt };
})();

/* ── DOM Ready ───────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => App.boot());
