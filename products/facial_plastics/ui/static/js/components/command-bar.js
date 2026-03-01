/**
 * Ontic Facial Plastics — Command Palette
 *
 * Fuzzy search over all modes, actions, shortcuts.
 * Activated with / key.
 */

"use strict";

const CommandBar = (() => {
  let _overlay = null;
  let _input = null;
  let _results = null;
  let _items = [];
  let _focusIdx = -1;

  function init() {
    _overlay = document.querySelector(".command-palette-overlay");
    if (!_overlay) return;
    _input = _overlay.querySelector("input");
    _results = _overlay.querySelector(".command-palette-results");

    _overlay.addEventListener("click", (e) => {
      if (e.target === _overlay) close();
    });

    _input.addEventListener("input", () => _search(_input.value));
    _input.addEventListener("keydown", _onInputKey);

    _buildItems();
  }

  function _buildItems() {
    _items = [];
    // Mode navigation
    for (const m of Router.MODES) {
      _items.push({ label: `Go to ${m.label}`, shortcut: m.key, action: () => Router.navigate(m.id), category: "Navigate" });
    }
    // Actions
    _items.push({ label: "Toggle Inspector", shortcut: "Ctrl+J", action: () => { Store.set("ui.inspectorOpen", !Store.get("ui.inspectorOpen")); }, category: "UI" });
    _items.push({ label: "Toggle Sidebar", shortcut: "Ctrl+B", action: () => { Store.set("ui.sidebarOpen", !Store.get("ui.sidebarOpen")); Store.savePrefs(); }, category: "UI" });
    _items.push({ label: "Refresh Cases", shortcut: "", action: () => { Router.navigate("case-library"); CaseLibrary.load(); }, category: "Action" });
    _items.push({ label: "New Case", shortcut: "N", action: () => CaseLibrary.openCreateModal(), category: "Action" });
  }

  function open() {
    if (!_overlay || _overlay.classList.contains("visible")) return;
    _overlay.classList.add("visible");
    _input.value = "";
    _focusIdx = -1;
    _search("");
    setTimeout(() => _input.focus(), 50);
  }

  function close() {
    if (!_overlay) return;
    _overlay.classList.remove("visible");
  }

  function _search(query) {
    const q = query.toLowerCase().trim();
    const filtered = q ? _items.filter(i => i.label.toLowerCase().includes(q)) : _items;
    _focusIdx = filtered.length > 0 ? 0 : -1;
    _renderResults(filtered);
  }

  function _renderResults(items) {
    if (!_results) return;
    if (items.length === 0) {
      _results.innerHTML = `<div class="command-palette-item" style="color:var(--text-muted)">No results</div>`;
      return;
    }
    _results.innerHTML = items.map((item, i) => `
      <div class="command-palette-item${i === _focusIdx ? " focused" : ""}" data-idx="${i}">
        <span class="cp-label">${_esc(item.label)}</span>
        ${item.shortcut ? `<span class="cp-shortcut">${_esc(item.shortcut)}</span>` : ""}
      </div>
    `).join("");

    _results.querySelectorAll(".command-palette-item").forEach(el => {
      el.addEventListener("click", () => {
        const idx = parseInt(el.dataset.idx);
        if (items[idx]) { items[idx].action(); close(); }
      });
    });
  }

  function _onInputKey(e) {
    const items = _getFilteredItems();
    if (e.key === "ArrowDown") {
      e.preventDefault();
      _focusIdx = Math.min(_focusIdx + 1, items.length - 1);
      _renderResults(items);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      _focusIdx = Math.max(_focusIdx - 1, 0);
      _renderResults(items);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (_focusIdx >= 0 && items[_focusIdx]) { items[_focusIdx].action(); close(); }
    } else if (e.key === "Escape") {
      close();
    }
  }

  function _getFilteredItems() {
    const q = _input.value.toLowerCase().trim();
    return q ? _items.filter(i => i.label.toLowerCase().includes(q)) : _items;
  }

  function _esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

  return { init, open, close };
})();
