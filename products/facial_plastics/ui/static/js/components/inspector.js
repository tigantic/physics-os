/**
 * HyperTensor Facial Plastics — Inspector Drawer
 *
 * Bottom panel for JSON tree, raw text, diff views.
 * Toggle with Ctrl+J. Resizable via drag.
 */

"use strict";

const Inspector = (() => {
  let _el = null;
  let _body = null;

  function init() {
    _el = document.getElementById("inspector");
    if (!_el) return;
    _body = _el.querySelector(".inspector-body");

    Store.subscribe("ui.inspectorOpen", (open) => {
      _el.classList.toggle("open", open);
    });

    Store.subscribe("inspector", () => _render());

    // Tab switching
    _el.querySelectorAll(".inspector-tab").forEach(tab => {
      tab.addEventListener("click", () => {
        Store.set("ui.inspectorTab", tab.dataset.tab);
        _el.querySelectorAll(".inspector-tab").forEach(t => t.classList.toggle("active", t === tab));
        _render();
      });
    });

    // Drag resize
    const header = _el.querySelector(".inspector-header");
    if (header) {
      let dragging = false;
      let startY = 0;
      let startH = 0;
      header.addEventListener("mousedown", (e) => {
        dragging = true;
        startY = e.clientY;
        startH = _el.getBoundingClientRect().height;
        document.body.style.cursor = "ns-resize";
        e.preventDefault();
      });
      document.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const delta = startY - e.clientY;
        const newH = Math.max(120, Math.min(window.innerHeight * 0.5, startH + delta));
        _el.style.height = newH + "px";
      });
      document.addEventListener("mouseup", () => {
        if (dragging) { dragging = false; document.body.style.cursor = ""; }
      });
    }
  }

  function show(title, content, tab) {
    Store.set("inspector", { title: title || "", content: typeof content === "object" ? JSON.stringify(content, null, 2) : String(content || "") });
    Store.set("ui.inspectorOpen", true);
    if (tab) Store.set("ui.inspectorTab", tab);
  }

  function _render() {
    if (!_body) return;
    const data = Store.get("inspector");
    const tab = Store.get("ui.inspectorTab");
    const titleEl = _el.querySelector(".inspector-title");
    if (titleEl) titleEl.textContent = data.title || "Inspector";

    if (tab === "json") {
      try {
        const parsed = JSON.parse(data.content);
        _body.textContent = JSON.stringify(parsed, null, 2);
      } catch (_) {
        _body.textContent = data.content;
      }
    } else {
      _body.textContent = data.content;
    }
  }

  return { init, show };
})();
