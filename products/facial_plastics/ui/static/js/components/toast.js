/**
 * Ontic Facial Plastics — Toast Notification System
 *
 * Queue-based, auto-dismiss, 4 severity levels.
 * Max 5 visible at once. Click to dismiss.
 */

"use strict";

const Toast = (() => {
  const MAX_VISIBLE = 5;
  const ICONS = { info: "ℹ", success: "✓", warning: "⚠", error: "✗" };

  function show(message, type, duration) {
    type = type || "info";
    duration = duration || 4000;
    const container = document.getElementById("toast-container");
    if (!container) return;

    // Trim excess
    const toasts = container.querySelectorAll(".toast:not(.removing)");
    if (toasts.length >= MAX_VISIBLE) {
      _dismiss(toasts[toasts.length - 1]);
    }

    const el = document.createElement("div");
    el.className = `toast toast-${type}`;
    el.innerHTML = `<span class="toast-icon">${ICONS[type] || ICONS.info}</span><div class="toast-body"><div class="toast-message">${_escHtml(message)}</div></div>`;
    el.addEventListener("click", () => _dismiss(el));
    container.appendChild(el);

    if (duration > 0) {
      setTimeout(() => _dismiss(el), duration);
    }
  }

  function _dismiss(el) {
    if (el.classList.contains("removing")) return;
    el.classList.add("removing");
    setTimeout(() => el.remove(), 200);
  }

  function _escHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function info(msg, dur) { show(msg, "info", dur); }
  function success(msg, dur) { show(msg, "success", dur); }
  function warning(msg, dur) { show(msg, "warning", dur); }
  function error(msg, dur) { show(msg, "error", dur); }

  return { show, info, success, warning, error };
})();
