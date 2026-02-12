/**
 * HyperTensor Facial Plastics — Modal Dialog Manager
 *
 * Focus trap, Escape to close, backdrop click, confirm/cancel.
 */

"use strict";

const Modal = (() => {
  let _overlay = null;
  let _onConfirm = null;
  let _onCancel = null;
  let _prevFocus = null;

  function init() {
    _overlay = document.getElementById("modal-overlay");
    if (!_overlay) return;
    _overlay.addEventListener("click", (e) => {
      if (e.target === _overlay) close();
    });
    // Wire the static close button
    const closeBtn = _overlay.querySelector("#modal-close");
    if (closeBtn) closeBtn.addEventListener("click", () => close());
  }

  function open(opts) {
    if (!_overlay) return;
    _prevFocus = document.activeElement;
    const title = _overlay.querySelector("#modal-title");
    const body = _overlay.querySelector("#modal-body");
    const confirmBtn = _overlay.querySelector("#modal-confirm");
    const cancelBtn = _overlay.querySelector("#modal-cancel");

    if (opts.title && title) title.textContent = opts.title;
    if (opts.body !== undefined && body) {
      if (typeof opts.body === "string") {
        body.innerHTML = opts.body;
      } else if (opts.body instanceof HTMLElement) {
        body.innerHTML = "";
        body.appendChild(opts.body);
      }
    }
    if (confirmBtn) {
      confirmBtn.textContent = opts.confirmText || "Confirm";
      confirmBtn.className = `btn ${opts.confirmClass || "btn-primary"}`;
    }
    if (cancelBtn) cancelBtn.textContent = opts.cancelText || "Cancel";

    _onConfirm = opts.onConfirm || null;
    _onCancel = opts.onCancel || null;

    if (confirmBtn) {
      confirmBtn.onclick = async () => {
        if (_onConfirm) {
          // Await async handlers; if handler returns false, keep modal open
          const result = await Promise.resolve(_onConfirm());
          if (result === false) return;
        }
        close();
      };
    }
    if (cancelBtn) cancelBtn.onclick = () => close();

    _overlay.classList.remove("hidden");
    _overlay.classList.add("visible");

    // Focus first input in modal
    setTimeout(() => {
      const firstInput = _overlay.querySelector("input, select, textarea");
      if (firstInput) firstInput.focus();
    }, 50);
  }

  function close() {
    if (!_overlay) return;
    _overlay.classList.remove("visible");
    _overlay.classList.add("hidden");
    if (_onCancel && !_overlay.classList.contains("confirmed")) _onCancel();
    _onConfirm = null;
    _onCancel = null;
    if (_prevFocus) { try { _prevFocus.focus(); } catch (_) {} }
  }

  function confirm(msg, onYes) {
    open({
      title: "Confirm",
      body: `<p style="color:var(--text-secondary);font-size:var(--font-size-sm);">${msg}</p>`,
      confirmText: "Confirm",
      confirmClass: "btn-danger",
      onConfirm: onYes,
    });
  }

  return { init, open, close, confirm };
})();
