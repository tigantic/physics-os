/**
 * Svelte action: focus trap for modal dialogs.
 *
 * Usage:
 *   <div use:focusTrap>
 *     ... modal content ...
 *   </div>
 *
 * Traps Tab/Shift+Tab focus within the element.
 * Restores focus to the previously focused element on destroy.
 * Closes on Escape if an `on:close` event is dispatched.
 */

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

/**
 * @param {HTMLElement} node - The container element to trap focus within.
 * @param {{ onClose?: () => void }} [params] - Optional config with onClose callback.
 */
export function focusTrap(node, params = {}) {
  const previouslyFocused = /** @type {HTMLElement | null} */ (document.activeElement);

  function getFocusableElements() {
    return /** @type {HTMLElement[]} */ (
      Array.from(node.querySelectorAll(FOCUSABLE_SELECTOR)).filter(
        (el) => !el.closest('[aria-hidden="true"]') && el.offsetParent !== null,
      )
    );
  }

  function handleKeydown(e) {
    if (e.key === 'Escape') {
      e.preventDefault();
      if (params.onClose) params.onClose();
      return;
    }

    if (e.key !== 'Tab') return;

    const focusable = getFocusableElements();
    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (e.shiftKey) {
      if (document.activeElement === first || !node.contains(document.activeElement)) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (document.activeElement === last || !node.contains(document.activeElement)) {
        e.preventDefault();
        first.focus();
      }
    }
  }

  // Focus first focusable element on mount
  requestAnimationFrame(() => {
    const focusable = getFocusableElements();
    if (focusable.length > 0) {
      focusable[0].focus();
    }
  });

  node.addEventListener('keydown', handleKeydown);

  return {
    update(newParams) {
      params = newParams || {};
    },
    destroy() {
      node.removeEventListener('keydown', handleKeydown);
      // Restore focus
      if (previouslyFocused && typeof previouslyFocused.focus === 'function') {
        previouslyFocused.focus();
      }
    },
  };
}
