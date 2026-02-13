/**
 * Plan History — undo/redo stack for surgical plan editing.
 *
 * Maintains a linear history of PlanDict snapshots. Each mutation
 * pushes a snapshot; undo pops to the previous; redo re-applies.
 *
 * Uses structuredClone for immutable snapshots — no shared references.
 */

import { writable, derived, get } from 'svelte/store';
import type { PlanDict } from './api-client';

const MAX_HISTORY = 50;

// ── Internal State ───────────────────────────────────────────────

interface HistoryState {
  /** All snapshots. Index 0 is oldest. */
  stack: PlanDict[];
  /** Current position in the stack (0-based). -1 = empty. */
  cursor: number;
}

const history = writable<HistoryState>({ stack: [], cursor: -1 });

// ── Public Stores ────────────────────────────────────────────────

/** Whether undo is available. */
export const canUndo = derived(history, ($h) => $h.cursor > 0);

/** Whether redo is available. */
export const canRedo = derived(history, ($h) => $h.cursor < $h.stack.length - 1);

/** Current undo depth (number of undo steps available). */
export const undoDepth = derived(history, ($h) => $h.cursor);

/** Current redo depth (number of redo steps available). */
export const redoDepth = derived(history, ($h) =>
  $h.stack.length - 1 - $h.cursor,
);

// ── Actions ──────────────────────────────────────────────────────

/**
 * Push a new plan snapshot onto the history stack.
 * Discards any redo states (fork in history).
 */
export function pushHistory(plan: PlanDict): void {
  history.update((h) => {
    // Truncate any redo states
    const stack = h.stack.slice(0, h.cursor + 1);
    stack.push(structuredClone(plan));

    // Enforce max history size — drop oldest
    if (stack.length > MAX_HISTORY) {
      stack.shift();
      return { stack, cursor: stack.length - 1 };
    }

    return { stack, cursor: stack.length - 1 };
  });
}

/**
 * Undo — move cursor back one step.
 * @returns The previous PlanDict snapshot, or null if at bottom.
 */
export function undo(): PlanDict | null {
  const h = get(history);
  if (h.cursor <= 0) return null;

  const newCursor = h.cursor - 1;
  history.update((s) => ({ ...s, cursor: newCursor }));
  return structuredClone(h.stack[newCursor]);
}

/**
 * Redo — move cursor forward one step.
 * @returns The next PlanDict snapshot, or null if at top.
 */
export function redo(): PlanDict | null {
  const h = get(history);
  if (h.cursor >= h.stack.length - 1) return null;

  const newCursor = h.cursor + 1;
  history.update((s) => ({ ...s, cursor: newCursor }));
  return structuredClone(h.stack[newCursor]);
}

/**
 * Clear the entire history stack. Call when starting a fresh plan.
 */
export function clearHistory(): void {
  history.set({ stack: [], cursor: -1 });
}

/**
 * Initialize history with an initial plan state.
 * Clears existing history and pushes the initial state.
 */
export function initHistory(plan: PlanDict): void {
  history.set({
    stack: [structuredClone(plan)],
    cursor: 0,
  });
}
