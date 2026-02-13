/**
 * Plan Persistence — localStorage-backed save/load for surgical plans.
 *
 * Plans are stored as a JSON array under the key `fps_saved_plans`.
 * Each entry includes the full PlanDict plus metadata (saved timestamp, ID).
 *
 * Storage format:
 *   SavedPlan { id, savedAt, plan: PlanDict }
 *
 * No backend endpoints required — pure client-side persistence.
 */

import type { PlanDict } from './api-client';

const STORAGE_KEY = 'fps_saved_plans';
const MAX_PLANS = 50;

// ── Types ────────────────────────────────────────────────────────

export interface SavedPlan {
  /** Unique ID for this saved plan (crypto.randomUUID or fallback). */
  id: string;
  /** ISO 8601 timestamp of when the plan was saved. */
  savedAt: string;
  /** The full plan data at time of save. */
  plan: PlanDict;
}

// ── ID Generation ────────────────────────────────────────────────

function generateId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for environments without crypto.randomUUID
  return (
    Date.now().toString(36) +
    '-' +
    Math.random().toString(36).substring(2, 10)
  );
}

// ── Storage Helpers ──────────────────────────────────────────────

function readAll(): SavedPlan[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    return [];
  }
}

function writeAll(plans: SavedPlan[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(plans));
  } catch (err) {
    console.error('[plan-storage] Failed to write localStorage:', err);
  }
}

// ── Public API ───────────────────────────────────────────────────

/**
 * List all saved plans, most recent first.
 */
export function listSavedPlans(): SavedPlan[] {
  return readAll().sort(
    (a, b) => new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime(),
  );
}

/**
 * Save a plan to localStorage. If a plan with the same ID exists, it is
 * overwritten. Otherwise a new entry is created.
 *
 * @returns The SavedPlan record (with id and savedAt).
 */
export function savePlan(plan: PlanDict, existingId?: string): SavedPlan {
  const plans = readAll();
  const id = existingId ?? generateId();
  const savedAt = new Date().toISOString();
  const entry: SavedPlan = { id, savedAt, plan: structuredClone(plan) };

  const existingIndex = plans.findIndex((p) => p.id === id);
  if (existingIndex >= 0) {
    plans[existingIndex] = entry;
  } else {
    plans.unshift(entry);
  }

  // Enforce max limit — drop oldest
  if (plans.length > MAX_PLANS) {
    plans.sort(
      (a, b) => new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime(),
    );
    plans.length = MAX_PLANS;
  }

  writeAll(plans);
  return entry;
}

/**
 * Load a saved plan by ID.
 * @returns The SavedPlan, or null if not found.
 */
export function loadSavedPlan(id: string): SavedPlan | null {
  const plans = readAll();
  return plans.find((p) => p.id === id) ?? null;
}

/**
 * Delete a saved plan by ID.
 * @returns true if found and deleted, false otherwise.
 */
export function deleteSavedPlan(id: string): boolean {
  const plans = readAll();
  const before = plans.length;
  const filtered = plans.filter((p) => p.id !== id);
  if (filtered.length === before) return false;
  writeAll(filtered);
  return true;
}

/**
 * Rename a saved plan.
 * @returns The updated SavedPlan, or null if not found.
 */
export function renameSavedPlan(
  id: string,
  newName: string,
): SavedPlan | null {
  const plans = readAll();
  const entry = plans.find((p) => p.id === id);
  if (!entry) return null;
  entry.plan.name = newName;
  entry.savedAt = new Date().toISOString();
  writeAll(plans);
  return entry;
}

/**
 * Export a plan as a downloadable JSON file.
 */
export function exportPlanAsJson(plan: PlanDict, filename?: string): void {
  const name = filename ?? `${(plan.name || 'plan').replace(/\s+/g, '_')}.json`;
  const blob = new Blob([JSON.stringify(plan, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Import a plan from a JSON file. Returns the parsed PlanDict or throws.
 */
export function parsePlanFromJson(jsonString: string): PlanDict {
  const parsed = JSON.parse(jsonString);
  // Validate required fields
  if (typeof parsed !== 'object' || parsed === null) {
    throw new Error('Invalid plan file: not an object');
  }
  if (typeof parsed.name !== 'string') {
    throw new Error('Invalid plan file: missing "name" field');
  }
  if (typeof parsed.procedure !== 'string') {
    throw new Error('Invalid plan file: missing "procedure" field');
  }
  if (!Array.isArray(parsed.steps)) {
    throw new Error('Invalid plan file: missing "steps" array');
  }
  return {
    name: parsed.name,
    procedure: parsed.procedure,
    description: parsed.description ?? '',
    n_steps: parsed.steps.length,
    steps: parsed.steps,
    content_hash: parsed.content_hash ?? '',
  };
}
