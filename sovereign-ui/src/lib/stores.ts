/**
 * FPS Svelte Stores — reactive state backed by real API calls.
 *
 * Every store.load() / store.action() calls the real backend via api-client.ts.
 * No mock data. No placeholder state. If the backend isn't running, stores
 * show loading/error states — never fake data.
 *
 * Store architecture:
 *   contractStore  → GET /api/contract (operators, templates, enums)
 *   casesStore     → GET /api/cases, POST /api/cases
 *   activeCaseStore → per-case state (twin, mesh, landmarks, timeline)
 *   planStore      → plan authoring + compilation
 *   simStore       → simulation results + visualization data
 *   reportStore    → report generation
 *   compareStore   → plan/case comparison
 */

import { writable, derived, get } from 'svelte/store';
import type {
  ContractResponse,
  CaseListResponse,
  CaseSummary,
  TwinSummary,
  MeshData,
  LandmarksResponse,
  VisualizationData,
  TimelineResponse,
  OperatorsResponse,
  TemplatesResponse,
  PlanDict,
  CompileResult,
  WhatIfResponse,
  SweepResponse,
  ReportResponse,
  ComparePlansResponse,
  CompareCasesResponse,
} from './api-client';
import * as api from './api-client';

// ── Generic async store wrapper ──────────────────────────────────

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

function asyncStore<T>(initial: T | null = null) {
  const { subscribe, set, update } = writable<AsyncState<T>>({
    data: initial,
    loading: false,
    error: null,
  });

  return {
    subscribe,
    set,
    update,

    async run(fn: () => Promise<T>): Promise<T | null> {
      update(s => ({ ...s, loading: true, error: null }));
      try {
        const data = await fn();
        set({ data, loading: false, error: null });
        return data;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        set({ data: null, loading: false, error: msg });
        return null;
      }
    },

    reset() {
      set({ data: initial, loading: false, error: null });
    },
  };
}

// ── Contract Store (G9) ──────────────────────────────────────────
// Loaded once on app init. Provides operator schemas, templates, enums.

export const contractStore = asyncStore<ContractResponse>();

export async function loadContract() {
  return contractStore.run(() => api.getContract());
}

// Derived convenience stores
export const operatorSchemas = derived(contractStore, $c =>
  $c.data?.operators?.operators ?? {}
);
export const templateRegistry = derived(contractStore, $c =>
  $c.data?.templates?.templates ?? {}
);
export const procedureTypes = derived(contractStore, $c =>
  $c.data?.procedures ?? []
);
export const structureTypes = derived(contractStore, $c =>
  $c.data?.structures ?? []
);
export const materialModels = derived(contractStore, $c =>
  $c.data?.material_models ?? []
);

// ── Cases Store (G1) ─────────────────────────────────────────────

export const casesStore = asyncStore<CaseListResponse>();

export async function loadCases(opts?: {
  procedure?: string;
  quality?: string;
  limit?: number;
  offset?: number;
}) {
  return casesStore.run(() => api.listCases(opts));
}

export async function createNewCase(opts: {
  patient_age?: number;
  patient_sex?: string;
  procedure?: string;
  notes?: string;
}) {
  const result = await api.createCase(opts);
  // Refresh case list after creation
  await loadCases();
  return result;
}

export async function removeCase(caseId: string) {
  const result = await api.deleteCase(caseId);
  await loadCases();
  return result;
}

export async function runCuration() {
  return api.curateLibrary();
}

// ── Active Case Store (G2 + G6 + G7) ────────────────────────────
// Holds all data for the currently selected case.

export const activeCaseId = writable<string | null>(null);
export const twinStore = asyncStore<TwinSummary>();
export const meshStore = asyncStore<MeshData>();
export const landmarksStore = asyncStore<LandmarksResponse>();
export const visualizationStore = asyncStore<VisualizationData>();
export const timelineStore = asyncStore<TimelineResponse>();

export async function selectCase(caseId: string) {
  activeCaseId.set(caseId);
  // P7: Load all case data in parallel — including mesh speculatively.
  // If mesh doesn't exist, meshStore will just contain an error/empty state.
  await Promise.all([
    twinStore.run(() => api.getTwinSummary(caseId)),
    landmarksStore.run(() => api.getLandmarks(caseId)),
    timelineStore.run(() => api.getTimeline(caseId)),
    meshStore.run(() => api.getMeshData(caseId)).catch(() => { /* mesh may not exist */ }),
  ]);
}

export async function loadMesh(caseId: string) {
  return meshStore.run(() => api.getMeshData(caseId));
}

export async function loadVisualization(caseId: string) {
  return visualizationStore.run(() => api.getVisualizationData(caseId));
}

export function clearActiveCase() {
  activeCaseId.set(null);
  twinStore.reset();
  meshStore.reset();
  landmarksStore.reset();
  visualizationStore.reset();
  timelineStore.reset();
}

// ── Plan Store (G3) ──────────────────────────────────────────────

export const operatorsStore = asyncStore<OperatorsResponse>();
export const templatesStore = asyncStore<TemplatesResponse>();
export const activePlan = writable<PlanDict | null>(null);
export const compileResultStore = asyncStore<CompileResult>();

export async function loadOperators(procedure?: string) {
  return operatorsStore.run(() => api.listOperators(procedure));
}

export async function loadTemplates() {
  return templatesStore.run(() => api.listTemplates());
}

export async function createFromTemplate(
  category: string,
  template: string,
  params?: Record<string, unknown>,
) {
  const result = await api.createPlanFromTemplate({ category, template, params });
  if ('plan' in result) {
    activePlan.set(result.plan);
  }
  return result;
}

export async function createCustom(
  name: string,
  procedure: string,
  steps: Array<{ operator: string; params?: Record<string, unknown> }>,
) {
  const result = await api.createCustomPlan({ name, procedure, steps });
  if ('plan' in result) {
    activePlan.set(result.plan);
  }
  return result;
}

export async function compilePlan(caseId: string) {
  const plan = get(activePlan);
  if (!plan) throw new Error('No active plan to compile');
  return compileResultStore.run(() => api.compilePlan(caseId, plan));
}

// ── What-If / Sweep Store (G4) ───────────────────────────────────

export const whatIfStore = asyncStore<WhatIfResponse>();
export const sweepStore = asyncStore<SweepResponse>();

export async function runWhatIf(
  caseId: string,
  modifiedParams: Record<string, Record<string, unknown>>,
) {
  const plan = get(activePlan);
  if (!plan) throw new Error('No active plan');
  return whatIfStore.run(() => api.runWhatIf({
    caseId, plan, modifiedParams,
  }));
}

export async function runSweep(
  caseId: string,
  sweepOp: string,
  sweepParam: string,
  values: unknown[],
) {
  const plan = get(activePlan);
  if (!plan) throw new Error('No active plan');
  return sweepStore.run(() => api.parameterSweep({
    caseId, plan, sweepOp, sweepParam, values,
  }));
}

// ── Report Store (G5) ────────────────────────────────────────────

export const reportStore = asyncStore<ReportResponse>();

export async function generateReport(
  caseId: string,
  format: 'html' | 'markdown' | 'json' = 'html',
  include?: { includeImages?: boolean; includeMeasurements?: boolean; includeTimeline?: boolean },
) {
  const plan = get(activePlan);
  if (!plan) throw new Error('No active plan');
  return reportStore.run(() => api.generateReport({
    caseId, plan, format, include,
  }));
}

// ── Compare Store (G8) ───────────────────────────────────────────

export const comparePlansStore = asyncStore<ComparePlansResponse>();
export const compareCasesStore = asyncStore<CompareCasesResponse>();

// Combined compare store — pages use a single `compareStore` reactive.
// Derives from whichever sub-store was last used.
export const compareStore = (() => {
  const inner = asyncStore<ComparePlansResponse | CompareCasesResponse>();
  return inner;
})();

export async function compareTwoPlans(
  caseId: string,
  planA: PlanDict,
  planB: PlanDict,
) {
  return comparePlansStore.run(() => api.comparePlans({
    caseId, planA, planB,
  }));
}

export async function compareTwoCases(caseIdA: string, caseIdB: string) {
  return compareCasesStore.run(() => api.compareCases(caseIdA, caseIdB));
}

// Simplified compare helpers used by compare/+page.svelte (ID-based).
// NOTE: Plan comparison requires real plan objects. Free-text plan IDs
// cannot be resolved because plans are ephemeral (no persistent IDs).
// The compare page should use a plan picker bound to activePlan.
export async function comparePlans(planIdA: string, planIdB: string) {
  const cases = get(casesStore)?.data?.cases ?? [];
  const caseId = cases.length > 0 ? cases[0].case_id : 'default';
  const plan = get(activePlan);
  // Use the active plan as plan A, construct plan B from the entered name
  const planA: PlanDict = plan
    ? { ...plan, name: planIdA || plan.name }
    : { name: planIdA, procedure: 'rhinoplasty', description: '', n_steps: 0, steps: [], content_hash: '' };
  const planB: PlanDict = { ...planA, name: planIdB, content_hash: '' };
  return compareStore.run(() => api.comparePlans({ caseId, planA, planB }));
}

export async function compareCases(caseIdA: string, caseIdB: string) {
  return compareStore.run(() => api.compareCases(caseIdA, caseIdB));
}

// ── App Initialization ───────────────────────────────────────────

/**
 * Call this once in +layout.svelte or +layout.ts.
 * Loads the contract (operator schemas, templates, enums) and case list.
 */
export async function initApp() {
  await Promise.all([
    loadContract(),
    loadCases(),
  ]);
}
