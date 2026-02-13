/**
 * FPS API Client — typed wrapper for every backend endpoint.
 *
 * RULE: Every function in this file calls a real HTTP endpoint.
 *       No mock data. No hardcoded values. No placeholder responses.
 *
 * Maps 1:1 to server.py routes → api.py UIApplication methods.
 *
 * 21 endpoints total:
 *   GET  /api/contract              → getContract()
 *   GET  /api/cases                 → listCases()
 *   GET  /api/cases/:id             → getCase()
 *   GET  /api/cases/:id/twin        → getTwinSummary()
 *   GET  /api/cases/:id/mesh        → getMeshData()
 *   GET  /api/cases/:id/landmarks   → getLandmarks()
 *   GET  /api/cases/:id/visualization → getVisualizationData()
 *   GET  /api/cases/:id/timeline    → getTimeline()
 *   GET  /api/operators             → listOperators()
 *   GET  /api/templates             → listTemplates()
 *   POST /api/cases                 → createCase()
 *   POST /api/cases/:id/delete      → deleteCase()
 *   POST /api/curate                → curateLibrary()
 *   POST /api/plan/template         → createPlanFromTemplate()
 *   POST /api/plan/custom           → createCustomPlan()
 *   POST /api/plan/compile          → compilePlan()
 *   POST /api/whatif                → runWhatIf()
 *   POST /api/sweep                 → parameterSweep()
 *   POST /api/report                → generateReport()
 *   POST /api/compare/plans         → comparePlans()
 *   POST /api/compare/cases         → compareCases()
 */

// ── Configuration ────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_BASE ?? '';
const API_KEY = import.meta.env.VITE_API_KEY ?? '';

// ── Types (derived from api.py return shapes) ────────────────────

export interface ApiError {
  error: string;
}

export interface CaseSummary {
  case_id: string;
  procedure_type?: string;
  quality_level?: string;
  [key: string]: unknown;
}

export interface CaseListResponse {
  total: number;
  offset: number;
  limit: number;
  cases: CaseSummary[];
}

export interface CaseDetailResponse {
  case_id: string;
  metadata: Record<string, unknown>;
}

export interface CreateCaseResponse {
  case_id: string;
  status: string;
}

export interface DeleteCaseResponse {
  case_id: string;
  status: string;
}

export interface CurateResponse {
  [key: string]: unknown;
}

export interface MeshRegion {
  structure: string;
  material: string;
}

export interface TwinSummary {
  case_id: string;
  mesh: {
    n_nodes: number;
    n_elements: number;
    element_type: string;
    n_regions: number;
    regions: Record<string, MeshRegion>;
  } | null;
  landmarks: Record<string, number[]>;
  segmentation: {
    shape: number[];
    n_labels: number;
  } | null;
}

export interface MeshData {
  case_id: string;
  positions: number[][];
  indices: number[][];
  n_vertices: number;
  n_triangles: number;
  region_ids: number[];
}

export interface LandmarkData {
  type: string;
  position: number[];
  confidence: number;
}

export interface LandmarksResponse {
  case_id: string;
  landmarks: LandmarkData[];
}

export interface VisualizationData {
  mesh: MeshData;
  landmarks?: LandmarksResponse;
  region_colors?: Record<string, string>;
}

export interface TimelineEvent {
  [key: string]: unknown;
}

export interface TimelineResponse {
  case_id: string;
  events: TimelineEvent[];
  n_events: number;
}

export interface ParamDef {
  name: string;
  param_type: string;
  unit: string;
  description: string;
  default: unknown;
  min_value: number | null;
  max_value: number | null;
  enum_values: string[] | null;
}

export interface OperatorSchema {
  name: string;
  category: string;
  procedure: string;
  params: Record<string, unknown>;
  affected_structures: string[];
  description: string;
  param_defs: Record<string, ParamDef>;
}

export interface OperatorsResponse {
  operators: Record<string, OperatorSchema>;
  count: number;
}

export interface TemplatesResponse {
  templates: Record<string, string[]>;
}

export interface PlanDict {
  name: string;
  procedure: string;
  description?: string;
  n_steps: number;
  steps: OperatorSchema[];
  content_hash: string;
}

export interface PlanResponse {
  plan: PlanDict;
}

export interface CompileResult {
  n_bcs?: number;
  n_material_mods?: number;
  n_mesh_mods?: number;
  content_hash?: string;
  [key: string]: unknown;
}

export interface WhatIfResponse {
  scenario: string;
  modified_operators: string[];
  result: CompileResult;
}

export interface SweepPoint {
  value: unknown;
  result: CompileResult;
}

export interface SweepResponse {
  sweep_op: string;
  sweep_param: string;
  n_points: number;
  results: SweepPoint[];
}

export interface ReportResponse {
  format: string;
  content: string;
}

export interface ComparePlansResponse {
  case_id: string;
  plan_a: { name: string; result: CompileResult };
  plan_b: { name: string; result: CompileResult };
  delta?: {
    n_bcs_diff: number;
    n_material_mods_diff: number;
    n_mesh_mods_diff: number;
  };
}

export interface CompareCasesResponse {
  case_a: TwinSummary;
  case_b: TwinSummary;
  mesh_diff: {
    comparable: boolean;
    node_diff?: number;
    element_diff?: number;
    region_diff?: number;
    reason?: string;
  };
}

export interface ContractResponse {
  version: string;
  modes: Record<string, { actions: string[] }>;
  operators: OperatorsResponse;
  templates: TemplatesResponse;
  procedures: string[];
  structures: string[];
  material_models: string[];
}

// ── HTTP helpers ─────────────────────────────────────────────────

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  let url = `${API_BASE}${path}`;
  if (params) {
    const qs = new URLSearchParams(params).toString();
    if (qs) url += `?${qs}`;
  }
  if (import.meta.env.DEV) console.debug(`[api] GET ${url}`);
  const res = await fetch(url, {
    headers: { 'X-API-Key': API_KEY },
  });
  if (import.meta.env.DEV) console.debug(`[api] GET ${url} → ${res.status}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    if (import.meta.env.DEV) console.error(`[api] GET ${url} error:`, body);
    throw new Error(body.error ?? `HTTP ${res.status}`);
  }
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(data.error ?? `HTTP ${res.status}`);
  }
  return res.json();
}

// ── API Client ───────────────────────────────────────────────────

/** G9: Get the full interaction contract (operator schemas, templates, enums) */
export async function getContract(): Promise<ContractResponse> {
  return get('/api/contract');
}

// ── G1: Case Library ─────────────────────────────────────────────

export async function listCases(opts?: {
  procedure?: string;
  quality?: string;
  limit?: number;
  offset?: number;
}): Promise<CaseListResponse> {
  const params: Record<string, string> = {};
  if (opts?.procedure) params.procedure = opts.procedure;
  if (opts?.quality) params.quality = opts.quality;
  if (opts?.limit !== undefined) params.limit = String(opts.limit);
  if (opts?.offset !== undefined) params.offset = String(opts.offset);
  return get('/api/cases', params);
}

export async function getCase(caseId: string): Promise<CaseDetailResponse> {
  return get(`/api/cases/${caseId}`);
}

export async function createCase(opts: {
  patient_age?: number;
  patient_sex?: string;
  procedure?: string;
  notes?: string;
}): Promise<CreateCaseResponse> {
  return post('/api/cases', opts);
}

export async function deleteCase(caseId: string): Promise<DeleteCaseResponse> {
  return post(`/api/cases/${caseId}/delete`, {});
}

export async function curateLibrary(): Promise<CurateResponse> {
  return post('/api/curate', {});
}

// ── G2: Twin Inspect ─────────────────────────────────────────────

export async function getTwinSummary(caseId: string): Promise<TwinSummary> {
  return get(`/api/cases/${caseId}/twin`);
}

export async function getMeshData(caseId: string): Promise<MeshData> {
  return get(`/api/cases/${caseId}/mesh`);
}

export async function getLandmarks(caseId: string): Promise<LandmarksResponse> {
  return get(`/api/cases/${caseId}/landmarks`);
}

// ── G3: Plan Author ──────────────────────────────────────────────

export async function listOperators(procedure?: string): Promise<OperatorsResponse> {
  const params: Record<string, string> = {};
  if (procedure) params.procedure = procedure;
  return get('/api/operators', params);
}

export async function listTemplates(): Promise<TemplatesResponse> {
  return get('/api/templates');
}

export async function createPlanFromTemplate(opts: {
  category: string;
  template: string;
  params?: Record<string, unknown>;
}): Promise<PlanResponse> {
  return post('/api/plan/template', opts);
}

export async function createCustomPlan(opts: {
  name: string;
  procedure: string;
  steps: Array<{ operator: string; params?: Record<string, unknown> }>;
}): Promise<PlanResponse> {
  return post('/api/plan/custom', opts);
}

export async function compilePlan(
  caseId: string,
  plan: PlanDict,
): Promise<CompileResult> {
  return post('/api/plan/compile', { case_id: caseId, plan });
}

// ── G4: Consult (What-if) ────────────────────────────────────────

export async function runWhatIf(opts: {
  caseId: string;
  plan: PlanDict;
  modifiedParams: Record<string, Record<string, unknown>>;
}): Promise<WhatIfResponse> {
  return post('/api/whatif', {
    case_id: opts.caseId,
    plan: opts.plan,
    modified_params: opts.modifiedParams,
  });
}

export async function parameterSweep(opts: {
  caseId: string;
  plan: PlanDict;
  sweepOp: string;
  sweepParam: string;
  values: unknown[];
}): Promise<SweepResponse> {
  return post('/api/sweep', {
    case_id: opts.caseId,
    plan: opts.plan,
    sweep_op: opts.sweepOp,
    sweep_param: opts.sweepParam,
    values: opts.values,
  });
}

// ── G5: Report ───────────────────────────────────────────────────

export async function generateReport(opts: {
  caseId: string;
  plan: PlanDict;
  format?: 'html' | 'markdown' | 'json';
  include?: { includeImages?: boolean; includeMeasurements?: boolean; includeTimeline?: boolean };
}): Promise<ReportResponse> {
  return post('/api/report', {
    case_id: opts.caseId,
    plan: opts.plan,
    format: opts.format ?? 'html',
    include: opts.include ?? {},
  });
}

// ── G6: Visualization ────────────────────────────────────────────

export async function getVisualizationData(
  caseId: string,
): Promise<VisualizationData> {
  return get(`/api/cases/${caseId}/visualization`);
}

// ── G7: Timeline ─────────────────────────────────────────────────

export async function getTimeline(caseId: string): Promise<TimelineResponse> {
  return get(`/api/cases/${caseId}/timeline`);
}

// ── G8: Compare ──────────────────────────────────────────────────

export async function comparePlans(opts: {
  caseId: string;
  planA: PlanDict;
  planB: PlanDict;
}): Promise<ComparePlansResponse> {
  return post('/api/compare/plans', {
    case_id: opts.caseId,
    plan_a: opts.planA,
    plan_b: opts.planB,
  });
}

export async function compareCases(
  caseIdA: string,
  caseIdB: string,
): Promise<CompareCasesResponse> {
  return post('/api/compare/cases', {
    case_id_a: caseIdA,
    case_id_b: caseIdB,
  });
}

// ── Endpoint registry (for validation) ───────────────────────────

export const ENDPOINT_REGISTRY = [
  { method: 'GET',  path: '/api/contract',         fn: 'getContract' },
  { method: 'GET',  path: '/api/cases',             fn: 'listCases' },
  { method: 'GET',  path: '/api/cases/:id',          fn: 'getCase' },
  { method: 'GET',  path: '/api/cases/:id/twin',     fn: 'getTwinSummary' },
  { method: 'GET',  path: '/api/cases/:id/mesh',     fn: 'getMeshData' },
  { method: 'GET',  path: '/api/cases/:id/landmarks', fn: 'getLandmarks' },
  { method: 'GET',  path: '/api/cases/:id/visualization', fn: 'getVisualizationData' },
  { method: 'GET',  path: '/api/cases/:id/timeline',  fn: 'getTimeline' },
  { method: 'GET',  path: '/api/operators',          fn: 'listOperators' },
  { method: 'GET',  path: '/api/templates',          fn: 'listTemplates' },
  { method: 'POST', path: '/api/cases',              fn: 'createCase' },
  { method: 'POST', path: '/api/cases/:id/delete',   fn: 'deleteCase' },
  { method: 'POST', path: '/api/curate',             fn: 'curateLibrary' },
  { method: 'POST', path: '/api/plan/template',      fn: 'createPlanFromTemplate' },
  { method: 'POST', path: '/api/plan/custom',        fn: 'createCustomPlan' },
  { method: 'POST', path: '/api/plan/compile',       fn: 'compilePlan' },
  { method: 'POST', path: '/api/whatif',             fn: 'runWhatIf' },
  { method: 'POST', path: '/api/sweep',              fn: 'parameterSweep' },
  { method: 'POST', path: '/api/report',             fn: 'generateReport' },
  { method: 'POST', path: '/api/compare/plans',      fn: 'comparePlans' },
  { method: 'POST', path: '/api/compare/cases',      fn: 'compareCases' },
] as const;
