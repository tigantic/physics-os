/**
 * api-client-ext.ts — Phase 7 API extensions
 *
 * Extends the base api-client.ts with surgical simulation endpoints.
 * Import alongside the original: import { ... } from '$lib/api-client-ext';
 *
 * These endpoints may not all exist in the backend yet.
 * The UI is built to gracefully degrade: if an endpoint returns 404/500,
 * the corresponding panel shows "not available" with a trigger to run the computation.
 */

const BASE = import.meta.env.VITE_API_BASE ?? '';

// ── Helpers ──────────────────────────────────────────────────

async function get<T>(path: string): Promise<T | null> {
  try {
    const resp = await fetch(`${BASE}${path}`);
    if (!resp.ok) return null;
    return await resp.json();
  } catch { return null; }
}

async function post<T>(path: string, body: any = {}): Promise<T | null> {
  try {
    const resp = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch { return null; }
}

// ── Types ────────────────────────────────────────────────────

export interface LayerMeshData {
  positions: number[][];
  indices: number[][];
  region_ids?: number[];
  layer_type: string;
  n_vertices: number;
  n_triangles: number;
}

export interface TissueLayersResponse {
  case_id: string;
  layers: Record<string, LayerMeshData>;
  layer_order: string[];
}

export interface StreamlineData {
  lines: Array<{
    points: number[][];
    velocities?: number[];
    radius?: number;
  }>;
  velocity_min: number;
  velocity_max: number;
  opacity?: number;
}

export interface CfdResults {
  case_id: string;
  streamlines: StreamlineData;
  summary: {
    peak_velocity: number;
    mean_velocity: number;
    flow_rate: number;
    reynolds: number;
  };
  resistance: {
    left: number;
    right: number;
    total: number;
  };
  solver?: {
    method: string;
    mesh_elements: number;
    iterations: number;
    convergence: string;
    compute_time: string;
  };
  velocity_min: number;
  velocity_max: number;
}

export interface ScalarFieldData {
  field: string;
  layer?: string;
  values: number[];
  min: number;
  max: number;
  mean: number;
  std: number;
  colormap?: string;
  by_region?: Record<string, { min: number; max: number; mean: number }>;
}

export interface FemResults {
  case_id: string;
  fields: Record<string, ScalarFieldData>;
  solver?: {
    type: string;
    elements: number;
    nodes: number;
    dof: number;
    iterations: number;
    residual: string;
    compute_time: string;
  };
  boundary_conditions?: Array<{
    type: string;
    location: string;
    value: string;
  }>;
  material_properties?: Array<{
    tissue: string;
    youngs_modulus: number;
    poisson_ratio: number;
    density?: number;
    unit?: string;
  }>;
}

export interface DicomSeriesData {
  case_id: string;
  dimensions: [number, number, number];
  volume?: Int16Array;
  slices?: Record<string, Array<{ width: number; height: number; pixels: Int16Array }>>;
  metadata?: {
    modality: string;
    slice_thickness: number;
    pixel_spacing: string;
    dimensions: number[];
  };
}

export interface IncisionResult {
  incision_id: string;
  path: number[][];
  depth: number;
  affected_layers: string[];
  updated_mesh?: Record<string, LayerMeshData>;
}

export interface OsteotomyResult {
  osteotomy_id: string;
  cut_plane: { normal: number[]; point: number[] };
  displacement: number[];
  rotation: number[];
  updated_bone?: LayerMeshData;
  stress_field?: ScalarFieldData;
}

export interface GraftResult {
  graft_id: string;
  position: number[];
  dimensions: { length: number; width: number; thickness: number };
  graft_mesh?: LayerMeshData;
  donor_site?: string;
}

export interface PostOpPrediction {
  case_id: string;
  plan_id: string;
  layers: Record<string, LayerMeshData>;
  landmarks_predicted: any;
  cfd_predicted?: CfdResults;
  fem_predicted?: FemResults;
}

// ── API Functions ────────────────────────────────────────────

/** Get tissue layers (bone, cartilage, muscle, skin, fascia) for a case */
export function loadTissueLayers(caseId: string): Promise<TissueLayersResponse | null> {
  return get(`/api/cases/${caseId}/layers`);
}

/** Get CFD airflow results for a case (pre-op or post-plan) */
export function loadCfdResults(caseId: string, planHash?: string): Promise<CfdResults | null> {
  const q = planHash ? `?plan_hash=${planHash}` : '';
  return get(`/api/cases/${caseId}/cfd${q}`);
}

/** Run CFD simulation */
export function runCfdSimulation(caseId: string, planHash?: string): Promise<CfdResults | null> {
  return post(`/api/simulate/cfd`, { case_id: caseId, plan_hash: planHash });
}

/** Get FEM results for a case */
export function loadFemResults(caseId: string, planHash?: string): Promise<FemResults | null> {
  const q = planHash ? `?plan_hash=${planHash}` : '';
  return get(`/api/cases/${caseId}/fem${q}`);
}

/** Run FEM simulation */
export function runFemSimulation(caseId: string, planHash?: string): Promise<FemResults | null> {
  return post(`/api/simulate/fem`, { case_id: caseId, plan_hash: planHash });
}

/** Get DICOM volume data for a case */
export function loadDicomData(caseId: string): Promise<DicomSeriesData | null> {
  return get(`/api/cases/${caseId}/dicom`);
}

/** Upload DICOM files for a case */
export async function uploadDicom(caseId: string, files: FileList): Promise<{ success: boolean } | null> {
  try {
    // Convert files to base64 array for JSON upload (server expects JSON, not multipart)
    const fileDataArray: string[] = [];
    for (let i = 0; i < files.length; i++) {
      const buf = await files[i].arrayBuffer();
      const bytes = new Uint8Array(buf);
      let binary = '';
      for (let j = 0; j < bytes.byteLength; j++) {
        binary += String.fromCharCode(bytes[j]);
      }
      fileDataArray.push(btoa(binary));
    }
    return post(`/api/dicom/upload`, { case_id: caseId, data: fileDataArray });
  } catch { return null; }
}

/** Execute incision on tissue layers */
export function executeIncision(caseId: string, points: number[][], depth: number, layers?: string[]): Promise<IncisionResult | null> {
  return post(`/api/simulate/incision`, { case_id: caseId, points, depth, layers });
}

/** Execute osteotomy (bone cut + displacement) */
export function executeOsteotomy(
  caseId: string,
  cutPlane: { normal: number[]; point: number[] },
  displacement: number[],
  rotation: number[],
): Promise<OsteotomyResult | null> {
  return post(`/api/simulate/osteotomy`, {
    case_id: caseId,
    plane: cutPlane,
    movement: { displacement, rotation },
  });
}

/** Place cartilage graft */
export function placeGraft(caseId: string, position: number[], normal: number[], graftType: string, dimensions: { length: number; width: number; thickness: number }): Promise<GraftResult | null> {
  return post(`/api/simulate/graft`, { case_id: caseId, position, normal, graft_type: graftType, dimensions });
}

/** Generate post-op prediction from compiled plan */
export function generatePostOpPrediction(caseId: string, planHash: string): Promise<PostOpPrediction | null> {
  return post(`/api/predict/postop`, { case_id: caseId, plan_hash: planHash });
}

/** Get post-op prediction if already computed */
export function loadPostOpPrediction(caseId: string, planHash: string): Promise<PostOpPrediction | null> {
  return get(`/api/cases/${caseId}/prediction?plan_hash=${planHash}`);
}

// ── Phase 8: Analytics endpoints ─────────────────────────────

export interface SafetyReport {
  case_id: string;
  overall_safety_index: number;
  is_safe: boolean;
  skin_tension: {
    max_tension_pa: number;
    mean_tension_pa: number;
    threshold_pa: number;
    safe: boolean;
  };
  vascular_risk: {
    risk_level: string;
    max_depth_mm: number;
    safe: boolean;
  };
  structural_integrity: {
    min_thickness_mm: number;
    safe: boolean;
  };
  stress_violations: Array<{
    region: string;
    value: number;
    threshold: number;
  }>;
  [key: string]: unknown;
}

export interface AestheticsReport {
  case_id: string;
  nasofrontal_angle: number;
  nasolabial_angle: number;
  goode_ratio: number;
  dorsal_line_deviation: number;
  symmetry_score: number;
  overall_aesthetic_score: number;
  [key: string]: unknown;
}

export interface FunctionalReport {
  case_id: string;
  reynolds_number: number;
  flow_regime: string;
  nasal_resistance: {
    left: number;
    right: number;
    total: number;
  };
  wall_shear_stress: {
    max_pa: number;
    mean_pa: number;
  };
  cross_sectional_areas: Array<{
    position_mm: number;
    area_mm2: number;
  }>;
  [key: string]: unknown;
}

export interface HealingTimeline {
  case_id: string;
  milestones: Array<{
    day: number;
    label: string;
    description: string;
    edema_fraction: number;
    structural_integrity: number;
    [key: string]: unknown;
  }>;
  edema_curve: Array<{
    day: number;
    edema_fraction: number;
  }>;
  [key: string]: unknown;
}

export interface UncertaintyResult {
  case_id: string;
  n_samples: number;
  parameters: Array<{
    name: string;
    distribution: string;
    nominal: number;
    bounds: [number, number];
  }>;
  statistics: Record<string, {
    mean: number;
    std: number;
    ci_lower: number;
    ci_upper: number;
  }>;
  sobol_indices?: Record<string, Record<string, number>>;
  compute_time_s: number;
  [key: string]: unknown;
}

export interface OptimizationResult {
  case_id: string;
  n_generations: number;
  pareto_front: Array<{
    objectives: Record<string, number>;
    parameters: Record<string, number>;
    constraints_satisfied: boolean;
  }>;
  best_compromise: {
    objectives: Record<string, number>;
    parameters: Record<string, number>;
  };
  compute_time_s: number;
  [key: string]: unknown;
}

/** Evaluate surgical safety (skin tension, vascular risk, structural integrity) */
export function evaluateSafety(caseId: string, planHash?: string): Promise<SafetyReport | null> {
  return post('/api/evaluate/safety', { case_id: caseId, plan_hash: planHash });
}

/** Evaluate aesthetic angles and proportions */
export function evaluateAesthetics(caseId: string, postopLandmarks?: Record<string, number[]>): Promise<AestheticsReport | null> {
  return post('/api/evaluate/aesthetics', { case_id: caseId, postop_landmarks: postopLandmarks });
}

/** Evaluate functional airflow metrics */
export function evaluateFunctional(caseId: string, planHash?: string): Promise<FunctionalReport | null> {
  return post('/api/evaluate/functional', { case_id: caseId, plan_hash: planHash });
}

/** Get healing timeline with milestones and edema curve */
export function getHealingTimeline(caseId: string, planHash?: string): Promise<HealingTimeline | null> {
  return post('/api/predict/healing', { case_id: caseId, plan_hash: planHash });
}

// ── Phase 9: UQ + Optimization endpoints ─────────────────────

/** Run Monte Carlo uncertainty quantification with Sobol sensitivity */
export function quantifyUncertainty(
  caseId: string,
  planHash?: string,
  nSamples?: number,
  computeSobol?: boolean,
): Promise<UncertaintyResult | null> {
  return post('/api/uncertainty', {
    case_id: caseId,
    plan_hash: planHash,
    n_samples: nSamples ?? 32,
    compute_sobol: computeSobol ?? true,
  });
}

/** Run NSGA-II multi-objective optimization */
export function optimizePlan(
  caseId: string,
  template?: string,
  populationSize?: number,
  nGenerations?: number,
): Promise<OptimizationResult | null> {
  return post('/api/optimize', {
    case_id: caseId,
    template: template ?? 'reduction_rhinoplasty',
    population_size: populationSize ?? 20,
    n_generations: nGenerations ?? 20,
  });
}
