/**
 * HyperTensor Simulation Types
 * 
 * Type definitions for CFD simulation management.
 * Uses camelCase for TypeScript conventions.
 */

// ============================================
// SIMULATION STATUS & STATE
// ============================================

export type SimulationStatus = 
  | 'pending' 
  | 'running' 
  | 'paused' 
  | 'completed' 
  | 'failed';

export type SolverType = 'steady' | 'transient' | 'pseudo-transient';

export type TurbulenceModel = 
  | 'laminar' 
  | 'ke-standard'
  | 'ke-realizable'
  | 'kw-sst'
  | 'sa'
  | 'les-smagorinsky'
  | 'des';

// ============================================
// SOLVER SETTINGS
// ============================================

export interface SolverSettings {
  solverType: SolverType;
  turbulenceModel: TurbulenceModel;
  maxIterations: number;
  convergenceTolerance: number;
  cflNumber: number;
  timeStep?: number;
  endTime?: number;
  relaxationFactors: {
    pressure: number;
    velocity: number;
    energy: number;
    turbulence: number;
  };
  discretization: {
    convection: 'upwind' | 'central' | 'quick' | 'muscl' | 'weno';
    gradient: 'green-gauss' | 'least-squares' | 'weighted-least-squares';
    time?: 'euler-implicit' | 'crank-nicolson' | 'bdf2' | 'rk4';
  };
  gpuAcceleration: boolean;
  precision: 'fp32' | 'fp64' | 'mixed';
}

export const DEFAULT_SOLVER_SETTINGS: SolverSettings = {
  solverType: 'steady',
  turbulenceModel: 'kw-sst',
  maxIterations: 5000,
  convergenceTolerance: 1e-6,
  cflNumber: 0.9,
  relaxationFactors: {
    pressure: 0.3,
    velocity: 0.7,
    energy: 0.9,
    turbulence: 0.7,
  },
  discretization: {
    convection: 'muscl',
    gradient: 'green-gauss',
  },
  gpuAcceleration: true,
  precision: 'mixed',
};

// ============================================
// PERFORMANCE METRICS
// ============================================

export interface PerformanceMetrics {
  throughput: number;              // Mcells/s
  gpuUtilization: number;          // 0-100
  vramUsedGb: number;
  vramTotalGb: number;
  wallTimeSeconds: number;
}

// ============================================
// RESIDUAL DATA (CONVERGENCE)
// ============================================

/**
 * Residual data point for convergence tracking.
 * Uses snake_case to match chart data keys.
 */
export interface ResidualPoint {
  iteration: number;
  continuity: number;
  momentum_x?: number;
  momentum_y?: number;
  momentum_z?: number;
  energy?: number;
  turbulent_ke?: number;
  turbulent_omega?: number;
  // Aliases for camelCase access
  momentumX?: number;
  momentumY?: number;
  momentumZ?: number;
  turbulentKe?: number;
  turbulentOmega?: number;
}

// ============================================
// SIMULATION ENTITIES
// ============================================

export interface SimulationSummary {
  id: string;
  name: string;
  status: SimulationStatus;
  meshId?: string;
  meshName?: string;
  currentIteration?: number;
  maxIterations?: number;
  createdAt: string;
  updatedAt?: string;
}

/**
 * Full simulation object with all details.
 * 
 * Note: Properties use both camelCase (for components) and 
 * snake_case aliases (for API response mapping).
 */
export interface Simulation extends SimulationSummary {
  description?: string;
  settings: SolverSettings;
  performance?: PerformanceMetrics;
  
  // Iteration tracking (snake_case for API compatibility)
  iteration: number;
  max_iterations: number;
  
  // Time tracking for transient simulations
  current_time: number;
  end_time: number;
  currentTime?: number;
  endTime?: number;
  
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

export interface SimulationCreate {
  name: string;
  description?: string;
  meshId: string;
  settings?: Partial<SolverSettings>;
}

// ============================================
// API RESPONSE TYPES
// ============================================

export interface SimulationListResponse {
  items: SimulationSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface SimulationFilters {
  status?: SimulationStatus;
  limit?: number;
  offset?: number;
  search?: string;
}
