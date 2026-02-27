/**
 * HyperTensor Types - Central Export
 * 
 * Re-exports all domain types for clean imports throughout the application.
 */

// ============================================
// DOMAIN TYPE EXPORTS
// ============================================

export * from './simulation';
export * from './mesh';
export * from './fields';

// ============================================
// SYSTEM TYPES
// ============================================

export type SystemHealth = 'healthy' | 'degraded' | 'unhealthy';

export interface SystemStatus {
  status: SystemHealth;
  version: string;
  uptime: number;
  activeSimulations: number;
  queuedSimulations: number;
  gpuCount: number;
  gpuUtilization: number;
  memoryUsedGb: number;
  memoryTotalGb: number;
}

export interface GPUInfo {
  index: number;
  name: string;
  driverVersion?: string;
  cudaVersion?: string;
  memoryTotal: number;
  memoryUsed: number;
  memoryFree?: number;
  utilization: number;
  temperature?: number;
  powerDraw?: number;
}

// ============================================
// WEBSOCKET EVENT TYPES
// ============================================

export type WebSocketChannel = 
  | `simulation.${string}.status`
  | `simulation.${string}.residuals`
  | `simulation.${string}.field`
  | 'system.gpus';

export interface WebSocketMessage<T = unknown> {
  channel: WebSocketChannel;
  data: T;
  timestamp: string;
}

// ============================================
// COMMON UTILITY TYPES
// ============================================

/**
 * Makes all properties of T optional recursively
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Extract the resolved type from a Promise
 */
export type Awaited<T> = T extends Promise<infer U> ? U : T;

/**
 * Make specific keys required
 */
export type RequiredKeys<T, K extends keyof T> = T & Required<Pick<T, K>>;

/**
 * Make specific keys optional
 */
export type OptionalKeys<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

/**
 * Loading state wrapper
 */
export interface LoadingState<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
}
