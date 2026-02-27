/**
 * HyperTensor Field Types
 * 
 * Type definitions for CFD field data - velocity, pressure, temperature,
 * and turbulence quantities flowing from HyperFOAM solver.
 */

// ============================================
// FIELD TYPES
// ============================================

export type ScalarField = 
  | 'pressure'
  | 'temperature'
  | 'turbulent_ke'      // Turbulent kinetic energy (k)
  | 'turbulent_omega'   // Specific dissipation rate (ω)
  | 'turbulent_epsilon' // Dissipation rate (ε)
  | 'velocity_mag';     // Velocity magnitude (derived)

export type VectorField = 'velocity';

export type FieldName = ScalarField | VectorField;

// ============================================
// FIELD DATA STRUCTURES
// ============================================

export interface FieldMetadata {
  field: FieldName;
  shape: [number, number, number] | [number, number, number, number]; // Scalar or vector
  dtype: 'float32' | 'float64';
  timestep: number;
  min: number;
  max: number;
}

export interface FieldStats {
  field: FieldName;
  min: number;
  max: number;
  mean: number;
  std: number;
  timestep: number;
}

// ============================================
// SLICE / VISUALIZATION OPTIONS
// ============================================

export type SlicePlane = 'xy' | 'xz' | 'yz';

export interface SliceSpec {
  plane: SlicePlane;
  position: number;     // Position along normal axis (meters)
}

export type ColormapName = 
  | 'viridis'
  | 'plasma'
  | 'inferno'
  | 'magma'
  | 'coolwarm'
  | 'jet'
  | 'turbo'
  | 'rainbow';

export interface ColorRange {
  mode: 'auto' | 'manual';
  min?: number;
  max?: number;
}

export interface VisualizationOptions {
  colormap: ColormapName;
  colorRange: ColorRange;
  opacity: number;
  showMesh: boolean;
  showBoundaries: boolean;
  showStreamlines: boolean;
  streamlineCount: number;
}

export const DEFAULT_VISUALIZATION_OPTIONS: VisualizationOptions = {
  colormap: 'viridis',
  colorRange: { mode: 'auto' },
  opacity: 1.0,
  showMesh: false,
  showBoundaries: true,
  showStreamlines: false,
  streamlineCount: 100,
};

// ============================================
// EXPORT OPTIONS
// ============================================

export type ExportFormat = 'png' | 'mp4' | 'csv' | 'vtk' | 'npz';

export interface ExportOptions {
  format: ExportFormat;
  field: FieldName;
  timesteps?: 'current' | 'all' | number[];
  resolution?: [number, number];  // For image/video
  fps?: number;                   // For video
}
