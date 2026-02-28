/**
 * HyperTensor Mesh Types
 * 
 * Type definitions for computational mesh management.
 */

// ============================================
// BOUNDARY PATCH TYPES
// ============================================

export type PatchType = 
  | 'inlet'
  | 'outlet'
  | 'wall'
  | 'symmetry'
  | 'periodic'
  | 'empty';

export type FaceDirection = 'x-' | 'x+' | 'y-' | 'y+' | 'z-' | 'z+';

export interface BoundaryPatch {
  id?: string;
  name: string;
  type: PatchType;
  faceCount?: number;
  velocity?: [number, number, number];
  pressure?: number;
  temperature?: number;
}

/**
 * Boundary patch creation payload for the API.
 * Uses snake_case for API compatibility.
 */
export interface BoundaryPatchCreate {
  name: string;
  face: FaceDirection;
  patch_type: PatchType;
  velocity?: [number, number, number];
  temperature?: number;
  pressure?: number;
}

// ============================================
// MESH ENTITIES
// ============================================

export interface MeshSummary {
  id: string;
  name: string;
  cellCount: number;
  patchCount: number;
  createdAt: string;
}

/**
 * Viewer boundary patch with face range information
 */
export interface ViewerBoundaryPatch extends BoundaryPatch {
  face?: FaceDirection;
  patch_type?: PatchType;
  i_range?: [number, number];
  j_range?: [number, number];
  k_range?: [number, number];
}

export interface Mesh extends MeshSummary {
  description?: string;
  dimensions: {
    x: [number, number];
    y: [number, number];
    z: [number, number];
  };
  resolution: {
    nx: number;
    ny: number;
    nz: number;
  };
  patches: ViewerBoundaryPatch[];
  qualityMetrics?: {
    minOrthogonality: number;
    maxSkewness: number;
    maxAspectRatio: number;
  };
  // Viewer-friendly properties (computed from dimensions)
  domain_size?: [number, number, number];
  cell_count?: number;
}

export interface MeshCreate {
  name: string;
  description?: string;
  dimensions: {
    x: [number, number];
    y: [number, number];
    z: [number, number];
  };
  resolution: {
    nx: number;
    ny: number;
    nz: number;
  };
  patches?: BoundaryPatch[];
}

// ============================================
// GEOMETRY PRIMITIVES (CSG)
// ============================================

export type GeometryChannel = 'x' | 'y' | 'z';

export interface BoxPrimitive {
  type: 'box';
  min: [number, number, number];
  max: [number, number, number];
}

export interface CylinderPrimitive {
  type: 'cylinder';
  center: [number, number, number];
  radius: number;
  height: number;
  axis: GeometryChannel;
}

export interface SpherePrimitive {
  type: 'sphere';
  center: [number, number, number];
  radius: number;
}

export type GeometryPrimitive = BoxPrimitive | CylinderPrimitive | SpherePrimitive;

export interface CSGOperation {
  operation: 'union' | 'subtract' | 'intersect';
  primitives: GeometryPrimitive[];
}
