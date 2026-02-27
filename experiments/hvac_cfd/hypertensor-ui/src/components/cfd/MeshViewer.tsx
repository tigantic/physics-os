/**
 * MeshViewer - Three.js HyperGrid Visualization
 * 
 * Renders HyperGrid mesh geometry with colormap support for scalar fields.
 * Features:
 * - Volume fraction visualization
 * - SDF field display
 * - Temperature/velocity magnitude coloring
 * - Boundary patch highlighting
 * - OrbitControls for camera manipulation
 */

'use client';

import { Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport, Grid, PerspectiveCamera } from '@react-three/drei';
import { useViewerStore } from '@/stores';
import { Skeleton } from '@/components/ui/skeleton';
import type { Mesh as MeshData, ViewerBoundaryPatch } from '@/types';

// ============================================
// PROPS INTERFACE
// ============================================

interface MeshViewerProps {
  mesh?: MeshData | null;
  isLoading?: boolean;
  className?: string;
}

// ============================================
// MAIN COMPONENT
// ============================================

export function MeshViewer({ mesh, isLoading = false, className = '' }: MeshViewerProps) {
  const { showAxes, showGrid, showBoundaries } = useViewerStore();

  if (isLoading) {
    return <MeshViewerSkeleton className={className} />;
  }

  if (!mesh) {
    return <MeshViewerEmpty className={className} />;
  }

  // Calculate domain bounds for camera positioning
  const domainSize: [number, number, number] = mesh.domain_size ?? [
    mesh.dimensions.x[1] - mesh.dimensions.x[0],
    mesh.dimensions.y[1] - mesh.dimensions.y[0],
    mesh.dimensions.z[1] - mesh.dimensions.z[0],
  ];
  const [lx, ly, lz] = domainSize;
  const maxDim = Math.max(lx, ly, lz);
  const cameraDistance = maxDim * 1.5;
  
  const resolution: [number, number, number] = [
    mesh.resolution.nx,
    mesh.resolution.ny,
    mesh.resolution.nz,
  ];
  
  const cellCount = mesh.cell_count ?? mesh.cellCount;

  return (
    <div className={`relative h-full w-full rounded-lg border bg-slate-950 overflow-hidden ${className}`}>
      <Canvas>
        <Suspense fallback={null}>
          <PerspectiveCamera
            makeDefault
            position={[cameraDistance, cameraDistance * 0.8, cameraDistance]}
            fov={50}
          />

          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 10, 5]} intensity={0.8} />
          <directionalLight position={[-5, 5, -5]} intensity={0.3} />

          {/* Domain bounding box */}
          <DomainBox size={domainSize} />

          {/* Boundary patches */}
          {showBoundaries && mesh.patches.map((patch) => (
            <BoundaryPatchMesh
              key={patch.name}
              patch={patch}
              domainSize={domainSize}
              resolution={resolution}
            />
          ))}

          {/* Grid */}
          {showGrid && (
            <Grid
              args={[maxDim * 2, maxDim * 2]}
              cellSize={maxDim / 10}
              cellThickness={0.5}
              cellColor="#334155"
              sectionSize={maxDim / 2}
              sectionThickness={1}
              sectionColor="#475569"
              fadeDistance={maxDim * 3}
              position={[lx / 2, 0, lz / 2]}
            />
          )}

          {/* Camera controls */}
          <OrbitControls
            makeDefault
            enableDamping
            dampingFactor={0.05}
            target={[lx / 2, ly / 2, lz / 2]}
          />

          {/* Axes gizmo */}
          {showAxes && (
            <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
              <GizmoViewport labelColor="white" axisHeadScale={1} />
            </GizmoHelper>
          )}
        </Suspense>
      </Canvas>

      {/* Overlay info */}
      <div className="absolute bottom-2 left-2 text-xs text-slate-400 font-mono">
        {resolution.join(' × ')} cells | {(cellCount / 1e6).toFixed(2)}M
      </div>
    </div>
  );
}

// ============================================
// DOMAIN BOUNDING BOX
// ============================================

interface DomainBoxProps {
  size: [number, number, number];
}

function DomainBox({ size }: DomainBoxProps) {
  const [lx, ly, lz] = size;

  return (
    <mesh position={[lx / 2, ly / 2, lz / 2]}>
      <boxGeometry args={size} />
      <meshBasicMaterial color="#3b82f6" wireframe opacity={0.3} transparent />
    </mesh>
  );
}

// ============================================
// BOUNDARY PATCH MESH
// ============================================

interface BoundaryPatchMeshProps {
  patch: ViewerBoundaryPatch;
  domainSize: [number, number, number];
  resolution: [number, number, number];
}

function BoundaryPatchMesh({ patch, domainSize, resolution }: BoundaryPatchMeshProps) {
  const [lx, ly, lz] = domainSize;
  const [nx, ny, nz] = resolution;
  const dx = lx / nx;
  const dy = ly / ny;
  const dz = lz / nz;

  // Calculate patch position and size based on face and ranges
  const { position, size, color } = useMemo(() => {
    const patchColors: Record<string, string> = {
      inlet: '#22c55e',
      outlet: '#ef4444',
      wall: '#6b7280',
      symmetry: '#8b5cf6',
    };

    let pos: [number, number, number] = [0, 0, 0];
    let sz: [number, number, number] = [0.01, 0.01, 0.01];

    const [i0, i1] = patch.i_range ?? [0, nx];
    const [j0, j1] = patch.j_range ?? [0, ny];
    const [k0, k1] = patch.k_range ?? [0, nz];
    
    const face = patch.face ?? 'x-';

    switch (face) {
      case 'x-':
        pos = [0, (j0 + j1) / 2 * dy, (k0 + k1) / 2 * dz];
        sz = [0.02, (j1 - j0) * dy, (k1 - k0) * dz];
        break;
      case 'x+':
        pos = [lx, (j0 + j1) / 2 * dy, (k0 + k1) / 2 * dz];
        sz = [0.02, (j1 - j0) * dy, (k1 - k0) * dz];
        break;
      case 'y-':
        pos = [(i0 + i1) / 2 * dx, 0, (k0 + k1) / 2 * dz];
        sz = [(i1 - i0) * dx, 0.02, (k1 - k0) * dz];
        break;
      case 'y+':
        pos = [(i0 + i1) / 2 * dx, ly, (k0 + k1) / 2 * dz];
        sz = [(i1 - i0) * dx, 0.02, (k1 - k0) * dz];
        break;
      case 'z-':
        pos = [(i0 + i1) / 2 * dx, (j0 + j1) / 2 * dy, 0];
        sz = [(i1 - i0) * dx, (j1 - j0) * dy, 0.02];
        break;
      case 'z+':
        pos = [(i0 + i1) / 2 * dx, (j0 + j1) / 2 * dy, lz];
        sz = [(i1 - i0) * dx, (j1 - j0) * dy, 0.02];
        break;
    }
    
    const patchType = patch.patch_type ?? patch.type ?? 'wall';

    return { position: pos, size: sz, color: patchColors[patchType] || '#3b82f6' };
  }, [patch, lx, ly, lz, dx, dy, dz, nx, ny, nz]);

  return (
    <mesh position={position}>
      <boxGeometry args={size} />
      <meshStandardMaterial color={color} opacity={0.7} transparent />
    </mesh>
  );
}

// ============================================
// LOADING / EMPTY STATES
// ============================================

function MeshViewerSkeleton({ className = '' }: { className?: string }) {
  return (
    <div className={`relative h-full w-full rounded-lg border bg-slate-950 ${className}`}>
      <Skeleton className="absolute inset-4 bg-slate-800" />
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-slate-500 animate-pulse">Loading mesh...</div>
      </div>
    </div>
  );
}

function MeshViewerEmpty({ className = '' }: { className?: string }) {
  return (
    <div className={`relative h-full w-full rounded-lg border bg-slate-950 flex items-center justify-center ${className}`}>
      <div className="text-center text-slate-500">
        <div className="text-4xl mb-2">🧊</div>
        <div>No mesh loaded</div>
        <div className="text-xs mt-1">Create or select a HyperGrid mesh</div>
      </div>
    </div>
  );
}
