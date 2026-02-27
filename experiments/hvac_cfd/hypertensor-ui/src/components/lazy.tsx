/**
 * Lazy Components - Dynamic Imports for Code Splitting
 * 
 * Heavy components that should be loaded on demand to reduce initial bundle size.
 * Per Article IX of the Constitution.
 */

'use client';

import dynamic from 'next/dynamic';
import { Skeleton } from '@/components/ui/skeleton';

// ============================================
// LOADING FALLBACKS
// ============================================

function MeshViewerSkeleton() {
  return (
    <div className="w-full h-full min-h-[300px] bg-muted rounded-lg flex items-center justify-center">
      <Skeleton className="w-full h-full" />
    </div>
  );
}

function ChartSkeleton() {
  return (
    <div className="w-full h-[200px] bg-muted rounded-lg animate-pulse" />
  );
}

function EditorSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-10 w-full" />
      <Skeleton className="h-32 w-full" />
      <Skeleton className="h-10 w-32" />
    </div>
  );
}

// ============================================
// LAZY COMPONENTS
// ============================================

/**
 * Lazy-loaded 3D Mesh Viewer (Three.js)
 * Heavy component - loads react-three-fiber on demand
 */
export const LazyMeshViewer = dynamic(
  () => import('@/components/cfd/MeshViewer').then((mod) => ({ default: mod.MeshViewer })),
  {
    loading: () => <MeshViewerSkeleton />,
    ssr: false, // Three.js doesn't work with SSR
  }
);

/**
 * Lazy-loaded Residual Chart (Recharts)
 */
export const LazyResidualChart = dynamic(
  () => import('@/components/cfd/ResidualChart').then((mod) => ({ default: mod.ResidualChart })),
  {
    loading: () => <ChartSkeleton />,
    ssr: false,
  }
);

/**
 * Lazy-loaded Mini Residual Chart
 */
export const LazyResidualChartMini = dynamic(
  () => import('@/components/cfd/ResidualChart').then((mod) => ({ default: mod.ResidualChartMini })),
  {
    loading: () => <Skeleton className="h-16 w-24" />,
    ssr: false,
  }
);

/**
 * Lazy-loaded Boundary Editor
 */
export const LazyBoundaryEditor = dynamic(
  () => import('@/components/cfd/BoundaryEditor').then((mod) => ({ default: mod.BoundaryEditor })),
  {
    loading: () => <EditorSkeleton />,
  }
);

/**
 * Lazy-loaded Color Legend (for post-processing)
 */
export const LazyColorLegend = dynamic(
  () => import('@/components/cfd/ColorLegend').then((mod) => ({ default: mod.ColorLegend })),
  {
    loading: () => <Skeleton className="h-4 w-full" />,
  }
);
