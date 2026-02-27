/**
 * Meshes Loading State
 * 
 * Shows skeleton UI while meshes list is loading.
 */

import { Skeleton } from '@/components/ui/skeleton';

export default function MeshesLoading() {
  return (
    <div className="flex-1 p-6 lg:p-8 space-y-6">
      {/* Page Header Skeleton */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-8 w-32" />
          <Skeleton className="h-4 w-56" />
        </div>
        <Skeleton className="h-10 w-32" />
      </div>

      {/* Filters Row Skeleton */}
      <div className="flex items-center gap-4">
        <Skeleton className="h-10 w-64" />
        <div className="flex-1" />
        <Skeleton className="h-10 w-10" />
        <Skeleton className="h-10 w-10" />
      </div>

      {/* Grid Skeleton */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="rounded-xl border bg-card p-4 space-y-4">
            {/* Preview Area */}
            <Skeleton className="h-32 w-full rounded-lg" />
            
            {/* Mesh Info */}
            <div className="space-y-2">
              <Skeleton className="h-5 w-3/4" />
              <div className="flex gap-2">
                <Skeleton className="h-5 w-16 rounded-full" />
                <Skeleton className="h-5 w-20 rounded-full" />
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-2">
              <div className="text-center space-y-1">
                <Skeleton className="h-3 w-12 mx-auto" />
                <Skeleton className="h-5 w-16 mx-auto" />
              </div>
              <div className="text-center space-y-1">
                <Skeleton className="h-3 w-12 mx-auto" />
                <Skeleton className="h-5 w-16 mx-auto" />
              </div>
              <div className="text-center space-y-1">
                <Skeleton className="h-3 w-12 mx-auto" />
                <Skeleton className="h-5 w-16 mx-auto" />
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-2">
              <Skeleton className="h-9 flex-1" />
              <Skeleton className="h-9 w-9" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
