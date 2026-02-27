/**
 * Simulations Loading State
 * 
 * Shows skeleton UI while simulations list is loading.
 */

import { Skeleton } from '@/components/ui/skeleton';

export default function SimulationsLoading() {
  return (
    <div className="flex-1 p-6 lg:p-8 space-y-6">
      {/* Page Header Skeleton */}
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-8 w-40" />
          <Skeleton className="h-4 w-64" />
        </div>
        <Skeleton className="h-10 w-36" />
      </div>

      {/* Filters Row Skeleton */}
      <div className="flex items-center gap-4">
        <Skeleton className="h-10 w-64" /> {/* Search */}
        <Skeleton className="h-10 w-32" /> {/* Status filter */}
        <div className="flex-1" />
        <Skeleton className="h-10 w-10" /> {/* Grid toggle */}
        <Skeleton className="h-10 w-10" /> {/* List toggle */}
      </div>

      {/* Table Skeleton */}
      <div className="rounded-xl border bg-card">
        {/* Table Header */}
        <div className="border-b p-4 flex gap-4">
          <Skeleton className="h-4 w-1/4" />
          <Skeleton className="h-4 w-1/6" />
          <Skeleton className="h-4 w-1/6" />
          <Skeleton className="h-4 w-1/6" />
          <Skeleton className="h-4 w-1/6" />
          <Skeleton className="h-4 w-20" />
        </div>

        {/* Table Rows */}
        <div className="divide-y">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="p-4 flex items-center gap-4">
              <Skeleton className="h-5 w-1/4" />
              <Skeleton className="h-6 w-20 rounded-full" /> {/* Badge */}
              <div className="w-1/6 flex items-center gap-2">
                <Skeleton className="h-2 w-16" />
                <Skeleton className="h-4 w-8" />
              </div>
              <Skeleton className="h-4 w-1/6" />
              <Skeleton className="h-4 w-1/6" />
              <Skeleton className="h-8 w-8" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
