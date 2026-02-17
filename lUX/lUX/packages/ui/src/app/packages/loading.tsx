import { Skeleton } from "@/ds/components/Skeleton";

export default function PackagesLoading() {
  return (
    <div
      className="min-h-screen bg-[var(--color-bg-base)]"
      role="status"
      aria-busy="true"
      aria-label="Loading packages"
    >
      <span className="sr-only">Loading packages…</span>
      <div className="mx-auto max-w-[1200px] px-4 py-8 md:px-6 2xl:max-w-[1400px]">
        {/* Header skeleton */}
        <div className="mb-8">
          <div className="lux-shimmer-bg h-5 w-40 animate-lux-shimmer rounded" />
          <div className="lux-shimmer-bg mt-2 h-3 w-24 animate-lux-shimmer rounded" />
        </div>

        {/* Search bar skeleton */}
        <div className="lux-shimmer-bg mb-4 h-10 w-full animate-lux-shimmer rounded-[var(--radius-control)]" />

        {/* Table skeleton */}
        <Skeleton rows={8} />
      </div>
    </div>
  );
}
