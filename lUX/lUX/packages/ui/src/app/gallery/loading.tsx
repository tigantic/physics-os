/**
 * Gallery loading skeleton — shown while the proof package is resolved on the server.
 * Mirrors the ProofWorkspace layout structure so the shift is minimal on hydration.
 */
export default function GalleryLoading() {
  const sh = "lux-shimmer-bg animate-lux-shimmer rounded";
  return (
    <div
      className="min-h-screen bg-[var(--color-bg-base)]"
      role="status"
      aria-busy="true"
      aria-label="Loading proof package"
    >
      <span className="sr-only">Loading proof package…</span>
      {/* Identity strip skeleton */}
      <div className="h-12 border-b border-[var(--color-border-base)]">
        <div className="mx-auto flex max-w-[1400px] items-center gap-4 px-6 py-3">
          <div className={`h-4 w-32 ${sh}`} />
          <div className={`h-4 w-24 ${sh}`} />
        </div>
      </div>

      {/* Mode dial area */}
      <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6 pt-4">
        <div className={`h-3 w-20 ${sh}`} />
        <div className="flex gap-2">
          {Array.from({ length: 4 }, (_, i) => (
            <div key={i} className={`lux-shimmer-bg h-7 w-20 animate-lux-shimmer rounded-full`} />
          ))}
        </div>
      </div>

      {/* Three-column layout skeleton */}
      <div className="mx-auto flex max-w-[1400px] gap-0 px-0 pt-4">
        {/* Left rail */}
        <div className="w-[260px] shrink-0 border-r border-[var(--color-border-base)] p-4">
          <div className="space-y-3">
            {Array.from({ length: 6 }, (_, i) => (
              <div key={i} className={sh} style={{ width: `${70 + (i % 3) * 10}%`, height: 16 }} />
            ))}
          </div>
        </div>

        {/* Center canvas */}
        <main id="main-content" className="flex-1 px-6 py-6">
          <div className="space-y-4">
            <div className={`h-6 w-48 ${sh}`} />
            <div className={`lux-shimmer-bg h-40 animate-lux-shimmer rounded-lg`} />
            <div className={`lux-shimmer-bg h-32 animate-lux-shimmer rounded-lg`} />
          </div>
        </main>

        {/* Right rail */}
        <div className="w-[280px] shrink-0 border-l border-[var(--color-border-base)] p-4">
          <div className="space-y-3">
            {Array.from({ length: 5 }, (_, i) => (
              <div key={i} className={sh} style={{ width: `${60 + (i % 4) * 10}%`, height: 16 }} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
