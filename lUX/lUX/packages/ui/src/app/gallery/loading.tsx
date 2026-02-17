/**
 * Gallery loading skeleton — shown while the proof package is resolved on the server.
 * Mirrors the ProofWorkspace responsive layout so the shift is minimal on hydration.
 *
 * Breakpoints:
 *   xs-sm : Single column — LeftRail hidden, RightRail hidden
 *   md    : Two-column — LeftRail visible, Center fills
 *   lg+   : Three-column — LeftRail + Center + RightRail
 *   xl+   : Wider rails, max-w-[1600px] expansion
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
        <div className="mx-auto flex max-w-[1400px] items-center gap-4 px-4 py-3 md:px-6 2xl:max-w-[1600px]">
          <div className={`h-4 w-32 ${sh}`} />
          <div className={`hidden h-4 w-24 sm:block ${sh}`} />
        </div>
      </div>

      {/* Mode dial area */}
      <div className="mx-auto flex max-w-[1400px] items-center justify-between px-4 pt-4 md:px-6 2xl:max-w-[1600px]">
        <div className={`h-3 w-20 ${sh}`} />
        <div className="flex gap-2 overflow-x-auto">
          {Array.from({ length: 4 }, (_, i) => (
            <div key={i} className={`lux-shimmer-bg h-10 w-20 animate-lux-shimmer rounded-full sm:h-7`} />
          ))}
        </div>
      </div>

      {/* Responsive column layout skeleton */}
      <div className="mx-auto flex max-w-[1400px] flex-col gap-0 lg:flex-row 2xl:max-w-[1600px]">
        {/* Left rail — hidden below md */}
        <div className="hidden w-[260px] shrink-0 border-r border-[var(--color-border-base)] p-4 md:block md:p-6 lg:w-[280px] xl:w-[300px]">
          <div className="space-y-3">
            {Array.from({ length: 6 }, (_, i) => (
              <div key={i} className={sh} style={{ width: `${70 + (i % 3) * 10}%`, height: 16 }} />
            ))}
          </div>
        </div>

        {/* Center canvas */}
        <main id="main-content" className="flex-1 px-4 py-4 md:px-6 md:py-6">
          <div className="space-y-4">
            <div className={`h-6 w-48 ${sh}`} />
            <div className={`lux-shimmer-bg h-40 animate-lux-shimmer rounded-lg`} />
            <div className={`lux-shimmer-bg h-32 animate-lux-shimmer rounded-lg`} />
          </div>
        </main>

        {/* Right rail — hidden below lg */}
        <div className="hidden w-[360px] shrink-0 border-l border-[var(--color-border-base)] p-4 lg:block lg:p-6 xl:w-[400px]">
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
