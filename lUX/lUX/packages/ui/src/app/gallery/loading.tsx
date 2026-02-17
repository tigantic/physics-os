/**
 * Gallery loading skeleton — shown while the proof package is resolved on the server.
 * Mirrors the ProofWorkspace layout structure so the shift is minimal on hydration.
 */
export default function GalleryLoading() {
  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] animate-pulse">
      {/* Identity strip skeleton */}
      <div className="h-12 border-b border-[var(--color-border)]">
        <div className="mx-auto flex max-w-[1400px] items-center gap-4 px-6 py-3">
          <div className="h-4 w-32 rounded bg-[var(--color-bg-raised)]" />
          <div className="h-4 w-24 rounded bg-[var(--color-bg-raised)]" />
        </div>
      </div>

      {/* Mode dial area */}
      <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6 pt-4">
        <div className="h-3 w-20 rounded bg-[var(--color-bg-raised)]" />
        <div className="flex gap-2">
          {Array.from({ length: 4 }, (_, i) => (
            <div key={i} className="h-7 w-20 rounded-full bg-[var(--color-bg-raised)]" />
          ))}
        </div>
      </div>

      {/* Three-column layout skeleton */}
      <div className="mx-auto flex max-w-[1400px] gap-0 px-0 pt-4">
        {/* Left rail */}
        <div className="w-[260px] shrink-0 border-r border-[var(--color-border)] p-4">
          <div className="space-y-3">
            {Array.from({ length: 6 }, (_, i) => (
              <div key={i} className="h-4 rounded bg-[var(--color-bg-raised)]" style={{ width: `${70 + (i % 3) * 10}%` }} />
            ))}
          </div>
        </div>

        {/* Center canvas */}
        <main className="flex-1 px-6 py-6">
          <div className="space-y-4">
            <div className="h-6 w-48 rounded bg-[var(--color-bg-raised)]" />
            <div className="h-40 rounded-lg bg-[var(--color-bg-raised)]" />
            <div className="h-32 rounded-lg bg-[var(--color-bg-raised)]" />
          </div>
        </main>

        {/* Right rail */}
        <div className="w-[280px] shrink-0 border-l border-[var(--color-border)] p-4">
          <div className="space-y-3">
            {Array.from({ length: 5 }, (_, i) => (
              <div key={i} className="h-4 rounded bg-[var(--color-bg-raised)]" style={{ width: `${60 + (i % 4) * 10}%` }} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
