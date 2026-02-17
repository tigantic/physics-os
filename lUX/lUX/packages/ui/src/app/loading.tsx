/**
 * Root loading skeleton — shown during top-level route transitions.
 */
export default function Loading() {
  return (
    <div
      role="status"
      aria-busy="true"
      aria-label="Loading application"
      className="flex min-h-screen items-center justify-center bg-[var(--color-bg-base)]"
    >
      <span className="sr-only">Loading…</span>
      <div aria-hidden="true" className="animate-lux-shimmer text-sm tracking-wide text-[var(--color-text-tertiary)]">
        Loading…
      </div>
    </div>
  );
}
