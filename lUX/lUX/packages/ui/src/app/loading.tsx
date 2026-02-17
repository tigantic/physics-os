/**
 * Root loading skeleton — shown during top-level route transitions.
 */
export default function Loading() {
  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] flex items-center justify-center">
      <div className="animate-pulse text-sm text-[var(--color-text-tertiary)] tracking-wide">
        Loading…
      </div>
    </div>
  );
}
