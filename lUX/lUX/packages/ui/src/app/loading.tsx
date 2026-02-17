/**
 * Root loading skeleton — shown during top-level route transitions.
 */
export default function Loading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--color-bg-base)]">
      <div className="animate-pulse text-sm tracking-wide text-[var(--color-text-tertiary)]">Loading…</div>
    </div>
  );
}
