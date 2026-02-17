"use client";

/**
 * Gallery-level error boundary — catches proof loading / rendering failures.
 * Provides structured diagnostics and retry capability.
 */
export default function GalleryError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  // Log for production error tracking
  console.error("[lUX] Gallery error boundary:", error);

  return (
    <div
      className="min-h-screen bg-[var(--color-bg-base)] p-10 text-[var(--color-text-primary)]"
      role="alert"
      aria-live="assertive"
    >
      <div className="mx-auto max-w-[900px] rounded-[var(--radius-outer)] border bg-[var(--color-bg-raised)] p-6">
        <div className="text-xs uppercase tracking-wider text-[var(--color-text-tertiary)]">Render Halted</div>
        <h1 className="mt-2 text-lg font-semibold">Viewer Error</h1>
        <pre className="mt-4 whitespace-pre-wrap break-words font-mono text-xs text-[var(--color-verdict-fail)]">
          {error.message}
        </pre>
        {error.digest && (
          <div className="mt-2 text-[11px] text-[var(--color-text-tertiary)]">Digest: {error.digest}</div>
        )}
        <button
          type="button"
          onClick={reset}
          className="mt-5 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-base)] px-5 py-2 text-sm font-medium text-[var(--color-text-primary)] transition-colors hover:bg-[var(--color-bg-raised)]"
        >
          Retry
        </button>
      </div>
    </div>
  );
}
