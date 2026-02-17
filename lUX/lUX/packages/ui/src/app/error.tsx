"use client";

/**
 * Root-level error boundary — catches any error thrown outside /gallery.
 */
export default function RootError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] text-[var(--color-text-primary)] flex items-center justify-center p-10">
      <div className="mx-auto max-w-[600px] rounded-[var(--radius-outer)] border bg-[var(--color-bg-raised)] p-6">
        <div className="text-xs uppercase tracking-wider text-[var(--color-text-tertiary)]">
          Application Error
        </div>
        <div className="mt-2 text-lg font-semibold">Something went wrong</div>
        <pre className="mt-4 font-mono text-xs text-[var(--color-verdict-fail)] whitespace-pre-wrap break-words">
          {error.message}
        </pre>
        {error.digest && (
          <div className="mt-2 text-[11px] text-[var(--color-text-tertiary)]">
            Digest: {error.digest}
          </div>
        )}
        <button
          type="button"
          onClick={reset}
          className="mt-5 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-base)] px-5 py-2 text-sm font-medium text-[var(--color-text-primary)] hover:bg-[var(--color-bg-raised)] transition-colors"
        >
          Retry
        </button>
      </div>
    </div>
  );
}
