"use client";

/**
 * Root-level error boundary — catches any error thrown outside /gallery.
 */
export default function RootError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  // Log for production error tracking (Sentry, Datadog, etc. can hook console.error)
  console.error("[lUX] Root error boundary:", error);

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--color-bg-base)] p-10 text-[var(--color-text-primary)]">
      <div className="mx-auto max-w-[600px] rounded-[var(--radius-outer)] border bg-[var(--color-bg-raised)] p-6">
        <div className="text-xs uppercase tracking-wider text-[var(--color-text-tertiary)]">Application Error</div>
        <div className="mt-2 text-lg font-semibold">Something went wrong</div>
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
