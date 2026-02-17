"use client";

import * as React from "react";
import { reportError } from "@/lib/reportError";

export default function PackagesListError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const retryRef = React.useRef<HTMLButtonElement>(null);

  React.useEffect(() => {
    retryRef.current?.focus();
  }, []);

  React.useEffect(() => {
    reportError(error, "PackagesListError");
  }, [error]);

  return (
    <div
      className="min-h-screen bg-[var(--color-bg-base)] p-10 text-[var(--color-text-primary)]"
      role="alert"
      aria-live="assertive"
    >
      <div className="mx-auto max-w-[900px] rounded-[var(--radius-outer)] border bg-[var(--color-bg-raised)] p-6">
        <div className="text-xs uppercase tracking-wider text-[var(--color-text-tertiary)]">Render Halted</div>
        <h1 className="mt-2 text-lg font-semibold">Package List Error</h1>
        <p className="mt-2 text-sm text-[var(--color-text-secondary)]">
          Failed to load the package index. This may be a temporary data-provider issue.
        </p>
        <pre className="mt-4 whitespace-pre-wrap break-words font-mono text-xs text-[var(--color-status-fail)]">
          {error.message}
        </pre>
        {error.digest && (
          <div className="mt-2 text-[11px] text-[var(--color-text-tertiary)]">Digest: {error.digest}</div>
        )}
        <div className="mt-5 flex gap-3">
          <button
            ref={retryRef}
            type="button"
            onClick={reset}
            className="min-h-[44px] rounded-md border border-[var(--color-border)] bg-[var(--color-bg-base)] px-5 py-2 text-sm font-medium text-[var(--color-text-primary)] transition-colors hover:bg-[var(--color-bg-raised)] sm:min-h-0"
          >
            Retry
          </button>
        </div>
      </div>
    </div>
  );
}
