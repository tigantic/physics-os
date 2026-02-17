"use client";

export default function Error({ error }: { error: Error & { digest?: string } }) {
  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] text-[var(--color-text-primary)] p-10">
      <div className="mx-auto max-w-[900px] rounded-[var(--radius-outer)] border bg-[var(--color-bg-raised)] p-6">
        <div className="text-xs uppercase tracking-wider text-[var(--color-text-tertiary)]">Render Halted</div>
        <div className="mt-2 text-lg font-semibold">Viewer Error</div>
        <div className="mt-4 font-mono text-xs text-[var(--color-verdict-fail)] whitespace-pre-wrap">{error.message}</div>
      </div>
    </div>
  );
}
