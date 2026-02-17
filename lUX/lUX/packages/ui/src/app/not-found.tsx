import Link from "next/link";

export default function NotFound() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[var(--color-bg-base)] text-[var(--color-text-primary)]">
      <div className="space-y-4 text-center">
        <h1 className="text-4xl font-bold tracking-tight">404</h1>
        <p className="text-[var(--color-text-secondary)]">Page not found</p>
        <Link
          href="/gallery"
          aria-label="Return to proof gallery"
          className="inline-block rounded-[var(--radius-inner)] bg-[var(--color-accent-gold)] px-4 py-2 text-sm font-medium text-[var(--color-bg-base)] transition-colors duration-[var(--motion-fastMs)] ease-[var(--motion-easeOut)] hover:opacity-90"
        >
          Go to Gallery
        </Link>
      </div>
    </main>
  );
}
