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
          className="inline-block min-h-[44px] rounded-[var(--radius-inner)] bg-[var(--color-accent-gold)] px-6 py-3 text-sm font-medium leading-[1.5] text-[var(--color-bg-base)] transition-colors duration-[var(--motion-fastMs)] ease-[var(--motion-easeOut)] hover:opacity-90 sm:min-h-0 sm:px-4 sm:py-2"
        >
          Go to Gallery
        </Link>
      </div>
    </main>
  );
}
