import Link from "next/link";
import type { ProofPackage, ProofMode } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { Chip } from "@/ds/components/Chip";
import { cn } from "@/config/utils";

export function LeftRail({ proof, fixture, mode }: { proof: ProofPackage; fixture: string; mode: ProofMode }) {
  if (mode === "EXECUTIVE") {
    return (
      <nav
        aria-label="Proof fixtures"
        className="w-full px-4 py-4 md:w-[260px] md:shrink-0 md:px-6 md:py-6 lg:w-[280px] xl:w-[300px]"
      >
        <Card>
          <CardHeader>
            <div className="text-sm text-[var(--color-text-primary)]">Fixtures</div>
            <div className="text-xs text-[var(--color-text-tertiary)]">Select proof package</div>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {["pass", "fail", "warn", "incomplete", "tampered"].map((id) => (
                <li key={id}>
                  <Link
                    className="block"
                    href={`/packages/${encodeURIComponent(id)}?mode=${mode}`}
                    aria-current={id === fixture ? "page" : undefined}
                  >
                    <div
                      className={cn(
                        "flex items-center justify-between rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2 transition-all duration-hover ease-lux-out",
                        id === fixture
                          ? "border-l-2 border-l-[var(--color-accent)] shadow-[0_0_12px_rgba(75,123,245,0.08)]"
                          : "hover:bg-[var(--color-bg-hover)]",
                      )}
                    >
                      <span className="text-sm text-[var(--color-text-secondary)]">{id}</span>
                      {id === fixture ? <Chip tone="gold">Active</Chip> : null}
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </nav>
    );
  }

  return (
    <aside
      aria-label="Claims list"
      className="w-full px-4 py-4 md:w-[280px] md:shrink-0 md:px-6 md:py-6 lg:w-[320px] xl:w-[360px]"
    >
      <Card>
        <CardHeader>
          <div className="text-sm text-[var(--color-text-primary)]">Claims</div>
          <div className="text-xs text-[var(--color-text-tertiary)]">{Object.keys(proof.claims).length} total</div>
        </CardHeader>
        <CardContent>
          <ul className="max-h-[60vh] space-y-2 overflow-y-auto">
            {Object.values(proof.claims).map((c) => (
              <li key={c.id} className="rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2">
                <div className="font-mono text-xs text-[var(--color-text-tertiary)]">{c.id}</div>
                <div className="text-sm text-[var(--color-text-secondary)]">{c.statement}</div>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </aside>
  );
}
