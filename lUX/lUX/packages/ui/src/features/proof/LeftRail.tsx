import Link from "next/link";
import type { ProofPackage, ProofMode } from "@luxury/core";
import { Card, CardContent, CardHeader } from "@/ds/components/Card";
import { Chip } from "@/ds/components/Chip";

export function LeftRail({ proof, fixture, mode }: { proof: ProofPackage; fixture: string; mode: ProofMode }) {
  if (mode === "EXECUTIVE") {
    return (
      <div className="w-[260px] shrink-0 px-6 py-6">
        <Card>
          <CardHeader>
            <div className="text-sm text-[var(--color-text-primary)]">Fixtures</div>
            <div className="text-xs text-[var(--color-text-tertiary)]">Select proof package</div>
          </CardHeader>
          <CardContent className="space-y-2">
            {["pass","fail","warn","incomplete","tampered"].map(id => (
              <Link key={id} className="block" href={`/gallery?fixture=${id}&mode=${mode}`}>
                <div className="flex items-center justify-between rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2">
                  <span className="text-sm text-[var(--color-text-secondary)]">{id}</span>
                  {id === fixture ? <Chip tone="gold">Active</Chip> : null}
                </div>
              </Link>
            ))}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="w-[320px] shrink-0 px-6 py-6">
      <Card>
        <CardHeader>
          <div className="text-sm text-[var(--color-text-primary)]">Claims</div>
          <div className="text-xs text-[var(--color-text-tertiary)]">{Object.keys(proof.claims).length} total</div>
        </CardHeader>
        <CardContent className="space-y-2">
          {Object.values(proof.claims).map(c => (
            <div key={c.id} className="rounded-[var(--radius-inner)] border bg-[var(--color-bg-surface)] px-3 py-2">
              <div className="text-xs font-mono text-[var(--color-text-tertiary)]">{c.id}</div>
              <div className="text-sm text-[var(--color-text-secondary)]">{c.statement}</div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
