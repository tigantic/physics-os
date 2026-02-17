import "server-only";

import type { Metadata } from "next";
import { getProvider } from "@/config/provider";
import { logger } from "@/lib/logger";
import { startTimer } from "@/lib/timing";
import { env } from "@/config/env";
import { PackageList } from "./PackageList";

export const revalidate = env.revalidate;

export const metadata: Metadata = {
  title: "Packages · lUX Proof Viewer",
  description: "Browse available proof packages",
  openGraph: {
    title: "Packages · lUX Proof Viewer",
    description: "Browse and inspect deterministic proof packages",
  },
};

export default async function PackagesPage() {
  const timer = startTimer("list_packages");
  const provider = await getProvider();
  const packages = await provider.listPackages();
  const timing = timer.stop();

  logger.info("packages.list", {
    count: packages.length,
    durationMs: timing.durationMs,
  });

  return (
    <div className="min-h-screen bg-[var(--color-bg-base)]">
      <div className="mx-auto max-w-[1200px] px-4 py-8 md:px-6 2xl:max-w-[1400px]">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-lg font-semibold text-[var(--color-text-primary)]">Proof Packages</h1>
          <p className="mt-1 text-xs text-[var(--color-text-tertiary)]">
            {packages.length} package{packages.length !== 1 ? "s" : ""} available
          </p>
        </div>

        <main id="main-content">
          <PackageList packages={packages} />
        </main>
      </div>
    </div>
  );
}
