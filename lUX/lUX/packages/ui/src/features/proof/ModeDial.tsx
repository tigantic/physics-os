"use client";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const MODES = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"] as const;

export function ModeDial() {
  const sp = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const mode = (sp.get("mode") ?? "REVIEW") as string;
  const fixture = sp.get("fixture") ?? "pass";
  const baseline = sp.get("baseline") ?? "pass";

  function setMode(m: string) {
    const next = new URLSearchParams(sp.toString());
    next.set("mode", m);
    next.set("fixture", fixture);
    next.set("baseline", baseline);
    router.push(`${pathname}?${next.toString()}`);
  }

  return (
    <div role="tablist" aria-label="Proof viewing mode" className="flex items-center gap-2 overflow-x-auto">
      {MODES.map((m) => (
        <Button
          key={m}
          role="tab"
          aria-selected={m === mode}
          tabIndex={m === mode ? 0 : -1}
          variant={m === mode ? "gold" : "default"}
          size="sm"
          onClick={() => setMode(m)}
          className={cn("h-8")}
        >
          {m}
        </Button>
      ))}
    </div>
  );
}
