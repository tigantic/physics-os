import { Badge } from "@/components/ui/badge";

type BadgeVariant = "default" | "gold" | "pass" | "fail" | "warn";

const STATUS_VARIANT: Record<string, BadgeVariant> = {
  PASS: "pass",
  FAIL: "fail",
  WARN: "warn",
  INCOMPLETE: "default",
};

const VERIFICATION_VARIANT: Record<string, BadgeVariant> = {
  VERIFIED: "gold",
  BROKEN_CHAIN: "fail",
  UNVERIFIED: "default",
  UNSUPPORTED: "warn",
};

export function VerdictSeal({
  status,
  verification,
}: {
  status: "PASS" | "FAIL" | "WARN" | "INCOMPLETE";
  verification: string;
}) {
  const s: BadgeVariant = STATUS_VARIANT[status] ?? "default";
  const v: BadgeVariant = VERIFICATION_VARIANT[verification] ?? "default";

  return (
    <div
      role="status"
      aria-label={`Verdict: ${status}, Verification: ${verification}`}
      className="flex animate-lux-scale-in items-center gap-2"
    >
      <Badge variant={s}>{status}</Badge>
      <Badge variant={v}>{verification}</Badge>
    </div>
  );
}
