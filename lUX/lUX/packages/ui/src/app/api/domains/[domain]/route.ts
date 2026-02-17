import { NextResponse } from "next/server";
import { z } from "zod";
import { getProvider } from "@/config/provider";

const ParamsSchema = z.object({
  domain: z
    .string()
    .min(1)
    .regex(/^[a-zA-Z0-9._-]+$/, "Invalid domain ID"),
});

/**
 * GET /api/domains/[domain] — Load a domain pack.
 *
 * Accepts either a TPC domain_id (e.g. "II.2") or a direct pack ID
 * (e.g. "com.physics.euler_3d"). The provider resolves via manifest
 * lookup with fallback to direct ID match.
 *
 * Response:
 *   200: DomainPack
 *   400: { error: string } — invalid domain ID
 *   404: { error: string } — domain pack not found
 *   500: { error: string } — server error
 */
export async function GET(_request: Request, { params }: { params: { domain: string } }) {
  const parsed = ParamsSchema.safeParse(params);
  if (!parsed.success) {
    return NextResponse.json(
      { error: `Invalid domain ID: ${parsed.error.issues.map((i) => i.message).join(", ")}` },
      { status: 400 },
    );
  }

  try {
    const provider = await getProvider();
    const domainPack = await provider.loadDomainPack(parsed.data.domain);
    return NextResponse.json(domainPack, {
      status: 200,
      headers: {
        "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Failed to load domain pack";
    const status = message.includes("not found") || message.includes("not found") ? 404 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}

export const runtime = "nodejs";
