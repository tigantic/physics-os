import { NextResponse } from "next/server";
import { z } from "zod";
import { getProvider } from "@/config/provider";

const ParamsSchema = z.object({
  id: z
    .string()
    .min(1)
    .regex(/^[a-zA-Z0-9._-]+$/, "Invalid package ID"),
});

/**
 * GET /api/packages/[id] — Load a full proof package by ID.
 *
 * Returns the complete ProofPackage JSON, including verification results.
 *
 * Response:
 *   200: ProofPackage
 *   400: { error: string } — invalid ID
 *   404: { error: string } — package not found
 *   500: { error: string } — server error
 */
export async function GET(_request: Request, { params }: { params: { id: string } }) {
  const parsed = ParamsSchema.safeParse(params);
  if (!parsed.success) {
    return NextResponse.json(
      { error: `Invalid package ID: ${parsed.error.issues.map((i) => i.message).join(", ")}` },
      { status: 400 },
    );
  }

  try {
    const provider = await getProvider();
    const proof = await provider.loadPackage(parsed.data.id);
    return NextResponse.json(proof, {
      status: 200,
      headers: {
        "Cache-Control": "public, s-maxage=300, stale-while-revalidate=600",
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Failed to load package";
    const status = message.includes("not found") ? 404 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}

export const runtime = "nodejs";
