import { NextResponse } from "next/server";
import { z } from "zod";
import { getProvider } from "@/config/provider";

const ParamsSchema = z.object({
  id: z
    .string()
    .min(1)
    .regex(/^[a-zA-Z0-9._-]+$/, "Invalid package ID"),
  path: z.array(z.string().min(1)).min(1, "Artifact path required"),
});

/**
 * GET /api/packages/[id]/artifacts/[...path] — Stream artifact bytes.
 *
 * Serves raw artifact files (CSV, log, etc.) for a given proof package.
 * Returns the raw bytes with appropriate Content-Type and hash headers.
 *
 * Response:
 *   200: Raw bytes with Content-Type and X-Artifact-Hash headers
 *   400: { error: string } — invalid params
 *   404: { error: string } — artifact not found
 *   500: { error: string } — server error
 */
export async function GET(_request: Request, { params }: { params: { id: string; path: string[] } }) {
  const parsed = ParamsSchema.safeParse(params);
  if (!parsed.success) {
    return NextResponse.json(
      { error: `Invalid parameters: ${parsed.error.issues.map((i) => i.message).join(", ")}` },
      { status: 400 },
    );
  }

  const artifactUri = parsed.data.path.join("/");

  try {
    const provider = await getProvider();
    const result = await provider.readArtifact(parsed.data.id, artifactUri);

    if (!result.ok) {
      return NextResponse.json({ error: result.reason }, { status: 404 });
    }

    return new NextResponse(Buffer.from(result.bytes), {
      status: 200,
      headers: {
        "Content-Type": result.mimeType,
        "Content-Length": String(result.bytes.byteLength),
        "X-Artifact-Hash": result.hash,
        "Cache-Control": "public, s-maxage=86400, stale-while-revalidate=604800, immutable",
      },
    });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Failed to read artifact" },
      { status: 500 },
    );
  }
}

export const runtime = "nodejs";
