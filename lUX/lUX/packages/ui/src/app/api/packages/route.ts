import { NextResponse } from "next/server";
import { getProvider } from "@/config/provider";

/**
 * GET /api/packages — List available proof packages.
 *
 * Returns a JSON array of PackageSummary objects with lightweight metadata
 * for each available proof package. Does not load full proof data.
 *
 * Response:
 *   200: { packages: PackageSummary[] }
 *   500: { error: string }
 */
export async function GET() {
  try {
    const provider = await getProvider();
    const packages = await provider.listPackages();
    return NextResponse.json(
      { packages },
      {
        status: 200,
        headers: {
          "Cache-Control": "public, s-maxage=60, stale-while-revalidate=120",
        },
      },
    );
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Failed to list packages" },
      { status: 500 },
    );
  }
}

export const runtime = "nodejs";
