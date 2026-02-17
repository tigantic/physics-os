import { redirect } from "next/navigation";

/**
 * Legacy /gallery route — redirects to canonical /packages routes.
 *
 * Preserves query params: ?fixture=X&mode=Y&baseline=Z → /packages/X?mode=Y&baseline=Z
 * If no fixture specified, redirects to the packages index.
 */
export default function GalleryPage({
  searchParams,
}: {
  searchParams: Record<string, string | string[] | undefined>;
}) {
  const fixture = searchParams.fixture ? String(searchParams.fixture) : undefined;

  if (!fixture) {
    redirect("/packages");
  }

  const params = new URLSearchParams();
  const mode = searchParams.mode ? String(searchParams.mode) : undefined;
  const baseline = searchParams.baseline ? String(searchParams.baseline) : undefined;

  if (mode) params.set("mode", mode);
  if (baseline) params.set("baseline", baseline);

  const qs = params.toString();
  redirect(`/packages/${encodeURIComponent(fixture)}${qs ? `?${qs}` : ""}`);
}
