import type { MetadataRoute } from "next";
import { env } from "@/config/env";

/**
 * Dynamic sitemap generated at build / request time.
 * Lists all known package × mode combinations as canonical URLs.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const base = env.baseUrl;
  const packages = ["pass", "fail", "warn", "incomplete", "tampered"];
  const modes = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"];

  const urls: MetadataRoute.Sitemap = [
    {
      url: base,
      lastModified: new Date(),
      changeFrequency: "monthly",
      priority: 1,
    },
    {
      url: `${base}/packages`,
      lastModified: new Date(),
      changeFrequency: "weekly",
      priority: 0.9,
    },
  ];

  for (const pkg of packages) {
    for (const mode of modes) {
      urls.push({
        url: `${base}/packages/${pkg}?mode=${mode}`,
        lastModified: new Date(),
        changeFrequency: "weekly",
        priority: 0.8,
      });
    }
  }

  return urls;
}
