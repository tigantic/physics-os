import type { MetadataRoute } from "next";
import { env } from "@/config/env";

/**
 * Dynamic sitemap generated at build / request time.
 * Lists all fixture × mode combinations as gallery URLs.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const base = env.baseUrl;
  const fixtures = ["pass", "fail", "warn", "incomplete", "tampered"];
  const modes = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"];

  const urls: MetadataRoute.Sitemap = [
    {
      url: base,
      lastModified: new Date(),
      changeFrequency: "monthly",
      priority: 1,
    },
  ];

  for (const fixture of fixtures) {
    for (const mode of modes) {
      urls.push({
        url: `${base}/gallery?fixture=${fixture}&mode=${mode}`,
        lastModified: new Date(),
        changeFrequency: "weekly",
        priority: 0.8,
      });
    }
  }

  return urls;
}
