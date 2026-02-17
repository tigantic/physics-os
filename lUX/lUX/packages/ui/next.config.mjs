import withBundleAnalyzer from "@next/bundle-analyzer";

const analyze = withBundleAnalyzer({
  // eslint-disable-next-line no-undef
  enabled: process.env.ANALYZE === "true",
});

/** @type {import("next").NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ["@luxury/core"],
  output: "standalone",
  // Disable built-in compression — the standalone server's compress middleware
  // emits `Vary: Accept-Encoding` without actually compressing cached/static
  // responses, producing a misleading header. Compression is handled by the
  // reverse proxy (nginx ingress / CDN) in production.
  compress: false,
  // Remove X-Powered-By header (information disclosure).
  poweredByHeader: false,
  eslint: {
    // Lint is handled externally via eslint.config.js (ESLint 9).
    // next build's built-in lint uses ESLint 8 API which is incompatible.
    ignoreDuringBuilds: true,
  },
  // Security headers are set via middleware.ts (CSP nonces require per-request generation).
  experimental: {
    optimizePackageImports: ["lucide-react", "@radix-ui/react-tooltip"],
  },

  // ── Cache headers for CDN / reverse-proxy ────────────────────────────────
  async headers() {
    return [
      {
        // Hashed static assets (_next/static) — immutable, long-lived cache.
        // Next.js sets these by default for its own server, but explicit
        // headers ensure CDN/reverse-proxy layers inherit the directive.
        source: "/_next/static/:path*",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
      {
        // Public static files (icons, manifest, PWA assets)
        source: "/:file(icon\\.svg|manifest\\.json|apple-touch-icon\\.png|favicon\\.ico)",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=86400, stale-while-revalidate=604800",
          },
        ],
      },
      {
        // Font files self-hosted by next/font
        source: "/_next/static/media/:path*",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
      {
        // API routes — never cache
        source: "/api/:path*",
        headers: [
          {
            key: "Cache-Control",
            value: "no-store, no-cache, must-revalidate",
          },
        ],
      },
    ];
  },
};

export default analyze(nextConfig);
