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
  eslint: {
    // Lint is handled externally via eslint.config.js (ESLint 9).
    // next build's built-in lint uses ESLint 8 API which is incompatible.
    ignoreDuringBuilds: true,
  },
  // Security headers are set via middleware.ts (CSP nonces require per-request generation).
  experimental: {
    optimizePackageImports: ["lucide-react", "@radix-ui/react-tooltip"],
  },
};

export default analyze(nextConfig);
