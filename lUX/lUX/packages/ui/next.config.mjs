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
};

export default nextConfig;
