/** @type {import('next').NextConfig} */

// Bundle analyzer (run with ANALYZE=true npm run build)
const withBundleAnalyzer = process.env.ANALYZE === 'true'
  ? require('@next/bundle-analyzer')({ enabled: true })
  : (config) => config;

const nextConfig = {
  // Strict mode for catching bugs early
  reactStrictMode: true,

  // Experimental features
  experimental: {
    // NOTE: typedRoutes disabled due to incompatibility with dynamic routes
    // like /simulations/[id] and /results/[id]. Re-enable when Next.js
    // improves typed routes support for template literal types.
    // typedRoutes: true,
    
    // Optimize package imports
    optimizePackageImports: [
      'lucide-react',
      '@radix-ui/react-icons',
      '@radix-ui/react-dialog',
      '@radix-ui/react-dropdown-menu',
      '@radix-ui/react-tabs',
      '@radix-ui/react-tooltip',
      '@tanstack/react-query',
      'recharts',
    ],
  },

  // Image optimization domains (add your CDN/storage domains here)
  images: {
    remotePatterns: [
      // Example: Add your image domains
      // {
      //   protocol: 'https',
      //   hostname: 'your-cdn.com',
      // },
    ],
    // Enable image optimization
    formats: ['image/avif', 'image/webp'],
  },

  // Powered by header removal
  poweredByHeader: false,

  // Compiler options for production
  compiler: {
    // Remove console.log in production
    removeConsole: process.env.NODE_ENV === 'production' ? { exclude: ['error', 'warn'] } : false,
  },

  // Headers for security
  async headers() {
    // Build CSP based on environment
    const isDev = process.env.NODE_ENV === 'development';
    
    // Content Security Policy
    // In development, we need to allow eval for hot reload
    const cspDirectives = [
      "default-src 'self'",
      // Scripts: self + inline for Next.js hydration + unsafe-eval only in dev
      `script-src 'self' 'unsafe-inline'${isDev ? " 'unsafe-eval'" : ''}`,
      // Styles: self + inline for Tailwind
      "style-src 'self' 'unsafe-inline'",
      // Images: self + data URIs + blob for Three.js textures
      "img-src 'self' data: blob:",
      // Fonts: self + Google Fonts
      "font-src 'self' https://fonts.gstatic.com",
      // Connect: API endpoints + WebSocket
      `connect-src 'self' ${isDev ? 'ws://localhost:* http://localhost:*' : ''} wss://* https://*`,
      // Frame: only same origin
      "frame-src 'self'",
      // Media: self
      "media-src 'self'",
      // Object: none
      "object-src 'none'",
      // Base URI: self
      "base-uri 'self'",
      // Form action: self
      "form-action 'self'",
      // Frame ancestors: same origin
      "frame-ancestors 'self'",
      // Upgrade insecure requests in production
      ...(isDev ? [] : ['upgrade-insecure-requests']),
    ].join('; ');

    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=(), browsing-topics=()',
          },
          {
            key: 'Content-Security-Policy',
            value: cspDirectives,
          },
        ],
      },
      // Cache static assets
      {
        source: '/static/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      // Cache fonts
      {
        source: '/fonts/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
    ];
  },
};

module.exports = withBundleAnalyzer(nextConfig);
