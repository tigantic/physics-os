import type { Metadata, Viewport } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import { Providers } from '@/lib/providers';
import { Toaster } from '@/components/ui/toaster';
import { SkipLink } from '@/components/common';
import './globals.css';

// ============================================
// FONTS
// ============================================

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-sans',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-mono',
});

// ============================================
// METADATA
// ============================================

export const metadata: Metadata = {
  title: {
    default: 'HyperTensor CFD',
    template: '%s | HyperTensor',
  },
  description: 'GPU-Accelerated CFD with QTT Compression - HyperGrid Engine',
  keywords: ['CFD', 'HyperGrid', 'QTT', 'GPU', 'simulation', 'fluid dynamics', 'tensor'],
  authors: [{ name: 'TiganticLabz' }],
  creator: 'TiganticLabz',
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_APP_URL ?? 'http://localhost:3001'
  ),
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: '/',
    siteName: 'HyperTensor CFD',
    title: 'HyperTensor CFD Engine',
    description: 'GPU-Accelerated CFD with QTT Compression',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'HyperTensor CFD Engine',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'HyperTensor CFD',
    description: 'GPU-Accelerated CFD with QTT Compression',
    images: ['/og-image.png'],
    creator: '@tiganticlabz',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  icons: {
    icon: '/favicon.ico',
    shortcut: '/favicon-16x16.png',
    apple: '/apple-touch-icon.png',
  },
  manifest: '/site.webmanifest',
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: 'white' },
    { media: '(prefers-color-scheme: dark)', color: 'black' },
  ],
  width: 'device-width',
  initialScale: 1,
};

// ============================================
// ROOT LAYOUT
// ============================================

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-background font-sans antialiased">
        {/* Skip to main content for accessibility */}
        <SkipLink />
        
        <Providers>
          {children}
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}
