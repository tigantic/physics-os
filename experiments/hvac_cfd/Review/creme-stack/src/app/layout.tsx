import type { Metadata, Viewport } from 'next';
import { GeistSans } from 'geist/font/sans';
import { GeistMono } from 'geist/font/mono';
import { Providers } from '@/lib/providers';
import './globals.css';

// ============================================
// METADATA
// ============================================

export const metadata: Metadata = {
  title: {
    default: 'Your App Name',
    template: '%s | Your App Name',
  },
  description: 'Your app description goes here',
  keywords: ['your', 'keywords', 'here'],
  authors: [{ name: 'Your Name' }],
  creator: 'Your Name',
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_APP_URL ?? 'http://localhost:3000'
  ),
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: '/',
    siteName: 'Your App Name',
    title: 'Your App Name',
    description: 'Your app description goes here',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'Your App Name',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Your App Name',
    description: 'Your app description goes here',
    images: ['/og-image.png'],
    creator: '@yourtwitterhandle',
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
      className={`${GeistSans.variable} ${GeistMono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-background font-sans antialiased">
        {/* Skip to main content for accessibility */}
        <a href="#main-content" className="skip-to-main">
          Skip to main content
        </a>
        
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}
