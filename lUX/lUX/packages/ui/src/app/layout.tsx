import "./globals.css";
import type { Metadata, Viewport } from "next";
import { IBM_Plex_Sans, JetBrains_Mono } from "next/font/google";
import { env } from "@/config/env";
import { WebVitalsReporter } from "@/lib/WebVitalsReporter";

const sans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
  variable: "--font-sans",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
  variable: "--font-mono",
});

const siteTitle = "Luxury Physics Viewer";
const siteDesc = "Deterministic proof package viewer for Trustless Physics Certificates";

export const metadata: Metadata = {
  title: {
    default: siteTitle,
    template: `%s · ${siteTitle}`,
  },
  description: siteDesc,
  manifest: "/manifest.json",
  icons: { icon: "/icon.svg" },
  metadataBase: new URL(env.baseUrl),
  openGraph: {
    type: "website",
    locale: "en_US",
    url: env.baseUrl,
    siteName: siteTitle,
    title: siteTitle,
    description: siteDesc,
  },
  twitter: {
    card: "summary",
    title: siteTitle,
    description: siteDesc,
  },
  robots: {
    index: true,
    follow: true,
  },
};

export const viewport: Viewport = {
  themeColor: "#C9A96E",
  colorScheme: "dark",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body className={sans.className}>
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[100] focus:rounded-md focus:bg-[var(--color-accent-gold)] focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-[var(--color-bg-base)]"
        >
          Skip to content
        </a>
        <WebVitalsReporter />
        {children}
      </body>
    </html>
  );
}
