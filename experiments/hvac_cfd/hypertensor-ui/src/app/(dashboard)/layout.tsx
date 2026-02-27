/**
 * Dashboard Layout
 * 
 * Shared layout for all dashboard pages with network status monitoring.
 */

'use client';

import { NetworkStatusBanner } from '@/components/common/NetworkStatusBanner';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      {children}
      <NetworkStatusBanner />
    </>
  );
}
