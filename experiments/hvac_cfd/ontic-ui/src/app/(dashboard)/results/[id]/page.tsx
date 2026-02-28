/**
 * Results Detail Page
 * 
 * Redirects to simulation detail page since results are per-simulation.
 * This is a convenience route for /results/[id] → /simulations/[id]
 */

'use client';

import { use, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Loader2 } from 'lucide-react';

export default function ResultsDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();

  useEffect(() => {
    // Redirect to simulation detail page
    router.replace(`/simulations/${id}`);
  }, [id, router]);

  return (
    <div className="flex h-screen items-center justify-center">
      <div className="text-center">
        <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Redirecting to simulation details...</p>
      </div>
    </div>
  );
}
