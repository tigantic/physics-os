/**
 * Dashboard Error Boundary
 * 
 * Catches runtime errors in dashboard routes and displays recovery UI.
 * Next.js App Router error boundary.
 */

'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { AlertTriangle, RefreshCw, Home, FileQuestion } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';

interface ErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function DashboardError({ error, reset }: ErrorProps) {
  useEffect(() => {
    // Log error to console (and optionally to error tracking service)
    console.error('Dashboard error:', error);
  }, [error]);

  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 h-14 w-14 rounded-full bg-destructive/10 flex items-center justify-center">
            <AlertTriangle className="h-7 w-7 text-destructive" />
          </div>
          <CardTitle className="text-xl">Something went wrong</CardTitle>
          <CardDescription>
            We encountered an unexpected error while loading this page.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="rounded-lg bg-muted p-4 text-center">
            <p className="text-sm text-muted-foreground">
              Error: {error.message || 'Unknown error'}
            </p>
            {error.digest && (
              <p className="text-xs text-muted-foreground mt-1">
                Reference: {error.digest}
              </p>
            )}
          </div>
        </CardContent>

        <CardFooter className="flex flex-col gap-3">
          <Button onClick={reset} className="w-full">
            <RefreshCw className="h-4 w-4 mr-2" />
            Try Again
          </Button>
          <div className="flex gap-2 w-full">
            <Button asChild variant="outline" className="flex-1">
              <Link href="/">
                <Home className="h-4 w-4 mr-2" />
                Dashboard
              </Link>
            </Button>
            <Button asChild variant="outline" className="flex-1">
              <Link href="/simulations">
                <FileQuestion className="h-4 w-4 mr-2" />
                Simulations
              </Link>
            </Button>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}
