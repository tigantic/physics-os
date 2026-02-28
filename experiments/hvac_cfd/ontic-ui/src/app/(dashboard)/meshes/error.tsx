/**
 * Meshes Error Boundary
 * 
 * Catches runtime errors in mesh routes.
 */

'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { AlertTriangle, RefreshCw, ArrowLeft, Upload } from 'lucide-react';
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

export default function MeshesError({ error, reset }: ErrorProps) {
  useEffect(() => {
    console.error('Meshes error:', error);
  }, [error]);

  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 h-14 w-14 rounded-full bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
            <AlertTriangle className="h-7 w-7 text-blue-600 dark:text-blue-400" />
          </div>
          <CardTitle className="text-xl">Mesh Error</CardTitle>
          <CardDescription>
            There was a problem loading the mesh data.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="rounded-lg bg-muted p-4">
            <p className="text-sm text-center text-muted-foreground">
              {error.message || 'Failed to load mesh'}
            </p>
          </div>
        </CardContent>

        <CardFooter className="flex flex-col gap-3">
          <Button onClick={reset} className="w-full">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
          <div className="flex gap-2 w-full">
            <Button asChild variant="outline" className="flex-1">
              <Link href="/meshes">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Meshes
              </Link>
            </Button>
            <Button asChild variant="ghost" className="flex-1">
              <Link href="/meshes">
                <Upload className="h-4 w-4 mr-2" />
                Upload New
              </Link>
            </Button>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}
