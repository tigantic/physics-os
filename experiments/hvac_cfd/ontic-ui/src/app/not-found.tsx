/**
 * 404 Not Found Page
 * 
 * Displayed when a route is not found.
 */

import Link from 'next/link';
import { FileQuestion, Home, Search, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-background">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 h-20 w-20 rounded-full bg-muted flex items-center justify-center">
            <FileQuestion className="h-10 w-10 text-muted-foreground" />
          </div>
          <CardTitle className="text-3xl font-bold">404</CardTitle>
          <CardDescription className="text-lg">
            Page not found
          </CardDescription>
        </CardHeader>

        <CardContent>
          <p className="text-center text-muted-foreground">
            The page you&apos;re looking for doesn&apos;t exist or has been moved.
          </p>
        </CardContent>

        <CardFooter className="flex flex-col gap-3">
          <Button asChild className="w-full">
            <Link href="/">
              <Home className="h-4 w-4 mr-2" />
              Go to Dashboard
            </Link>
          </Button>
          
          <div className="flex gap-2 w-full">
            <Button asChild variant="outline" className="flex-1">
              <Link href="/simulations">
                <Search className="h-4 w-4 mr-2" />
                Simulations
              </Link>
            </Button>
            <Button asChild variant="ghost" className="flex-1">
              <Link href="/meshes">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Meshes
              </Link>
            </Button>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}
