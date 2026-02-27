/**
 * Global Error Boundary
 * 
 * Catches errors in the root layout. Must render its own html/body.
 */

'use client';

import { useEffect } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface GlobalErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function GlobalError({ error, reset }: GlobalErrorProps) {
  useEffect(() => {
    console.error('Global error:', error);
  }, [error]);

  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 dark:bg-gray-950 flex items-center justify-center p-6">
        <div className="w-full max-w-md bg-white dark:bg-gray-900 rounded-xl shadow-lg p-8 text-center">
          <div className="mx-auto mb-6 h-16 w-16 rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
            <AlertTriangle className="h-8 w-8 text-red-600 dark:text-red-400" />
          </div>
          
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            Application Error
          </h1>
          
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            A critical error occurred. Please try refreshing the page.
          </p>
          
          <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 mb-6">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {error.message || 'Unknown error'}
            </p>
            {error.digest && (
              <p className="text-xs text-gray-500 mt-1">
                Reference: {error.digest}
              </p>
            )}
          </div>
          
          <button
            onClick={reset}
            className="inline-flex items-center justify-center px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Try Again
          </button>
        </div>
      </body>
    </html>
  );
}
