/**
 * NetworkStatusBanner - Offline/Slow Connection Indicator
 * 
 * Displays a banner when network connectivity is degraded or unavailable.
 * Provides retry functionality and connection status information.
 */

'use client';

import { useState, useEffect } from 'react';
import { WifiOff, Wifi, AlertTriangle, RefreshCw, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useNetworkStatus } from '@/hooks/useNetworkStatus';

interface NetworkStatusBannerProps {
  className?: string;
}

export function NetworkStatusBanner({ className }: NetworkStatusBannerProps) {
  const { isOnline, isApiReachable, isSlowConnection, checkApiConnectivity } = useNetworkStatus();
  const [isDismissed, setIsDismissed] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);

  // Reset dismissed state when status changes
  useEffect(() => {
    if (!isOnline || !isApiReachable) {
      setIsDismissed(false);
    }
  }, [isOnline, isApiReachable]);

  // Don't show if everything is fine or dismissed
  if ((isOnline && isApiReachable && !isSlowConnection) || isDismissed) {
    return null;
  }

  const handleRetry = async () => {
    setIsRetrying(true);
    await checkApiConnectivity();
    setIsRetrying(false);
  };

  // Determine banner variant
  const isOffline = !isOnline;
  const isApiDown = isOnline && !isApiReachable;

  const bannerConfig = {
    offline: {
      icon: WifiOff,
      message: 'You are offline. Some features may not be available.',
      variant: 'destructive' as const,
    },
    apiDown: {
      icon: AlertTriangle,
      message: 'Cannot connect to server. Please check your connection.',
      variant: 'warning' as const,
    },
    slow: {
      icon: Wifi,
      message: 'Slow connection detected. Some features may be delayed.',
      variant: 'warning' as const,
    },
  };

  const config = isOffline
    ? bannerConfig.offline
    : isApiDown
    ? bannerConfig.apiDown
    : bannerConfig.slow;

  const Icon = config.icon;

  return (
    <div
      className={cn(
        'fixed bottom-4 left-1/2 -translate-x-1/2 z-50',
        'flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg',
        'bg-background border max-w-md',
        config.variant === 'destructive' && 'border-destructive bg-destructive/10',
        config.variant === 'warning' && 'border-orange-500 bg-orange-50 dark:bg-orange-950/20',
        className
      )}
      role="alert"
      aria-live="polite"
    >
      <Icon
        className={cn(
          'h-5 w-5 flex-shrink-0',
          config.variant === 'destructive' && 'text-destructive',
          config.variant === 'warning' && 'text-orange-600 dark:text-orange-400'
        )}
      />

      <p className="text-sm flex-1">{config.message}</p>

      <div className="flex items-center gap-2">
        {!isOffline && (
          <Button
            size="sm"
            variant="ghost"
            onClick={handleRetry}
            disabled={isRetrying}
          >
            <RefreshCw className={cn('h-4 w-4', isRetrying && 'animate-spin')} />
            <span className="sr-only">Retry connection</span>
          </Button>
        )}

        {isSlowConnection && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setIsDismissed(true)}
          >
            <X className="h-4 w-4" />
            <span className="sr-only">Dismiss</span>
          </Button>
        )}
      </div>
    </div>
  );
}

export default NetworkStatusBanner;
