/**
 * useNetworkStatus - Network State Detection Hook
 * 
 * Monitors online/offline status and provides network state info.
 * Enables graceful degradation when network is unavailable.
 */

'use client';

import { useState, useEffect, useCallback } from 'react';

interface NetworkStatus {
  /** Whether browser reports being online */
  isOnline: boolean;
  /** Whether we've verified API connectivity */
  isApiReachable: boolean;
  /** Last time we checked connectivity */
  lastChecked: Date | null;
  /** Current connection type (if available) */
  connectionType: string | null;
  /** Effective bandwidth estimate */
  effectiveType: string | null;
  /** Whether connection is slow */
  isSlowConnection: boolean;
}

interface UseNetworkStatusOptions {
  /** API endpoint to ping for connectivity check */
  pingEndpoint?: string;
  /** How often to check connectivity (ms) */
  pingInterval?: number;
  /** Enable periodic connectivity checks */
  enablePing?: boolean;
}

const defaultOptions: UseNetworkStatusOptions = {
  pingEndpoint: '/api/v1/health',
  pingInterval: 30000, // 30 seconds
  enablePing: true,
};

export function useNetworkStatus(options: UseNetworkStatusOptions = {}) {
  const config = { ...defaultOptions, ...options };
  
  const [status, setStatus] = useState<NetworkStatus>({
    isOnline: typeof navigator !== 'undefined' ? navigator.onLine : true,
    isApiReachable: true,
    lastChecked: null,
    connectionType: null,
    effectiveType: null,
    isSlowConnection: false,
  });

  // Check API connectivity
  const checkApiConnectivity = useCallback(async () => {
    if (!status.isOnline) {
      setStatus((prev) => ({
        ...prev,
        isApiReachable: false,
        lastChecked: new Date(),
      }));
      return false;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(config.pingEndpoint!, {
        method: 'HEAD',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const isReachable = response.ok;
      setStatus((prev) => ({
        ...prev,
        isApiReachable: isReachable,
        lastChecked: new Date(),
      }));
      
      return isReachable;
    } catch {
      setStatus((prev) => ({
        ...prev,
        isApiReachable: false,
        lastChecked: new Date(),
      }));
      return false;
    }
  }, [status.isOnline, config.pingEndpoint]);

  // Handle online/offline events
  useEffect(() => {
    const handleOnline = () => {
      setStatus((prev) => ({ ...prev, isOnline: true }));
      checkApiConnectivity();
    };

    const handleOffline = () => {
      setStatus((prev) => ({
        ...prev,
        isOnline: false,
        isApiReachable: false,
      }));
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [checkApiConnectivity]);

  // Monitor connection quality
  useEffect(() => {
    if (typeof navigator === 'undefined') return;

    const connection = (navigator as any).connection || 
                       (navigator as any).mozConnection || 
                       (navigator as any).webkitConnection;

    if (!connection) return;
    
    const updateConnectionInfo = () => {
      setStatus((prev) => ({
        ...prev,
        connectionType: connection.type || null,
        effectiveType: connection.effectiveType || null,
        isSlowConnection: ['slow-2g', '2g'].includes(connection.effectiveType),
      }));
    };

    updateConnectionInfo();
    connection.addEventListener('change', updateConnectionInfo);

    return () => {
      connection.removeEventListener('change', updateConnectionInfo);
    };
  }, []);

  // Periodic connectivity check
  useEffect(() => {
    if (!config.enablePing) return;

    // Initial check
    checkApiConnectivity();

    // Set up interval
    const intervalId = setInterval(checkApiConnectivity, config.pingInterval!);

    return () => clearInterval(intervalId);
  }, [config.enablePing, config.pingInterval, checkApiConnectivity]);

  return {
    ...status,
    checkApiConnectivity,
  };
}

export default useNetworkStatus;
