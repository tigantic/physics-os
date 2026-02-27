/**
 * WebSocket Hook for Real-time Updates
 * 
 * Provides real-time simulation residual updates via WebSocket connection.
 */

'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { queryKeys } from './useApi';
import type { ResidualPoint } from '@/types';

// ============================================
// TYPES
// ============================================

interface WebSocketMessage {
  channel: string;
  data: unknown;
  timestamp: string;
}

interface UseWebSocketOptions {
  url?: string;
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
}

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

// ============================================
// WEBSOCKET HOOK
// ============================================

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url = 'ws://localhost:8000/ws',
    onMessage,
    onConnect,
    onDisconnect,
    autoReconnect = true,
    reconnectInterval = 3000,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setStatus('connecting');

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setStatus('connected');
        onConnect?.();
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          setLastMessage(message);
          onMessage?.(message);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onclose = () => {
        setStatus('disconnected');
        onDisconnect?.();
        wsRef.current = null;

        if (autoReconnect) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = () => {
        setStatus('error');
      };

      wsRef.current = ws;
    } catch (e) {
      setStatus('error');
      console.error('WebSocket connection failed:', e);
    }
  }, [url, onMessage, onConnect, onDisconnect, autoReconnect, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const send = useCallback((data: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const subscribe = useCallback((channel: string) => {
    send({ action: 'subscribe', channel });
  }, [send]);

  const unsubscribe = useCallback((channel: string) => {
    send({ action: 'unsubscribe', channel });
  }, [send]);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    status,
    lastMessage,
    send,
    subscribe,
    unsubscribe,
    connect,
    disconnect,
  };
}

// ============================================
// SIMULATION RESIDUALS SUBSCRIPTION
// ============================================

export function useSimulationWebSocket(simulationId: string | null) {
  const queryClient = useQueryClient();
  const [liveResiduals, setLiveResiduals] = useState<ResidualPoint[]>([]);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    // Handle residual updates
    if (message.channel === `simulation.${simulationId}.residuals`) {
      const residual = message.data as ResidualPoint;
      setLiveResiduals((prev) => [...prev.slice(-500), residual]);
      
      // Update React Query cache
      queryClient.setQueryData(
        queryKeys.simulations.residuals(simulationId ?? ''),
        (old: ResidualPoint[] | undefined) => {
          if (!old) return [residual];
          return [...old.slice(-500), residual];
        }
      );
    }

    // Handle simulation status updates
    if (message.channel === `simulation.${simulationId}.status`) {
      queryClient.invalidateQueries({
        queryKey: queryKeys.simulations.detail(simulationId ?? ''),
      });
    }
  }, [simulationId, queryClient]);

  const { status, subscribe, unsubscribe } = useWebSocket({
    onMessage: handleMessage,
    onConnect: () => {
      if (simulationId) {
        subscribe(`simulation.${simulationId}.residuals`);
        subscribe(`simulation.${simulationId}.status`);
      }
    },
  });

  // Subscribe when simulation ID changes
  useEffect(() => {
    if (simulationId && status === 'connected') {
      subscribe(`simulation.${simulationId}.residuals`);
      subscribe(`simulation.${simulationId}.status`);
    }
    
    return () => {
      if (simulationId) {
        unsubscribe(`simulation.${simulationId}.residuals`);
        unsubscribe(`simulation.${simulationId}.status`);
      }
    };
  }, [simulationId, status, subscribe, unsubscribe]);

  return {
    status,
    liveResiduals,
    isConnected: status === 'connected',
  };
}
