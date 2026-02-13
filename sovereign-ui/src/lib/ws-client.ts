/**
 * WebSocket Client — real-time update channel for the FPS platform.
 *
 * Connects to the backend WebSocket endpoint proxied through Vite
 * at /ws. Provides a reactive Svelte store for connection status
 * and an event dispatcher for incoming messages.
 *
 * Message protocol: JSON frames with { type: string, payload: unknown }.
 * Known message types:
 *   - case_updated    → case data changed
 *   - case_deleted    → case removed
 *   - plan_compiled   → compilation result ready
 *   - mesh_ready      → mesh data available for case
 *   - heartbeat       → keep-alive ping
 */

import { writable, get } from 'svelte/store';

// ── Types ────────────────────────────────────────────────────────

export interface WsMessage {
  type: string;
  payload: unknown;
  timestamp?: string;
}

export type WsMessageHandler = (msg: WsMessage) => void;

export type WsStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

// ── Configuration ────────────────────────────────────────────────

const WS_RECONNECT_BASE_MS = 1000;
const WS_RECONNECT_MAX_MS = 30000;
const WS_HEARTBEAT_INTERVAL_MS = 30000;
const WS_HEARTBEAT_TIMEOUT_MS = 10000;

// ── Stores ───────────────────────────────────────────────────────

/** Current WebSocket connection status. */
export const wsStatus = writable<WsStatus>('disconnected');

/** Last message received (for debugging / reactivity). */
export const wsLastMessage = writable<WsMessage | null>(null);

/** Number of reconnection attempts since last successful connect. */
export const wsReconnectCount = writable<number>(0);

// ── Internal State ───────────────────────────────────────────────

let socket: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
let heartbeatTimeout: ReturnType<typeof setTimeout> | null = null;
let reconnectAttempts = 0;
let intentionalClose = false;

/** Registered message handlers by type. Use '*' for catch-all. */
const handlers = new Map<string, Set<WsMessageHandler>>();

// ── Public API ───────────────────────────────────────────────────

/**
 * Register a handler for a specific message type.
 * Use '*' to receive all messages.
 * @returns An unsubscribe function.
 */
export function onWsMessage(
  type: string,
  handler: WsMessageHandler,
): () => void {
  if (!handlers.has(type)) {
    handlers.set(type, new Set());
  }
  handlers.get(type)!.add(handler);

  return () => {
    const set = handlers.get(type);
    if (set) {
      set.delete(handler);
      if (set.size === 0) handlers.delete(type);
    }
  };
}

/**
 * Connect to the WebSocket endpoint.
 * Safe to call multiple times — no-ops if already connected/connecting.
 */
export function connectWs(): void {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  intentionalClose = false;
  wsStatus.set('connecting');

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${window.location.host}/ws`;

  try {
    socket = new WebSocket(url);
  } catch (err) {
    console.error('[ws] Failed to create WebSocket:', err);
    wsStatus.set('error');
    scheduleReconnect();
    return;
  }

  socket.onopen = () => {
    wsStatus.set('connected');
    wsReconnectCount.set(0);
    reconnectAttempts = 0;
    startHeartbeat();
    if (import.meta.env.DEV) {
      console.debug('[ws] Connected');
    }
  };

  socket.onmessage = (event) => {
    try {
      const msg: WsMessage = JSON.parse(event.data);
      if (!msg.type) return;

      wsLastMessage.set(msg);
      resetHeartbeatTimeout();

      // Dispatch to type-specific handlers
      const typeHandlers = handlers.get(msg.type);
      if (typeHandlers) {
        for (const handler of typeHandlers) {
          try {
            handler(msg);
          } catch (err) {
            console.error(`[ws] Handler error for "${msg.type}":`, err);
          }
        }
      }

      // Dispatch to catch-all handlers
      const wildcardHandlers = handlers.get('*');
      if (wildcardHandlers) {
        for (const handler of wildcardHandlers) {
          try {
            handler(msg);
          } catch (err) {
            console.error('[ws] Wildcard handler error:', err);
          }
        }
      }
    } catch {
      // Not JSON or malformed — ignore
    }
  };

  socket.onclose = (event) => {
    stopHeartbeat();
    socket = null;

    if (intentionalClose) {
      wsStatus.set('disconnected');
    } else {
      wsStatus.set('error');
      if (import.meta.env.DEV) {
        console.debug(`[ws] Closed (code=${event.code}). Reconnecting...`);
      }
      scheduleReconnect();
    }
  };

  socket.onerror = () => {
    // onclose will fire after onerror — handle reconnect there
    wsStatus.set('error');
  };
}

/**
 * Disconnect from the WebSocket. Stops reconnection attempts.
 */
export function disconnectWs(): void {
  intentionalClose = true;
  clearReconnectTimer();
  stopHeartbeat();

  if (socket) {
    socket.close(1000, 'Client disconnect');
    socket = null;
  }

  wsStatus.set('disconnected');
}

/**
 * Send a message to the backend via WebSocket.
 * @returns true if sent, false if not connected.
 */
export function sendWsMessage(type: string, payload: unknown = {}): boolean {
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;

  try {
    socket.send(
      JSON.stringify({ type, payload, timestamp: new Date().toISOString() }),
    );
    return true;
  } catch {
    return false;
  }
}

// ── Reconnection ─────────────────────────────────────────────────

function scheduleReconnect(): void {
  clearReconnectTimer();
  reconnectAttempts++;
  wsReconnectCount.set(reconnectAttempts);

  // Exponential backoff with jitter
  const delay = Math.min(
    WS_RECONNECT_BASE_MS * Math.pow(2, reconnectAttempts - 1) +
      Math.random() * 500,
    WS_RECONNECT_MAX_MS,
  );

  reconnectTimer = setTimeout(() => {
    connectWs();
  }, delay);
}

function clearReconnectTimer(): void {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

// ── Heartbeat ────────────────────────────────────────────────────

function startHeartbeat(): void {
  stopHeartbeat();
  heartbeatTimer = setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      sendWsMessage('heartbeat');
      heartbeatTimeout = setTimeout(() => {
        // No response within timeout — assume dead connection
        if (import.meta.env.DEV) {
          console.debug('[ws] Heartbeat timeout — reconnecting');
        }
        socket?.close(4000, 'Heartbeat timeout');
      }, WS_HEARTBEAT_TIMEOUT_MS);
    }
  }, WS_HEARTBEAT_INTERVAL_MS);
}

function stopHeartbeat(): void {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  }
  if (heartbeatTimeout) {
    clearTimeout(heartbeatTimeout);
    heartbeatTimeout = null;
  }
}

function resetHeartbeatTimeout(): void {
  if (heartbeatTimeout) {
    clearTimeout(heartbeatTimeout);
    heartbeatTimeout = null;
  }
}
