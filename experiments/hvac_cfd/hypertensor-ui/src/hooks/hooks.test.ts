/**
 * Hooks Tests
 * 
 * Tests for useWebSocket, useNetworkStatus
 * Constitutional Compliance: Article III Testing Protocols
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useNetworkStatus } from './useNetworkStatus';

// ============================================
// USE NETWORK STATUS HOOK TESTS
// ============================================

describe('useNetworkStatus', () => {
  const originalOnLine = navigator.onLine;
  const addEventListenerSpy = vi.spyOn(window, 'addEventListener');
  const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener');

  beforeEach(() => {
    addEventListenerSpy.mockClear();
    removeEventListenerSpy.mockClear();
  });

  afterEach(() => {
    Object.defineProperty(navigator, 'onLine', {
      value: originalOnLine,
      writable: true,
      configurable: true,
    });
  });

  it('should return initial online status', () => {
    Object.defineProperty(navigator, 'onLine', {
      value: true,
      writable: true,
      configurable: true,
    });
    
    const { result } = renderHook(() => useNetworkStatus());
    
    expect(result.current.isOnline).toBe(true);
  });

  it('should return initial offline status', () => {
    Object.defineProperty(navigator, 'onLine', {
      value: false,
      writable: true,
      configurable: true,
    });
    
    const { result } = renderHook(() => useNetworkStatus());
    
    expect(result.current.isOnline).toBe(false);
  });

  it('should add event listeners on mount', () => {
    renderHook(() => useNetworkStatus());
    
    expect(addEventListenerSpy).toHaveBeenCalledWith('online', expect.any(Function));
    expect(addEventListenerSpy).toHaveBeenCalledWith('offline', expect.any(Function));
  });

  it('should remove event listeners on unmount', () => {
    const { unmount } = renderHook(() => useNetworkStatus());
    
    unmount();
    
    expect(removeEventListenerSpy).toHaveBeenCalledWith('online', expect.any(Function));
    expect(removeEventListenerSpy).toHaveBeenCalledWith('offline', expect.any(Function));
  });
});

// ============================================
// USE WEBSOCKET HOOK TESTS  
// Note: WebSocket mocking is complex in jsdom
// Testing the hook interface and exports
// ============================================

describe('useWebSocket', () => {
  let mockWebSocket: any;
  const originalWebSocket = global.WebSocket;

  beforeEach(() => {
    mockWebSocket = {
      send: vi.fn(),
      close: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      readyState: 1, // OPEN
      CONNECTING: 0,
      OPEN: 1,
      CLOSING: 2,
      CLOSED: 3,
    };
    
    global.WebSocket = vi.fn(() => mockWebSocket) as any;
    (global.WebSocket as any).CONNECTING = 0;
    (global.WebSocket as any).OPEN = 1;
    (global.WebSocket as any).CLOSING = 2;
    (global.WebSocket as any).CLOSED = 3;
  });

  afterEach(() => {
    global.WebSocket = originalWebSocket;
    vi.clearAllMocks();
  });

  it('should export useWebSocket function', async () => {
    const { useWebSocket } = await import('./useWebSocket');
    expect(typeof useWebSocket).toBe('function');
  });

  it('should create websocket on mount', async () => {
    const { useWebSocket } = await import('./useWebSocket');
    
    renderHook(() => useWebSocket());
    
    expect(global.WebSocket).toHaveBeenCalled();
  });

  it('should return status and send function', async () => {
    const { useWebSocket } = await import('./useWebSocket');
    
    const { result } = renderHook(() => useWebSocket());
    
    expect(result.current).toHaveProperty('status');
    expect(result.current).toHaveProperty('send');
  });
});
