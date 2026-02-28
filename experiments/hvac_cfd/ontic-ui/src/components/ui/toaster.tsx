/**
 * Toaster Component
 * 
 * Renders toast notifications from the use-toast hook.
 */

'use client';

import { useEffect, useState } from 'react';
import { X, CheckCircle2, AlertCircle, Info } from 'lucide-react';
import { cn } from '@/lib/utils';

// ============================================
// TYPES
// ============================================

interface Toast {
  id: string;
  title: string;
  description?: string;
  variant?: 'default' | 'destructive';
}

// ============================================
// GLOBAL STATE
// ============================================

let toastListeners: Array<(toasts: Toast[]) => void> = [];
let toasts: Toast[] = [];

function notify() {
  toastListeners.forEach((listener) => listener([...toasts]));
}

export function toast(options: Omit<Toast, 'id'>) {
  const id = Math.random().toString(36).slice(2);
  const newToast: Toast = { id, ...options };
  toasts = [...toasts, newToast];
  notify();

  // Auto-remove after 5 seconds
  setTimeout(() => {
    dismissToast(id);
  }, 5000);

  return id;
}

export function dismissToast(id: string) {
  toasts = toasts.filter((t) => t.id !== id);
  notify();
}

// ============================================
// HOOK
// ============================================

export function useToast() {
  return { toast, dismiss: dismissToast };
}

// ============================================
// TOAST ITEM
// ============================================

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Animate in
    requestAnimationFrame(() => setIsVisible(true));
  }, []);

  const handleDismiss = () => {
    setIsVisible(false);
    setTimeout(onDismiss, 150);
  };

  const Icon = toast.variant === 'destructive' ? AlertCircle : CheckCircle2;

  return (
    <div
      className={cn(
        'pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-lg border bg-background p-4 shadow-lg transition-all duration-150',
        isVisible ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0',
        toast.variant === 'destructive' && 'border-destructive/50 bg-destructive/5'
      )}
    >
      <Icon
        className={cn(
          'h-5 w-5 shrink-0 mt-0.5',
          toast.variant === 'destructive' ? 'text-destructive' : 'text-emerald-500'
        )}
      />
      <div className="flex-1 space-y-1">
        <p className="text-sm font-medium">{toast.title}</p>
        {toast.description && (
          <p className="text-xs text-muted-foreground">{toast.description}</p>
        )}
      </div>
      <button
        onClick={handleDismiss}
        className="shrink-0 rounded-md p-1 hover:bg-muted transition-colors"
      >
        <X className="h-4 w-4 text-muted-foreground" />
      </button>
    </div>
  );
}

// ============================================
// TOASTER
// ============================================

export function Toaster() {
  const [currentToasts, setCurrentToasts] = useState<Toast[]>([]);

  useEffect(() => {
    const listener = (newToasts: Toast[]) => setCurrentToasts(newToasts);
    toastListeners.push(listener);
    // Sync with any existing toasts
    setCurrentToasts([...toasts]);
    return () => {
      toastListeners = toastListeners.filter((l) => l !== listener);
    };
  }, []);

  if (currentToasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      {currentToasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={() => dismissToast(t.id)} />
      ))}
    </div>
  );
}
