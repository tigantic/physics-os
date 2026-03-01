/**
 * The Physics OS Stores - Central Export
 * 
 * Zustand stores for client-side state management.
 * Server state is handled by TanStack Query.
 */

import { create } from 'zustand';
import { persist, devtools } from 'zustand/middleware';

// Re-export domain stores
export { useSimulationStore, selectActiveSimulation, selectResiduals, selectPerformance, selectIsRunning, selectProgress } from './simulationStore';
export { useViewerStore, selectVisualizationOptions } from './viewerStore';

// ============================================
// AUTH STORE
// ============================================

interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role: 'user' | 'admin';
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set) => ({
        user: null,
        isAuthenticated: false,
        isLoading: true,

        setUser: (user) =>
          set(
            { user, isAuthenticated: !!user, isLoading: false },
            false,
            'setUser'
          ),

        setLoading: (isLoading) => set({ isLoading }, false, 'setLoading'),

        logout: () => {
          if (typeof window !== 'undefined') {
            localStorage.removeItem('auth_token');
          }
          set(
            { user: null, isAuthenticated: false, isLoading: false },
            false,
            'logout'
          );
        },
      }),
      {
        name: 'auth-storage',
        partialize: (state) => ({
          user: state.user,
          isAuthenticated: state.isAuthenticated,
        }),
      }
    ),
    { name: 'AuthStore' }
  )
);

// ============================================
// UI STORE
// ============================================

interface UIState {
  // Sidebar
  sidebarOpen: boolean;
  sidebarCollapsed: boolean;

  // Modals
  activeModal: string | null;
  modalData: unknown;

  // Theme
  theme: 'light' | 'dark' | 'system';

  // Actions
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  openModal: (modalId: string, data?: unknown) => void;
  closeModal: () => void;
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    persist(
      (set) => ({
        sidebarOpen: true,
        sidebarCollapsed: false,
        activeModal: null,
        modalData: null,
        theme: 'system',

        toggleSidebar: () =>
          set((state) => ({ sidebarOpen: !state.sidebarOpen }), false, 'toggleSidebar'),

        setSidebarCollapsed: (collapsed) =>
          set({ sidebarCollapsed: collapsed }, false, 'setSidebarCollapsed'),

        openModal: (modalId, data) =>
          set({ activeModal: modalId, modalData: data }, false, 'openModal'),

        closeModal: () =>
          set({ activeModal: null, modalData: null }, false, 'closeModal'),

        setTheme: (theme) =>
          set({ theme }, false, 'setTheme'),
      }),
      {
        name: 'ui-storage',
        partialize: (state) => ({
          sidebarCollapsed: state.sidebarCollapsed,
          theme: state.theme,
        }),
      }
    ),
    { name: 'UIStore' }
  )
);
