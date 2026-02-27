import { create } from 'zustand';
import { persist, devtools } from 'zustand/middleware';

// ============================================
// AUTH STORE EXAMPLE
// ============================================

interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role: 'user' | 'admin';
}

interface AuthState {
  // State
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  // Actions
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set) => ({
        // Initial state
        user: null,
        isAuthenticated: false,
        isLoading: true,

        // Actions
        setUser: (user) =>
          set(
            { user, isAuthenticated: !!user, isLoading: false },
            false,
            'setUser'
          ),

        setLoading: (isLoading) => set({ isLoading }, false, 'setLoading'),

        logout: () => {
          // Clear token from storage
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
        name: 'auth-storage', // localStorage key
        partialize: (state) => ({
          // Only persist these fields
          user: state.user,
          isAuthenticated: state.isAuthenticated,
        }),
      }
    ),
    { name: 'AuthStore' }
  )
);

// ============================================
// UI STORE EXAMPLE
// ============================================

interface UIState {
  // Sidebar
  sidebarOpen: boolean;
  sidebarCollapsed: boolean;

  // Modals
  activeModal: string | null;
  modalData: unknown;

  // Actions
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  openModal: (modalId: string, data?: unknown) => void;
  closeModal: () => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    (set) => ({
      // Initial state
      sidebarOpen: true,
      sidebarCollapsed: false,
      activeModal: null,
      modalData: null,

      // Actions
      toggleSidebar: () =>
        set((state) => ({ sidebarOpen: !state.sidebarOpen }), false, 'toggleSidebar'),

      setSidebarCollapsed: (collapsed) =>
        set({ sidebarCollapsed: collapsed }, false, 'setSidebarCollapsed'),

      openModal: (modalId, data) =>
        set({ activeModal: modalId, modalData: data }, false, 'openModal'),

      closeModal: () =>
        set({ activeModal: null, modalData: null }, false, 'closeModal'),
    }),
    { name: 'UIStore' }
  )
);
