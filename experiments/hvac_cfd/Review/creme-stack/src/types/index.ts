// ============================================
// COMMON TYPES
// ============================================

/**
 * Makes all properties of T optional recursively
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Extract the resolved type from a Promise
 */
export type Awaited<T> = T extends Promise<infer U> ? U : T;

/**
 * Make specific keys required
 */
export type RequiredKeys<T, K extends keyof T> = T & Required<Pick<T, K>>;

/**
 * Make specific keys optional
 */
export type OptionalKeys<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

// ============================================
// API TYPES
// ============================================

/**
 * Standard API response wrapper
 */
export interface ApiResponse<T> {
  data: T;
  message?: string;
}

/**
 * Standard API error
 */
export interface ApiError {
  code: string;
  message: string;
  details?: unknown[];
  requestId?: string;
}

/**
 * Paginated response
 */
export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    total: number;
    page: number;
    limit: number;
    hasMore: boolean;
    nextCursor?: string;
  };
}

/**
 * Standard query params for lists
 */
export interface ListQueryParams {
  page?: number;
  limit?: number;
  search?: string;
  sort?: string;
  order?: 'asc' | 'desc';
}

// ============================================
// USER TYPES
// ============================================

export type UserRole = 'user' | 'admin' | 'super_admin';

export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role: UserRole;
  createdAt: string;
  updatedAt: string;
}

// ============================================
// UTILITY TYPES FOR REACT
// ============================================

/**
 * Props that include children
 */
export interface WithChildren {
  children: React.ReactNode;
}

/**
 * Props that include optional className
 */
export interface WithClassName {
  className?: string;
}

/**
 * Common component props
 */
export interface ComponentProps extends WithClassName {
  id?: string;
}
