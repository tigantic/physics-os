/**
 * DashboardShell - Page Layout Wrapper
 * 
 * Provides consistent page structure with title and actions.
 */

'use client';

import { ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';
import Link from 'next/link';
import { cn } from '@/lib/utils';

// ============================================
// BREADCRUMB
// ============================================

interface BreadcrumbItem {
  label: string;
  href?: string;
}

interface BreadcrumbProps {
  items: BreadcrumbItem[];
  className?: string;
}

export function Breadcrumb({ items, className }: BreadcrumbProps) {
  return (
    <nav className={cn('flex items-center text-sm text-muted-foreground', className)}>
      {items.map((item, index) => (
        <span key={index} className="flex items-center">
          {index > 0 && <ChevronRight className="h-4 w-4 mx-1" />}
          {item.href ? (
            <Link
              href={item.href}
              className="hover:text-foreground transition-colors"
            >
              {item.label}
            </Link>
          ) : (
            <span className="text-foreground font-medium">{item.label}</span>
          )}
        </span>
      ))}
    </nav>
  );
}

// ============================================
// PAGE HEADER
// ============================================

interface PageHeaderProps {
  title: string;
  description?: string;
  breadcrumbs?: BreadcrumbItem[];
  actions?: ReactNode;
  className?: string;
}

export function PageHeader({
  title,
  description,
  breadcrumbs,
  actions,
  className,
}: PageHeaderProps) {
  return (
    <div className={cn('space-y-2', className)}>
      {breadcrumbs && <Breadcrumb items={breadcrumbs} />}
      <div className="flex items-center justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
          {description && (
            <p className="text-sm text-muted-foreground">{description}</p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
    </div>
  );
}

// ============================================
// DASHBOARD SHELL
// ============================================

interface DashboardShellProps {
  children: ReactNode;
  className?: string;
}

export function DashboardShell({ children, className }: DashboardShellProps) {
  return (
    <main
      className={cn(
        'flex-1 overflow-auto p-6 lg:p-8',
        className
      )}
    >
      {children}
    </main>
  );
}

// ============================================
// SECTION CARD
// ============================================

interface SectionCardProps {
  title?: ReactNode;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  noPadding?: boolean;
}

export function SectionCard({
  title,
  description,
  actions,
  children,
  className,
  noPadding = false,
}: SectionCardProps) {
  return (
    <div
      className={cn(
        'rounded-xl border bg-card shadow-sm',
        className
      )}
    >
      {(title || actions) && (
        <div className="flex items-center justify-between border-b px-6 py-4">
          <div className="space-y-0.5">
            {title && <h2 className="font-semibold">{title}</h2>}
            {description && (
              <p className="text-sm text-muted-foreground">{description}</p>
            )}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className={cn(noPadding ? '' : 'p-6')}>{children}</div>
    </div>
  );
}

// ============================================
// STAT CARD
// ============================================

interface StatCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon?: ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  className?: string;
}

export function StatCard({
  title,
  value,
  description,
  icon,
  trend,
  className,
}: StatCardProps) {
  return (
    <div className={cn('rounded-xl border bg-card p-6', className)}>
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-muted-foreground">{title}</span>
        {icon && (
          <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
            {icon}
          </div>
        )}
      </div>
      <div className="mt-3">
        <span className="text-3xl font-bold">{value}</span>
        {trend && (
          <span
            className={cn(
              'ml-2 text-sm font-medium',
              trend.isPositive ? 'text-emerald-600' : 'text-red-500'
            )}
          >
            {trend.isPositive ? '+' : ''}
            {trend.value}%
          </span>
        )}
      </div>
      {description && (
        <p className="mt-1 text-sm text-muted-foreground">{description}</p>
      )}
    </div>
  );
}

// ============================================
// EMPTY STATE
// ============================================

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center py-12 text-center',
        className
      )}
    >
      {icon && (
        <div className="mb-4 h-12 w-12 rounded-full bg-muted flex items-center justify-center text-muted-foreground">
          {icon}
        </div>
      )}
      <h3 className="text-lg font-semibold">{title}</h3>
      {description && (
        <p className="mt-1 text-sm text-muted-foreground max-w-sm">
          {description}
        </p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
