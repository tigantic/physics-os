/**
 * Sidebar - Main Navigation Component
 * 
 * The Physics OS UI navigation with collapsible sections.
 */

'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Box,
  Play,
  BarChart3,
  Settings,
  Layers,
  FileJson,
  Cpu,
  ChevronLeft,
  ChevronRight,
  Zap,
  Beaker,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { useUIStore } from '@/stores';

// ============================================
// NAVIGATION ITEMS
// ============================================

interface NavItem {
  label: string;
  href: string;
  icon: React.ElementType;
  badge?: number;
  disabled?: boolean;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    title: 'Overview',
    items: [
      { label: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
      { label: 'System Status', href: '/status', icon: Cpu },
    ],
  },
  {
    title: 'Simulation',
    items: [
      { label: 'Meshes', href: '/meshes', icon: Box },
      { label: 'Simulations', href: '/simulations', icon: Play },
      { label: 'Results', href: '/results', icon: BarChart3 },
    ],
  },
  {
    title: 'Advanced',
    items: [
      { label: 'QTT Compression', href: '/qtt', icon: Layers, disabled: true },
      { label: 'Benchmarks', href: '/benchmarks', icon: Zap, disabled: true },
      { label: 'API Explorer', href: '/api', icon: FileJson, disabled: true },
    ],
  },
];

// ============================================
// MAIN COMPONENT
// ============================================

interface SidebarProps {
  className?: string;
}

export function Sidebar({ className }: SidebarProps) {
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <TooltipProvider delayDuration={0}>
      <aside
        className={cn(
          'flex flex-col h-full bg-background border-r transition-all duration-300',
          isCollapsed ? 'w-16' : 'w-64',
          className
        )}
        aria-label="Main navigation"
      >
        {/* Logo Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b">
          {!isCollapsed && (
            <Link href="/" className="flex items-center gap-2" aria-label="The Physics OS CFD Home">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                <Beaker className="h-5 w-5 text-white" aria-hidden="true" />
              </div>
              <div className="flex flex-col">
                <span className="font-bold text-sm">The Physics OS</span>
                <span className="text-[10px] text-muted-foreground">CFD Engine</span>
              </div>
            </Link>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="h-8 w-8"
            aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-expanded={!isCollapsed}
          >
            {isCollapsed ? (
              <ChevronRight className="h-4 w-4" aria-hidden="true" />
            ) : (
              <ChevronLeft className="h-4 w-4" aria-hidden="true" />
            )}
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-4" aria-label="Primary">
          {NAV_SECTIONS.map((section) => (
            <div key={section.title} className="mb-6" role="group" aria-labelledby={`nav-section-${section.title.toLowerCase().replace(/\s/g, '-')}`}>
              {!isCollapsed && (
                <h3 
                  id={`nav-section-${section.title.toLowerCase().replace(/\s/g, '-')}`}
                  className="px-4 mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider"
                >
                  {section.title}
                </h3>
              )}
              <ul className="space-y-1 px-2" role="list">
                {section.items.map((item) => {
                  const isActive = pathname === item.href;
                  const Icon = item.icon;

                  // Disabled items render as spans, not links
                  if (item.disabled) {
                    const disabledContent = (
                      <span
                        className={cn(
                          'flex items-center gap-3 rounded-lg px-3 py-2 text-sm cursor-not-allowed opacity-50',
                          'text-muted-foreground',
                          isCollapsed && 'justify-center px-2'
                        )}
                        aria-disabled="true"
                        title="Coming soon"
                      >
                        <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
                        {!isCollapsed && (
                          <span className="flex-1">{item.label}</span>
                        )}
                      </span>
                    );

                    if (isCollapsed) {
                      return (
                        <li key={item.href}>
                          <Tooltip>
                            <TooltipTrigger asChild>{disabledContent}</TooltipTrigger>
                            <TooltipContent side="right">
                              {item.label} (Coming soon)
                            </TooltipContent>
                          </Tooltip>
                        </li>
                      );
                    }
                    return <li key={item.href}>{disabledContent}</li>;
                  }

                  const linkContent = (
                    <Link
                      href={item.href}
                      className={cn(
                        'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
                        isActive
                          ? 'bg-primary/10 text-primary font-medium'
                          : 'text-muted-foreground hover:bg-muted hover:text-foreground',
                        isCollapsed && 'justify-center px-2'
                      )}
                      aria-current={isActive ? 'page' : undefined}
                      aria-label={isCollapsed ? item.label : undefined}
                    >
                      <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
                      {!isCollapsed && (
                        <>
                          <span className="flex-1">{item.label}</span>
                          {item.badge !== undefined && (
                            <span 
                              className="flex h-5 w-5 items-center justify-center rounded-full bg-primary/20 text-xs font-medium"
                              aria-label={`${item.badge} items`}
                            >
                              {item.badge}
                            </span>
                          )}
                        </>
                      )}
                    </Link>
                  );

                  if (isCollapsed) {
                    return (
                      <li key={item.href}>
                        <Tooltip>
                          <TooltipTrigger asChild>{linkContent}</TooltipTrigger>
                          <TooltipContent side="right">
                            {item.label}
                            {item.badge !== undefined && ` (${item.badge})`}
                          </TooltipContent>
                        </Tooltip>
                      </li>
                    );
                  }

                  return <li key={item.href}>{linkContent}</li>;
                })}
              </ul>
            </div>
          ))}
        </nav>

        {/* Settings Footer */}
        <div className="border-t p-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Link
                href="/settings"
                className={cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground',
                  isCollapsed && 'justify-center px-2',
                  pathname === '/settings' && 'bg-primary/10 text-primary font-medium'
                )}
              >
                <Settings className="h-4 w-4" />
                {!isCollapsed && <span>Settings</span>}
              </Link>
            </TooltipTrigger>
            {isCollapsed && <TooltipContent side="right">Settings</TooltipContent>}
          </Tooltip>
        </div>
      </aside>
    </TooltipProvider>
  );
}
