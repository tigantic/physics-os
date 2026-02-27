/**
 * Header - Top Navigation Bar
 * 
 * Global header with search, notifications, and user menu.
 */

'use client';

import { useState } from 'react';
import Link from 'next/link';
import {
  Search,
  Bell,
  Moon,
  Sun,
  User,
  LogOut,
  Settings,
  HelpCircle,
  Activity,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { useUIStore, useSimulationStore } from '@/stores';
import { useSystemStatus } from '@/hooks';

// ============================================
// MAIN COMPONENT
// ============================================

interface HeaderProps {
  className?: string;
}

export function Header({ className = '' }: HeaderProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const { theme, setTheme } = useUIStore();
  const isConnected = useSimulationStore((s) => s.isConnected);
  const { data: systemStatus } = useSystemStatus();

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  return (
    <TooltipProvider delayDuration={0}>
      <header
        className={`h-16 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 ${className}`}
        role="banner"
      >
        <div className="flex h-full items-center justify-between px-4 gap-4">
          {/* Search */}
          <div className="flex-1 max-w-md" role="search">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
              <Input
                type="search"
                placeholder="Search simulations, meshes..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 bg-muted/50"
                aria-label="Search simulations and meshes"
              />
            </div>
          </div>

          {/* Right Actions */}
          <div className="flex items-center gap-2">
            {/* Connection Status */}
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs ${
                    isConnected
                      ? 'text-emerald-600 bg-emerald-500/10'
                      : 'text-muted-foreground bg-muted'
                  }`}
                  role="status"
                  aria-live="polite"
                  aria-label={isConnected ? 'WebSocket connected' : 'WebSocket disconnected'}
                >
                  {isConnected ? (
                    <Wifi className="h-3 w-3" aria-hidden="true" />
                  ) : (
                    <WifiOff className="h-3 w-3" aria-hidden="true" />
                  )}
                  <span className="hidden sm:inline">
                    {isConnected ? 'Connected' : 'Offline'}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                {isConnected
                  ? 'Real-time updates active'
                  : 'WebSocket disconnected - attempting reconnect'}
              </TooltipContent>
            </Tooltip>

            {/* System Health */}
            {systemStatus && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Link
                    href="/status"
                    className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs bg-muted hover:bg-muted/80 transition-colors"
                  >
                    <Activity className="h-3 w-3 text-cyan-500" />
                    <span className="hidden sm:inline font-mono">
                      {systemStatus.gpuUtilization?.toFixed(0) ?? '--'}%
                    </span>
                  </Link>
                </TooltipTrigger>
                <TooltipContent>GPU Utilization</TooltipContent>
              </Tooltip>
            )}

            {/* Notifications */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="relative" aria-label="Notifications (3 unread)">
                  <Bell className="h-4 w-4" aria-hidden="true" />
                  <Badge
                    variant="destructive"
                    className="absolute -top-1 -right-1 h-4 w-4 p-0 flex items-center justify-center text-[10px]"
                    aria-hidden="true"
                  >
                    3
                  </Badge>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-80">
                <DropdownMenuLabel>Notifications</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="flex flex-col items-start gap-1 py-2">
                  <span className="font-medium">Simulation Completed</span>
                  <span className="text-xs text-muted-foreground">
                    "HVAC Room v2" finished in 4m 32s
                  </span>
                </DropdownMenuItem>
                <DropdownMenuItem className="flex flex-col items-start gap-1 py-2">
                  <span className="font-medium">Convergence Warning</span>
                  <span className="text-xs text-muted-foreground">
                    "Duct Flow" residuals plateaued at 1e-4
                  </span>
                </DropdownMenuItem>
                <DropdownMenuItem className="flex flex-col items-start gap-1 py-2">
                  <span className="font-medium">GPU Driver Update</span>
                  <span className="text-xs text-muted-foreground">
                    NVIDIA 545.23.08 available
                  </span>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="text-center text-sm text-primary">
                  View all notifications
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Theme Toggle */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  onClick={toggleTheme}
                  aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                >
                  {theme === 'dark' ? (
                    <Sun className="h-4 w-4" aria-hidden="true" />
                  ) : (
                    <Moon className="h-4 w-4" aria-hidden="true" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                Switch to {theme === 'dark' ? 'light' : 'dark'} mode
              </TooltipContent>
            </Tooltip>

            {/* Help */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" asChild aria-label="Documentation">
                  <Link href="/docs">
                    <HelpCircle className="h-4 w-4" aria-hidden="true" />
                  </Link>
                </Button>
              </TooltipTrigger>
              <TooltipContent>Documentation</TooltipContent>
            </Tooltip>

            {/* User Menu */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="rounded-full" aria-label="User menu">
                  <div className="h-8 w-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                    <User className="h-4 w-4 text-white" aria-hidden="true" />
                  </div>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuLabel>
                  <div className="flex flex-col">
                    <span>Brad Tigani</span>
                    <span className="text-xs font-normal text-muted-foreground">
                      brad@tiganticlabz.com
                    </span>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <Link href="/settings" className="cursor-pointer">
                    <Settings className="mr-2 h-4 w-4" />
                    Settings
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/api" className="cursor-pointer">
                    <Activity className="mr-2 h-4 w-4" />
                    API Keys
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="text-destructive focus:text-destructive">
                  <LogOut className="mr-2 h-4 w-4" />
                  Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </header>
    </TooltipProvider>
  );
}
