'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui';
import {
  LayoutDashboard,
  Users,
  Settings,
  FileText,
  BarChart3,
  Menu,
  X,
  ChevronLeft,
  Moon,
  Sun,
  LogOut,
} from 'lucide-react';
import { useTheme } from 'next-themes';
import { useIsMobile } from '@/hooks';

// ============================================
// NAVIGATION ITEMS
// ============================================

const navItems = [
  {
    title: 'Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
  },
  {
    title: 'Analytics',
    href: '/dashboard/analytics',
    icon: BarChart3,
  },
  {
    title: 'Users',
    href: '/dashboard/users',
    icon: Users,
  },
  {
    title: 'Documents',
    href: '/dashboard/documents',
    icon: FileText,
  },
  {
    title: 'Settings',
    href: '/dashboard/settings',
    icon: Settings,
  },
];

// ============================================
// DASHBOARD LAYOUT
// ============================================

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const isMobile = useIsMobile();

  const toggleSidebar = () => setSidebarOpen(!sidebarOpen);
  const toggleMobileMenu = () => setMobileMenuOpen(!mobileMenuOpen);

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 flex flex-col border-r bg-card transition-all duration-300',
          sidebarOpen ? 'w-64' : 'w-16',
          isMobile && !mobileMenuOpen && '-translate-x-full',
          isMobile && mobileMenuOpen && 'w-64 translate-x-0'
        )}
      >
        {/* Logo */}
        <div className="flex h-16 items-center justify-between border-b px-4">
          {(sidebarOpen || mobileMenuOpen) && (
            <Link href="/" className="text-xl font-bold">
              YourApp
            </Link>
          )}
          {!isMobile && (
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={toggleSidebar}
              className={cn(!sidebarOpen && 'mx-auto')}
            >
              <ChevronLeft
                className={cn(
                  'h-4 w-4 transition-transform',
                  !sidebarOpen && 'rotate-180'
                )}
              />
            </Button>
          )}
          {isMobile && (
            <Button variant="ghost" size="icon-sm" onClick={toggleMobileMenu}>
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 p-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => isMobile && setMobileMenuOpen(false)}
                className={cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
                  !sidebarOpen && !mobileMenuOpen && 'justify-center'
                )}
              >
                <item.icon className="h-5 w-5 shrink-0" />
                {(sidebarOpen || mobileMenuOpen) && <span>{item.title}</span>}
              </Link>
            );
          })}
        </nav>

        {/* Bottom Actions */}
        <div className="border-t p-2">
          <Button
            variant="ghost"
            size={sidebarOpen || mobileMenuOpen ? 'default' : 'icon'}
            className={cn(
              'w-full',
              (sidebarOpen || mobileMenuOpen) && 'justify-start'
            )}
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            {(sidebarOpen || mobileMenuOpen) && (
              <span className="ml-3">Toggle theme</span>
            )}
          </Button>
          <Button
            variant="ghost"
            size={sidebarOpen || mobileMenuOpen ? 'default' : 'icon'}
            className={cn(
              'w-full text-destructive hover:bg-destructive/10 hover:text-destructive',
              (sidebarOpen || mobileMenuOpen) && 'justify-start'
            )}
          >
            <LogOut className="h-5 w-5" />
            {(sidebarOpen || mobileMenuOpen) && (
              <span className="ml-3">Logout</span>
            )}
          </Button>
        </div>
      </aside>

      {/* Main Content */}
      <div
        className={cn(
          'flex flex-col transition-all duration-300',
          !isMobile && (sidebarOpen ? 'lg:pl-64' : 'lg:pl-16')
        )}
      >
        {/* Top Bar */}
        <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b bg-background px-4 lg:px-6">
          {isMobile && (
            <Button variant="ghost" size="icon" onClick={toggleMobileMenu}>
              <Menu className="h-5 w-5" />
            </Button>
          )}
          <div className="flex-1" />
          {/* Add header actions here (search, notifications, user menu) */}
        </header>

        {/* Page Content */}
        <main id="main-content" className="flex-1 p-4 lg:p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
