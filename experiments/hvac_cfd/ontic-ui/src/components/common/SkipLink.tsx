/**
 * SkipLink - Accessibility Skip Navigation
 * 
 * Allows keyboard users to skip directly to main content.
 * Only visible when focused via keyboard navigation.
 */

'use client';

import { cn } from '@/lib/utils';

interface SkipLinkProps {
  targetId?: string;
  className?: string;
}

export function SkipLink({ 
  targetId = 'main-content', 
  className 
}: SkipLinkProps) {
  return (
    <a
      href={`#${targetId}`}
      className={cn(
        'skip-to-main',
        'fixed top-0 left-0 z-[100]',
        'px-4 py-2 m-3',
        'bg-primary text-primary-foreground',
        'font-medium rounded-md shadow-lg',
        'transform -translate-y-full opacity-0',
        'focus:translate-y-0 focus:opacity-100',
        'transition-all duration-200',
        'focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
        className
      )}
    >
      Skip to main content
    </a>
  );
}

export default SkipLink;
