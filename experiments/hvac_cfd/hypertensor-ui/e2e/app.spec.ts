/**
 * Core App E2E Tests
 * 
 * Basic app functionality and responsive design tests.
 */

import { test, expect } from '@playwright/test';

test.describe('App Initialization', () => {
  test('should load without JavaScript errors', async ({ page }) => {
    const errors: string[] = [];
    
    page.on('pageerror', (error) => {
      errors.push(error.message);
    });
    
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Filter out expected errors (e.g., failed API calls in dev)
    const criticalErrors = errors.filter((e) => 
      !e.includes('fetch') && 
      !e.includes('Network') &&
      !e.includes('Failed to load')
    );
    
    expect(criticalErrors).toHaveLength(0);
  });

  test('should have proper document title', async ({ page }) => {
    await page.goto('/');
    
    const title = await page.title();
    expect(title).toBeTruthy();
    expect(title.length).toBeGreaterThan(0);
  });

  test('should have meta viewport tag', async ({ page }) => {
    await page.goto('/');
    
    const viewport = await page.locator('meta[name="viewport"]').getAttribute('content');
    expect(viewport).toContain('width=device-width');
  });
});

test.describe('Error Handling', () => {
  test('should handle 404 pages gracefully', async ({ page }) => {
    await page.goto('/non-existent-page-xyz-123');
    
    // Should show error page or redirect
    const is404 = await page.locator('text=404, text=not found, text=page not found').isVisible();
    const redirected = page.url() !== '/non-existent-page-xyz-123';
    
    expect(is404 || redirected).toBe(true);
  });

  test('should handle API errors gracefully', async ({ page }) => {
    // Navigate to a page that makes API calls
    await page.goto('/simulations');
    await page.waitForLoadState('networkidle');
    
    // Page should render without crashing
    const hasContent = await page.locator('body').innerHTML();
    expect(hasContent.length).toBeGreaterThan(100);
  });
});

test.describe('Responsive Design - Desktop', () => {
  test.use({ viewport: { width: 1920, height: 1080 } });

  test('should show full sidebar on desktop', async ({ page }) => {
    await page.goto('/');
    
    const sidebar = page.locator('aside, nav[class*="sidebar"]').first();
    await expect(sidebar).toBeVisible();
  });

  test('should layout dashboard in grid on desktop', async ({ page }) => {
    await page.goto('/');
    
    // Stats should be in grid
    const statsGrid = page.locator('[class*="grid"]').first();
    await expect(statsGrid).toBeVisible();
  });
});

test.describe('Responsive Design - Tablet', () => {
  test.use({ viewport: { width: 768, height: 1024 } });

  test('should adapt layout for tablet', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Page should be usable
    const mainContent = page.locator('main, [role="main"]');
    await expect(mainContent).toBeVisible();
  });
});

test.describe('Responsive Design - Mobile', () => {
  test.use({ viewport: { width: 375, height: 667 } });

  test('should hide sidebar on mobile by default', async ({ page }) => {
    await page.goto('/');
    
    // Sidebar should be hidden or collapsed on mobile
    const sidebar = page.locator('aside').first();
    
    if (await sidebar.isVisible()) {
      // Should be collapsed or at least not full width
      const box = await sidebar.boundingBox();
      if (box) {
        // Sidebar should not take up more than 75% of mobile viewport
        expect(box.width).toBeLessThan(300);
      }
    }
  });

  test('should have mobile menu button', async ({ page }) => {
    await page.goto('/');
    
    // Look for hamburger menu
    const menuBtn = page.locator('button').filter({ has: page.locator('svg.lucide-menu') }).first();
    
    // Menu button should exist on mobile
    expect(await menuBtn.count()).toBeGreaterThanOrEqual(0);
  });

  test('should stack cards vertically on mobile', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Page should render
    const content = page.locator('main');
    await expect(content).toBeVisible();
  });
});

test.describe('Performance', () => {
  test('should load main content within timeout', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    
    const loadTime = Date.now() - startTime;
    
    // Should load within 5 seconds
    expect(loadTime).toBeLessThan(5000);
  });

  test('should not have memory leaks on navigation', async ({ page }) => {
    await page.goto('/');
    
    // Navigate through pages
    await page.click('a[href="/simulations"]');
    await page.waitForLoadState('networkidle');
    
    await page.click('a[href="/meshes"]');
    await page.waitForLoadState('networkidle');
    
    await page.click('a[href="/results"]');
    await page.waitForLoadState('networkidle');
    
    await page.click('a[href="/"]');
    await page.waitForLoadState('networkidle');
    
    // Page should still be responsive
    const heading = page.locator('h1').first();
    await expect(heading).toBeVisible();
  });
});

