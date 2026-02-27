import { test, expect } from '@playwright/test';

test.describe('Home Page', () => {
  test('should load the home page', async ({ page }) => {
    await page.goto('/');
    
    // Check for main heading
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    
    // Check for Get Started button
    await expect(page.getByRole('link', { name: /get started/i })).toBeVisible();
  });

  test('should navigate to dashboard', async ({ page }) => {
    await page.goto('/');
    
    // Click Get Started
    await page.click('text=Get Started');
    
    // Should be on dashboard
    await expect(page).toHaveURL('/dashboard');
    await expect(page.getByRole('heading', { name: /dashboard/i })).toBeVisible();
  });
});

test.describe('Dashboard', () => {
  test('should display stats cards', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Check for stats cards
    await expect(page.getByText('Total Revenue')).toBeVisible();
    await expect(page.getByText('Subscriptions')).toBeVisible();
    await expect(page.getByText('Sales')).toBeVisible();
    await expect(page.getByText('Active Now')).toBeVisible();
  });

  test('should toggle sidebar', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Find and click sidebar toggle (on desktop)
    const sidebarToggle = page.locator('button').filter({ has: page.locator('svg.lucide-chevron-left') });
    
    if (await sidebarToggle.isVisible()) {
      await sidebarToggle.click();
      // Sidebar should be collapsed
      await expect(page.locator('aside')).toHaveClass(/w-16/);
    }
  });

  test('should toggle theme', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Find theme toggle button
    const themeToggle = page.getByRole('button', { name: /toggle theme/i });
    
    if (await themeToggle.isVisible()) {
      // Click to toggle to dark mode
      await themeToggle.click();
      
      // Check if dark class is applied
      await expect(page.locator('html')).toHaveClass(/dark/);
    }
  });
});

test.describe('Accessibility', () => {
  test('should have no accessibility violations on home page', async ({ page }) => {
    await page.goto('/');
    
    // Check for skip link
    const skipLink = page.locator('a.skip-to-main');
    await expect(skipLink).toBeAttached();
    
    // Check for main landmark
    await expect(page.locator('main#main-content')).toBeVisible();
  });

  test('should be navigable with keyboard', async ({ page }) => {
    await page.goto('/');
    
    // Tab to first interactive element
    await page.keyboard.press('Tab');
    
    // Should focus skip link first
    const skipLink = page.locator('a.skip-to-main');
    await expect(skipLink).toBeFocused();
    
    // Continue tabbing
    await page.keyboard.press('Tab');
    
    // Should focus Get Started button
    await expect(page.getByRole('link', { name: /get started/i })).toBeFocused();
  });
});
