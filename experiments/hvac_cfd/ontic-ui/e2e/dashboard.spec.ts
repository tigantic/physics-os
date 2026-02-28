/**
 * Dashboard/Landing Page E2E Tests
 * 
 * Tests the landing page (simulations list) since root "/" redirects to "/simulations".
 * Also tests the dashboard page at "/dashboard" if accessed directly.
 */

import { test, expect } from '@playwright/test';

test.describe('Landing Page (Root Redirect)', () => {
  test('should redirect root to simulations', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/\/simulations/);
  });

  test('should display Simulations page header after redirect', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: /simulations/i })).toBeVisible();
  });

  test('should display New Simulation button', async ({ page }) => {
    await page.goto('/');
    const newSimBtn = page.locator('text=New Simulation').first();
    await expect(newSimBtn).toBeVisible({ timeout: 10000 });
  });
});

test.describe('Dashboard Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('should display Dashboard header', async ({ page }) => {
    await expect(page.getByRole('heading', { name: /dashboard/i })).toBeVisible();
  });

  test('should display stat cards', async ({ page }) => {
    // Wait for stats to load - use first() to handle multiple matches
    await expect(page.getByText('Active Simulations').first()).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Completed Today').first()).toBeVisible();
    // Use role-based selector for stat cards to be more specific
    await expect(page.locator('[class*="stat"], [class*="card"]').filter({ hasText: 'Meshes' }).first()).toBeVisible();
    await expect(page.getByText('GPU Utilization').first()).toBeVisible();
  });

  test('should display active simulations section', async ({ page }) => {
    await expect(page.getByText('Active Simulations').first()).toBeVisible({ timeout: 10000 });
    
    // Should have a "New Simulation" button
    const newSimBtn = page.locator('text=New Simulation').first();
    await expect(newSimBtn).toBeVisible();
  });

  test('should display recent activity section', async ({ page }) => {
    await expect(page.getByText('Recent Activity').first()).toBeVisible({ timeout: 10000 });
  });

  test('should display quick actions section', async ({ page }) => {
    await expect(page.getByText('Quick Actions').first()).toBeVisible({ timeout: 10000 });
  });

  test('should handle empty or populated state', async ({ page }) => {
    // Either empty state or simulation cards should be visible
    const emptyState = page.locator('text=No active simulations');
    const simContent = page.locator('[class*="simulation"], a[href^="/simulations/sim-"]').first();
    
    await expect(emptyState.or(simContent)).toBeVisible({ timeout: 10000 });
  });
});

test.describe('Dashboard Interactions', () => {
  test('should navigate to new simulation from dashboard', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Click "New Simulation" button
    await page.click('text=New Simulation');
    await expect(page).toHaveURL('/simulations/new');
  });

  test('should navigate to simulation detail when clicking simulation link', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Wait for content to load
    await page.waitForLoadState('networkidle');
    
    // Try to click a simulation link if one exists
    const simLink = page.locator('a[href^="/simulations/sim-"]').first();
    if (await simLink.isVisible()) {
      await simLink.click();
      await expect(page).toHaveURL(/\/simulations\/sim-/);
    }
  });
});
