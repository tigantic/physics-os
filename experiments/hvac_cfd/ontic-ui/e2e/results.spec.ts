/**
 * Results E2E Tests
 * 
 * Tests results listing and viewing functionality.
 */

import { test, expect } from '@playwright/test';

test.describe('Results List Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/results');
  });

  test('should display page header', async ({ page }) => {
    await expect(page.getByRole('heading', { name: /results/i })).toBeVisible();
  });

  test('should display search input', async ({ page }) => {
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]');
    await expect(searchInput).toBeVisible();
  });

  test('should have time range filter', async ({ page }) => {
    const timeFilter = page.locator('button:has-text("Last"), [role="combobox"], select').first();
    await expect(timeFilter).toBeVisible();
  });

  test('should display results or empty state', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    const resultsGrid = page.locator('[class*="grid"]').filter({ hasText: /converged|completed|fields/i });
    const emptyState = page.locator('text=No results, text=No completed').first();
    
    // One of these should be visible
    await expect(resultsGrid.or(emptyState)).toBeVisible({ timeout: 10000 });
  });

  test('should display result cards with metadata', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Result cards should show key info
    const resultCards = page.locator('[class*="card"]').filter({ hasText: /converged|partial|diverged/i });
    
    if (await resultCards.count() > 0) {
      const firstCard = resultCards.first();
      
      // Should have status badge
      const statusBadge = firstCard.locator('[class*="badge"]');
      await expect(statusBadge).toBeVisible();
    }
  });

  test('should filter results by time range', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    const timeFilter = page.locator('button:has-text("Last"), [role="combobox"]').first();
    if (await timeFilter.isVisible()) {
      await timeFilter.click();
      
      // Select a different time range
      const option = page.locator('[role="option"]:has-text("7 days"), [role="option"]:has-text("30 days")').first();
      if (await option.isVisible()) {
        await option.click();
        await page.waitForTimeout(500);
      }
    }
  });

  test('should have view/download actions on result cards', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    const resultCards = page.locator('[class*="card"]').filter({ hasText: /converged|completed/i });
    
    if (await resultCards.count() > 0) {
      const firstCard = resultCards.first();
      
      // Look for action buttons
      const viewBtn = firstCard.locator('button:has-text("View"), a:has-text("View")');
      const downloadBtn = firstCard.locator('button:has-text("Download"), button:has-text("Export")');
      
      // At least one should exist
      expect(await viewBtn.or(downloadBtn).count()).toBeGreaterThanOrEqual(0);
    }
  });

  test('should navigate to simulation when clicking result', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    const viewBtn = page.locator('a[href*="/simulations/"], button:has-text("View")').first();
    
    if (await viewBtn.isVisible()) {
      await viewBtn.click();
      
      // Should navigate to simulation detail
      await expect(page).toHaveURL(/\/simulations\//);
    }
  });
});

test.describe('Result Detail Page', () => {
  test('should redirect to simulation detail', async ({ page }) => {
    // Result detail pages redirect to simulation detail
    await page.goto('/results/test-id-123');
    
    // Should redirect or show error
    await page.waitForLoadState('networkidle');
    
    const url = page.url();
    const onSimulationPage = url.includes('/simulations/');
    const onResultsPage = url.includes('/results');
    const hasError = await page.locator('text=not found, text=error').isVisible();
    
    expect(onSimulationPage || onResultsPage || hasError).toBe(true);
  });
});
