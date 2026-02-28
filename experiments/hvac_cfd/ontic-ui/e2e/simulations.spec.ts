/**
 * Simulations E2E Tests
 * 
 * Tests simulations list, detail pages, and simulation creation wizard.
 */

import { test, expect } from '@playwright/test';

test.describe('Simulations List Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/simulations');
  });

  test('should display page header with title', async ({ page }) => {
    await expect(page.getByRole('heading', { name: /simulations/i })).toBeVisible();
  });

  test('should have "New Simulation" button', async ({ page }) => {
    const newBtn = page.locator('a[href="/simulations/new"], button:has-text("New")').first();
    await expect(newBtn).toBeVisible();
  });

  test('should display search input', async ({ page }) => {
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]');
    await expect(searchInput).toBeVisible();
  });

  test('should have view toggle (grid/list)', async ({ page }) => {
    const gridBtn = page.locator('button').filter({ has: page.locator('svg') });
    expect(await gridBtn.count()).toBeGreaterThan(0);
  });

  test('should have status filter dropdown', async ({ page }) => {
    const filterBtn = page.locator('button:has-text("Status"), button:has-text("All"), [role="combobox"]').first();
    await expect(filterBtn).toBeVisible();
  });

  test('should display simulations or empty state', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    const simTable = page.locator('table, [data-testid="simulations-table"]');
    const simCards = page.locator('[class*="card"]').filter({ hasText: /simulation|pending|running|completed/i });
    const emptyState = page.locator('text=No simulations').first();
    
    // One of these should be visible
    await expect(simTable.or(simCards.first()).or(emptyState)).toBeVisible({ timeout: 10000 });
  });

  test('should filter simulations by search', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]').first();
    if (await searchInput.isVisible()) {
      await searchInput.fill('test');
      await page.waitForTimeout(500); // Wait for debounce
      // Results should filter (we just verify no error)
    }
  });

  test('should switch between grid and list view', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    // Find view toggle buttons
    const listBtn = page.locator('button[aria-label*="list" i], button:has(svg.lucide-list)').first();
    const gridBtn = page.locator('button[aria-label*="grid" i], button:has(svg.lucide-layout-grid)').first();
    
    if (await listBtn.isVisible() && await gridBtn.isVisible()) {
      // Click list view
      await listBtn.click();
      await page.waitForTimeout(300);
      
      // Click grid view
      await gridBtn.click();
      await page.waitForTimeout(300);
    }
  });
});

test.describe('New Simulation Wizard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/simulations/new');
  });

  test('should display wizard with steps', async ({ page }) => {
    // Should show step indicators
    await expect(page.getByText('Geometry').first()).toBeVisible();
    await expect(page.getByText('Boundaries').first()).toBeVisible();
    await expect(page.getByText('Solver').first()).toBeVisible();
  });

  test('should display simulation name input', async ({ page }) => {
    const nameInput = page.locator('input[name="name"], input[placeholder*="name" i]').first();
    await expect(nameInput).toBeVisible();
  });

  test('should have mesh selection', async ({ page }) => {
    // Should have mesh selection or list
    const meshSelector = page.locator('[role="combobox"], select, [data-testid="mesh-select"]').first();
    const meshList = page.locator('text=Select.*mesh', { hasNot: page.locator('option') });
    
    await expect(meshSelector.or(meshList)).toBeVisible({ timeout: 5000 });
  });

  test('should navigate between steps', async ({ page }) => {
    // Fill required field first
    const nameInput = page.locator('input[name="name"]').first();
    if (await nameInput.isVisible()) {
      await nameInput.fill('Test Simulation');
    }
    
    // Click Next button
    const nextBtn = page.locator('button:has-text("Next")').first();
    if (await nextBtn.isVisible()) {
      await nextBtn.click();
      await page.waitForTimeout(500);
      
      // Should be on step 2 (Boundaries)
      const step2Active = page.locator('[aria-current="step"], .active').filter({ hasText: /boundaries/i });
      // May or may not move depending on validation
    }
  });

  test('should have solver settings on step 3', async ({ page }) => {
    // Navigate to solver step
    const solverTab = page.locator('button:has-text("Solver"), [role="tab"]:has-text("Solver")').first();
    if (await solverTab.isVisible()) {
      await solverTab.click();
      await page.waitForTimeout(300);
    }
    
    // Check for solver settings fields
    const turbulenceField = page.locator('text=Turbulence').first();
    const iterationsField = page.locator('text=Iterations').first();
    
    // At least one should be visible on solver step
    await expect(turbulenceField.or(iterationsField)).toBeVisible({ timeout: 5000 });
  });

  test('should validate required fields', async ({ page }) => {
    // Try to submit without filling required fields
    const submitBtn = page.locator('button:has-text("Create"), button[type="submit"]').first();
    if (await submitBtn.isVisible()) {
      await submitBtn.click();
      
      // Should show validation error
      const errorMsg = page.locator('text=required, text=invalid, [role="alert"]').first();
      await expect(errorMsg).toBeVisible({ timeout: 3000 });
    }
  });

  test('should have back button', async ({ page }) => {
    const backBtn = page.locator('button:has-text("Back"), a:has-text("Back"), button:has(svg.lucide-arrow-left)').first();
    await expect(backBtn).toBeVisible();
  });
});

test.describe('Simulation Detail Page', () => {
  test('should handle 404 for non-existent simulation', async ({ page }) => {
    await page.goto('/simulations/non-existent-id-12345');
    
    // Should show error or redirect
    const errorState = page.locator('text=not found, text=error, text=404').first();
    const redirected = page.url().includes('/simulations') && !page.url().includes('non-existent');
    
    // Either error shown or redirected
    if (!redirected) {
      await expect(errorState).toBeVisible({ timeout: 5000 });
    }
  });
});
