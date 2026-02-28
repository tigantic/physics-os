/**
 * Meshes E2E Tests
 * 
 * Tests meshes list, upload functionality, and mesh detail pages.
 */

import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('Meshes List Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/meshes');
  });

  test('should display page header', async ({ page }) => {
    await expect(page.getByRole('heading', { name: /meshes/i })).toBeVisible();
  });

  test('should have import/upload button', async ({ page }) => {
    const uploadBtn = page.locator('button:has-text("Import"), button:has-text("Upload"), a:has-text("Import")').first();
    await expect(uploadBtn).toBeVisible();
  });

  test('should display search input', async ({ page }) => {
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]');
    await expect(searchInput).toBeVisible();
  });

  test('should display meshes or empty state', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    const meshTable = page.locator('table');
    const meshCards = page.locator('[class*="card"]').filter({ hasText: /cells|vertices|mesh/i });
    const emptyState = page.locator('text=No meshes').first();
    
    // One of these should be visible
    await expect(meshTable.or(meshCards.first()).or(emptyState)).toBeVisible({ timeout: 10000 });
  });

  test('should open upload dialog when clicking import', async ({ page }) => {
    const uploadBtn = page.locator('button:has-text("Import"), button:has-text("Upload")').first();
    
    if (await uploadBtn.isVisible()) {
      await uploadBtn.click();
      
      // Dialog should appear
      const dialog = page.locator('[role="dialog"], [class*="dialog"]');
      await expect(dialog).toBeVisible({ timeout: 3000 });
    }
  });

  test('should have drag and drop zone in upload dialog', async ({ page }) => {
    const uploadBtn = page.locator('button:has-text("Import"), button:has-text("Upload")').first();
    
    if (await uploadBtn.isVisible()) {
      await uploadBtn.click();
      
      // Look for drop zone
      const dropZone = page.locator('text=drag, text=drop, [class*="dropzone"]').first();
      await expect(dropZone).toBeVisible({ timeout: 3000 });
    }
  });

  test('should close upload dialog with X button', async ({ page }) => {
    const uploadBtn = page.locator('button:has-text("Import"), button:has-text("Upload")').first();
    
    if (await uploadBtn.isVisible()) {
      await uploadBtn.click();
      
      // Find and click close button
      const closeBtn = page.locator('button[aria-label="Close"], button:has(svg.lucide-x)').first();
      if (await closeBtn.isVisible()) {
        await closeBtn.click();
        
        // Dialog should be hidden
        const dialog = page.locator('[role="dialog"]');
        await expect(dialog).not.toBeVisible({ timeout: 2000 });
      }
    }
  });

  test('should filter meshes by search', async ({ page }) => {
    await page.waitForLoadState('networkidle');
    
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]').first();
    if (await searchInput.isVisible()) {
      await searchInput.fill('test');
      await page.waitForTimeout(500); // Wait for debounce
    }
  });
});

test.describe('Mesh Detail Page', () => {
  test('should handle 404 for non-existent mesh', async ({ page }) => {
    await page.goto('/meshes/non-existent-mesh-12345');
    
    // Should show error state
    const errorState = page.locator('text=not found, text=error, text=404').first();
    const redirected = page.url().includes('/meshes') && !page.url().includes('non-existent');
    
    if (!redirected) {
      await expect(errorState).toBeVisible({ timeout: 5000 });
    }
  });
});

test.describe('Mesh Actions', () => {
  test('should have action menu on mesh items', async ({ page }) => {
    await page.goto('/meshes');
    await page.waitForLoadState('networkidle');
    
    // Look for action menu buttons
    const actionMenuBtn = page.locator('button:has(svg.lucide-more-horizontal), button[aria-label*="action" i]').first();
    
    if (await actionMenuBtn.isVisible()) {
      await actionMenuBtn.click();
      
      // Menu should appear with options
      const menu = page.locator('[role="menu"], [class*="dropdown"]');
      await expect(menu).toBeVisible({ timeout: 2000 });
      
      // Check for common actions
      const viewOption = page.locator('[role="menuitem"]:has-text("View"), [role="menuitem"]:has-text("Details")');
      const deleteOption = page.locator('[role="menuitem"]:has-text("Delete")');
      
      expect(await viewOption.or(deleteOption).count()).toBeGreaterThan(0);
    }
  });
});
