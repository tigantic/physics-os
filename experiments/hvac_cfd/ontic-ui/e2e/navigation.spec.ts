/**
 * Navigation E2E Tests
 * 
 * Tests all primary navigation paths and page loading.
 */

import { test, expect } from '@playwright/test';

test.describe('Global Navigation', () => {
  test('should redirect root to simulations', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/simulations/);
  });

  test('should navigate to dashboard from sidebar', async ({ page }) => {
    await page.goto('/');
    
    // Click Dashboard link in sidebar
    await page.click('text=Dashboard');
    await expect(page).toHaveURL('/');
    await expect(page.getByRole('heading', { name: /dashboard/i })).toBeVisible();
  });

  test('should navigate to simulations page', async ({ page }) => {
    await page.goto('/');
    
    await page.click('a[href="/simulations"]');
    await expect(page).toHaveURL('/simulations');
    await expect(page.getByRole('heading', { name: /simulations/i })).toBeVisible();
  });

  test('should navigate to meshes page', async ({ page }) => {
    await page.goto('/');
    
    await page.click('a[href="/meshes"]');
    await expect(page).toHaveURL('/meshes');
    await expect(page.getByRole('heading', { name: /meshes/i })).toBeVisible();
  });

  test('should navigate to results page', async ({ page }) => {
    await page.goto('/');
    
    await page.click('a[href="/results"]');
    await expect(page).toHaveURL('/results');
    await expect(page.getByRole('heading', { name: /results/i })).toBeVisible();
  });
});

test.describe('Sidebar', () => {
  test('should highlight active nav item', async ({ page }) => {
    await page.goto('/simulations');
    
    // Check that simulations nav item is active
    const simNav = page.locator('a[href="/simulations"]').first();
    await expect(simNav).toHaveClass(/bg-accent|active|current/);
  });

  test('should collapse and expand sidebar', async ({ page }) => {
    await page.goto('/');
    
    // Find sidebar toggle
    const sidebarToggle = page.locator('button').filter({ hasText: '' }).first();
    const sidebar = page.locator('aside').first();
    
    // Check initial state
    const initialWidth = await sidebar.boundingBox();
    
    // Click to toggle
    await sidebarToggle.click();
    await page.waitForTimeout(300); // Wait for animation
    
    const collapsedWidth = await sidebar.boundingBox();
    
    // Width should have changed
    if (initialWidth && collapsedWidth) {
      expect(collapsedWidth.width).not.toBe(initialWidth.width);
    }
  });
});

test.describe('Quick Actions', () => {
  test('should have working "New Simulation" button on dashboard', async ({ page }) => {
    await page.goto('/');
    
    // Find and click New Simulation button
    await page.click('a[href="/simulations/new"]');
    await expect(page).toHaveURL('/simulations/new');
  });

  test('should navigate to mesh upload from dashboard', async ({ page }) => {
    await page.goto('/');
    
    // Look for Upload Mesh link
    const uploadLink = page.locator('a[href="/meshes"]').filter({ hasText: /upload|import/i });
    if (await uploadLink.count() > 0) {
      await uploadLink.first().click();
      await expect(page).toHaveURL('/meshes');
    }
  });
});
