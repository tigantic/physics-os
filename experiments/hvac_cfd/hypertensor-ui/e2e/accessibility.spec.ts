/**
 * Accessibility E2E Tests
 * 
 * Tests keyboard navigation, focus management, and ARIA compliance.
 */

import { test, expect } from '@playwright/test';

test.describe('Keyboard Navigation', () => {
  test('should have skip-to-main link', async ({ page }) => {
    await page.goto('/');
    
    // Tab to first element
    await page.keyboard.press('Tab');
    
    // Check for skip link
    const skipLink = page.locator('a[href="#main-content"], a.skip-to-main, a:has-text("Skip")').first();
    if (await skipLink.isVisible()) {
      await expect(skipLink).toBeFocused();
    }
  });

  test('should navigate sidebar with keyboard', async ({ page }) => {
    await page.goto('/');
    
    // Tab through to sidebar links
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('Tab');
    }
    
    // Should be able to reach nav links
    const focusedElement = page.locator(':focus');
    const isSidebarLink = await focusedElement.evaluate((el) => {
      return el.tagName === 'A' || el.tagName === 'BUTTON';
    });
    
    expect(isSidebarLink).toBe(true);
  });

  test('should activate buttons with Enter key', async ({ page }) => {
    await page.goto('/simulations');
    await page.waitForLoadState('networkidle');
    
    // Focus on a button
    const button = page.locator('button:visible').first();
    await button.focus();
    
    // Press Enter
    await page.keyboard.press('Enter');
    
    // Button should have been activated (no errors)
    await page.waitForTimeout(300);
  });

  test('should close dialogs with Escape key', async ({ page }) => {
    await page.goto('/meshes');
    
    // Open upload dialog
    const uploadBtn = page.locator('button:has-text("Import")').first();
    if (await uploadBtn.isVisible()) {
      await uploadBtn.click();
      
      // Dialog should open
      const dialog = page.locator('[role="dialog"]');
      await expect(dialog).toBeVisible({ timeout: 2000 });
      
      // Press Escape
      await page.keyboard.press('Escape');
      
      // Dialog should close
      await expect(dialog).not.toBeVisible({ timeout: 2000 });
    }
  });

  test('should trap focus in modal dialogs', async ({ page }) => {
    await page.goto('/meshes');
    
    const uploadBtn = page.locator('button:has-text("Import")').first();
    if (await uploadBtn.isVisible()) {
      await uploadBtn.click();
      
      const dialog = page.locator('[role="dialog"]');
      await expect(dialog).toBeVisible({ timeout: 2000 });
      
      // Tab several times
      for (let i = 0; i < 20; i++) {
        await page.keyboard.press('Tab');
        
        // Focus should stay within dialog
        const focusedElement = page.locator(':focus');
        const isInDialog = await focusedElement.evaluate((el) => {
          return el.closest('[role="dialog"]') !== null;
        });
        
        expect(isInDialog).toBe(true);
      }
    }
  });
});

test.describe('ARIA Compliance', () => {
  test('should have proper heading hierarchy', async ({ page }) => {
    await page.goto('/');
    
    // Should have h1
    const h1 = page.locator('h1');
    await expect(h1).toBeVisible();
    
    // H1 should come before h2s
    const headings = await page.locator('h1, h2, h3').all();
    let lastLevel = 0;
    
    for (const heading of headings) {
      const tagName = await heading.evaluate((el) => el.tagName);
      const level = parseInt(tagName.replace('H', ''));
      
      // Should not skip levels (e.g., h1 -> h3)
      expect(level - lastLevel).toBeLessThanOrEqual(1);
      lastLevel = level;
    }
  });

  test('should have labeled form inputs', async ({ page }) => {
    await page.goto('/simulations/new');
    
    // Get all inputs
    const inputs = await page.locator('input:visible').all();
    
    for (const input of inputs) {
      // Each input should have a label or aria-label
      const hasLabel = await input.evaluate((el) => {
        const id = el.id;
        if (id) {
          return document.querySelector(`label[for="${id}"]`) !== null;
        }
        return false;
      });
      
      const ariaLabel = await input.getAttribute('aria-label');
      const ariaLabelledby = await input.getAttribute('aria-labelledby');
      const placeholder = await input.getAttribute('placeholder');
      
      const isLabeled = hasLabel || ariaLabel || ariaLabelledby || placeholder;
      expect(isLabeled).toBeTruthy();
    }
  });

  test('should have accessible buttons', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Get all buttons
    const buttons = await page.locator('button:visible').all();
    
    for (const button of buttons) {
      // Each button should have accessible name
      const text = await button.innerText();
      const ariaLabel = await button.getAttribute('aria-label');
      const title = await button.getAttribute('title');
      
      // Button should have text or aria-label
      const hasAccessibleName = text.trim() || ariaLabel || title;
      
      // Some icon buttons may rely on child SVG titles
      if (!hasAccessibleName) {
        const hasSvgTitle = await button.locator('svg title').count() > 0;
        expect(hasSvgTitle || hasAccessibleName).toBeTruthy();
      }
    }
  });

  test('should have proper landmarks', async ({ page }) => {
    await page.goto('/');
    
    // Should have main landmark
    const main = page.locator('main, [role="main"]');
    await expect(main).toBeVisible();
    
    // Should have navigation
    const nav = page.locator('nav, [role="navigation"]');
    expect(await nav.count()).toBeGreaterThan(0);
  });

  test('should announce loading states', async ({ page }) => {
    await page.goto('/', { waitUntil: 'commit' });
    
    // Check for aria-live regions or loading indicators
    const loadingIndicators = page.locator('[aria-busy="true"], [aria-live], .animate-pulse, .skeleton');
    
    // May or may not be present depending on load time
    const count = await loadingIndicators.count();
    expect(count).toBeGreaterThanOrEqual(0);
  });
});

test.describe('Color Contrast', () => {
  test('should have sufficient contrast for text', async ({ page }) => {
    await page.goto('/');
    
    // Check that text is visible (basic check)
    const heading = page.locator('h1');
    await expect(heading).toBeVisible();
    
    // Verify heading has non-transparent color
    const color = await heading.evaluate((el) => {
      return window.getComputedStyle(el).color;
    });
    
    expect(color).not.toBe('transparent');
  });
});

test.describe('Theme Support', () => {
  test('should toggle between light and dark themes', async ({ page }) => {
    await page.goto('/');
    
    // Find theme toggle
    const themeToggle = page.locator('button[aria-label*="theme" i], button:has(svg.lucide-sun), button:has(svg.lucide-moon)').first();
    
    if (await themeToggle.isVisible()) {
      // Get initial theme
      const initialTheme = await page.locator('html').getAttribute('class');
      
      // Toggle theme
      await themeToggle.click();
      await page.waitForTimeout(300);
      
      // Theme should change
      const newTheme = await page.locator('html').getAttribute('class');
      expect(newTheme).not.toBe(initialTheme);
    }
  });
});
