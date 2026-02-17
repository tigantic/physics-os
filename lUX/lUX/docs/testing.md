# lUX — Testing Guide

## Test Strategy

lUX follows a testing pyramid with three layers:

| Layer | Tool | Count | Speed | Scope |
|-------|------|-------|-------|-------|
| Unit | Vitest | 639+ | ~4s | Functions, components, hooks |
| Integration | Vitest | Included above | — | API routes, middleware, providers |
| E2E | Playwright | 35+ | ~30s | Full browser rendering, navigation |

## Running Tests

```bash
# All unit tests
pnpm -w run test:unit

# With coverage
pnpm -w run test:coverage

# Schema validation tests only
pnpm -w run test:schema

# E2E tests (all browsers)
pnpm -w run test:e2e

# E2E single browser
cd packages/ui && npx playwright test --project=chromium

# Run a specific test file
cd packages/ui && npx vitest run tests/unit/auth.test.ts

# Watch mode (development)
cd packages/ui && npx vitest tests/unit/auth.test.ts
```

## Test File Organization

```
packages/core/tests/
├── *.test.ts              # Core unit tests (schemas, providers)
└── fixtures/              # Test data (proof packages, domain packs)

packages/ui/tests/
├── unit/                  # Vitest unit tests
│   ├── auth.test.ts       # Auth module tests
│   ├── etag.test.ts       # ETag utility tests
│   ├── apiRoutes.test.ts  # API route handler tests
│   ├── modeDial.test.tsx  # ModeDial component tests
│   ├── reactMemo.test.tsx # React.memo wrapper verification
│   └── ...
├── e2e/                   # Playwright E2E tests
│   ├── gallery.spec.ts
│   ├── modes.spec.ts
│   └── ...
└── setup.ts               # Test environment setup
```

## Writing Unit Tests

### Convention

- File: `tests/unit/<module>.test.ts(x)`
- Use `.tsx` extension when testing React components
- Use `.ts` for pure logic and API routes

### Test Structure

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";

describe("ModuleName", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does something specific", () => {
    // Arrange
    const input = createTestInput();

    // Act
    const result = functionUnderTest(input);

    // Assert
    expect(result).toBe(expectedValue);
  });
});
```

### Mocking Guidelines

```typescript
// ✅ Mock I/O boundaries
vi.mock("server-only", () => ({}));
vi.mock("@/config/provider", () => ({
  getProvider: () => Promise.resolve(mockProvider),
}));

// ✅ Mock unavailable browser APIs
vi.mock("@/features/math/MathBlock", () => ({
  MathBlock: ({ latex }: { latex: string }) => <div>{latex}</div>,
}));

// ❌ Don't mock the module under test
// ❌ Don't mock purely to avoid complexity — test the real logic
```

### Testing API Routes

API route handlers are tested by importing the `GET`/`POST` function and calling it directly:

```typescript
import { describe, it, expect, vi } from "vitest";

vi.mock("server-only", () => ({}));
vi.mock("@/config/provider", () => ({
  getProvider: () => Promise.resolve(mockProvider),
}));

describe("GET /api/packages", () => {
  it("returns packages with ETag header", async () => {
    const { GET } = await import("@/app/api/packages/route");
    mockProvider.listPackages.mockResolvedValue([{ id: "pkg-1" }]);

    const response = await GET(new Request("http://localhost/api/packages", {
      headers: { "x-request-id": "test-id" },
    }));

    expect(response.status).toBe(200);
    expect(response.headers.get("ETag")).toMatch(/^W\/"[a-f0-9]{16}"$/);
  });
});
```

### Testing React Components

```typescript
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

it("renders component with correct content", () => {
  render(<MyComponent prop="value" />);
  expect(screen.getByText("Expected Text")).toBeInTheDocument();
});

it("handles user interaction", async () => {
  const user = userEvent.setup();
  render(<MyComponent onClick={mockHandler} />);
  await user.click(screen.getByRole("button"));
  expect(mockHandler).toHaveBeenCalledOnce();
});
```

## Coverage Thresholds

Coverage thresholds are enforced in `vitest.config.ts` and CI. They only go up, never down.

| Package | Statements | Branches | Lines |
|---------|-----------|----------|-------|
| Core | 96%+ | 87%+ | 96%+ |
| UI | 70%+ | — | — |

Check current coverage:

```bash
pnpm -w run test:coverage
# Reports generated in packages/*/coverage/
```

## E2E Testing

E2E tests run against a production build in real browsers:

```bash
# Build first (required for E2E)
pnpm -w run build

# Run all E2E tests
pnpm -w run test:e2e

# Run specific browser
cd packages/ui && npx playwright test --project=chromium
cd packages/ui && npx playwright test --project=firefox
cd packages/ui && npx playwright test --project=mobile-chrome
```

### Writing E2E Tests

```typescript
import { test, expect } from "@playwright/test";

test("gallery loads proof packages", async ({ page }) => {
  await page.goto("/gallery");
  await expect(page.getByRole("heading", { name: /Physics Proof/i })).toBeVisible();
});
```

## CI Integration

Tests run automatically on every push and PR:

- **ci.yml**: Unit tests, coverage, schema tests
- **e2e.yml**: 3-browser matrix (Chromium, Firefox, Mobile Chrome)

Coverage reports are uploaded as GitHub artifacts (14-day retention).
