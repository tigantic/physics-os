import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { HamburgerButton } from "@/features/proof/HamburgerButton";
import { DrawerToggleContext } from "@/features/proof/ResponsiveShell";

describe("HamburgerButton", () => {
  it("renders nothing without context", () => {
    const { container } = render(<HamburgerButton />);
    expect(container.innerHTML).toBe("");
  });

  it("renders menu button when context provides toggle", () => {
    const toggle = vi.fn();
    render(
      <DrawerToggleContext.Provider value={toggle}>
        <HamburgerButton />
      </DrawerToggleContext.Provider>,
    );
    const btn = screen.getByLabelText("Open navigation menu");
    expect(btn).toBeInTheDocument();
  });

  it("calls toggle when clicked", () => {
    const toggle = vi.fn();
    render(
      <DrawerToggleContext.Provider value={toggle}>
        <HamburgerButton />
      </DrawerToggleContext.Provider>,
    );
    screen.getByLabelText("Open navigation menu").click();
    expect(toggle).toHaveBeenCalledOnce();
  });

  it("has md:hidden class for desktop hiding", () => {
    const toggle = vi.fn();
    render(
      <DrawerToggleContext.Provider value={toggle}>
        <HamburgerButton />
      </DrawerToggleContext.Provider>,
    );
    const btn = screen.getByLabelText("Open navigation menu");
    expect(btn.className).toContain("md:hidden");
  });

  it("has min 40px touch target", () => {
    const toggle = vi.fn();
    render(
      <DrawerToggleContext.Provider value={toggle}>
        <HamburgerButton />
      </DrawerToggleContext.Provider>,
    );
    const btn = screen.getByLabelText("Open navigation menu");
    expect(btn.className).toContain("h-10");
    expect(btn.className).toContain("w-10");
  });
});
