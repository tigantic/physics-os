import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ResponsiveShell, DrawerToggleContext, useDrawerToggle } from "@/features/proof/ResponsiveShell";

// Mock MobileDrawer to avoid complex DOM
vi.mock("@/ds/components/MobileDrawer", () => ({
  MobileDrawer: ({
    open,
    onClose,
    label,
    children,
  }: {
    open: boolean;
    onClose: () => void;
    label: string;
    children: React.ReactNode;
  }) =>
    open ? (
      <div data-testid="mobile-drawer" data-label={label}>
        <button onClick={onClose}>Close</button>
        {children}
      </div>
    ) : null,
}));

function TestToggleButton() {
  const toggle = useDrawerToggle();
  return toggle ? (
    <button data-testid="hamburger" onClick={toggle}>
      Menu
    </button>
  ) : null;
}

describe("ResponsiveShell", () => {
  it("renders header, left rail, center content, and right rail", () => {
    render(
      <ResponsiveShell
        header={<div data-testid="header">Header</div>}
        leftRail={<div data-testid="left-rail">Left</div>}
        rightRail={<div data-testid="right-rail">Right</div>}
      >
        <div data-testid="center">Center</div>
      </ResponsiveShell>,
    );
    expect(screen.getByTestId("header")).toBeInTheDocument();
    expect(screen.getByTestId("center")).toBeInTheDocument();
    // LeftRail appears in both inline (hidden md:block) and in the drawer content
    // RightRail appears in both the lg: inline and the mobile collapsible section
    const leftRails = screen.getAllByTestId("left-rail");
    expect(leftRails.length).toBeGreaterThanOrEqual(1);
    const rightRails = screen.getAllByTestId("right-rail");
    expect(rightRails.length).toBeGreaterThanOrEqual(1);
  });

  it("mobile right rail toggles on button click", () => {
    render(
      <ResponsiveShell
        header={<div>Header</div>}
        leftRail={<div>Left</div>}
        rightRail={<div data-testid="right-content">Integrity</div>}
      >
        <div>Center</div>
      </ResponsiveShell>,
    );
    const expandBtn = screen.getByRole("button", { name: /integrity details/i });
    expect(expandBtn).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(expandBtn);
    expect(expandBtn).toHaveAttribute("aria-expanded", "true");
  });

  it("opens mobile drawer via context callback", () => {
    render(
      <ResponsiveShell
        header={<TestToggleButton />}
        leftRail={<div data-testid="drawer-left">Left</div>}
        rightRail={<div>Right</div>}
      >
        <div>Center</div>
      </ResponsiveShell>,
    );
    // Drawer should not be visible initially
    expect(screen.queryByTestId("mobile-drawer")).not.toBeInTheDocument();

    // Click hamburger to open drawer
    fireEvent.click(screen.getByTestId("hamburger"));
    expect(screen.getByTestId("mobile-drawer")).toBeInTheDocument();
    expect(screen.getByTestId("mobile-drawer")).toHaveAttribute("data-label", "Navigation");
  });

  it("closes mobile drawer via close button", () => {
    render(
      <ResponsiveShell header={<TestToggleButton />} leftRail={<div>Left</div>} rightRail={<div>Right</div>}>
        <div>Center</div>
      </ResponsiveShell>,
    );
    fireEvent.click(screen.getByTestId("hamburger"));
    expect(screen.getByTestId("mobile-drawer")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Close"));
    expect(screen.queryByTestId("mobile-drawer")).not.toBeInTheDocument();
  });
});

describe("DrawerToggleContext", () => {
  it("provides null when no provider", () => {
    function Consumer() {
      const toggle = useDrawerToggle();
      return <div data-testid="val">{toggle === null ? "null" : "fn"}</div>;
    }
    render(<Consumer />);
    expect(screen.getByTestId("val")).toHaveTextContent("null");
  });

  it("provides function when provider present", () => {
    const fn = vi.fn();
    function Consumer() {
      const toggle = useDrawerToggle();
      return <div data-testid="val">{typeof toggle === "function" ? "fn" : "null"}</div>;
    }
    render(
      <DrawerToggleContext.Provider value={fn}>
        <Consumer />
      </DrawerToggleContext.Provider>,
    );
    expect(screen.getByTestId("val")).toHaveTextContent("fn");
  });
});
