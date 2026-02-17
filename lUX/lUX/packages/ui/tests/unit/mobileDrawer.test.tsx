import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MobileDrawer } from "@/ds/components/MobileDrawer";

describe("MobileDrawer", () => {
  const onClose = vi.fn();

  beforeEach(() => {
    onClose.mockClear();
  });

  afterEach(() => {
    document.body.style.overflow = "";
  });

  it("renders as dialog with aria-modal", () => {
    render(
      <MobileDrawer open={true} onClose={onClose} label="Navigation">
        <div>Content</div>
      </MobileDrawer>,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAttribute("aria-label", "Navigation");
  });

  it("is not visible when closed", () => {
    render(
      <MobileDrawer open={false} onClose={onClose} label="Nav">
        <div>Hidden</div>
      </MobileDrawer>,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog.className).toContain("-translate-x-full");
  });

  it("translates to visible position when open (left side)", () => {
    render(
      <MobileDrawer open={true} onClose={onClose} side="left" label="Nav">
        <div>Visible</div>
      </MobileDrawer>,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog.className).toContain("translate-x-0");
    expect(dialog.className).toContain("left-0");
  });

  it("translates to visible position when open (right side)", () => {
    render(
      <MobileDrawer open={true} onClose={onClose} side="right" label="Nav">
        <div>Visible</div>
      </MobileDrawer>,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog.className).toContain("translate-x-0");
    expect(dialog.className).toContain("right-0");
  });

  it("calls onClose when close button is clicked", () => {
    render(
      <MobileDrawer open={true} onClose={onClose} label="Navigation">
        <div>Content</div>
      </MobileDrawer>,
    );
    const closeBtn = screen.getByLabelText("Close Navigation");
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("calls onClose on Escape key", () => {
    render(
      <MobileDrawer open={true} onClose={onClose} label="Nav">
        <div>Content</div>
      </MobileDrawer>,
    );
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("calls onClose when backdrop is clicked", () => {
    render(
      <MobileDrawer open={true} onClose={onClose} label="Nav">
        <div>Content</div>
      </MobileDrawer>,
    );
    // The backdrop has aria-hidden="true" — query by that
    const backdrop = document.querySelector("[aria-hidden='true']") as HTMLElement;
    fireEvent.click(backdrop);
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("locks body scroll when open", () => {
    const { rerender } = render(
      <MobileDrawer open={true} onClose={onClose} label="Nav">
        <div>Content</div>
      </MobileDrawer>,
    );
    expect(document.body.style.overflow).toBe("hidden");

    rerender(
      <MobileDrawer open={false} onClose={onClose} label="Nav">
        <div>Content</div>
      </MobileDrawer>,
    );
    expect(document.body.style.overflow).toBe("");
  });

  it("renders children content", () => {
    render(
      <MobileDrawer open={true} onClose={onClose} label="Nav">
        <div data-testid="drawer-inner">Inside drawer</div>
      </MobileDrawer>,
    );
    expect(screen.getByTestId("drawer-inner")).toHaveTextContent("Inside drawer");
  });

  it("renders drawer label in header", () => {
    render(
      <MobileDrawer open={true} onClose={onClose} label="Navigation">
        <div>Content</div>
      </MobileDrawer>,
    );
    expect(screen.getByText("Navigation")).toBeInTheDocument();
  });
});
