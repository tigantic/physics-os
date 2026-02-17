import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { DetailDrawer } from "@/ds/components/DetailDrawer";

describe("DetailDrawer", () => {
  const onClose = vi.fn();

  beforeEach(() => {
    onClose.mockClear();
  });

  it("renders nothing when closed", () => {
    const { container } = render(
      <DetailDrawer open={false} onClose={onClose} title="Test">
        Content
      </DetailDrawer>,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders title and content when open", () => {
    render(
      <DetailDrawer open={true} onClose={onClose} title="Gate Detail" subtitle="g-001">
        <div>Detail content</div>
      </DetailDrawer>,
    );
    expect(screen.getByText("Gate Detail")).toBeInTheDocument();
    expect(screen.getByText("g-001")).toBeInTheDocument();
    expect(screen.getByText("Detail content")).toBeInTheDocument();
  });

  it("has dialog role with aria-modal", () => {
    render(
      <DetailDrawer open={true} onClose={onClose} title="Test">
        Content
      </DetailDrawer>,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAttribute("aria-label", "Test");
  });

  it("calls onClose when close button is clicked", () => {
    render(
      <DetailDrawer open={true} onClose={onClose} title="Test">
        Content
      </DetailDrawer>,
    );
    fireEvent.click(screen.getByLabelText("Close drawer"));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("calls onClose on Escape key", () => {
    render(
      <DetailDrawer open={true} onClose={onClose} title="Test">
        Content
      </DetailDrawer>,
    );
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledOnce();
  });
});
