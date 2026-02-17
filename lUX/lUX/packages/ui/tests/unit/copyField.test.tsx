import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import React from "react";
import { CopyField } from "@/ds/components/CopyField";
import { addBreadcrumb } from "@/lib/reportError";

vi.mock("@/lib/reportError", () => ({
  addBreadcrumb: vi.fn(),
  reportError: vi.fn(),
}));

describe("CopyField", () => {
  let originalClipboard: Clipboard;

  beforeEach(() => {
    originalClipboard = navigator.clipboard;
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.mocked(addBreadcrumb).mockClear();
  });

  afterEach(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: originalClipboard,
      writable: true,
      configurable: true,
    });
    vi.useRealTimers();
  });

  function mockClipboard(writeText: () => Promise<void>) {
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      writable: true,
      configurable: true,
    });
  }

  it("renders label and value", () => {
    mockClipboard(() => Promise.resolve());
    render(<CopyField label="Hash" value="0xabc123" />);
    expect(screen.getByText("Hash")).toBeInTheDocument();
    expect(screen.getByText("0xabc123")).toBeInTheDocument();
  });

  it("renders a Copy button with aria-label", () => {
    mockClipboard(() => Promise.resolve());
    render(<CopyField label="Digest" value="sha256:dead" />);
    const btn = screen.getByRole("button", { name: "Copy Digest" });
    expect(btn).toBeInTheDocument();
    expect(screen.getByText("Copy")).toBeInTheDocument();
  });

  it('shows "Copied" after successful clipboard write', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    mockClipboard(writeText);

    render(<CopyField label="Hash" value="0xabc" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy Hash" }));
    });

    expect(writeText).toHaveBeenCalledWith("0xabc");
    expect(screen.getByText("Copied")).toBeInTheDocument();
  });

  it("shows checkmark SVG icon on copy success", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    mockClipboard(writeText);

    const { container } = render(<CopyField label="Hash" value="0xabc" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy Hash" }));
    });

    const svg = container.querySelector("svg[aria-hidden='true']");
    expect(svg).toBeInTheDocument();
    expect(svg?.getAttribute("class")).toContain("animate-lux-scale-in");
  });

  it("shows X SVG icon on copy failure", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("Denied"));
    mockClipboard(writeText);

    const { container } = render(<CopyField label="Hash" value="0xabc" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy Hash" }));
    });

    const svg = container.querySelector("svg[aria-hidden='true']");
    expect(svg).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("logs breadcrumb on successful copy", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    mockClipboard(writeText);

    render(<CopyField label="Hash" value="0xabc" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy Hash" }));
    });

    expect(addBreadcrumb).toHaveBeenCalledWith("action", "Copied Hash");
  });

  it("does not log breadcrumb on failed copy", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("Denied"));
    mockClipboard(writeText);

    render(<CopyField label="Hash" value="0xabc" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy Hash" }));
    });

    expect(addBreadcrumb).not.toHaveBeenCalled();
  });

  it('"Copied" resets back to "Copy" after 1200ms', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    mockClipboard(writeText);

    render(<CopyField label="Hash" value="0xabc" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy Hash" }));
    });
    expect(screen.getByText("Copied")).toBeInTheDocument();

    // Not yet reset at 1100ms
    act(() => {
      vi.advanceTimersByTime(1100);
    });
    expect(screen.getByText("Copied")).toBeInTheDocument();

    // Reset at 1200ms
    act(() => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.getByText("Copy")).toBeInTheDocument();
  });

  it('shows "Failed" when clipboard write rejects', async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("Denied"));
    mockClipboard(writeText);

    render(<CopyField label="Hash" value="0xabc" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy Hash" }));
    });

    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it('"Failed" resets back to "Copy" after 1800ms', async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("Denied"));
    mockClipboard(writeText);

    render(<CopyField label="Hash" value="0xabc" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy Hash" }));
    });
    expect(screen.getByText("Failed")).toBeInTheDocument();

    // Not yet reset at 1700ms
    act(() => {
      vi.advanceTimersByTime(1700);
    });
    expect(screen.getByText("Failed")).toBeInTheDocument();

    // Reset at 1800ms
    act(() => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.getByText("Copy")).toBeInTheDocument();
  });

  it("cleans up timer on unmount", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    mockClipboard(writeText);

    const { unmount } = render(<CopyField label="Hash" value="0xabc" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy Hash" }));
    });

    // Unmount before timeout fires — should not throw or warn
    unmount();
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    // If timer cleanup works, no errors are thrown
  });

  it("returns focus to the button after copy", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    mockClipboard(writeText);

    render(<CopyField label="Hash" value="0xabc" />);
    const btn = screen.getByRole("button", { name: "Copy Hash" });
    await act(async () => {
      fireEvent.click(btn);
    });

    expect(document.activeElement).toBe(btn);
  });
});
