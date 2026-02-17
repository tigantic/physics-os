import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import React from "react";
import { CopyField } from "@/ds/components/CopyField";

describe("CopyField", () => {
  let originalClipboard: Clipboard;

  beforeEach(() => {
    originalClipboard = navigator.clipboard;
    vi.useFakeTimers({ shouldAdvanceTime: true });
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

  it('"Copied" resets back to "Copy" after timeout', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    mockClipboard(writeText);

    render(<CopyField label="Hash" value="0xabc" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy Hash" }));
    });
    expect(screen.getByText("Copied")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(1000);
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

  it('"Failed" resets back to "Copy" after timeout', async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("Denied"));
    mockClipboard(writeText);

    render(<CopyField label="Hash" value="0xabc" />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy Hash" }));
    });
    expect(screen.getByText("Failed")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.getByText("Copy")).toBeInTheDocument();
  });
});
