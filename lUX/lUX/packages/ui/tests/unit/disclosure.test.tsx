import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { Disclosure } from "@/ds/components/Disclosure";

describe("Disclosure", () => {
  it("renders the title", () => {
    render(<Disclosure title="Details">Content</Disclosure>);
    expect(screen.getByText("Details")).toBeInTheDocument();
  });

  it("hides children by default", () => {
    render(
      <Disclosure title="Details">
        <p>Secret content</p>
      </Disclosure>,
    );
    expect(screen.queryByText("Secret content")).not.toBeInTheDocument();
  });

  it('shows a "Show" toggle button', () => {
    render(<Disclosure title="Details">Content</Disclosure>);
    expect(screen.getByRole("button", { name: /show/i })).toBeInTheDocument();
  });

  it("shows children when toggle is clicked", async () => {
    const user = userEvent.setup();
    render(
      <Disclosure title="Details">
        <p>Secret content</p>
      </Disclosure>,
    );

    await user.click(screen.getByRole("button", { name: /show/i }));
    expect(screen.getByText("Secret content")).toBeInTheDocument();
  });

  it('button text changes to "Hide" when open', async () => {
    const user = userEvent.setup();
    render(<Disclosure title="Details">Content</Disclosure>);

    const btn = screen.getByRole("button", { name: /show/i });
    await user.click(btn);
    expect(btn).toHaveTextContent("Hide");
  });

  it("hides children again on second click", async () => {
    const user = userEvent.setup();
    render(
      <Disclosure title="Details">
        <p>Secret content</p>
      </Disclosure>,
    );

    const btn = screen.getByRole("button", { name: /show/i });
    await user.click(btn);
    expect(screen.getByText("Secret content")).toBeInTheDocument();

    await user.click(btn);
    expect(screen.queryByText("Secret content")).not.toBeInTheDocument();
  });

  it("aria-expanded toggles with open state", async () => {
    const user = userEvent.setup();
    render(<Disclosure title="Details">Content</Disclosure>);

    const btn = screen.getByRole("button", { name: /show/i });
    expect(btn).toHaveAttribute("aria-expanded", "false");

    await user.click(btn);
    expect(btn).toHaveAttribute("aria-expanded", "true");

    await user.click(btn);
    expect(btn).toHaveAttribute("aria-expanded", "false");
  });

  it("aria-controls references the panel id", async () => {
    const user = userEvent.setup();
    render(
      <Disclosure title="Details">
        <p>Inner</p>
      </Disclosure>,
    );

    const btn = screen.getByRole("button", { name: /show/i });
    const controlsId = btn.getAttribute("aria-controls");
    expect(controlsId).toBeTruthy();

    await user.click(btn);
    const panel = document.getElementById(controlsId!);
    expect(panel).toBeInTheDocument();
    expect(panel).toHaveAttribute("role", "region");
    expect(panel).toHaveAttribute("aria-label", "Details");
  });
});
