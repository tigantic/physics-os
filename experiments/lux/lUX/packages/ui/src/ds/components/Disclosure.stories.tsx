import type { Meta, StoryObj } from "@storybook/react";
import { within, userEvent, expect } from "@storybook/test";
import { Disclosure } from "./Disclosure";

const meta: Meta<typeof Disclosure> = {
  title: "DS/Disclosure",
  component: Disclosure,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof Disclosure>;

export const Default: Story = {
  args: {
    title: "Advanced Details",
    children: "Hidden content revealed on toggle.",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const toggle = canvas.getByRole("button", { name: /show/i });

    // Initially closed — content not visible
    expect(canvas.queryByText("Hidden content revealed on toggle.")).toBeNull();

    // Open disclosure
    await userEvent.click(toggle);
    expect(canvas.getByText("Hidden content revealed on toggle.")).toBeVisible();
    expect(toggle).toHaveAttribute("aria-expanded", "true");

    // Close disclosure
    await userEvent.click(canvas.getByRole("button", { name: /hide/i }));
    expect(canvas.queryByText("Hidden content revealed on toggle.")).toBeNull();
  },
};

export const WithRichContent: Story = {
  render: () => (
    <Disclosure title="Attestation Metadata">
      <dl className="grid grid-cols-2 gap-2 text-xs">
        <dt className="font-medium">Pipeline</dt>
        <dd>Trustless Physics v10</dd>
        <dt className="font-medium">Timestamp</dt>
        <dd>2026-02-16T12:00:00Z</dd>
        <dt className="font-medium">Hash</dt>
        <dd className="truncate font-mono">sha256:a1b2c3d4e5f6</dd>
      </dl>
    </Disclosure>
  ),
};

export const MultipleStacked: Story = {
  render: () => (
    <div className="flex max-w-md flex-col gap-2">
      <Disclosure title="Conservation Laws">
        <p className="text-xs">All conservation laws satisfied within tolerance.</p>
      </Disclosure>
      <Disclosure title="Numerical Stability">
        <p className="text-xs">CFL condition maintained throughout simulation.</p>
      </Disclosure>
      <Disclosure title="Raw Diagnostics">
        <pre className="text-xs">{'{ "residual": 1.2e-12, "iterations": 4200 }'}</pre>
      </Disclosure>
    </div>
  ),
};

export const LongTitle: Story = {
  args: {
    title: "Extended Conservation Verification Diagnostics and Detailed Breakdown of Numerical Results",
    children: "Content under a very long title.",
  },
};

export const EmptyContent: Story = {
  args: {
    title: "Empty Section",
    children: null,
  },
};

export const NestedDisclosures: Story = {
  render: () => (
    <Disclosure title="Outer Disclosure">
      <Disclosure title="Inner Disclosure">
        <p className="text-xs">Deeply nested content.</p>
      </Disclosure>
    </Disclosure>
  ),
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // Open outer disclosure
    const outerToggle = canvas.getByRole("button", { name: /show/i });
    await userEvent.click(outerToggle);

    // Inner disclosure should now be visible
    const innerToggle = canvas.getAllByRole("button", { name: /show/i })[0];
    expect(innerToggle).toBeVisible();

    // Open inner disclosure
    await userEvent.click(innerToggle);
    expect(canvas.getByText("Deeply nested content.")).toBeVisible();

    // Escape inside inner disclosure closes it
    await userEvent.keyboard("{Escape}");

    // Outer still open — its region should be visible
    expect(canvas.getByRole("button", { name: /show/i })).toBeVisible();
  },
};
