import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { EmptyState } from "./EmptyState";

const meta: Meta<typeof EmptyState> = {
  title: "DS/EmptyState",
  component: EmptyState,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof EmptyState>;

export const Default: Story = {
  args: {
    title: "No data available",
    description: "There are no items to display at this time.",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("No data available")).toBeVisible();
    expect(canvas.getByText("There are no items to display at this time.")).toBeVisible();
  },
};

export const TitleOnly: Story = {
  args: {
    title: "Empty",
  },
};

export const WithAction: Story = {
  args: {
    title: "No baseline selected",
    description: "Select a baseline proof to enable comparison.",
    action: <button type="button" className="rounded bg-[var(--color-accent)] px-3 py-1 text-xs text-white">Select Baseline</button>,
  },
};

export const WithIcon: Story = {
  args: {
    title: "No gates evaluated",
    description: "This proof has no gate results.",
    icon: (
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none" aria-hidden="true">
        <circle cx="24" cy="24" r="20" stroke="currentColor" strokeWidth="1.5" className="text-[var(--color-text-tertiary)]" />
        <path d="M16 24h16M24 16v16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" className="text-[var(--color-text-tertiary)]" />
      </svg>
    ),
  },
};
