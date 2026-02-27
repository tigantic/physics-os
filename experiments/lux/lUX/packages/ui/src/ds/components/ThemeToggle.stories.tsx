import type { Meta, StoryObj } from "@storybook/react";
import { within, expect, userEvent } from "@storybook/test";
import { ThemeToggle } from "./ThemeToggle";

const meta: Meta<typeof ThemeToggle> = {
  title: "DS/ThemeToggle",
  component: ThemeToggle,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof ThemeToggle>;

export const Default: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const btn = canvas.getByRole("button", { name: /switch to/i });
    expect(btn).toBeVisible();
  },
};

export const Toggle: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const btn = canvas.getByRole("button", { name: /switch to/i });
    await userEvent.click(btn);
    expect(btn).toHaveAttribute("aria-label", expect.stringContaining("Switch to"));
  },
};

export const WithClassName: Story = {
  args: { className: "border border-dashed" },
};
