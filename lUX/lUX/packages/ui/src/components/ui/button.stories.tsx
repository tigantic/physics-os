import type { Meta, StoryObj } from "@storybook/react";
import { within, userEvent, expect } from "@storybook/test";
import { Button } from "./button";

const meta: Meta<typeof Button> = {
  title: "UI/Button",
  component: Button,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof Button>;

export const Default: Story = {
  args: { children: "Button" },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const btn = canvas.getByRole("button", { name: "Button" });
    expect(btn).toBeVisible();
    // Verify focus ring appears on focus
    await userEvent.tab();
    expect(btn).toHaveFocus();
  },
};

export const Gold: Story = {
  args: { children: "Gold", variant: "gold" },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const btn = canvas.getByRole("button", { name: "Gold" });
    expect(btn).toBeVisible();
    await userEvent.click(btn);
    expect(btn).toHaveFocus();
  },
};
