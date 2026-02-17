import type { Meta, StoryObj } from "@storybook/react";
import { Chip } from "./Chip";

const meta: Meta<typeof Chip> = {
  title: "DS/Chip",
  component: Chip,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof Chip>;

export const Default: Story = {
  args: { children: "Default" },
};

export const Gold: Story = {
  args: { children: "Verified", tone: "gold" },
};

export const Fail: Story = {
  args: { children: "Failed", tone: "fail" },
};

export const Warn: Story = {
  args: { children: "Warning", tone: "warn" },
};

export const AllTones: Story = {
  render: () => (
    <div className="flex flex-wrap gap-2">
      <Chip tone="default">Default</Chip>
      <Chip tone="gold">Gold</Chip>
      <Chip tone="fail">Fail</Chip>
      <Chip tone="warn">Warn</Chip>
    </div>
  ),
};

export const LongLabel: Story = {
  args: { children: "Conservation Law Verified via Trustless Physics Pipeline", tone: "gold" },
};

export const NoToneProp: Story = {
  args: { children: "Implicit default (no tone prop)" },
};

export const WithEmoji: Story = {
  args: { children: "✓ Pass", tone: "gold" },
};

export const InlineUsage: Story = {
  render: () => (
    <p className="text-sm">
      Status: <Chip tone="gold">PASS</Chip> — Verification: <Chip tone="fail">BROKEN_CHAIN</Chip>
    </p>
  ),
};
