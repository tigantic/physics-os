import type { Meta, StoryObj } from "@storybook/react";
import { Badge } from "./badge";

const meta: Meta<typeof Badge> = {
  title: "UI/Badge",
  component: Badge,
};

export default meta;
type Story = StoryObj<typeof Badge>;

export const Default: Story = {
  args: { children: "Default" },
};

export const Gold: Story = {
  args: { children: "Verified", variant: "gold" },
};

export const Pass: Story = {
  args: { children: "PASS", variant: "pass" },
};

export const Fail: Story = {
  args: { children: "FAIL", variant: "fail" },
};

export const Warn: Story = {
  args: { children: "WARN", variant: "warn" },
};

export const AllVariants: Story = {
  render: () => (
    <div className="flex flex-wrap gap-2">
      <Badge variant="default">Default</Badge>
      <Badge variant="gold">Gold</Badge>
      <Badge variant="pass">Pass</Badge>
      <Badge variant="fail">Fail</Badge>
      <Badge variant="warn">Warn</Badge>
    </div>
  ),
};

export const CustomClassName: Story = {
  args: { children: "Custom", variant: "gold", className: "text-base px-4 py-1" },
};

export const NoVariantProp: Story = {
  args: { children: "Implicit default" },
};

export const LongLabel: Story = {
  args: { children: "Conservation Law Attestation Verified", variant: "pass" },
};

export const WithHTMLAttributes: Story = {
  args: { children: "Clickable", variant: "gold", role: "button", tabIndex: 0 },
};
