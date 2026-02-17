import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { DataValueNumberView } from "./DataValueView";
import { FIXTURE_DOMAIN } from "@/__fixtures__/storybook";

const meta: Meta<typeof DataValueNumberView> = {
  title: "Features/Proof/DataValueNumberView",
  component: DataValueNumberView,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof DataValueNumberView>;

export const FixedFormat: Story = {
  args: {
    dv: { status: "ok", value: 1.3 },
    metricId: "cd_error_pct",
    domain: FIXTURE_DOMAIN,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("1.3")).toBeVisible();
  },
};

export const ScientificFormat: Story = {
  args: {
    dv: { status: "ok", value: 3.2e-7 },
    metricId: "residual_l2",
    domain: FIXTURE_DOMAIN,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvasElement.textContent).toMatch(/3\.20e/);
  },
};

export const MissingData: Story = {
  args: {
    dv: { status: "missing", reason: "Metric missing" },
    metricId: "residual_l2",
    domain: FIXTURE_DOMAIN,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Data Unavailable")).toBeVisible();
  },
};

export const InvalidData: Story = {
  args: {
    dv: { status: "invalid", reason: "Parse error" },
    metricId: "residual_l2",
    domain: FIXTURE_DOMAIN,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Invalid")).toBeVisible();
  },
};

export const OutOfRange: Story = {
  args: {
    dv: { status: "ok", value: 150 },
    metricId: "cd_error_pct",
    domain: FIXTURE_DOMAIN,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Invalid")).toBeVisible();
  },
};

export const ConvergenceRate: Story = {
  args: {
    dv: { status: "ok", value: 0.92 },
    metricId: "convergence_rate",
    domain: FIXTURE_DOMAIN,
  },
};
