import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { CompareScreen } from "./Compare";
import { FIXTURE_PROOF_PASS, FIXTURE_PROOF_FAIL, FIXTURE_DOMAIN } from "@/__fixtures__/storybook";

const meta: Meta<typeof CompareScreen> = {
  title: "Features/Screens/Compare",
  component: CompareScreen,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof CompareScreen>;

export const WithBaseline: Story = {
  args: {
    proof: FIXTURE_PROOF_PASS,
    baseline: FIXTURE_PROOF_FAIL,
    domain: FIXTURE_DOMAIN,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Compare")).toBeVisible();
    expect(canvas.getByText("Current")).toBeVisible();
    expect(canvas.getByText("Baseline")).toBeVisible();
  },
};

export const NoBaseline: Story = {
  args: {
    proof: FIXTURE_PROOF_PASS,
    domain: FIXTURE_DOMAIN,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("No baseline")).toBeVisible();
    expect(canvas.getByText("Select a baseline proof to enable comparison.")).toBeVisible();
  },
};

export const WithBaselineSelector: Story = {
  args: {
    proof: FIXTURE_PROOF_PASS,
    domain: FIXTURE_DOMAIN,
    availableBaselineIds: ["hydro-sim-001", "hydro-sim-003-fail", "hydro-sim-004"],
    onBaselineSelect: () => {},
  },
};
