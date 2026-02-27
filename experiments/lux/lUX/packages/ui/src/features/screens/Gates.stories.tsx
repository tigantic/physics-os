import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { GatesScreen } from "./Gates";
import { FIXTURE_PROOF_PASS, FIXTURE_PROOF_FAIL } from "@/__fixtures__/storybook";

const meta: Meta<typeof GatesScreen> = {
  title: "Features/Screens/Gates",
  component: GatesScreen,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof GatesScreen>;

export const AllPassing: Story = {
  args: { proof: FIXTURE_PROOF_PASS },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Gates")).toBeVisible();
    expect(canvas.getByText("3 evaluated")).toBeVisible();
    const passChips = canvas.getAllByText("PASS");
    expect(passChips.length).toBe(3);
  },
};

export const WithFailure: Story = {
  args: { proof: FIXTURE_PROOF_FAIL },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("FAIL")).toBeVisible();
  },
};

export const Empty: Story = {
  args: {
    proof: {
      ...FIXTURE_PROOF_PASS,
      gate_results: {},
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("No gates evaluated")).toBeVisible();
  },
};
