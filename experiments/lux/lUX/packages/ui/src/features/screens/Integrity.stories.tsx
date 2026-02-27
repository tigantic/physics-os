import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { IntegrityScreen } from "./Integrity";
import { FIXTURE_PROOF_PASS, FIXTURE_PROOF_FAIL } from "@/__fixtures__/storybook";

const meta: Meta<typeof IntegrityScreen> = {
  title: "Features/Screens/Integrity",
  component: IntegrityScreen,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof IntegrityScreen>;

export const Verified: Story = {
  args: { proof: FIXTURE_PROOF_PASS },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Verification")).toBeVisible();
    expect(canvas.getByText("VERIFIED")).toBeVisible();
    expect(canvas.getByText("Chain Intact")).toBeVisible();
  },
};

export const BrokenChain: Story = {
  args: { proof: FIXTURE_PROOF_FAIL },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("BROKEN_CHAIN")).toBeVisible();
    expect(canvas.getByText("SIG_MISMATCH")).toBeVisible();
  },
};

export const Unverified: Story = {
  args: {
    proof: {
      ...FIXTURE_PROOF_PASS,
      verification: undefined,
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("UNVERIFIED")).toBeVisible();
  },
};
