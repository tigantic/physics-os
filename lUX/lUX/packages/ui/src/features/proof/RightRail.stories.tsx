import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { RightRail } from "./RightRail";
import { FIXTURE_PROOF_PASS, FIXTURE_PROOF_FAIL } from "@/__fixtures__/storybook";

const meta: Meta<typeof RightRail> = {
  title: "Features/Proof/RightRail",
  component: RightRail,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof RightRail>;

export const Verified: Story = {
  args: { proof: FIXTURE_PROOF_PASS },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByLabelText("Integrity details")).toBeVisible();
    expect(canvas.getByText("Integrity")).toBeVisible();
    expect(canvas.getByText("VERIFIED")).toBeVisible();
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
