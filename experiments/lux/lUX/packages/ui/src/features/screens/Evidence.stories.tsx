import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { EvidenceScreen } from "./Evidence";
import { FIXTURE_PROOF_PASS } from "@/__fixtures__/storybook";

const meta: Meta<typeof EvidenceScreen> = {
  title: "Features/Screens/Evidence",
  component: EvidenceScreen,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof EvidenceScreen>;

export const Default: Story = {
  args: { proof: FIXTURE_PROOF_PASS },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Evidence")).toBeVisible();
    expect(canvas.getByText("2 artifacts")).toBeVisible();
  },
};

export const SingleArtifact: Story = {
  args: {
    proof: {
      ...FIXTURE_PROOF_PASS,
      artifacts: {
        "artifact-convergence-plot": FIXTURE_PROOF_PASS.artifacts["artifact-convergence-plot"],
      },
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("1 artifacts")).toBeVisible();
  },
};

export const NoArtifacts: Story = {
  args: {
    proof: {
      ...FIXTURE_PROOF_PASS,
      artifacts: {},
    },
  },
};
