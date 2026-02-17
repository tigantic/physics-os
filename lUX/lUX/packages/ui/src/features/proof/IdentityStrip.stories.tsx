import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { IdentityStrip } from "./IdentityStrip";
import { FIXTURE_PROOF_PASS, FIXTURE_PROOF_FAIL } from "@/__fixtures__/storybook";

const meta: Meta<typeof IdentityStrip> = {
  title: "Features/Proof/IdentityStrip",
  component: IdentityStrip,
  tags: ["autodocs"],
  parameters: {
    nextjs: { appDirectory: true },
    layout: "fullscreen",
  },
};

export default meta;
type Story = StoryObj<typeof IdentityStrip>;

export const PassVerdict: Story = {
  args: { proof: FIXTURE_PROOF_PASS },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByLabelText("Proof identity")).toBeVisible();
    expect(canvas.getByText(/hydro-sim · fluid_dynamics/)).toBeVisible();
  },
};

export const FailVerdict: Story = {
  args: { proof: FIXTURE_PROOF_FAIL },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText(/hydro-sim · fluid_dynamics/)).toBeVisible();
  },
};

export const LongProjectId: Story = {
  args: {
    proof: {
      ...FIXTURE_PROOF_PASS,
      meta: {
        ...FIXTURE_PROOF_PASS.meta,
        project_id: "very-long-project-identifier-that-should-truncate-gracefully",
        domain_id: "extended_turbulence_simulation_domain_v2",
      },
    },
  },
};
