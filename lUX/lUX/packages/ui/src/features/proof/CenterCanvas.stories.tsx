import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { CenterCanvas } from "./CenterCanvas";
import { FIXTURE_PROOF_PASS, FIXTURE_PROOF_FAIL, FIXTURE_DOMAIN, DEFAULT_MODE } from "@/__fixtures__/storybook";

const meta: Meta<typeof CenterCanvas> = {
  title: "Features/Proof/CenterCanvas",
  component: CenterCanvas,
  tags: ["autodocs"],
  parameters: {
    nextjs: { appDirectory: true },
  },
};

export default meta;
type Story = StoryObj<typeof CenterCanvas>;

export const ReviewMode: Story = {
  args: {
    proof: FIXTURE_PROOF_PASS,
    domain: FIXTURE_DOMAIN,
    mode: "REVIEW",
    packageId: "hydro-sim-002",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByRole("tabpanel")).toBeVisible();
    expect(canvas.getByText(/viewing review mode/i)).toBeInTheDocument();
  },
};

export const ExecutiveMode: Story = {
  args: {
    proof: FIXTURE_PROOF_PASS,
    domain: FIXTURE_DOMAIN,
    mode: "EXECUTIVE",
    packageId: "hydro-sim-002",
  },
};

export const AuditMode: Story = {
  args: {
    proof: FIXTURE_PROOF_PASS,
    domain: FIXTURE_DOMAIN,
    mode: "AUDIT",
    packageId: "hydro-sim-002",
  },
};

export const WithBaseline: Story = {
  args: {
    proof: FIXTURE_PROOF_PASS,
    baseline: FIXTURE_PROOF_FAIL,
    domain: FIXTURE_DOMAIN,
    mode: "REVIEW",
    packageId: "hydro-sim-002",
  },
};
