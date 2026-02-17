import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { ProofWorkspace } from "./ProofWorkspace";
import { FIXTURE_PROOF_PASS, FIXTURE_PROOF_FAIL, FIXTURE_DOMAIN, DEFAULT_MODE } from "@/__fixtures__/storybook";

const meta: Meta<typeof ProofWorkspace> = {
  title: "Features/Proof/ProofWorkspace",
  component: ProofWorkspace,
  tags: ["autodocs"],
  parameters: {
    nextjs: { appDirectory: true, navigation: { pathname: "/packages/hydro-sim-002", query: { mode: "REVIEW", fixture: "pass", baseline: "pass" } } },
    layout: "fullscreen",
  },
};

export default meta;
type Story = StoryObj<typeof ProofWorkspace>;

export const Default: Story = {
  args: {
    proof: FIXTURE_PROOF_PASS,
    domain: FIXTURE_DOMAIN,
    fixture: "pass",
    mode: "REVIEW",
    packageId: "hydro-sim-002",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByLabelText("Proof identity")).toBeVisible();
    expect(canvas.getByText(/hydro-sim/)).toBeVisible();
  },
};

export const FailVerdict: Story = {
  args: {
    proof: FIXTURE_PROOF_FAIL,
    domain: FIXTURE_DOMAIN,
    fixture: "fail",
    mode: "REVIEW",
    packageId: "hydro-sim-003-fail",
  },
  parameters: {
    nextjs: { appDirectory: true, navigation: { pathname: "/packages/hydro-sim-003-fail", query: { mode: "REVIEW", fixture: "fail", baseline: "pass" } } },
  },
};

export const ExecutiveMode: Story = {
  args: {
    proof: FIXTURE_PROOF_PASS,
    domain: FIXTURE_DOMAIN,
    fixture: "pass",
    mode: "EXECUTIVE",
    packageId: "hydro-sim-002",
  },
  parameters: {
    nextjs: { appDirectory: true, navigation: { pathname: "/packages/hydro-sim-002", query: { mode: "EXECUTIVE", fixture: "pass", baseline: "pass" } } },
  },
};

export const WithBaseline: Story = {
  args: {
    proof: FIXTURE_PROOF_PASS,
    baseline: FIXTURE_PROOF_FAIL,
    domain: FIXTURE_DOMAIN,
    fixture: "pass",
    mode: "REVIEW",
    packageId: "hydro-sim-002",
  },
};
