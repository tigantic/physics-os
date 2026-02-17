import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { LeftRail } from "./LeftRail";
import { FIXTURE_PROOF_PASS, ALL_MODES } from "@/__fixtures__/storybook";

const meta: Meta<typeof LeftRail> = {
  title: "Features/Proof/LeftRail",
  component: LeftRail,
  tags: ["autodocs"],
  parameters: {
    nextjs: { appDirectory: true },
  },
};

export default meta;
type Story = StoryObj<typeof LeftRail>;

export const ExecutiveMode: Story = {
  args: { proof: FIXTURE_PROOF_PASS, fixture: "pass", mode: "EXECUTIVE" },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByLabelText("Proof fixtures")).toBeVisible();
    expect(canvas.getByText("Fixtures")).toBeVisible();
    expect(canvas.getByText("pass")).toBeVisible();
    expect(canvas.getByText("fail")).toBeVisible();
  },
};

export const ReviewMode: Story = {
  args: { proof: FIXTURE_PROOF_PASS, fixture: "pass", mode: "REVIEW" },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByLabelText("Claims list")).toBeVisible();
    expect(canvas.getByText("Claims")).toBeVisible();
  },
};

export const AuditMode: Story = {
  args: { proof: FIXTURE_PROOF_PASS, fixture: "pass", mode: "AUDIT" },
};

export const PublicationMode: Story = {
  args: { proof: FIXTURE_PROOF_PASS, fixture: "pass", mode: "PUBLICATION" },
};

export const ActiveFixtureHighlight: Story = {
  args: { proof: FIXTURE_PROOF_PASS, fixture: "fail", mode: "EXECUTIVE" },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const activeLink = canvas.getByText("fail").closest("a");
    expect(activeLink).toHaveAttribute("aria-current", "page");
  },
};
