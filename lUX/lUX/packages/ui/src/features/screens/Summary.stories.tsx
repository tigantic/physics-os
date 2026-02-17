import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { SummaryScreen } from "./Summary";
import { FIXTURE_PROOF_PASS, FIXTURE_PROOF_FAIL, FIXTURE_DOMAIN } from "@/__fixtures__/storybook";

const meta: Meta<typeof SummaryScreen> = {
  title: "Features/Screens/Summary",
  component: SummaryScreen,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof SummaryScreen>;

export const ReviewMode: Story = {
  args: { proof: FIXTURE_PROOF_PASS, domain: FIXTURE_DOMAIN, mode: "REVIEW" },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Overview")).toBeVisible();
    expect(canvas.getByText(/hydro-sim-002/)).toBeVisible();
    expect(canvas.getByText("PASS")).toBeVisible();
  },
};

export const PublicationMode: Story = {
  args: { proof: FIXTURE_PROOF_PASS, domain: FIXTURE_DOMAIN, mode: "PUBLICATION" },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Paper View")).toBeVisible();
    expect(canvas.getByRole("img", { name: /math equation/i })).toBeVisible();
  },
};

export const FailVerdict: Story = {
  args: { proof: FIXTURE_PROOF_FAIL, domain: FIXTURE_DOMAIN, mode: "REVIEW" },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("FAIL")).toBeVisible();
  },
};

export const ExecutiveMode: Story = {
  args: { proof: FIXTURE_PROOF_PASS, domain: FIXTURE_DOMAIN, mode: "EXECUTIVE" },
};
