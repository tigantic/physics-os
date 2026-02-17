import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { TimelineScreen } from "./Timeline";
import { FIXTURE_PROOF_PASS, FIXTURE_DOMAIN } from "@/__fixtures__/storybook";

const meta: Meta<typeof TimelineScreen> = {
  title: "Features/Screens/Timeline",
  component: TimelineScreen,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof TimelineScreen>;

export const Default: Story = {
  args: { proof: FIXTURE_PROOF_PASS, domain: FIXTURE_DOMAIN },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Timeline")).toBeVisible();
    expect(canvas.getByText("3 steps")).toBeVisible();
    expect(canvas.getByText("Step")).toBeVisible();
    expect(canvas.getByText("Conservation Residual")).toBeVisible();
  },
};

export const SingleStep: Story = {
  args: {
    proof: {
      ...FIXTURE_PROOF_PASS,
      timeline: {
        step_count: 1,
        steps: [FIXTURE_PROOF_PASS.timeline.steps[0]],
      },
    },
    domain: FIXTURE_DOMAIN,
  },
};
