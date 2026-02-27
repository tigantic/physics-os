import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { ReproduceScreen } from "./Reproduce";
import { FIXTURE_PROOF_PASS } from "@/__fixtures__/storybook";

const meta: Meta<typeof ReproduceScreen> = {
  title: "Features/Screens/Reproduce",
  component: ReproduceScreen,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof ReproduceScreen>;

export const ValidDigest: Story = {
  args: { proof: FIXTURE_PROOF_PASS },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Reproduce")).toBeVisible();
    expect(canvas.getByText("Deterministic command")).toBeVisible();
    expect(canvas.getByText(/docker run --rm/)).toBeVisible();
  },
};

export const InvalidDigest: Story = {
  args: {
    proof: {
      ...FIXTURE_PROOF_PASS,
      meta: {
        ...FIXTURE_PROOF_PASS.meta,
        environment: {
          ...FIXTURE_PROOF_PASS.meta.environment,
          container_digest: "not-a-valid-digest",
        },
      },
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Invalid reproduction metadata")).toBeVisible();
  },
};

export const InvalidSeed: Story = {
  args: {
    proof: {
      ...FIXTURE_PROOF_PASS,
      meta: {
        ...FIXTURE_PROOF_PASS.meta,
        environment: {
          ...FIXTURE_PROOF_PASS.meta.environment,
          seed: Number.MAX_SAFE_INTEGER + 1,
        },
      },
    },
  },
};
