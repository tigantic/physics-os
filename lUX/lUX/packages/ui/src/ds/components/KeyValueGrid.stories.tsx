import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { KeyValueGrid } from "./KeyValueGrid";

const meta: Meta<typeof KeyValueGrid> = {
  title: "DS/KeyValueGrid",
  component: KeyValueGrid,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof KeyValueGrid>;

export const Default: Story = {
  args: {
    entries: [
      { label: "Proof ID", value: "hydro-sim-002-2026-02-16", mono: true },
      { label: "Verdict", value: "PASS" },
      { label: "Domain", value: "fluid_dynamics" },
    ],
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Proof ID")).toBeVisible();
    expect(canvas.getByText("hydro-sim-002-2026-02-16")).toBeVisible();
  },
};

export const TwoColumns: Story = {
  args: {
    entries: [
      { label: "Solver", value: "HydroSolve v4.2.1" },
      { label: "Architecture", value: "x86_64", mono: true },
      { label: "Seed", value: "42", mono: true },
      { label: "Commit", value: "a1b2c3d4e5f6", mono: true, copyable: true },
    ],
    columns: 2,
  },
};

export const ThreeColumns: Story = {
  args: {
    entries: [
      { label: "Lines", value: "70%" },
      { label: "Functions", value: "65%" },
      { label: "Branches", value: "60%" },
      { label: "Statements", value: "70%" },
      { label: "E2E Specs", value: "66" },
      { label: "Unit Tests", value: "570" },
    ],
    columns: 3,
  },
};

export const WithCopyable: Story = {
  args: {
    entries: [
      { label: "Merkle Root", value: "sha256:e3b0c44298fc1c149afbf4c8996fb924", mono: true, copyable: true },
      { label: "Container Digest", value: "sha256:9f86d081884c7d659a2feaa0c55ad015", mono: true, copyable: true },
    ],
  },
};
