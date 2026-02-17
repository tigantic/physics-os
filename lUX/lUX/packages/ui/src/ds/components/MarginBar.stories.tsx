import type { Meta, StoryObj } from "@storybook/react";
import { MarginBar } from "./MarginBar";

const meta: Meta<typeof MarginBar> = {
  title: "DS/MarginBar",
  component: MarginBar,
};

export default meta;
type Story = StoryObj<typeof MarginBar>;

export const HighMargin: Story = {
  args: {
    margin: { status: "ok", value: 0.85 },
  },
};

export const MediumMargin: Story = {
  args: {
    margin: { status: "ok", value: 0.35 },
  },
};

export const LowMargin: Story = {
  args: {
    margin: { status: "ok", value: 0.05 },
  },
};

export const ZeroMargin: Story = {
  args: {
    margin: { status: "ok", value: 0 },
  },
};

export const FullMargin: Story = {
  args: {
    margin: { status: "ok", value: 1.0 },
  },
};

export const NegativeMargin: Story = {
  args: {
    margin: { status: "ok", value: -0.15 },
  },
};

export const OvershootMargin: Story = {
  args: {
    margin: { status: "ok", value: 1.5 },
  },
};

export const BoundaryWarn: Story = {
  args: {
    margin: { status: "ok", value: 0.1 },
  },
};

export const BoundaryGold: Story = {
  args: {
    margin: { status: "ok", value: 0.5 },
  },
};

export const MissingData: Story = {
  args: {
    margin: { status: "missing", reason: "Sensor offline" },
  },
};

export const InvalidData: Story = {
  args: {
    margin: { status: "invalid", reason: "NaN encountered", details: { raw: NaN } },
  },
};

export const AllStates: Story = {
  render: () => (
    <div className="flex max-w-md flex-col gap-4">
      <div>
        <p className="mb-1 text-xs font-medium">High (85%)</p>
        <MarginBar margin={{ status: "ok", value: 0.85 }} />
      </div>
      <div>
        <p className="mb-1 text-xs font-medium">Medium (35%)</p>
        <MarginBar margin={{ status: "ok", value: 0.35 }} />
      </div>
      <div>
        <p className="mb-1 text-xs font-medium">Low (5%)</p>
        <MarginBar margin={{ status: "ok", value: 0.05 }} />
      </div>
      <div>
        <p className="mb-1 text-xs font-medium">Missing</p>
        <MarginBar margin={{ status: "missing", reason: "No data" }} />
      </div>
      <div>
        <p className="mb-1 text-xs font-medium">Invalid</p>
        <MarginBar margin={{ status: "invalid", reason: "Parse error" }} />
      </div>
    </div>
  ),
};
