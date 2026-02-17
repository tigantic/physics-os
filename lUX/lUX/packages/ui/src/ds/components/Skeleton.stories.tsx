import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { Skeleton } from "./Skeleton";

const meta: Meta<typeof Skeleton> = {
  title: "DS/Skeleton",
  component: Skeleton,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof Skeleton>;

export const SingleLine: Story = {
  args: {},
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByRole("status", { name: /loading/i })).toBeVisible();
  },
};

export const MultipleRows: Story = {
  args: { rows: 4 },
};

export const TallRows: Story = {
  args: { rows: 3, heightClass: "h-8", gapClass: "gap-3" },
};

export const CardSkeleton: Story = {
  args: { rows: 1, heightClass: "h-40", roundedClass: "rounded-[12px]" },
};

export const PageSkeleton: Story = {
  render: () => (
    <div className="space-y-4">
      <Skeleton heightClass="h-10" roundedClass="rounded-[12px]" />
      <div className="grid grid-cols-3 gap-4">
        <Skeleton heightClass="h-32" roundedClass="rounded-[8px]" />
        <Skeleton heightClass="h-32" roundedClass="rounded-[8px]" />
        <Skeleton heightClass="h-32" roundedClass="rounded-[8px]" />
      </div>
      <Skeleton rows={5} heightClass="h-6" gapClass="gap-2" />
    </div>
  ),
};
