import type { Meta, StoryObj } from "@storybook/react";
import { VerdictSeal } from "./VerdictSeal";

const meta: Meta<typeof VerdictSeal> = {
  title: "DS/VerdictSeal",
  component: VerdictSeal,
};

export default meta;
type Story = StoryObj<typeof VerdictSeal>;

export const PassVerified: Story = {
  args: { status: "PASS", verification: "VERIFIED" },
};

export const PassUnverified: Story = {
  args: { status: "PASS", verification: "UNVERIFIED" },
};

export const FailBrokenChain: Story = {
  args: { status: "FAIL", verification: "BROKEN_CHAIN" },
};

export const FailVerified: Story = {
  args: { status: "FAIL", verification: "VERIFIED" },
};

export const WarnUnsupported: Story = {
  args: { status: "WARN", verification: "UNSUPPORTED" },
};

export const WarnVerified: Story = {
  args: { status: "WARN", verification: "VERIFIED" },
};

export const IncompleteUnverified: Story = {
  args: { status: "INCOMPLETE", verification: "UNVERIFIED" },
};

export const IncompleteBrokenChain: Story = {
  args: { status: "INCOMPLETE", verification: "BROKEN_CHAIN" },
};

export const UnknownVerification: Story = {
  args: { status: "PASS", verification: "SOME_FUTURE_STATUS" },
};

export const AllCombinations: Story = {
  render: () => {
    const statuses = ["PASS", "FAIL", "WARN", "INCOMPLETE"] as const;
    const verifications = ["VERIFIED", "BROKEN_CHAIN", "UNVERIFIED", "UNSUPPORTED"];
    return (
      <div className="flex flex-col gap-3">
        {statuses.map((s) =>
          verifications.map((v) => (
            <div key={`${s}-${v}`} className="flex items-center gap-4">
              <span className="w-40 font-mono text-xs">
                {s} / {v}
              </span>
              <VerdictSeal status={s} verification={v} />
            </div>
          )),
        )}
      </div>
    );
  },
};
