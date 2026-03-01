import type { Meta, StoryObj } from "@storybook/react";
import { within, userEvent, expect, fn } from "@storybook/test";
import { CopyField } from "./CopyField";

const meta: Meta<typeof CopyField> = {
  title: "DS/CopyField",
  component: CopyField,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof CopyField>;

export const Default: Story = {
  args: {
    label: "SHA-256",
    value: "sha256:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // Mock clipboard API for Storybook environment
    const writeText = fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    const copyBtn = canvas.getByRole("button", { name: /copy sha-256/i });
    expect(copyBtn).toBeVisible();

    // Click copy — should show "Copied" feedback
    await userEvent.click(copyBtn);
    expect(canvas.getByText("Copied")).toBeVisible();

    // Focus should return to the copy button
    expect(copyBtn).toHaveFocus();
  },
};

export const ShortValue: Story = {
  args: {
    label: "Node ID",
    value: "node-42",
  },
};

export const LongHash: Story = {
  args: {
    label: "Attestation Hash",
    value: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855deadbeef",
  },
};

export const URL: Story = {
  args: {
    label: "Endpoint",
    value: "https://physics-os.example.com/api/v1/attestations/verify",
  },
};

export const MultipleFields: Story = {
  render: () => (
    <div className="flex max-w-md flex-col gap-3">
      <CopyField label="Block Hash" value="0xdeadbeefcafebabe1234567890abcdef" />
      <CopyField label="Transaction ID" value="tx_9f8e7d6c5b4a3210" />
      <CopyField label="Validator Key" value="val_pub_key_abc123def456" />
    </div>
  ),
};

export const EmptyValue: Story = {
  args: {
    label: "Pending",
    value: "",
  },
};
