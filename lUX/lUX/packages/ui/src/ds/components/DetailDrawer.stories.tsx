import type { Meta, StoryObj } from "@storybook/react";
import { within, expect, userEvent } from "@storybook/test";
import { DetailDrawer } from "./DetailDrawer";
import * as React from "react";

function DrawerWrapper({ defaultOpen = true }: { defaultOpen?: boolean }) {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div>
      <button type="button" onClick={() => setOpen(true)}>Open Drawer</button>
      <DetailDrawer
        open={open}
        onClose={() => setOpen(false)}
        title="Artifact Details"
        subtitle="convergence-plot.png"
      >
        <div className="space-y-3 p-4">
          <div className="text-sm text-[var(--color-text-primary)]">Hash: sha256:abc123</div>
          <div className="text-sm text-[var(--color-text-secondary)]">MIME: image/png</div>
          <div className="text-sm text-[var(--color-text-tertiary)]">Size: 45,200 bytes</div>
        </div>
      </DetailDrawer>
    </div>
  );
}

const meta: Meta<typeof DetailDrawer> = {
  title: "DS/DetailDrawer",
  component: DetailDrawer,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof DetailDrawer>;

export const Open: Story = {
  render: () => <DrawerWrapper defaultOpen />,
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Artifact Details")).toBeVisible();
    expect(canvas.getByText("convergence-plot.png")).toBeVisible();
  },
};

export const Closed: Story = {
  render: () => <DrawerWrapper defaultOpen={false} />,
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const openBtn = canvas.getByText("Open Drawer");
    expect(openBtn).toBeVisible();
    await userEvent.click(openBtn);
    expect(canvas.getByText("Artifact Details")).toBeVisible();
  },
};

export const EscapeClose: Story = {
  render: () => <DrawerWrapper defaultOpen />,
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Artifact Details")).toBeVisible();
    await userEvent.keyboard("{Escape}");
  },
};
