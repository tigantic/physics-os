import type { Meta, StoryObj } from "@storybook/react";
import { within, expect, userEvent } from "@storybook/test";
import { ModeDial } from "./ModeDial";

const meta: Meta<typeof ModeDial> = {
  title: "Features/Proof/ModeDial",
  component: ModeDial,
  tags: ["autodocs"],
  parameters: {
    nextjs: { appDirectory: true, navigation: { pathname: "/packages/test-id", query: { mode: "REVIEW", fixture: "pass", baseline: "pass" } } },
  },
};

export default meta;
type Story = StoryObj<typeof ModeDial>;

export const Default: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const tablist = canvas.getByRole("tablist", { name: /proof viewing mode/i });
    expect(tablist).toBeVisible();
    const tabs = canvas.getAllByRole("tab");
    expect(tabs).toHaveLength(4);
    expect(tabs.map((t) => t.textContent)).toEqual(["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"]);
  },
};

export const ExecutiveMode: Story = {
  parameters: {
    nextjs: { appDirectory: true, navigation: { pathname: "/packages/test-id", query: { mode: "EXECUTIVE", fixture: "pass", baseline: "pass" } } },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const exec = canvas.getByRole("tab", { name: "EXECUTIVE" });
    expect(exec).toHaveAttribute("aria-selected", "true");
  },
};

export const AuditMode: Story = {
  parameters: {
    nextjs: { appDirectory: true, navigation: { pathname: "/packages/test-id", query: { mode: "AUDIT", fixture: "pass", baseline: "pass" } } },
  },
};

export const PublicationMode: Story = {
  parameters: {
    nextjs: { appDirectory: true, navigation: { pathname: "/packages/test-id", query: { mode: "PUBLICATION", fixture: "pass", baseline: "pass" } } },
  },
};

export const KeyboardNavigation: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const tabs = canvas.getAllByRole("tab");
    const selected = tabs.find((t) => t.getAttribute("aria-selected") === "true");
    expect(selected).toBeTruthy();
    await userEvent.tab();
    await userEvent.keyboard("{ArrowRight}");
  },
};
