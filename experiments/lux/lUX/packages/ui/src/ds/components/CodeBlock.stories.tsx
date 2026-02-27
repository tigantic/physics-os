import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { CodeBlock } from "./CodeBlock";

const meta: Meta<typeof CodeBlock> = {
  title: "DS/CodeBlock",
  component: CodeBlock,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof CodeBlock>;

export const Bash: Story = {
  args: {
    children: "docker run --rm sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 --seed 42",
    language: "bash",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText(/docker run/)).toBeVisible();
  },
};

export const TypeScript: Story = {
  args: {
    children: `export function greet(name: string): string {
  return \`Hello, \${name}!\`;
}`,
    language: "typescript",
  },
};

export const JSON: Story = {
  args: {
    children: `{
  "schema_version": "1.0.0",
  "verdict": { "status": "PASS", "quality_score": 0.94 },
  "attestation": { "merkle_root": "sha256:e3b0c44298fc1c..." }
}`,
    language: "json",
  },
};

export const WithLineNumbers: Story = {
  args: {
    children: `const MODES = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"] as const;
type Mode = (typeof MODES)[number];

function isMode(s: string): s is Mode {
  return (MODES as readonly string[]).includes(s);
}`,
    language: "typescript",
    lineNumbers: true,
  },
};

export const NoCopy: Story = {
  args: {
    children: "read-only content, no copy button",
    language: "text",
    copyable: false,
  },
};

export const LongContent: Story = {
  args: {
    children: Array.from({ length: 50 }, (_, i) => `line ${i + 1}: ${("x").repeat(80)}`).join("\n"),
    language: "text",
    maxHeight: 200,
  },
};
