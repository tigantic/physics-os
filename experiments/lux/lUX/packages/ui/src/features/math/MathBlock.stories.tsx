import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { MathBlock } from "./MathBlock";

const meta: Meta<typeof MathBlock> = {
  title: "Features/Math/MathBlock",
  component: MathBlock,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof MathBlock>;

export const SimpleIntegral: Story = {
  args: { latex: "\\int f(\\mathbf{x},\\mathbf{v})\\,d\\mathbf{x}\\,d\\mathbf{v} = M" },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const img = canvas.getByRole("img");
    expect(img).toBeVisible();
    expect(img).toHaveAttribute("aria-label", expect.stringContaining("Math equation"));
  },
};

export const NavierStokes: Story = {
  args: { latex: "\\frac{\\partial \\mathbf{u}}{\\partial t} + (\\mathbf{u} \\cdot \\nabla)\\mathbf{u} = -\\frac{1}{\\rho}\\nabla p + \\nu \\nabla^2 \\mathbf{u}" },
};

export const EinsteinFieldEquation: Story = {
  args: { latex: "R_{\\mu\\nu} - \\frac{1}{2}Rg_{\\mu\\nu} + \\Lambda g_{\\mu\\nu} = \\frac{8\\pi G}{c^4}T_{\\mu\\nu}" },
};

export const SimpleExpression: Story = {
  args: { latex: "e^{i\\pi} + 1 = 0" },
};

export const Fraction: Story = {
  args: { latex: "\\frac{a}{b} + \\frac{c}{d} = \\frac{ad + bc}{bd}" },
};
