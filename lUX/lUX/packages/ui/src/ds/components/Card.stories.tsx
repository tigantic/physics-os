import type { Meta, StoryObj } from "@storybook/react";
import { Card, CardHeader, CardContent } from "./Card";

const meta: Meta<typeof Card> = {
  title: "DS/Card",
  component: Card,
};

export default meta;
type Story = StoryObj<typeof Card>;

export const Default: Story = {
  render: () => (
    <Card>
      <CardHeader>Header</CardHeader>
      <CardContent>Content body goes here.</CardContent>
    </Card>
  ),
};

export const HeaderOnly: Story = {
  render: () => (
    <Card>
      <CardHeader>Only a header, no content section</CardHeader>
    </Card>
  ),
};

export const ContentOnly: Story = {
  render: () => (
    <Card>
      <CardContent>Content without a header section.</CardContent>
    </Card>
  ),
};

export const CustomClassName: Story = {
  render: () => (
    <Card className="max-w-sm border-dashed">
      <CardHeader className="bg-black/5">Custom Header Styling</CardHeader>
      <CardContent className="text-sm italic">Custom content styling applied via className prop.</CardContent>
    </Card>
  ),
};

export const NestedCards: Story = {
  render: () => (
    <Card>
      <CardHeader>Outer Card</CardHeader>
      <CardContent>
        <Card>
          <CardHeader>Inner Card</CardHeader>
          <CardContent>Nested content.</CardContent>
        </Card>
      </CardContent>
    </Card>
  ),
};

export const LongContent: Story = {
  render: () => (
    <Card className="max-w-md">
      <CardHeader>Attestation Results</CardHeader>
      <CardContent>
        <p>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore
          magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
          consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
          pariatur.
        </p>
      </CardContent>
    </Card>
  ),
};

export const MultipleContentSections: Story = {
  render: () => (
    <Card>
      <CardHeader>Multi-Section</CardHeader>
      <CardContent>First content section.</CardContent>
      <CardContent>Second content section with different data.</CardContent>
    </Card>
  ),
};
