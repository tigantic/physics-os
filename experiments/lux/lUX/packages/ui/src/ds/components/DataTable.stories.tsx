import type { Meta, StoryObj } from "@storybook/react";
import { within, expect } from "@storybook/test";
import { DataTable, type DataTableColumn } from "./DataTable";

interface DemoRow {
  id: string;
  metric: string;
  value: number;
  status: string;
}

const DEMO_COLUMNS: DataTableColumn<DemoRow>[] = [
  { key: "metric", header: "Metric", cell: (r) => <span className="font-mono text-xs">{r.metric}</span> },
  { key: "value", header: "Value", cell: (r) => <span className="font-mono text-xs">{r.value.toFixed(4)}</span>, numeric: true },
  { key: "status", header: "Status", cell: (r) => <span className="text-xs">{r.status}</span> },
];

const DEMO_DATA: DemoRow[] = [
  { id: "1", metric: "residual_l2", value: 3.2e-7, status: "PASS" },
  { id: "2", metric: "cd_error_pct", value: 1.3, status: "PASS" },
  { id: "3", metric: "convergence_rate", value: 0.92, status: "PASS" },
  { id: "4", metric: "mass_balance", value: 0.0001, status: "PASS" },
  { id: "5", metric: "courant_max", value: 0.85, status: "WARN" },
];

const meta: Meta<typeof DataTable> = {
  title: "DS/DataTable",
  component: DataTable,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof DataTable>;

export const Default: Story = {
  render: () => (
    <DataTable<DemoRow>
      columns={DEMO_COLUMNS}
      data={DEMO_DATA}
      rowKey={(r) => r.id}
      caption="Simulation metrics"
    />
  ),
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    expect(canvas.getByText("Metric")).toBeVisible();
    expect(canvas.getByText("residual_l2")).toBeVisible();
    expect(canvas.getByText("PASS")).toBeVisible();
  },
};

export const Compact: Story = {
  render: () => (
    <DataTable<DemoRow>
      columns={DEMO_COLUMNS}
      data={DEMO_DATA}
      rowKey={(r) => r.id}
      caption="Compact metrics table"
      compact
    />
  ),
};

export const Striped: Story = {
  render: () => (
    <DataTable<DemoRow>
      columns={DEMO_COLUMNS}
      data={DEMO_DATA}
      rowKey={(r) => r.id}
      caption="Striped metrics table"
      striped
    />
  ),
};

export const WithMaxRows: Story = {
  render: () => (
    <DataTable<DemoRow>
      columns={DEMO_COLUMNS}
      data={DEMO_DATA}
      rowKey={(r) => r.id}
      caption="Limited rows"
      maxRows={3}
    />
  ),
};

export const Empty: Story = {
  render: () => (
    <DataTable<DemoRow>
      columns={DEMO_COLUMNS}
      data={[]}
      rowKey={(r) => r.id}
      caption="Empty table"
      emptyState={<div className="py-8 text-center text-sm text-[var(--color-text-tertiary)]">No data available</div>}
    />
  ),
};
