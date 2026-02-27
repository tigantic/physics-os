import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { DataTable, type DataTableColumn } from "@/ds/components/DataTable";

interface Row {
  id: string;
  name: string;
  value: number;
}

const columns: DataTableColumn<Row>[] = [
  { key: "id", header: "ID", cell: (r) => r.id, className: "w-24" },
  { key: "name", header: "Name", cell: (r) => r.name },
  { key: "value", header: "Value", cell: (r) => r.value.toFixed(2), numeric: true },
];

const data: Row[] = [
  { id: "g-001", name: "Stability", value: 0.95 },
  { id: "g-002", name: "Convergence", value: 0.87 },
];

describe("DataTable", () => {
  it("renders column headers", () => {
    render(<DataTable columns={columns} data={data} rowKey={(r) => r.id} />);
    expect(screen.getByText("ID")).toBeInTheDocument();
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Value")).toBeInTheDocument();
  });

  it("renders all data rows", () => {
    render(<DataTable columns={columns} data={data} rowKey={(r) => r.id} />);
    expect(screen.getByText("Stability")).toBeInTheDocument();
    expect(screen.getByText("Convergence")).toBeInTheDocument();
    expect(screen.getByText("0.95")).toBeInTheDocument();
  });

  it("renders caption as sr-only when provided", () => {
    const { container } = render(
      <DataTable columns={columns} data={data} rowKey={(r) => r.id} caption="Gate results" />,
    );
    const caption = container.querySelector("caption");
    expect(caption).toBeInTheDocument();
    expect(caption?.textContent).toBe("Gate results");
    expect(caption?.className).toContain("sr-only");
  });

  it("renders emptyState when data is empty", () => {
    render(
      <DataTable
        columns={columns}
        data={[]}
        rowKey={(r) => r.id}
        emptyState={<div>No data available</div>}
      />,
    );
    expect(screen.getByText("No data available")).toBeInTheDocument();
  });

  it("applies numeric alignment to numeric columns", () => {
    const { container } = render(
      <DataTable columns={columns} data={data} rowKey={(r) => r.id} />,
    );
    const ths = container.querySelectorAll("th");
    const valueTh = Array.from(ths).find((th) => th.textContent === "Value");
    expect(valueTh?.className).toContain("text-right");
    expect(valueTh?.className).toContain("tabular-nums");
  });

  it("caps visible rows with maxRows and shows 'Show all' button", () => {
    const manyRows: Row[] = Array.from({ length: 20 }, (_, i) => ({
      id: `r-${i}`,
      name: `Row ${i}`,
      value: i * 0.1,
    }));
    render(<DataTable columns={columns} data={manyRows} rowKey={(r) => r.id} maxRows={5} />);

    // Only 5 rows rendered initially
    const tbody = screen.getByRole("table").querySelector("tbody");
    expect(tbody?.querySelectorAll("tr")).toHaveLength(5);

    // "Show all" button present
    const showAll = screen.getByText(/Show all 20 rows/);
    expect(showAll).toBeInTheDocument();

    // Click expands to all rows
    fireEvent.click(showAll);
    expect(tbody?.querySelectorAll("tr")).toHaveLength(20);
  });

  it("does not show 'Show all' button when data fits within maxRows", () => {
    render(<DataTable columns={columns} data={data} rowKey={(r) => r.id} maxRows={10} />);
    expect(screen.queryByText(/Show all/)).toBeNull();
  });
});
