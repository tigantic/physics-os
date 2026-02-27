import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { PackageSummary } from "@luxury/core";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/packages",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("next/link", () => ({
  __esModule: true,
  default: ({ href, children, ...props }: { href: string; children: React.ReactNode;[k: string]: unknown }) => (
    <a href={href} {...props}>{children}</a>
  ),
}));

import { PackageList } from "@/app/packages/PackageList";

const mockPackages: readonly PackageSummary[] = [
  {
    id: "pass",
    domain_id: "com.physics.vlasov",
    verdict_status: "PASS",
    quality_score: 0.95,
    timestamp: "2025-01-15T12:00:00Z",
    solver_name: "vlasov-solver",
  },
  {
    id: "fail",
    domain_id: "com.physics.navier-stokes",
    verdict_status: "FAIL",
    quality_score: 0.23,
    timestamp: "2025-01-14T10:00:00Z",
    solver_name: "ns-solver",
  },
  {
    id: "warn",
    domain_id: "com.physics.vlasov",
    verdict_status: "WARN",
    quality_score: 0.67,
    timestamp: "2025-01-13T08:00:00Z",
    solver_name: "vlasov-solver",
  },
] as const;

describe("PackageList", () => {
  it("renders all packages in a table", () => {
    render(<PackageList packages={mockPackages} />);
    expect(screen.getByText("pass")).toBeInTheDocument();
    expect(screen.getByText("fail")).toBeInTheDocument();
    expect(screen.getByText("warn")).toBeInTheDocument();
  });

  it("renders package IDs as links to /packages/[id]", () => {
    render(<PackageList packages={mockPackages} />);
    const passLink = screen.getByRole("link", { name: "pass" });
    expect(passLink).toHaveAttribute("href", "/packages/pass");
    const failLink = screen.getByRole("link", { name: "fail" });
    expect(failLink).toHaveAttribute("href", "/packages/fail");
  });

  it("renders verdict chips", () => {
    render(<PackageList packages={mockPackages} />);
    expect(screen.getByText("PASS")).toBeInTheDocument();
    expect(screen.getByText("FAIL")).toBeInTheDocument();
    expect(screen.getByText("WARN")).toBeInTheDocument();
  });

  it("renders quality scores as percentages", () => {
    render(<PackageList packages={mockPackages} />);
    expect(screen.getByText("95%")).toBeInTheDocument();
    expect(screen.getByText("23%")).toBeInTheDocument();
    expect(screen.getByText("67%")).toBeInTheDocument();
  });

  it("renders solver names", () => {
    render(<PackageList packages={mockPackages} />);
    expect(screen.getAllByText("vlasov-solver").length).toBe(2);
    expect(screen.getByText("ns-solver")).toBeInTheDocument();
  });

  it("renders search input", () => {
    render(<PackageList packages={mockPackages} />);
    expect(screen.getByRole("searchbox")).toBeInTheDocument();
  });

  it("filters packages by search query", () => {
    render(<PackageList packages={mockPackages} />);
    const search = screen.getByRole("searchbox");
    fireEvent.change(search, { target: { value: "navier" } });
    // Only "fail" package has "navier-stokes" domain
    expect(screen.getByText("fail")).toBeInTheDocument();
    expect(screen.queryByText("pass")).not.toBeInTheDocument();
    expect(screen.queryByText("warn")).not.toBeInTheDocument();
  });

  it("shows empty state when no packages match", () => {
    render(<PackageList packages={mockPackages} />);
    const search = screen.getByRole("searchbox");
    fireEvent.change(search, { target: { value: "nonexistent" } });
    expect(screen.getByText("No matches")).toBeInTheDocument();
  });

  it("shows empty state when no packages at all", () => {
    render(<PackageList packages={[]} />);
    expect(screen.getByText("No packages found")).toBeInTheDocument();
  });

  it("renders accessible table caption", () => {
    render(<PackageList packages={mockPackages} />);
    // DataTable renders caption as sr-only
    expect(screen.getByText("Available proof packages")).toBeInTheDocument();
  });
});
