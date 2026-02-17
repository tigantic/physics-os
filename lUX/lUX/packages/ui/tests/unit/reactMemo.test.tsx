import { describe, it, expect, vi } from "vitest";

// MathJax does not work in jsdom
vi.mock("@/features/math/MathBlock", () => ({
  MathBlock: ({ latex }: { latex: string }) => <div data-testid="math-block">{`[LaTeX: ${latex}]`}</div>,
}));

import { SummaryScreen } from "@/features/screens/Summary";
import { TimelineScreen } from "@/features/screens/Timeline";
import { GatesScreen } from "@/features/screens/Gates";
import { EvidenceScreen } from "@/features/screens/Evidence";
import { IntegrityScreen } from "@/features/screens/Integrity";
import { CompareScreen } from "@/features/screens/Compare";
import { ReproduceScreen } from "@/features/screens/Reproduce";

describe("Screen component React.memo wrappers", () => {
  const components = [
    { name: "SummaryScreen", component: SummaryScreen },
    { name: "TimelineScreen", component: TimelineScreen },
    { name: "GatesScreen", component: GatesScreen },
    { name: "EvidenceScreen", component: EvidenceScreen },
    { name: "IntegrityScreen", component: IntegrityScreen },
    { name: "CompareScreen", component: CompareScreen },
    { name: "ReproduceScreen", component: ReproduceScreen },
  ];

  it.each(components)("$name is wrapped with React.memo", ({ component }) => {
    // React.memo components have $$typeof = Symbol.for("react.memo")
    expect((component as unknown as { $$typeof: symbol }).$$typeof).toBe(Symbol.for("react.memo"));
  });

  it.each(components)("$name has displayName set", ({ name, component }) => {
    expect(component.displayName).toBe(name);
  });
});
