import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";

// Mock next/image — jsdom can't run the real next/image loader
vi.mock("next/image", () => ({
  __esModule: true,
  default: function MockImage(props: Record<string, unknown>) {
    // Render a plain <img> with all passed props for assertion
    // eslint-disable-next-line @next/next/no-img-element, jsx-a11y/alt-text
    return <img {...props} />;
  },
}));

import { ProofImage } from "@/ds/components/ProofImage";

describe("ProofImage", () => {
  it("renders an <img> element with the given alt text", () => {
    render(
      <ProofImage src="/evidence/spectrum.png" alt="Power spectrum" width={800} height={400} />,
    );
    const img = screen.getByRole("img", { name: "Power spectrum" });
    expect(img).toBeInTheDocument();
  });

  it("passes src, width, and height through to the image", () => {
    render(
      <ProofImage src="/evidence/test.png" alt="Test" width={640} height={320} />,
    );
    const img = screen.getByRole("img", { name: "Test" });
    expect(img).toHaveAttribute("src", "/evidence/test.png");
    expect(img).toHaveAttribute("width", "640");
    expect(img).toHaveAttribute("height", "320");
  });

  it("sets blur placeholder props", () => {
    render(
      <ProofImage src="/evidence/test.png" alt="Test" width={100} height={100} />,
    );
    const img = screen.getByRole("img", { name: "Test" });
    expect(img).toHaveAttribute("placeholder", "blur");
    expect(img).toHaveAttribute("blurDataURL");
    const blurUrl = img.getAttribute("blurDataURL") ?? "";
    expect(blurUrl).toContain("data:image/svg+xml;base64,");
  });

  it("sets quality to 85", () => {
    render(
      <ProofImage src="/test.png" alt="Test" width={100} height={100} />,
    );
    const img = screen.getByRole("img", { name: "Test" });
    expect(img).toHaveAttribute("quality", "85");
  });

  it("defaults to lazy loading", () => {
    render(
      <ProofImage src="/test.png" alt="Test" width={100} height={100} />,
    );
    const img = screen.getByRole("img", { name: "Test" });
    expect(img).toHaveAttribute("loading", "lazy");
  });

  it("uses eager loading when priority is true", () => {
    render(
      <ProofImage src="/test.png" alt="Hero" width={100} height={100} priority />,
    );
    const img = screen.getByRole("img", { name: "Hero" });
    expect(img).toHaveAttribute("loading", "eager");
  });

  it("passes responsive sizes attribute", () => {
    render(
      <ProofImage src="/test.png" alt="Test" width={100} height={100} />,
    );
    const img = screen.getByRole("img", { name: "Test" });
    const sizes = img.getAttribute("sizes") ?? "";
    expect(sizes).toContain("max-width: 768px");
    expect(sizes).toContain("100vw");
  });

  it("wraps image in a div with design-system classes", () => {
    const { container } = render(
      <ProofImage src="/test.png" alt="Test" width={100} height={100} />,
    );
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper.tagName).toBe("DIV");
    expect(wrapper.className).toContain("overflow-hidden");
    expect(wrapper.className).toContain("rounded-");
  });

  it("merges wrapperClassName onto the outer div", () => {
    const { container } = render(
      <ProofImage
        src="/test.png"
        alt="Test"
        width={100}
        height={100}
        wrapperClassName="my-custom-class"
      />,
    );
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper.className).toContain("my-custom-class");
  });

  it("merges custom className onto the image", () => {
    render(
      <ProofImage
        src="/test.png"
        alt="Test"
        width={100}
        height={100}
        className="border-2"
      />,
    );
    const img = screen.getByRole("img", { name: "Test" });
    expect(img.className).toContain("border-2");
  });
});
