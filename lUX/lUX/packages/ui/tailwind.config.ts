import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["selector", "[data-theme='dark']"],
  content: ["./src/**/*.{ts,tsx}", "./.storybook/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "var(--color-bg-base)",
        foreground: "var(--color-text-primary)",
        muted: "var(--color-bg-raised)",
        border: "var(--color-border-base)",
        accent: {
          DEFAULT: "var(--color-accent)",
          dim: "var(--color-accent-dim)",
          border: "var(--color-accent-border)",
          strong: "var(--color-accent-strong)",
        },
        pass: "var(--color-status-pass)",
        fail: "var(--color-status-fail)",
        warn: "var(--color-status-warn)",
        surface: "var(--color-bg-surface)",
      },
      borderRadius: {
        lg: "var(--radius-outer)",
        md: "var(--radius-inner)",
        pill: "var(--radius-pill)",
        control: "var(--radius-control)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrainsMono", "ui-monospace", "monospace"],
      },
      fontSize: {
        "fluid-xs": "var(--type-fluid-xs)",
        "fluid-sm": "var(--type-fluid-sm)",
        "fluid-base": "var(--type-fluid-base)",
        "fluid-lg": "var(--type-fluid-lg)",
        "fluid-xl": "var(--type-fluid-xl)",
        "fluid-2xl": "var(--type-fluid-2xl)",
      },
      transitionTimingFunction: {
        "lux-out": "var(--motion-easeOut)",
        "lux-in-out": "var(--motion-easeInOut)",
      },
      transitionDuration: {
        hover: "var(--motion-hover)",
        base: "var(--motion-base)",
      },
      keyframes: {
        "lux-fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "lux-slide-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "lux-scale-in": {
          "0%": { opacity: "0", transform: "scale(0.95)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        "lux-shimmer": {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "lux-drawer-in": {
          "0%": { opacity: "0", transform: "translateX(100%)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
      },
      animation: {
        "lux-fade-in": "lux-fade-in var(--motion-base) var(--motion-easeOut) both",
        "lux-slide-up": "lux-slide-up var(--motion-base) var(--motion-easeOut) both",
        "lux-scale-in": "lux-scale-in var(--motion-base) var(--motion-easeOut) both",
        "lux-shimmer": "lux-shimmer 1.5s linear infinite",
        "lux-drawer-in": "lux-drawer-in var(--motion-base) var(--motion-easeOut) both",
      },
    },
  },
  plugins: [],
};

export default config;
