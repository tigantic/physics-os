import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx}", "./.storybook/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "var(--color-bg-base)",
        foreground: "var(--color-text-primary)",
        muted: "var(--color-bg-raised)",
        border: "var(--color-border-base)",
        accent: "var(--color-accent-gold)",
        pass: "var(--color-verdict-pass)",
        fail: "var(--color-verdict-fail)",
        warn: "var(--color-verdict-warn)"
      },
      borderRadius: {
        lg: "var(--radius-outer)",
        md: "var(--radius-inner)"
      },
      fontFamily: {
        sans: ["IBMPlexSans", "system-ui", "sans-serif"],
        mono: ["JetBrainsMono", "ui-monospace", "monospace"]
      }
    }
  },
  plugins: []
};

export default config;
