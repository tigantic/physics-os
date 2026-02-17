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
        warn: "var(--color-verdict-warn)",
      },
      borderRadius: {
        lg: "var(--radius-outer)",
        md: "var(--radius-inner)",
      },
      fontFamily: {
        sans: ["IBMPlexSans", "system-ui", "sans-serif"],
        mono: ["JetBrainsMono", "ui-monospace", "monospace"],
      },
      transitionTimingFunction: {
        "lux-out": "var(--motion-easeOut)",
        "lux-in-out": "var(--motion-easeInOut)",
      },
      transitionDuration: {
        fast: "180ms",
        base: "220ms",
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
      },
      animation: {
        "lux-fade-in": "lux-fade-in 220ms var(--motion-easeOut) both",
        "lux-slide-up": "lux-slide-up 220ms var(--motion-easeOut) both",
        "lux-scale-in": "lux-scale-in 220ms var(--motion-easeOut) both",
        "lux-shimmer": "lux-shimmer 1.5s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
