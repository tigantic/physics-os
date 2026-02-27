export const TOKENS = {
  color: {
    bg: {
      base: "#0B0C10",
      raised: "#13141A",
      hover: "#1B1C24",
      surface: "#20212B",
    },
    text: {
      primary: "#EAECF0",
      secondary: "#8A8EA0",
      tertiary: "#5A5E70",
    },
    accent: {
      base: "#4B7BF5",
      dim: "rgba(75,123,245,0.10)",
      border: "rgba(75,123,245,0.25)",
      strong: "#6B96FF",
    },
    status: {
      pass: "#34B870",
      fail: "#E05252",
      warn: "#E5A833",
      passBorder: "rgba(52,184,112,0.30)",
      failBorder: "rgba(224,82,82,0.30)",
      warnBorder: "rgba(229,168,51,0.30)",
    },
    border: {
      base: "rgba(255,255,255,0.06)",
      active: "rgba(75,123,245,0.30)",
    },
  },
  radius: {
    outer: 12,
    inner: 8,
    pill: 9999,
    control: 6,
  },
  space: {
    u: 8,
  },
  shadow: {
    raised: "0 1px 3px rgba(0,0,0,0.3), 0 4px 12px rgba(0,0,0,0.2)",
    floating: "0 4px 12px rgba(0,0,0,0.4), 0 12px 40px rgba(0,0,0,0.3)",
  },
  motion: {
    easeOut: "cubic-bezier(0.16, 1, 0.3, 1)",
    easeInOut: "cubic-bezier(0.33, 1, 0.68, 1)",
    hoverMs: 160,
    baseMs: 220,
  },
  type: {
    ui: "Inter",
    mono: "JetBrainsMono",
    math: "SVG",
    fluidXs: "clamp(0.6875rem, 0.65rem + 0.12vw, 0.75rem)",
    fluidSm: "clamp(0.75rem, 0.72rem + 0.1vw, 0.8125rem)",
    fluidBase: "clamp(0.8125rem, 0.78rem + 0.1vw, 0.875rem)",
    fluidLg: "clamp(0.875rem, 0.85rem + 0.08vw, 0.9375rem)",
    fluidXl: "clamp(1rem, 0.97rem + 0.1vw, 1.0625rem)",
    fluid2xl: "clamp(1.125rem, 1.06rem + 0.2vw, 1.25rem)",
  },
} as const;
