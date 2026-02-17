module.exports = {
  extends: ["stylelint-config-standard"],
  rules: {
    "at-rule-no-unknown": [true, { ignoreAtRules: ["tailwind", "apply", "layer", "responsive", "variants", "screen"] }],
    "color-no-hex": true,
    "function-disallowed-list": ["rgb", "rgba", "hsl", "hsla"],
    "declaration-property-value-disallowed-list": {
      "/color/": ["/^#/", "/rgb\\(/", "/rgba\\(/", "/hsl\\(/", "/hsla\\(/"]
    },
    "import-notation": "string",
    "value-keyword-case": ["lower", { ignoreProperties: ["text-rendering", "font-family"] }]
  },
  ignoreFiles: [
    "**/tokens.css",
    "**/tokens.ts",
    "design/tokens.json",
    "**/.next/**",
    "**/dist/**"
  ]
};
