import js from "@eslint/js";
import next from "@next/eslint-plugin-next";
import tseslint from "typescript-eslint";

export default [
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    plugins: { "@next/next": next },
    rules: {
      "@next/next/no-html-link-for-pages": "off",
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/consistent-type-imports": "error"
    }
  }
];
