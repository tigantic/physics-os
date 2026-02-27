export type DataStatus = "ok" | "missing" | "invalid";

export type DataValue<T> =
  | { status: "ok"; value: T }
  | { status: "missing"; reason: string }
  | { status: "invalid"; reason: string; details?: unknown };

export type SHA256 = `sha256:${string}`;
export type ISO8601 = string;
export type UUID = string;
export type SemVer = `${number}.${number}.${number}`;
