/**
 * Structured error thrown when a requested resource (proof package, domain pack,
 * artifact) does not exist in the data provider.
 *
 * API routes use `instanceof ProviderNotFoundError` to return 404 status,
 * eliminating fragile string matching on error messages.
 */
export class ProviderNotFoundError extends Error {
  /** The type of resource that was not found. */
  readonly resource: "package" | "domain" | "artifact";

  /** The identifier that was looked up. */
  readonly id: string;

  /** Human-readable resource labels for error messages. */
  private static readonly LABELS: Record<string, string> = {
    package: "Proof package",
    domain: "DomainPack",
    artifact: "Artifact",
  };

  constructor(resource: "package" | "domain" | "artifact", id: string) {
    const label = ProviderNotFoundError.LABELS[resource] ?? resource;
    super(`${label} not found: ${id}`);
    this.name = "ProviderNotFoundError";
    this.resource = resource;
    this.id = id;
  }
}
