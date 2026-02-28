# ADR-0022: MCP Server for AI Agent Integration

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

AI coding agents (GitHub Copilot, Claude, Cursor, Windsurf) are increasingly used to compose and execute engineering workflows. The Physics OS's physics simulation capabilities would reach a much larger audience if AI agents could invoke simulations, inspect results, and iterate on parameters — all within a chat or IDE session.

The Model Context Protocol (MCP) by Anthropic provides a standardized JSON-RPC interface for exposing tools to AI agents. Alternatives considered:

1. **Custom REST API only**: Requires each agent platform to write a bespoke integration.
2. **OpenAI function-calling format**: Vendor-specific, no standardized tool discovery.
3. **MCP server**: Standard tool discovery, invocation, and result formatting across all MCP-compatible agents.

## Decision

**physics-os exposes an MCP server that wraps the physics execution API.** Specifically:

1. The MCP server runs as a sidecar process (`physics_os.mcp.server`) alongside the main VM.
2. Tools exposed: `simulate`, `get_result`, `list_domain_packs`, `validate_mesh`, `get_certificate`.
3. Each tool has a JSON Schema input definition and returns structured results with optional visualization URLs.
4. The MCP server authenticates via API key passed in the MCP session header.
5. Rate limiting matches the main API (100 req/min per key during alpha).
6. The server supports both `stdio` transport (for local IDE use) and `SSE` transport (for remote agents).
7. Tool definitions are auto-generated from the REST API OpenAPI spec to prevent drift.

## Consequences

- **Easier:** Any MCP-compatible AI agent can run physics simulations without custom integration code.
- **Easier:** Developers iterate on simulation parameters conversationally — "increase mesh to 256³ and re-run."
- **Easier:** Tool discovery is automatic — agents see available tools and their schemas.
- **Harder:** MCP protocol is early-stage — breaking changes may require server updates.
- **Harder:** Structured result formatting must balance human readability and machine parsability.
- **Risk:** AI agents submitting unbounded compute workloads. Mitigated by per-session compute quotas and mandatory mesh-size limits in tool schemas.
