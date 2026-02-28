# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Constitutional Audit report with compliance tracking
- Comprehensive test suite for stores, hooks, and components
- Secure token storage replacing localStorage per Article IX
- Error Boundaries for graceful failure handling per Article VIII
- API transformation layer for snake_case/camelCase conversion
- Full README.md with architecture documentation
- Test utilities with mock factories
- Vitest configuration with 85% coverage thresholds

### Changed

- Updated API client to use secure token storage
- Improved type definitions with consistent naming

### Fixed

- Type errors resolved (0 TypeScript errors)
- Dev server port conflict (changed to 3010)

### Security

- Migrated auth tokens from localStorage to memory-first storage
- Added sessionStorage fallback with base64 encoding

## [0.1.0] - 2026-01-18

### Added

- Initial HyperTensor UI implementation
- Next.js 14 App Router with TypeScript strict mode
- CFD component library:
  - `MeshViewer` - 3D mesh visualization with React Three Fiber
  - `SimulationCard` - Simulation status display
  - `ResidualChart` - Live convergence plots with Recharts
  - `BoundaryEditor` - Boundary condition configuration
  - `RunControls` - Simulation control panel
  - `ParameterForm` - Solver settings form
- Layout components:
  - `DashboardShell` - Page wrapper with breadcrumbs
  - `Sidebar` - Navigation sidebar
  - `Header` - App header with theme toggle
- Dashboard pages:
  - Home dashboard with system overview
  - Simulations list and detail views
  - Meshes list and detail views
  - New simulation wizard
- State management:
  - TanStack Query for server state
  - Zustand for client state
  - React Hook Form + Zod for forms
- Type system:
  - Full TypeScript coverage
  - Simulation types with solver settings
  - Mesh types with boundary patches
  - API response types
- Infrastructure:
  - API client with fetch wrapper
  - React Query hooks for all CRUD operations
  - WebSocket types for real-time updates
  - Theme provider with dark/light mode
  - OpenAPI specification

### Dependencies

- Next.js 14.2.21
- React 18.3.1
- TypeScript 5.7.2
- Tailwind CSS 3.4.17
- TanStack Query 5.62.0
- Zustand 5.0.2
- React Three Fiber 8.15.0
- Recharts 2.10.0
- React Hook Form 7.54.0
- Zod 3.24.1
- shadcn/ui (Radix primitives)

---

## Types of Changes

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

[Unreleased]: https://github.com/physics_os/hypertensor-ui/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/physics_os/hypertensor-ui/releases/tag/v0.1.0
