# Spec: API Stability Contract

## Objective
Define what enterprise users can rely on.

## Stability tiers
- **Stable**: `qtenet.sdk.*`
- **Experimental**: everything else (until promoted)

## Promotion criteria (experimental → stable)
- docstring + docs page
- deterministic behavior defined (or explicitly non-deterministic)
- tests: golden + property
- performance envelope documented

## Deprecation policy
- Deprecations require warnings for at least one minor release.
- Removals only in major releases.
