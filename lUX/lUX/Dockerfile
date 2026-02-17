# ── Stage 1: Install dependencies ───────────────────────────────────────
FROM node:20-alpine AS deps
RUN corepack enable && corepack prepare pnpm@9.0.0 --activate
WORKDIR /app

# Copy only dependency-resolution files (maximise Docker layer caching)
COPY pnpm-lock.yaml pnpm-workspace.yaml package.json ./
COPY packages/core/package.json packages/core/package.json
COPY packages/ui/package.json packages/ui/package.json

RUN pnpm install --frozen-lockfile

# ── Stage 2: Build ─────────────────────────────────────────────────────
FROM node:20-alpine AS builder
RUN corepack enable && corepack prepare pnpm@9.0.0 --activate
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/packages/core/node_modules ./packages/core/node_modules
COPY --from=deps /app/packages/ui/node_modules ./packages/ui/node_modules
COPY . .

# Build core → ui (standalone output).
# Token CSS/TS files are pre-generated and committed to source; the build:tokens
# step requires tools/ and design/ which are excluded by .dockerignore.
RUN pnpm -r build

# ── Stage 3: Production runner ─────────────────────────────────────────
FROM node:20-alpine AS runner

# Build-time metadata (injected by CI)
ARG BUILD_VERSION=dev
ARG BUILD_COMMIT_SHA=unknown
ARG BUILD_DATE

LABEL org.opencontainers.image.title="lUX Proof Viewer" \
      org.opencontainers.image.description="Forensic inspection interface for HyperTensor TPC proof certificates" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.revision="${BUILD_COMMIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/tigantic/HyperTensor-VM" \
      org.opencontainers.image.licenses="MIT"

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"
ENV LUX_VERSION=${BUILD_VERSION}
ENV LUX_COMMIT_SHA=${BUILD_COMMIT_SHA}

# Copy Next.js standalone output + static + public assets
COPY --from=builder /app/packages/ui/.next/standalone ./
COPY --from=builder /app/packages/ui/.next/static ./packages/ui/.next/static
COPY --from=builder /app/packages/ui/public ./packages/ui/public

# Copy core fixtures (proof packages + domain packs for gallery)
COPY --from=builder /app/packages/core/tests/fixtures ./packages/core/tests/fixtures

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1:3000/api/ready || exit 1

CMD ["node", "packages/ui/server.js"]
