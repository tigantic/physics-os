# ── lUX Makefile ────────────────────────────────────────────────────────────────
# Common development and deployment commands.
# ────────────────────────────────────────────────────────────────────────────────

.PHONY: install dev build lint typecheck test test-coverage test-e2e \
        docker-build docker-run docker-stop docker-logs \
        k8s-apply k8s-delete clean help

# ── Variables ───────────────────────────────────────────────────────────────────
VERSION      ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT_SHA   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_DATE   ?= $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
IMAGE_NAME   ?= ghcr.io/tigantic/lux
IMAGE_TAG    ?= $(VERSION)
K8S_NS       ?= lux

# ── Development ─────────────────────────────────────────────────────────────────

install: ## Install all dependencies
	pnpm install

dev: ## Start development server
	pnpm dev

build: ## Build all packages (tokens → core → ui)
	pnpm -w run build

lint: ## Run linters (ESLint + Stylelint)
	pnpm -w run lint

typecheck: ## Run TypeScript type checking
	pnpm -w run typecheck

test: ## Run unit tests
	pnpm -w run test:unit

test-coverage: ## Run tests with coverage reporting
	pnpm -w run test:coverage

test-e2e: ## Run Playwright E2E tests
	pnpm -w run test:e2e

format: ## Format all files with Prettier
	pnpm -w run format

format-check: ## Check formatting without writing
	pnpm -w run format:check

# ── Quality gate (run all checks) ──────────────────────────────────────────────

ci: format-check lint typecheck test ## Run full CI quality gate locally
	@echo "✅ All checks passed"

# ── Docker ──────────────────────────────────────────────────────────────────────

docker-build: ## Build Docker image
	docker build \
		--build-arg BUILD_VERSION=$(VERSION) \
		--build-arg BUILD_COMMIT_SHA=$(COMMIT_SHA) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-t $(IMAGE_NAME):latest \
		.

docker-run: ## Run Docker container locally
	docker compose up -d

docker-stop: ## Stop Docker container
	docker compose down

docker-logs: ## Tail Docker container logs
	docker compose logs -f lux

docker-push: docker-build ## Build and push Docker image
	docker push $(IMAGE_NAME):$(IMAGE_TAG)
	docker push $(IMAGE_NAME):latest

# ── Kubernetes ──────────────────────────────────────────────────────────────────

k8s-apply: ## Apply Kubernetes manifests
	kubectl apply -f deployment/k8s/namespace.yaml
	kubectl apply -f deployment/k8s/configmap.yaml
	kubectl apply -f deployment/k8s/secret.yaml
	kubectl apply -f deployment/k8s/pvc.yaml
	kubectl apply -f deployment/k8s/deployment.yaml
	kubectl apply -f deployment/k8s/service.yaml
	kubectl apply -f deployment/k8s/ingress.yaml
	kubectl apply -f deployment/k8s/hpa.yaml

k8s-set-image: ## Update deployment image tag (usage: make k8s-set-image TAG=abc1234)
	kubectl -n $(K8S_NS) set image deployment/lux lux=$(IMAGE_NAME):$(TAG)

k8s-delete: ## Delete Kubernetes resources
	kubectl delete namespace $(K8S_NS) --ignore-not-found

k8s-rollout: ## Check rollout status
	kubectl -n $(K8S_NS) rollout status deployment/lux

k8s-restart: ## Trigger rolling restart
	kubectl -n $(K8S_NS) rollout restart deployment/lux

# ── Utility ─────────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf packages/ui/.next packages/ui/out
	rm -rf packages/core/dist
	rm -rf node_modules/.cache
	rm -rf packages/ui/coverage packages/core/coverage

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
