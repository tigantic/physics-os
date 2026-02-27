# FluidElite ZK Makefile
# Production-ready commands for building, testing, and deploying

.PHONY: all build test bench clean docker help

# Default target
all: build test

# ============================================
# Build targets
# ============================================

build:
	cargo build --release --features halo2

build-cli:
	cargo build --release --features halo2 --bin fluidelite-cli

build-server:
	cargo build --release --features production --bin fluidelite-server

build-gpu:
	cargo build --release --features production-gpu --bin fluidelite-server

build-python:
	maturin develop --features python

build-all: build-cli build-server
	@echo "All binaries built in target/release/"

# ============================================
# Test targets
# ============================================

test:
	cargo test --features halo2

test-verbose:
	cargo test --features halo2 -- --nocapture

test-lib:
	cargo test --features halo2 --lib

test-integration:
	cargo test --features halo2 --test integration

test-circuit:
	cargo test --features halo2 halo2_impl

# ============================================
# Benchmark targets
# ============================================

bench:
	cargo bench --features halo2 --bench proof_bench

bench-constraints:
	cargo bench --features halo2 --bench constraint_bench

# ============================================
# Run targets
# ============================================

run-server:
	cargo run --release --features production --bin fluidelite-server

run-server-gpu:
	ICICLE_BACKEND_INSTALL_DIR=/opt/icicle/lib/backend \
	cargo run --release --features production-gpu --bin fluidelite-server

run-server-test:
	cargo run --release --features production --bin fluidelite-server -- --test

run-production:
	./start-production.sh

run-cli:
	cargo run --release --features halo2 --bin fluidelite-cli -- $(ARGS)

prove:
	cargo run --release --features halo2 --bin fluidelite-cli -- prove --token 42

stats:
	cargo run --release --features halo2 --bin fluidelite-cli -- stats

# ============================================
# Docker targets
# ============================================

docker-build:
	docker build -t fluidelite-zk:latest .

docker-build-prod:
	docker build -f Dockerfile.prod -t fluidelite-zk:prod .

docker-run:
	docker run -p 8080:8080 fluidelite-zk:latest

docker-run-prod:
	docker run --gpus all -p 8080:8080 -p 9090:9090 \
		-e FLUIDELITE_API_KEY=$${FLUIDELITE_API_KEY:-} \
		fluidelite-zk:prod

docker-compose-up:
	docker-compose up -d

docker-compose-prod:
	docker compose -f docker-compose.prod.yml up -d

docker-compose-monitoring:
	docker compose -f docker-compose.prod.yml --profile monitoring up -d

docker-compose-down:
	docker-compose down

docker-compose-logs:
	docker-compose logs -f prover

# ============================================
# Development targets
# ============================================

check:
	cargo check --features halo2

clippy:
	cargo clippy --features halo2 -- -D warnings

fmt:
	cargo fmt

fmt-check:
	cargo fmt -- --check

doc:
	cargo doc --features halo2 --no-deps --open

clean:
	cargo clean
	rm -rf target/criterion

# ============================================
# Help
# ============================================

help:
	@echo "FluidElite ZK Makefile"
	@echo ""
	@echo "Build:"
	@echo "  make build          - Build library with Halo2"
	@echo "  make build-cli      - Build CLI tool"
	@echo "  make build-server   - Build prover node"
	@echo "  make build-python   - Build Python bindings"
	@echo "  make build-all      - Build everything"
	@echo ""
	@echo "Test:"
	@echo "  make test           - Run all tests"
	@echo "  make test-verbose   - Run tests with output"
	@echo "  make test-circuit   - Run circuit tests only"
	@echo ""
	@echo "Benchmark:"
	@echo "  make bench          - Run proof benchmarks"
	@echo ""
	@echo "Run:"
	@echo "  make run-server     - Start REST API server"
	@echo "  make prove          - Generate a proof"
	@echo "  make stats          - Show circuit stats"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run container"
	@echo "  make docker-compose-up - Start full stack"
	@echo ""
	@echo "Dev:"
	@echo "  make check          - Check compilation"
	@echo "  make clippy         - Run linter"
	@echo "  make fmt            - Format code"
	@echo "  make doc            - Generate docs"
	@echo "  make clean          - Clean build artifacts"
