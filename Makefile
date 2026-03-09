.PHONY: all build install test clean rust cuda

# Default target
all: build

# Build native extensions through Python packaging
build:
	@echo "Building BarqTrain backends via Python packaging..."
	pip install -e . --no-deps

# Rust backend note
rust:
	@echo "Rust backend is built automatically by: pip install -e ."

# CUDA backend note
cuda:
	@echo "CUDA backend is built automatically when CUDA_HOME is set."

# Install the package
install:
	@echo "Installing BarqTrain..."
	pip install -e .

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v

# Run benchmark
benchmark:
	@echo "Running baseline benchmark..."
	python -m barqtrain.benchmarks.baseline --steps 100

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete

# Format code
format:
	@echo "Formatting code..."
	black python/
	isort python/

# Lint code
lint:
	@echo "Linting code..."
	black --check python/
	isort --check-only python/

# Type check
typecheck:
	@echo "Type checking..."
	mypy python/

# Release checklist
release-check: clean format lint test
	@echo "Release checklist complete!"
