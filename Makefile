.PHONY: all build install test clean rust cuda

# Default target
all: build

# Build both CUDA and Rust extensions
build: rust cuda

# Build Rust extension
rust:
	@echo "Building Rust extension..."
	cd rust && maturin develop --release

# Build CUDA extension
cuda:
	@echo "Building CUDA extension..."
	python setup.py build_ext --inplace

# Install the package
install: build
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
