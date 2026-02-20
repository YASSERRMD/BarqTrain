# Contributing to BarqTrain

Thank you for your interest in contributing to BarqTrain! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA Toolkit 11.0+ (for CUDA development)
- Rust 1.70+ (for Rust development)
- Git

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/YASSERRMD/BarqTrain.git
cd BarqTrain

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install additional build tools
pip install maturin ninja cmake
```

### Building Extensions

```bash
# Build Rust extension
cd rust
maturin develop --release

# Build CUDA extension (requires CUDA toolkit)
cd ../csrc
mkdir build && cd build
cmake ..
make -j

# Or use the Makefile
cd ../../
make build
```

## Development Setup

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rmsnorm.py -v

# Run with coverage
pytest --cov=barqtrain tests/
```

### Code Formatting

```bash
# Format Python code
make format
# or
black python/
isort python/

# Check formatting
make lint
# or
black --check python/
isort --check-only python/
```

### Type Checking

```bash
# Run type checker
make typecheck
# or
mypy python/
```

## Contribution Guidelines

### Types of Contributions

We welcome the following types of contributions:

1. **Bug Fixes** - Fix reported issues
2. **New Features** - Add new functionality
3. **Performance Improvements** - Optimize existing code
4. **Documentation** - Improve docs and examples
5. **Tests** - Add or improve test coverage

### Before Contributing

1. Check existing [issues](https://github.com/YASSERRMD/BarqTrain/issues) and [pull requests](https://github.com/YASSERRMD/BarqTrain/pulls)
2. Discuss major changes in an issue first
3. Follow the coding standards below
4. Write tests for new functionality
5. Update documentation as needed

### Small Changes

For small changes (typos, bug fixes, minor improvements):
1. Fork the repository
2. Create a branch: `git checkout -b fix/description`
3. Make your changes
4. Commit with conventional commit format
5. Push and create a pull request

### Major Changes

For significant features or refactoring:
1. Open an issue to discuss the change first
2. Get feedback from maintainers
3. Follow the small changes process above

## Pull Request Process

### 1. Fork and Branch

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/BarqTrain.git
cd BarqTrain

# Add upstream remote
git remote add upstream https://github.com/YASSERRMD/BarqTrain.git

# Create a feature branch
git checkout -b feat/your-feature-name
```

### 2. Make Changes

- Write clear, concise code
- Add tests for new functionality
- Update documentation
- Follow coding standards
- Use atomic commits

### 3. Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `perf` - Performance improvement
- `refactor` - Code restructuring
- `test` - Adding tests
- `docs` - Documentation changes
- `build` - Build system changes
- `ci` - CI/CD changes
- `chore` - Other changes

**Examples:**
```
feat(cuda/rmsnorm): implement backward pass

Add gradient computation for fused RMSNorm kernel with
proper handling of weight gradients.

Fixes #123
```

### 4. Test Your Changes

```bash
# Run tests
make test

# Run benchmarks (if applicable)
make benchmark

# Check formatting
make lint
make typecheck
```

### 5. Create Pull Request

1. Push your branch: `git push origin feat/your-feature-name`
2. Visit: https://github.com/YASSERRMD/BarqTrain/pull/new/feat/your-feature-name
3. Fill out the PR template
4. Link related issues
5. Wait for review

### PR Review Process

- Maintainers will review your PR
- Address feedback comments
- Keep the conversation focused and constructive
- Update your PR as needed
- Once approved, maintainers will merge

## Coding Standards

### Python Code

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Add type hints for public APIs
- Write docstrings (Google style preferred)

```python
def fused_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Apply fused RMSNorm to input tensor.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor with same shape as input
        
    Example:
        >>> x = torch.randn(2, 128, 768)
        >>> weight = torch.ones(768)
        >>> output = fused_rms_norm(x, weight)
    """
    ...
```

### CUDA/C++ Code

- Use modern C++ (C++17)
- Follow CUDA best practices
- Add comments for complex kernel logic
- Use descriptive variable names
- Check for CUDA errors

```cpp
// Check for CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)
```

### Rust Code

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Document public APIs with rustdoc
- Handle errors appropriately with `Result`

### Git Conventions

- **Atomic commits** - One logical change per commit
- **No merge commits** - Use rebase to keep history clean
- **Sign commits** - Optional but recommended
- **Reference issues** - Use `Fixes #123` or `Closes #123`

### Documentation

- Keep README.md up to date
- Add docstrings to all public functions
- Include usage examples
- Update IMPLEMENTATION_PLAN.md if architecture changes
- Document breaking changes

## Testing Guidelines

### Unit Tests

- Test individual functions and methods
- Mock external dependencies
- Test edge cases and error conditions
- Aim for high code coverage

### Integration Tests

- Test component interactions
- Use real data when appropriate
- Test with CUDA device if available

### Numerical Parity Tests

- Compare custom kernels against PyTorch implementations
- Test with different data types (fp32, fp16, bf16)
- Verify gradients with gradcheck

```python
def test_fused_rmsnorm_forward_parity():
    """Test that forward pass matches PyTorch RMSNorm"""
    x = torch.randn(2, 128, 768, device='cuda', dtype=torch.float16)
    weight = torch.ones(768, device='cuda', dtype=torch.float16)
    
    # PyTorch baseline
    output_pt = torch.nn.functional.rms_norm(x, (768,), weight=weight)
    
    # BarqTrain implementation
    output_bt = fused_rms_norm(x, weight)
    
    # Check parity
    assert torch.allclose(output_pt, output_bt, rtol=1e-3, atol=1e-5)
```

### Benchmark Tests

- Track performance over time
- Compare against baselines
- Test with different input sizes
- Report tokens/sec and memory usage

## Getting Help

- **Issues**: https://github.com/YASSERRMD/BarqTrain/issues
- **Discussions**: https://github.com/YASSERRMD/BarqTrain/discussions
- **Email**: Create a GitHub discussion

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in significant feature announcements

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

Thank you for contributing to BarqTrain! 🚀
