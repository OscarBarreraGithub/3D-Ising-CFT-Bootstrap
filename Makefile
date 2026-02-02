.PHONY: install install-dev lock test clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package (production)"
	@echo "  install-dev  - Install package with dev dependencies"
	@echo "  lock         - Generate requirements.lock from pyproject.toml"
	@echo "  test         - Run pytest"
	@echo "  clean        - Remove build artifacts"
	@echo "  worktree     - Create a new worktree (usage: make worktree NAME=feature-name)"

# Install for production (using lock file if available)
install:
	@if [ -f requirements.lock ]; then \
		pip install -r requirements.lock && pip install -e . --no-deps; \
	else \
		pip install -e .; \
	fi

# Install for development
install-dev:
	pip install -e .[dev]

# Generate lock file from pyproject.toml
lock:
	pip install pip-tools
	pip-compile pyproject.toml -o requirements.lock --strip-extras
	@echo "Generated requirements.lock"

# Run tests
test:
	pytest tests/ -v

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Create a new worktree for parallel development
# Usage: make worktree NAME=blocks-implementation
worktree:
ifndef NAME
	$(error NAME is required. Usage: make worktree NAME=feature-name)
endif
	git worktree add ../ising-$(NAME) -b feature/$(NAME)
	@echo "Created worktree at ../ising-$(NAME)"
	@echo "Next steps:"
	@echo "  cd ../ising-$(NAME)"
	@echo "  pip install -e . (or make install-dev)"
	@echo "  claude"
