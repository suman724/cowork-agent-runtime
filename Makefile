.PHONY: help install lint format format-check typecheck test test-integration build check clean coverage

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -e ".[dev]" -e "../cowork-platform[sdk]"

lint: ## Run linter
	ruff check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

format-check: ## Check formatting without modifying
	ruff format --check src/ tests/

typecheck: ## Run type checker
	mypy src/

test: ## Run unit tests
	pytest -m "unit or not integration" -x -q

test-integration: ## Run integration tests
	pytest -m integration -x -q

build: ## Build package
	python -m build

check: lint format-check typecheck test ## CI gate: lint + format-check + typecheck + test

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

coverage: ## Run tests with coverage
	coverage run -m pytest -m "unit or not integration" -x -q
	coverage report
	coverage html
