.PHONY: help install run run-anthropic lint format format-check typecheck test test-integration test-jsonrpc test-chat test-chat-anthropic build check clean coverage

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -e ".[dev]" -e "../cowork-platform[sdk]"

run: ## Run the agent-runtime in stdio mode (sources .env)
	set -a && [ -f .env ] && . .env; set +a && .venv/bin/python -m agent_host.main

run-anthropic: ## Run the agent-runtime with Anthropic Claude (sources .env.anthropic)
	@[ -f .env.anthropic ] || (echo "ERROR: .env.anthropic not found. Copy the example and add your API key:" && echo "  cp .env.anthropic.example .env.anthropic" && exit 1)
	set -a && . .env.anthropic; set +a && .venv/bin/python -m agent_host.main

lint: ## Run linter
	.venv/bin/ruff check src/ tests/

format: ## Auto-format code
	.venv/bin/ruff format src/ tests/
	.venv/bin/ruff check --fix src/ tests/

format-check: ## Check formatting without modifying
	.venv/bin/ruff format --check src/ tests/

typecheck: ## Run type checker
	.venv/bin/mypy src/

test: ## Run unit tests
	.venv/bin/pytest -m "unit or not integration" -x -q

test-integration: ## Run integration tests
	.venv/bin/pytest -m integration -x -q

test-jsonrpc: ## Smoke test: CreateSession + Shutdown over JSON-RPC (needs backend services)
	set -a && [ -f .env ] && . .env; set +a && .venv/bin/python scripts/test-jsonrpc.py

test-chat: ## Full chat test: CreateSession + StartTask + LLM response (needs backend + LLM)
	set -a && [ -f .env ] && . .env; set +a && .venv/bin/python scripts/test-chat.py

test-chat-anthropic: ## Full chat test using Anthropic Claude (needs backend + Anthropic API key)
	@[ -f .env.anthropic ] || (echo "ERROR: .env.anthropic not found. Copy the example and add your API key:" && echo "  cp .env.anthropic.example .env.anthropic" && exit 1)
	set -a && . .env.anthropic; set +a && .venv/bin/python scripts/test-chat.py

build: ## Build package
	.venv/bin/python -m build

check: lint format-check typecheck test ## CI gate: lint + format-check + typecheck + test

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

coverage: ## Run tests with coverage
	.venv/bin/coverage run -m pytest -m "unit or not integration" -x -q
	.venv/bin/coverage report
	.venv/bin/coverage html
