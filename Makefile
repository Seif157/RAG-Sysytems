# Document RAG Platform — developer tasks.
#
# On Windows these targets run under Git Bash / WSL. The underlying commands are
# plain `uv run ...` invocations and can be typed directly if `make` is absent.

.DEFAULT_GOAL := help
UV := uv run

.PHONY: help install run fmt fmt-check lint types arch test test-unit cov check clean up down

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install the project with dev extras
	uv venv
	uv pip install -e ".[dev]"

fmt: ## Format the codebase
	$(UV) ruff format src tests
	$(UV) ruff check --fix src tests

fmt-check: ## Verify formatting without modifying files
	$(UV) ruff format --check src tests

lint: ## Lint the codebase
	$(UV) ruff check src tests

types: ## Static type check
	$(UV) mypy

arch: ## Enforce the layering contracts (architecture spec, section 4)
	$(UV) lint-imports --config importlinter.ini

run: ## Start the Streamlit app
	$(UV) streamlit run src/rag/presentation/ui/app.py

test: ## Run the full test suite
	$(UV) pytest

test-unit: ## Run only the fast, hermetic unit tests
	$(UV) pytest -m unit

cov: ## Run tests with a coverage report
	$(UV) pytest --cov --cov-report=term-missing

check: fmt-check lint types arch test ## Everything CI runs

up: ## Start supporting infrastructure (Qdrant, Postgres, Redis)
	docker compose up -d

down: ## Stop supporting infrastructure
	docker compose down

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
