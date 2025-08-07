.PHONY: help install test test-parallel test-subprocess test-ci lint type-check format clean build
.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	pip install -e .[dev]

test:
	pytest tests/ --cov=dftracer --cov-report=term-missing -v

test-parallel: ## Run tests with parallel execution
	pytest tests/ --cov=dftracer --cov-report=term-missing -v -n 2

test-subprocess: ## Run subprocess tests
	pytest tests/ -m subprocess -v -n 2

test-ci: ## Run complete CI pipeline locally
	@echo "Running tests with parallel execution and coverage..."
	pytest tests/ --cov=dftracer --cov-report=xml --cov-report=term-missing -v -n 2
	@echo ""
	@echo ""
	@echo "Running linting checks..."
	ruff check .
	ruff format --check .
	@echo ""
	@echo "Running type checking..."
	mypy dftracer --ignore-missing-imports --exclude dftracer/dynamo.py
	@echo ""
	@echo "All checks passed! ✅"

lint:
	ruff check .

lint-fix:
	ruff check . --fix

format:
	ruff format .

format-check:
	ruff format --check .

type-check:
	mypy dftracer --ignore-missing-imports --exclude dftracer/dynamo.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f coverage.xml
	rm -rf htmlcov/
	rm -rf .pytest_cache/

build:
	python -m build

check-all: lint format-check type-check test-parallel
fix-all: lint-fix format type-check

test-ci-quick: ## Run quick CI checks without coverage
	@echo "Running quick tests..."
	pytest tests/ -v -n 2
	@echo ""
	@echo "Running linting and type checks..."
	ruff check .
	ruff format --check .
	mypy dftracer --ignore-missing-imports --exclude dftracer/dynamo.py
	@echo ""
	@echo "Quick checks passed! ✅"
