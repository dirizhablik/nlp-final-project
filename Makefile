PYTHON ?= python3.12
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: setup test lint format clean

setup:  ## Create the virtualenv and install pinned dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

test:   ## Run the test suite
	$(BIN)/pytest

lint:   ## Check style and common bugs
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Auto-fix lint issues and format code
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

clean:  ## Remove caches (keeps checkpoints and the venv)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
