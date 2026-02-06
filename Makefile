.PHONY: install install-dev install-all clean test lint format typecheck demo help experiments experiments_dashboard

# Default Python version
PYTHON := python3.12
VENV := .venv
ACTIVATE := . $(VENV)/bin/activate

help: ## Show this help message
	@echo "Thermo Manifold - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using uv
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && uv sync


clean: ## Remove virtual environment and build artifacts
	rm -rf $(VENV)
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

test: ## Run all tests
	$(ACTIVATE) && pytest tests/ -v

test-cov: ## Run tests with coverage
	$(ACTIVATE) && pytest tests/ -v --cov=sensorium --cov-report=term-missing

lint: ## Run linter (ruff)
	$(ACTIVATE) && ruff check sensorium/

lint-fix: ## Run linter and fix issues
	$(ACTIVATE) && ruff check --fix sensorium/

format: ## Format code with black and isort
	$(ACTIVATE) && black sensorium/
	$(ACTIVATE) && isort sensorium/

format-check: ## Check formatting without making changes
	$(ACTIVATE) && black --check sensorium/
	$(ACTIVATE) && isort --check-only sensorium/

typecheck: ## Run type checking with mypy
	$(ACTIVATE) && mypy sensorium/

check: lint format-check typecheck ## Run all checks (lint, format, typecheck)

# ============================================================================
# SIMULATION
# ============================================================================

run: ## Run the physics simulation with dashboard
	$(ACTIVATE) && python run.py

run-profile: ## Run simulation with GPU profiling
	$(ACTIVATE) && python run.py --profile

# ============================================================================
# EXPERIMENTS (Metal/MPS kernel-based)
# ============================================================================

experiments: ## Run all kernel experiments
	@mkdir -p paper/figures paper/tables
	$(ACTIVATE) && python -m sensorium.experiments --experiment all

experiments_dashboard: ## Run all experiments with live dashboard + video recording
	@mkdir -p paper/figures paper/tables paper/videos
	$(ACTIVATE) && python -m sensorium.experiments --experiment all --dashboard --dashboard-fps 30

# ============================================================================
# PAPER GENERATION
# ============================================================================

paper: ## Run experiments and compile paper (single command)
	@echo "============================================"
	@echo "SENSORIUM MANIFOLD - PAPER GENERATION"
	@echo "============================================"
	@mkdir -p paper/figures paper/tables
	$(ACTIVATE) && python -m sensorium.experiments --experiment all
	@echo ""
	@echo "Compiling LaTeX..."
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && pdflatex -interaction=nonstopmode main.tex
	@echo ""
	@echo "============================================"
	@echo "Done! Paper available at: paper/main.pdf"
	@echo "============================================"

paper-experiments: ## Run experiments only (generate tables and figures)
	@mkdir -p paper/figures paper/tables
	$(ACTIVATE) && python -m sensorium.experiments --experiment all

paper-compile: ## Compile LaTeX only (assumes experiments already run)
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && bibtex main
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && pdflatex -interaction=nonstopmode main.tex
	@echo "Paper compiled: paper/main.pdf"

paper-clean: ## Clean paper build artifacts
	cd paper && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf
	rm -rf paper/figures/*.pdf paper/tables/*.tex
