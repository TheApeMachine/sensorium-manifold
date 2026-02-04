.PHONY: install install-dev install-all clean test lint format typecheck demo help

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
	$(ACTIVATE) && uv sync --extra experiments

install-dev: ## Install with development dependencies
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && uv sync --extra dev

install-viz: ## Install with visualization dependencies
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && uv sync --extra viz

install-all: ## Install all optional dependencies
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && uv sync --all-extras

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
	$(ACTIVATE) && pytest thermo_manifold/ -v

test-cov: ## Run tests with coverage
	$(ACTIVATE) && pytest thermo_manifold/ -v --cov=thermo_manifold --cov-report=term-missing

lint: ## Run linter (ruff)
	$(ACTIVATE) && ruff check thermo_manifold/

lint-fix: ## Run linter and fix issues
	$(ACTIVATE) && ruff check --fix thermo_manifold/

format: ## Format code with black and isort
	$(ACTIVATE) && black thermo_manifold/
	$(ACTIVATE) && isort thermo_manifold/

format-check: ## Check formatting without making changes
	$(ACTIVATE) && black --check thermo_manifold/
	$(ACTIVATE) && isort --check-only thermo_manifold/

typecheck: ## Run type checking with mypy
	$(ACTIVATE) && mypy thermo_manifold/

check: lint format-check typecheck ## Run all checks (lint, format, typecheck)

demo: ## Run the unified demo
	$(ACTIVATE) && python -m thermo_manifold.demos.unified_demo

demo-rule-shift: ## Run the rule-shift benchmark demo
	$(ACTIVATE) && python -m thermo_manifold.demos.rule_shift_demo

demo-multimodal: ## Run the unified multimodal demo (image encoding/decoding)
	$(ACTIVATE) && python -m thermo_manifold.demos.unified_multimodal_demo

demo-cross-modal: ## Run the cross-modal demo (text + image together)
	$(ACTIVATE) && python -m thermo_manifold.demos.cross_modal_demo

# ============================================================================
# SIMULATION
# ============================================================================

run: ## Run the physics simulation with dashboard
	$(ACTIVATE) && python run.py

run-profile: ## Run simulation with GPU profiling
	$(ACTIVATE) && python run.py --profile

run-bench: ## Run headless benchmark (no dashboard)
	$(ACTIVATE) && python run.py --no-dashboard --steps 1000 --profile

run-quick: ## Quick test run (100 steps, 500 particles)
	$(ACTIVATE) && python run.py --steps 100 --particles 500

run-continuous: ## Run indefinitely with random file injections
	$(ACTIVATE) && python run.py --continuous --particles 200 --dashboard-video artifacts/dashboard.mp4 --dashboard-fps 30

run-continuous-fast: ## Continuous mode with faster injections (5-20s)
	$(ACTIVATE) && python run.py --continuous --particles 100 --inject-min 5 --inject-max 20

run-script: ## Deterministic scripted injections (integrity check)
	$(ACTIVATE) && python run.py --particles 200 --dashboard-video artifacts/dashboard_script.mp4 --dashboard-fps 30 --seed 0 --injection-script artifacts/injection_script_example.json

# ============================================================================
# EXPERIMENTS (Metal/MPS kernel-based)
# ============================================================================

experiments: ## Run all kernel experiments
	@mkdir -p paper/figures paper/tables
	$(ACTIVATE) && python -m sensorium.experiments --experiment all

experiments-v: ## Run all kernel experiments (verbose)
	@mkdir -p paper/figures paper/tables
	$(ACTIVATE) && python -m sensorium.experiments --experiment all

exp-rule-shift: ## Run rule shift experiment only
	$(ACTIVATE) && python -m sensorium.experiments --experiment rule_shift

exp-ablation: ## Run ablation study only
	$(ACTIVATE) && python -m sensorium.experiments --experiment ablation

exp-continuous: ## Run continuous simulation experiment
	$(ACTIVATE) && python -m sensorium.experiments --experiment continuous

exp-timeseries: ## Run time series experiment only
	$(ACTIVATE) && python -m sensorium.experiments --experiment timeseries

exp-next-token: ## Run next token prediction experiment only
	$(ACTIVATE) && python -m sensorium.experiments --experiment next_token

exp-mnist: ## Run MNIST bytes experiment only
	$(ACTIVATE) && python -m sensorium.experiments --experiment mnist_bytes

exp-image: ## Run image generation experiment only
	$(ACTIVATE) && python -m sensorium.experiments --experiment image_gen

exp-audio: ## Run audio generation experiment only
	$(ACTIVATE) && python -m sensorium.experiments --experiment audio_gen

exp-text-diffusion: ## Run text diffusion experiment only
	$(ACTIVATE) && python -m sensorium.experiments --experiment text_diffusion

exp-cocktail: ## Run cocktail party (source separation) experiment
	$(ACTIVATE) && python -m sensorium.experiments --experiment cocktail_party

exp-collision: ## Run Universal Tokenizer TOY collision sweep
	$(ACTIVATE) && python -m sensorium.experiments --experiment collision

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
