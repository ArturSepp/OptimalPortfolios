# Convenience aliases for the commands documented in CONTRIBUTING.md.
#
# The recipes below hold those commands *verbatim*. CONTRIBUTING.md stays the source of
# truth: it explains why each flag is there (`--locked`, `--only-group lint`, the explicit
# `--select`, `--python 3.12`), and that reasoning does not survive being hidden behind an
# alias. If you change a command here, change it there too — and vice versa.
#
# Targets here are a convenience for contributors, not a CI entry point. The workflows in
# .github/workflows/ inline their own commands, because ci.yml varies the flags per matrix
# cell and cannot call a fixed target.
#
# Requires GNU make (any version; tested against the 3.81 that ships with macOS).

.DEFAULT_GOAL := help

.PHONY: help uv install test lint lint-ruff lint-interrogate cover audit clean check-uv

help: ## Show this help
	@echo "OptimalPortfolios — see CONTRIBUTING.md for what each command does and why."
	@echo
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------

uv: ## Install uv and uvx (skipped if uv is already on PATH)
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv already installed: $$(uv --version)"; \
		echo "To upgrade, run: uv self update"; \
	else \
		echo "Installing uv from https://astral.sh/uv ..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo; \
		echo "uv and uvx are installed (uvx ships with uv)."; \
		echo "Open a new shell, or add ~/.local/bin to your PATH, then re-run make."; \
	fi

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv not found on PATH. Run 'make uv' to install it."; exit 1; }

install: check-uv ## Editable install with the dev extra, versions from uv.lock
	uv sync --extra dev

# ---------------------------------------------------------------------------
# Gates — the same invocations ci.yml and static.yml run
# ---------------------------------------------------------------------------

test: check-uv ## Run the full test suite (a few minutes)
	uv run --locked pytest

lint: check-uv ## Run both source gates (ruff, then interrogate)
	@# Both gates run even when the first fails: they are independent checks that share a
	@# target only for convenience, exactly as the two steps in static.yml do.
	@status=0; \
	$(MAKE) --no-print-directory lint-ruff || status=1; \
	$(MAKE) --no-print-directory lint-interrogate || status=1; \
	exit $$status

lint-ruff: check-uv ## Ruff stack invariants only (not a bare ruff check — see CONTRIBUTING.md)
	uv run --locked --only-group lint ruff check --select TID251,TID253,ICN,F src/optimalportfolios/

lint-interrogate: check-uv ## Docstring coverage only, must stay at 100%
	uv run --locked --only-group lint interrogate -v

cover: check-uv ## Run the suite with coverage; the floor is fail_under = 100
	uv run --locked pytest --cov=optimalportfolios --cov-report=term-missing

# ---------------------------------------------------------------------------
# Dependency audit
# ---------------------------------------------------------------------------
# Two trees, because they answer different questions: `uv pip compile` is what a fresh
# install resolves to today from the floors in pyproject.toml, and `uv export --locked` is
# the exact pinned set uv sync --locked installs. A pin can sit on a vulnerable version
# long after the floor would resolve past it.
#
# Note that --only-group reinstalls the project virtualenv with that group alone; the next
# `make test` syncs the development environment back.
#
# Unix only. On Windows, use the PowerShell block in CONTRIBUTING.md.

audit: check-uv ## Audit both dependency trees with pip-audit (run when deps change)
	uv pip compile --all-extras --python-version 3.12 --quiet pyproject.toml -o /tmp/requirements-fresh.txt
	uv run --locked --only-group audit --python 3.12 pip-audit -r /tmp/requirements-fresh.txt
	uv export --locked --all-extras --no-emit-project --quiet --format requirements-txt -o /tmp/requirements-locked.txt
	uv run --locked --only-group audit --python 3.12 pip-audit -r /tmp/requirements-locked.txt

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

# Everything below is regenerable by `uv build`, `uv sync` or a test run. `build/` in
# particular is worth clearing: setuptools does not clear stale contents between builds, so
# an old copy of the source lingers there and gets picked up by anything that walks the tree
# from the repository root — interrogate counts it, and its numbers double.
#
# Deliberately NOT removed, because they are not cheap to regenerate and may hold work:
# `.venv/` (run `make install`) and `outputs/` (figures, factsheets, backtest results).
# Remove those by hand if you want them gone.

clean: ## Remove build, packaging and test-cache artefacts
	rm -rf build dist htmlcov .pytest_cache .ruff_cache .mypy_cache *.egg-info
	rm -f .coverage .coverage.*
	find src examples papers docs -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo
	@echo "Removed build, packaging and test-cache artefacts."
	@echo "Left alone: .venv/, outputs/"
	@echo "Clearing *.egg-info detaches the editable install — run 'make install' before the next test run."
