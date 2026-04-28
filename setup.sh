#!/usr/bin/env bash
# =============================================================================
# CognivCrew — Setup Script
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[setup]${RESET} $*"; }
success() { echo -e "${GREEN}[setup]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[setup]${RESET} $*"; }
error()   { echo -e "${RED}[setup]${RESET} $*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 1. Python version check
# -----------------------------------------------------------------------------
info "Checking Python version..."
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python 3.13+ from https://python.org"
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 13 ]]; then
    error "Python 3.13+ required. Found: $PY_VERSION"
fi
success "Python $PY_VERSION OK"

# -----------------------------------------------------------------------------
# 2. uv installation
# -----------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    info "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    success "uv installed"
else
    success "uv $(uv --version) already installed"
fi

# -----------------------------------------------------------------------------
# 3. Install dependencies
# -----------------------------------------------------------------------------
info "Installing dependencies with uv..."
uv sync
success "Dependencies installed"

# -----------------------------------------------------------------------------
# 4. Environment file
# -----------------------------------------------------------------------------
if [[ ! -f .env ]]; then
    info "Creating .env from .env.example..."
    cp .env.example .env
    warn ".env created — open it and set your ANTHROPIC_API_KEY before running."
else
    success ".env already exists"
fi

# -----------------------------------------------------------------------------
# 5. Output directory
# -----------------------------------------------------------------------------
mkdir -p outputs
success "outputs/ directory ready"

# -----------------------------------------------------------------------------
# 6. Validate configuration
# -----------------------------------------------------------------------------
info "Running configuration validation..."
if uv run python main.py validate 2>/dev/null; then
    success "Configuration valid"
else
    warn "Validation found issues — check your .env file (ANTHROPIC_API_KEY required)"
fi

echo ""
success "Setup complete!"
echo ""
echo -e "  Run the pipeline:  ${CYAN}uv run python main.py run \"Your project description\"${RESET}"
echo -e "  List past runs:    ${CYAN}uv run python main.py list${RESET}"
echo -e "  Show a run:        ${CYAN}uv run python main.py show <run-id>${RESET}"
echo -e "  Validate config:   ${CYAN}uv run python main.py validate${RESET}"
echo ""
