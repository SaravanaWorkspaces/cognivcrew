#!/usr/bin/env bash
# CognivCrew — Setup Script
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

ok()   { echo -e "${GREEN}✓${RESET}  $*"; }
info() { echo -e "${CYAN}→${RESET}  $*"; }
warn() { echo -e "${YELLOW}!${RESET}  $*"; }

echo ""
echo -e "${CYAN}CognivCrew — Setup${RESET}"
echo "────────────────────────────────────────"

# 1. Install uv if not present
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    ok "uv installed"
else
    ok "uv $(uv --version) already installed"
fi

# 2. Install dependencies
info "Installing Python dependencies..."
uv sync
ok "Dependencies installed"

# 3. Create .env from example if it doesn't exist
if [[ ! -f .env ]]; then
    info "Creating .env from .env.example..."
    cp .env.example .env
    warn ".env created — open it and set ANTHROPIC_API_KEY before running"
else
    ok ".env already exists"
fi

# 4. Print next steps
echo ""
echo "────────────────────────────────────────"
ok "Setup complete!"
echo ""
echo "Next steps:"
echo ""
echo    "  1. Edit .env and set your ANTHROPIC_API_KEY"
echo -e "  2. ${CYAN}uv run python main.py validate${RESET}   — check everything is ready"
echo -e "  3. ${CYAN}uv run python main.py run \"Your project description\"${RESET}"
echo ""
echo "Other commands:"
echo -e "  ${CYAN}uv run python main.py info${RESET}        — show agent roster and config"
echo -e "  ${CYAN}uv run python main.py list${RESET}        — list past runs"
echo -e "  ${CYAN}uv run python main.py show <id>${RESET}   — show all files for a run"
echo ""
