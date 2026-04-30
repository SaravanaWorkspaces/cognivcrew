"""
Execution mode configuration for CognivCrew.

Set EXECUTION_MODE in your .env file:
  EXECUTION_MODE=api          # Default — uses ANTHROPIC_API_KEY
  EXECUTION_MODE=pro_native   # Uses Claude Code CLI (Pro plan, no API key needed)
  EXECUTION_MODE=mock         # Zero LLM calls — pre-canned outputs for testing

Test layer:
  TEST_MODE=true              # Forces mock mode regardless of EXECUTION_MODE.
                              # Useful for CI, smoke tests, and pipeline validation.

Pro Native mode requirements:
  - Claude Code CLI installed: npm install -g @anthropic-ai/claude-code
  - Authenticated: run 'claude' once to log in via browser
  - Active Claude Pro plan subscription
  - No ANTHROPIC_API_KEY needed
"""
import os

# ── Active execution mode ─────────────────────────────────────────────────
# Override with EXECUTION_MODE env var or --mode CLI flag.
EXECUTION_MODE: str = os.getenv("EXECUTION_MODE", "api")

# ── Test layer ────────────────────────────────────────────────────────────
# When TEST_MODE=true, the pipeline is forced into mock mode regardless of
# EXECUTION_MODE. Zero LLM calls, zero cost — useful for CI and smoke tests.
TEST_MODE: bool = os.getenv("TEST_MODE", "false").lower().strip() in ("1", "true", "yes")

# ── Pro Native Config ─────────────────────────────────────────────────────
# Tokens are drawn from the Claude Pro plan subscription.
# Claude Code CLI must be installed and authenticated.
# Cost is $0 per run — included in the monthly Pro subscription.
PRO_NATIVE_CONFIG: dict = {
    # Model invoked by Claude Code CLI
    "model": "claude-sonnet-4-5",
    # Displayed in run summary instead of a dollar figure
    "cost_display": "$0 (included in Claude Pro plan)",
    # No Anthropic API key required
    "requires_api_key": False,
    # Claude Code CLI (npm install -g @anthropic-ai/claude-code) must be present
    "requires_claude_code": True,
    # Token estimation heuristic: 1 token ≈ 4 English characters
    # Pro plan does not expose exact token counts via the CLI.
    "chars_per_token": 4,
    # Pipeline agent groupings
    "planning_agents": ["ceo", "pm", "architect", "designer"],
    "codegen_agents": ["engineer", "qa"],
    # Timeout per Claude Code call (seconds).
    # Architect, Engineer, and QA generate long responses — 600s covers all agents.
    "call_timeout_seconds": 600,
}

# ── API Config (default) ──────────────────────────────────────────────────
# Uses ANTHROPIC_API_KEY for direct API access. Exact token counts available.
API_CONFIG: dict = {
    "model": "claude-sonnet-4-5",
    "cost_display": "calculated per token (see run summary)",
    "requires_api_key": True,
    "requires_claude_code": False,
}

# ── Mock Config ───────────────────────────────────────────────────────────
# Zero LLM calls. Pre-canned outputs for every stage. Always produces a
# hello_world.py in the engineer stage. Activated by TEST_MODE=true or
# EXECUTION_MODE=mock.
MOCK_CONFIG: dict = {
    "cost_display": "$0.00 (mock mode — no LLM calls)",
    "requires_api_key": False,
    "requires_claude_code": False,
    # Engineer stage always writes this file in addition to the plan.
    "engineer_deliverable": "hello_world.py",
}
