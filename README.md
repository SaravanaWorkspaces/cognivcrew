# CognivCrew

An AI-powered multi-agent pipeline that turns a plain-English software project description into a complete, QA-verified set of deliverables — strategy, product spec, architecture brief, design brief, and implementation plan — using a crew of specialised Claude agents coordinated by LangGraph.

---

## What it produces

Every run writes a timestamped output folder:

```
outputs/
└── 20240428_143022/
    ├── 00_project_summary.md      ← run index, QA verdict, deliverable checklist
    ├── 01_product_spec.md         ← PM: epics, user stories, acceptance criteria
    ├── 02_architect_brief.md      ← Architect: ADRs, tech stack, component diagram
    ├── 03_design_brief.md         ← Designer: user journeys, IA, interaction patterns
    ├── 04_implementation_plan.md  ← Engineer: architecture, data models, API design
    ├── 05_qa_report.md            ← QA: traceability matrix, issue log, verdict
    └── cognivcrew.log             ← full DEBUG log for the run
```

---

## How it works

```
  User Request
       │
       ▼
  ┌─────────┐    ┌──────────┐    ┌────────────────────────────────┐
  │  👔 CEO  │───▶│  📋 PM   │───▶│  🏗  Architect                  │
  └─────────┘    └──────────┘    └──────────────┬─────────────────┘
  Strategy       Product Spec                   │
                                       ┌────────▼────────┐
                                       │  Human Gate      │  ← A / R / M
                                       └────────┬────────┘
                                    Approved    │   Modify (feedback loop, max 3×)
                                                │
                              ┌─────────────────▼──────────────────┐
                              │  🎨 Designer                         │
                              └──────────────────┬─────────────────┘
                                                 │
                              ┌──────────────────▼─────────────────┐
                              │  ⚙️  Engineer                        │
                              └──────────────────┬─────────────────┘
                                                 │
                              ┌──────────────────▼─────────────────┐
                              │  🔍 QA                              │
                              └──────────┬──────────────────────────┘
                                PASS     │    FAIL (loops back, max 3×)
                                         │
                              ┌──────────▼────────┐
                              │  📦 Final          │
                              └────────────────────┘
                              00_project_summary.md
```

Each role is a LangGraph node. The graph is persisted to SQLite so pipeline state survives process restarts. The QA node loops the engineer up to 3 times before accepting the output.

---

## Setup

### Prerequisites

- Python 3.13+
- An Anthropic API key — [console.anthropic.com](https://console.anthropic.com)
- (Optional) A LangSmith API key — [smith.langchain.com](https://smith.langchain.com)

### Automated

```bash
git clone <repo-url>
cd cognivcrew
chmod +x setup.sh
./setup.sh
```

### Manual

```bash
# 1. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Set up environment
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum

# 4. Confirm everything is ready
uv run python main.py validate
```

---

## Human Approval Gate

After the Architect agent produces its brief, the pipeline pauses and presents the brief for human review. You have three options:

| Input | Action |
|-------|--------|
| `A`   | **Approve** — pipeline continues to Designer |
| `R`   | **Reject** — pipeline stops immediately |
| `M`   | **Modify** — you enter written feedback; Architect revises and presents again |

The gate allows up to **3 revision cycles**. If the limit is reached, the brief is auto-approved with a warning panel and the pipeline continues. The checkpoint is persisted to SQLite so you can resume a paused pipeline later with the same thread ID.

---

## Commands

```bash
# Run the full pipeline
uv run python main.py run "Build a task management app for remote teams"

# Run with LangSmith tracing enabled
uv run python main.py run --langsmith "Build a task management app for remote teams"

# List all past runs with request preview and QA verdict
uv run python main.py list

# Print full contents of all output files for a run
uv run python main.py show 20240428_143022

# Run full configuration checklist (env vars, prompt files, packages)
uv run python main.py validate

# Show agent roster, model, and loop caps
uv run python main.py info
```

---

## Configuration

All settings are in `config.py`. Key overrides via environment variables:

| Setting | Default | Env var |
|---------|---------|---------|
| Max QA iterations | 3 | `MAX_ITERATIONS` |
| Log level | INFO | `LOG_LEVEL` |

Agent temperatures (all in `config.py`): CEO 0.0 · PM 0.0 · Architect 0.0 · Designer 0.7 · Engineer 0.3 · QA 0.0

Pricing for the run summary cost estimate: $3.00/M input tokens · $15.00/M output tokens (Claude Sonnet).
