# CognivCrew

An AI-powered multi-agent pipeline that turns a plain-English software project description into a complete, QA-verified set of project deliverables — strategy, product spec, design brief, and implementation plan — using a crew of specialised Claude agents coordinated by LangGraph.

---

## What it is

CognivCrew simulates an AI software company. You describe a product you want to build. A pipeline of five specialised agents — CEO, PM, Designer, Engineer, and QA — each produce a professional deliverable, then a QA agent reviews everything and sends the engineer back for revisions if quality standards aren't met (up to 3 cycles). All output is saved to a timestamped folder as Markdown files.

---

## Architecture

```
  User Request
       │
       ▼
  ┌─────────┐      ┌──────────┐      ┌────────────┐      ┌──────────────┐
  │   CEO   │ ───▶ │    PM    │ ───▶ │  Designer  │ ───▶ │   Engineer   │
  └─────────┘      └──────────┘      └────────────┘      └──────┬───────┘
  Strategy         Product Spec      Design Brief               │
                                                         ┌───────▼───────┐
                                                         │      QA       │
                                                         └───────┬───────┘
                                                    PASS │       │ FAIL (< 3x)
                                                         │       └──────────▶ Engineer
                                                    ┌────▼────┐
                                                    │  Final  │
                                                    └─────────┘
                                                    00_project_summary.md
```

Each agent is a LangGraph node. The QA conditional edge either routes to `final` (PASS or max iterations reached) or loops back to `engineer` with structured feedback.

---

## Output structure

Every run produces a timestamped folder:

```
outputs/
└── 20240428_143022/
    ├── 00_project_summary.md      <- index of all deliverables
    ├── 01_product_spec.md         <- PM: epics, stories, acceptance criteria
    ├── 02_design_brief.md         <- Designer: journeys, IA, interaction patterns
    ├── 03_implementation_plan.md  <- Engineer: architecture, data models, API design
    ├── 04_qa_report.md            <- QA: traceability matrix, issue log, verdict
    └── cognivcrew.log             <- full DEBUG log for the run
```

---

## Setup

### Prerequisites

- Python 3.13+
- An Anthropic API key (https://console.anthropic.com)
- (Optional) A LangSmith API key (https://smith.langchain.com) for tracing

### Automated setup

```bash
git clone <repo-url>
cd cognivcrew
chmod +x setup.sh
./setup.sh
```

`setup.sh` installs `uv`, syncs dependencies, creates `.env` from `.env.example`, and validates the configuration.

### Manual setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY at minimum

# 4. Validate
uv run python main.py validate
```

---

## Usage

### Run the pipeline

```bash
uv run python main.py run "Build a task management app for distributed remote teams"
```

With LangSmith tracing:

```bash
uv run python main.py run --langsmith "Build a task management app for distributed remote teams"
```

### List past runs

```bash
uv run python main.py list
```

### Show a run summary

```bash
uv run python main.py show 20240428_143022
```

### Validate configuration

```bash
uv run python main.py validate
```

---

## Configuration

All settings live in `config.py`. Key values:

| Setting            | Default           | Override via          |
|--------------------|-------------------|-----------------------|
| Model              | claude-sonnet-4-5 | Edit config.py        |
| Max QA iterations  | 3                 | MAX_ITERATIONS env    |
| Log level          | INFO              | LOG_LEVEL env         |
| Output directory   | outputs/          | Edit config.py        |

Agent temperatures: CEO 0.0, PM 0.0, Designer 0.7, Engineer 0.3, QA 0.0.

---

## Run summary

After each run, CognivCrew prints a summary panel showing:

- Total wall-clock duration
- QA iterations used
- Input / output / total tokens
- Estimated cost at Claude Sonnet pricing ($3/M input, $15/M output)
- Per-agent timing breakdown
- Output directory path

---

## Retry behaviour

All Claude API calls use tenacity exponential backoff:

- Max 3 attempts
- Wait 2^n seconds between retries (2s then 4s)
- Retries on: RateLimitError, APITimeoutError, APIConnectionError
- Retry attempts logged at WARN level and written to the run log file
