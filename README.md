# 🧠 CognivCrew

**Your AI Software Company — From Idea to Project Package in One Command**

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-6B46C1?style=flat-square)
![Claude Powered](https://img.shields.io/badge/Claude-Powered-D97706?style=flat-square)

CognivCrew is a terminal-based multi-agent AI system that operates as a fully functional software company. Give it a project idea — it returns a complete project package produced by six specialised AI agents working in sequence. Built exclusively on Anthropic Claude models using LangGraph for production-grade agent orchestration.

---

## 📁 What It Produces

Every run creates a timestamped folder containing six Markdown deliverables:

```
outputs/20250429_143022/
  ├── 00_project_summary.md     Complete run overview
  ├── 01_product_spec.md        Epics, stories, acceptance criteria
  ├── 02_architect_brief.md     ADR-based tech stack and system design
  ├── 03_design_brief.md        ASCII wireframes and component library
  ├── 04_implementation_plan.md Epic-by-epic engineering plan
  └── 05_qa_report.md           Quality review with verdict
```

Each file is a standalone professional document. Together they form a complete brief a development team can act on immediately.

---

## 💡 Real World Use Case

A solo developer or small team with a new product idea can run CognivCrew before writing a single line of code. In under a minute they have a product spec with user stories, an architecture decision record, wireframes, and a full implementation plan. What used to take a team two weeks of meetings and documents now takes one command.

---

## ⚙️ How It Works

```
User Input
    │
    ▼
👔 CEO Agent           Strategic vision and scope
    │
    ▼
📋 PM Agent            Product spec with epics and user stories
    │
    ▼
🏛️  Architect Agent    ADR-based tech stack and system design
    │
    ▼
✋ Human Gate          YOU approve the architecture (A/R/M)
    │
    ▼
🎨 UI/UX Agent         Design brief with ASCII wireframes
    │
    ▼
⚙️  Engineer Agent     Epic-by-epic implementation planning
    │
    ▼
🔍 QA Agent            Review loop — iterates until standard met
    │
    ▼
📁 Complete Project Package
```

Each agent is a LangGraph node with a single job and a single output. Pipeline state is persisted to SQLite — if you interrupt a run, it can resume from the last checkpoint. The QA agent loops the engineer up to three times before accepting the output.

---

## ✋ The Human Approval Gate

CognivCrew pauses after the Architect Agent and presents the full architecture brief in the terminal. The tech stack is the highest-consequence decision in any software project — this gate exists because that decision is too important to delegate entirely to AI.

| Input | Action |
|-------|--------|
| `A`   | **Approve** — pipeline continues to UI/UX Agent |
| `R`   | **Reject** — Architect produces a completely fresh proposal |
| `M`   | **Modify** — you type specific feedback, Architect revises and presents again |

The gate allows up to **3 architect iterations**. If the limit is reached, the current brief is auto-approved with a warning and the pipeline continues. This is intentional — the gate is a quality checkpoint, not a blocker.

---

## ⚠️ Claude Models Only

> CognivCrew is designed and validated exclusively for Anthropic's Claude model family. Other LLM providers are not supported. All six agent prompts are engineered specifically for Claude's reasoning and instruction-following.
>
> **Recommended:** `claude-sonnet-4-5`
> **Minimum:** Any Claude model with a 100k+ context window
> **API access:** [console.anthropic.com](https://console.anthropic.com)

---

## 🛠️ Setup

**Requirements**

- Mac or Linux
- Python 3.11+
- Anthropic API key with active credits

**1. Clone the repo**

```bash
git clone https://github.com/you/cognivcrew
cd cognivcrew
```

**2. Run setup**

```bash
chmod +x setup.sh && ./setup.sh
```

`setup.sh` installs `uv` if not present, runs `uv sync`, and creates `.env` from `.env.example`.

**3. Add API keys to `.env`**

```bash
ANTHROPIC_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_key_here   # optional — enables LangSmith tracing
```

**4. Validate setup**

```bash
uv run python main.py validate
```

**5. Run your first project**

```bash
uv run python main.py run "your project idea here"
```

---

## 📋 Commands

| Command | Description |
|---------|-------------|
| `main.py run "project idea"` | Run full pipeline |
| `main.py list` | List all previous runs |
| `main.py show [timestamp]` | View all output files for a specific run |
| `main.py validate` | Check environment setup |
| `main.py info` | Show agent roster and configuration |

```bash
# Example: run with LangSmith tracing
uv run python main.py run --langsmith "Build a habit tracker for busy professionals"
```

---

## 🔬 Example Output

**Input:**

```
"Build a habit tracker for busy professionals"
```

**`01_product_spec.md`**
- 3 epics, 11 user stories, 28 acceptance criteria
- Personas: Remote worker, Freelancer, Engineering manager

**`02_architect_brief.md`**
- Pattern: Modular Monolith
- Stack: Next.js + FastAPI + PostgreSQL
- 5 ADRs with alternatives considered
- 4 risks identified with mitigations

**`03_design_brief.md`**
- 8 screens wireframed in ASCII
- 14 named components in PascalCase
- Accessibility notes per screen

**`04_implementation_plan.md`**
- All 11 stories planned with complexity ratings
- 3 technical risks flagged with mitigations
- Full dependency list

**`05_qa_report.md`**
- VERDICT: PASS (achieved on iteration 2)
- All epics covered
- 2 gaps identified and addressed in revision

---

## 🏭 Production Features

- Stateful multi-agent orchestration via LangGraph
- Human-in-the-loop with SQLite checkpoint persistence
- Conditional QA review loop with iteration cap
- Retry logic with exponential backoff on API failures
- Structured per-run logging to `outputs/[timestamp]/cognivcrew.log`
- Token usage and cost tracking per run
- LangSmith tracing — full prompt and response visibility
- Fail-fast startup validation with actionable error messages
- Config-centralised — all tunables in `config.py`

---

## 🧭 Philosophy

**1. Specialised agents outperform generalist ones**

Each agent has one job, one output, one area of ownership. No agent does another agent's work. Specialisation produces coherent, consistent, high-quality output at every stage.

**2. Human judgment belongs at consequence boundaries**

Full automation is not always the goal. Tech stack is the highest-consequence decision in any software project. The approval gate is a deliberate architectural feature, not a limitation.

**3. Prompts are the product**

The intelligence in CognivCrew lives in the prompt files, not the Python code. Improving CognivCrew means improving prompts. The code is the delivery mechanism — the prompts are the craft.

---

## 🗺️ Roadmap

**v1 — Complete ✅**
- Six-agent pipeline: CEO, PM, Architect, UI/UX, Engineer, QA
- Human-in-the-loop approval gate with SQLite checkpointing
- Conditional QA review loop with iteration cap (max 3)
- Rich CLI: `run`, `list`, `show`, `validate`, `info`
- LangSmith tracing and structured per-run logging
- Token usage and cost tracking

**v2 — Code Generation (Planned)**
- Dual execution: Claude Code for agentic coding, Anthropic API for reasoning — both supported
- Engineer Agent upgraded to real code generation
- Test Writer Agent — generates and runs pytest suites
- Docker sandboxed execution layer
- QA Agent upgraded to run real tests against real code
- Agent Skills System — pluggable skill modules per agent role that modify or enhance any pipeline stage without touching core agent logic
- Output package includes runnable codebase

**v3 — Full Autonomy (Planned)**
- Product Architect Agent as dedicated pipeline node
- Memory layer — learns and improves from past runs
- Streaming token-by-token terminal output per agent
- Multi-model routing across Claude model family
- Web UI — FastAPI backend with React frontend
- GitHub integration — push output directly to new repo

---

## ⚠️ Known Limitations

- Output quality scales with input specificity. Vague requests produce vague output. Be specific.
- v1 produces planning documents, not executable code. Output is designed to guide a development team.
- Complex requests over 200 words may approach context limits in downstream agents.
- LangSmith tracing is optional but strongly recommended for understanding and debugging agent behaviour.

---

## 🤝 Contributing

- Prompt improvements are the highest value contribution
- New skill modules for v2 agent roles
- Bug reports must include the relevant output files from `outputs/[timestamp]/` to be actionable
- Prompt contributions must include before and after output examples showing the improvement

---

*Built with LangGraph and Anthropic Claude*
*Designed for developers who think before they build*

**CognivCrew — because great software starts with great thinking**
