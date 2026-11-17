<div align="center">

# 🧠 Deep Agents with LangChain & LangGraph

<p align="center">
  <a href="https://readme-typing-svg.demolab.com">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&pause=1000&color=2563EB&center=true&vCenter=true&width=650&lines=Mastering+Deep+Agents+with+LangChain;Autonomous+Planning+%2B+Context+Offloading;State%2C+Filesystem%2C+%26+Store+Backends;Multi-Agent+Delegation+%26+Pydantic+Outputs;Production-Ready+Streamlit+Lab+Assistant" alt="Typing SVG" />
  </a>
</p>

[![Python Version](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v1.4%2B-1C3C3C?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langchain)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange?logo=diagram-project&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![DeepAgents](https://img.shields.io/badge/DeepAgents-v0.7.34%2B-7928CA?logo=openai&logoColor=white)](https://github.com/langchain-ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-App_Live-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>A comprehensive, production-grade guide and lab environment for building long-horizon, autonomous Deep Agents on LangGraph.</b>
  <br />
  Featuring explicit task planning, context offloading with virtual/persistent file backends, specialized subagent delegation, and modular skills.
</p>

[Explore Lab Notebooks](#-deep-agents-lab-notebooks) • [Architecture Overview](#-the-four-pillars-of-deep-agents) • [Streamlit App](#-interactive-streamlit-chatbot-app) • [Quickstart](#-quickstart-guide)

</div>

---

## 📌 Repository Description (for GitHub)

> **GitHub Tagline:** 🚀 Production-grade implementation & interactive lab for Deep Agents on LangGraph. Master the 4 pillars: Autonomous Planning (`write_todos`), Context Offloading (State, Filesystem & Store Backends), Subagent Delegation with Pydantic structured output, and Context Engineering with modular skills. Includes a full-featured Streamlit Chatbot & Lab Workbench.

**GitHub Topics:** `deep-agents`, `langchain`, `langgraph`, `autonomous-agents`, `context-engineering`, `multi-agent`, `streamlit`, `subagents`, `python-3-13`

---

## 📖 What is a Deep Agent?

Standard AI agents follow a simplistic reactive loop:
$$\text{User Query} \longrightarrow \text{LLM Call} \longrightarrow \text{Tool Execution} \longrightarrow \text{LLM Call} \longrightarrow \text{Final Answer}$$

While this works for trivial queries, it fails on **complex, multi-step, long-horizon tasks** (e.g. Deep Research, code refactoring, complex data analysis) due to **context window pollution**, **drifting focus**, and **lack of persistent state**.

Inspired by frontier AI systems like *Claude Code*, *Deep Research*, and *Manus*, a **Deep Agent** is built on **LangGraph** around **Four Architectural Pillars**:

```
                    ┌────────────────────────────────────────────────────────┐
                    │                      DEEP AGENT                        │
                    │               (create_deep_agent core)                 │
                    └───────────────────────────┬────────────────────────────┘
                                                │
         ┌──────────────────┬───────────────────┴──────────────────┬──────────────────┐
         ▼                  ▼                                      ▼                  ▼
   ┌───────────┐      ┌───────────┐                          ┌───────────┐      ┌───────────┐
   │  Pillar 1 │      │  Pillar 2 │                          │  Pillar 3 │      │  Pillar 4 │
   │  Planning │      │File System│                          │ Subagents │      │  Context  │
   │           │      │ Offload   │                          │           │      │Engineering│
   ├───────────┤      ├───────────┤                          ├───────────┤      ├───────────┤
   │write_todos│      │ls, read,  │                          │task tool  │      │AGENTS.md  │
   │track plan │      │write, edit│                          │isolated   │      │skills/    │
   │status     │      │State/FS/  │                          │contexts & │      │modular    │
   │           │      │Store      │                          │Pydantic   │      │knowledge  │
   └───────────┘      └───────────┘                          └───────────┘      └───────────┘
```

---

## 🏛️ The Four Pillars of Deep Agents

### 1. 📋 Autonomous Planning (`write_todos` & `TodoListMiddleware`)
- Deep agents maintain an explicit, stateful checklist (`pending`, `in_progress`, `completed`).
- Plans are written **before** executing complex tasks and updated in real-time as steps finish.
- Eliminates hallucinated progress and provides 100% transparent execution steps.

### 2. 🗂️ Filesystem & Context Offloading
- Rather than dumping hundreds of search results or large drafts directly into the conversation history, deep agents offload intermediate data into a virtual or disk-backed file system (`write_file`, `read_file`, `edit_file`, `ls`).
- Supports three interchangeable backends:
  - **`StateBackend`**: In-memory ephemeral virtual files held directly in graph state.
  - **`FilesystemBackend`**: Real disk files confined safely to a sandbox root (`virtual_mode=True`).
  - **`StoreBackend`**: Persistent, cross-thread storage backed by LangGraph's `BaseStore` (enabling durable cross-session memory).

### 3. 👥 Subagents & Task Delegation (`task` tool)
- The main agent can spawn specialized child agents with **isolated context windows**.
- Subagents execute focused tasks (e.g., deep search, code validation) without polluting the parent's token budget.
- Supports **Pydantic Structured Output** (e.g., `ResearchFindings` with confidence score and verified source links).

### 4. 🧠 Context Engineering & Modular Skills System
- **Project Memory (`AGENTS.md`)**: Injected into the agent's memory to anchor its architecture, operating guidelines, and constraints.
- **Skills System (`/skills/`)**: Dynamic skill folders (`aws`, `langgraph`, `python`, `report-writer`) containing `SKILL.md`, `instructions.md`, and `examples.md` that the agent dynamically discovers and loads when needed.

---

## 🗂️ Repository Structure

```plaintext
Deep-Agents-with-Langchain/
├── app.py                      # 🌟 Full-featured Streamlit Chatbot & Workbench
├── deep_agents_core.py         # ⚙️ Core orchestrator (Model, Backends, Skills, Subagents)
├── pyproject.toml              # 📦 Project dependencies (uv-managed, Python 3.13+)
├── uv.lock                     # 🔒 Pinned lockfile for reproducible environments
├── README.md                   # 📖 Repository Documentation
├── main.py                     # 🚀 Basic entrypoint script
│
└── deepagents_lab/             # 🔬 Interactive Deep Agents Laboratory
    ├── intro.ipynb             # 01: Deep Agents vs LangChain Agents + Tavily Search
    ├── backends.ipynb          # 02: StateBackend, FilesystemBackend & StoreBackend
    ├── context_engineering.ipynb # 03: AGENTS.md, MemoryMiddleware & Skills System
    ├── subagents.ipynb         # 04: Subagent delegation & Pydantic structured output
    │
    ├── projects/
    │   └── AGENTS.md           # 📘 Deep Agents architectural blueprint & guidelines
    │
    ├── skills/                 # ⚡ Modular Agent Skills
    │   ├── aws/                # AWS architecture, services, & cost guidelines
    │   ├── langgraph/          # Stateful workflows, memory, & conditional routing
    │   ├── python/             # Modern Python idioms, type annotations, & patterns
    │   └── report-writer/      # Standardized post-answer executive markdown reports
    │
    └── workspace/              # 📁 Sandbox directory for FilesystemBackend real disk tests
```

---

## 🔬 Deep Agents Lab Notebooks

| Notebook | Topic | Key Concepts Demonstrated |
|---|---|---|
| [`01_intro.ipynb`](file:///c:/Users/musta/OneDrive/Desktop/Deep%20Agents%20Book/deepagents_lab/intro.ipynb) | **Deep Agents Foundations** | Comparing `create_agent` vs `create_deep_agent`, OpenRouter/Groq model initialization, Tavily `internet_search` custom tool integration. |
| [`02_backends.ipynb`](file:///c:/Users/musta/OneDrive/Desktop/Deep%20Agents%20Book/deepagents_lab/backends.ipynb) | **Storage Backends & Offloading** | `StateBackend` (in-state virtual files), `FilesystemBackend` (real disk sandbox), `StoreBackend` (cross-thread persistence with `InMemoryStore`). |
| [`03_context_engineering.ipynb`](file:///c:/Users/musta/OneDrive/Desktop/Deep%20Agents%20Book/deepagents_lab/context_engineering.ipynb) | **Memory & Skills System** | Injecting `projects/AGENTS.md`, `MemoryMiddleware`, checkpointer isolation, and dynamic skills indexing (`/skills/langgraph`, `/skills/report-writer`). |
| [`04_subagents.ipynb`](file:///c:/Users/musta/OneDrive/Desktop/Deep%20Agents%20Book/deepagents_lab/subagents.ipynb) | **Subagent Delegation** | Spawning child agents via `task`, isolated context windows, and returning Pydantic `ResearchFindings` (summary, confidence score, source list). |

---

## 💻 Interactive Streamlit Chatbot App

This repository includes a production-grade **Streamlit application** (`app.py`) that brings every concept from the notebooks into a live, interactive UI.

### 🌟 App Capabilities

```
+-----------------------------------------------------------------------------------+
|  🤖 Deep Agents Lab Assistant                                                     |
|  [Planning: write_todos]  [Filesystem: Offload]  [Subagents: task]  [Memory: AGENTS] |
+-----------------------------------------------------------------------------------+
| SIDEBAR CONTROLS        | MAIN APPLICATION TABS                                   |
| - Provider & Model      |                                                         |
| - Backend Selector      | [💬 Chat] [📋 Plan Tracker] [🗂️ Filesystem] [📚 Skills]   |
|   (State/FS/Store)      |                                                         |
| - Memory & Skills Switch| > User: Research LLM Gateways using subagents.          |
| - Subagents & Output    | > Agent: [📋 Plan Generated]                             |
| - Thread/Session Mgr    |          [🔧 Spawning structured-researcher...]         |
|                         |          [📊 Confidence: 0.94 | 3 Sources]              |
|                         |          [📄 Saved to /reports/gateways.md]            |
+-----------------------------------------------------------------------------------+
```

1. **💬 Chat & Execution**: Real-time conversation with expandable tool trace inspects (`write_todos`, `read_file`, `write_file`, `task`, `internet_search`).
2. **📋 Live Plan & Todos**: Real-time progress bar and Kanban cards tracking `pending`, `in_progress`, and `completed` plan steps.
3. **🗂️ Virtual Filesystem**: Live file tree explorer, markdown/code viewer, file editor, and one-click file download.
4. **📚 Context & Skills Library**: Direct reader for `AGENTS.md` and interactive skill inspector (`SKILL.md`, `instructions.md`, `examples.md`).
5. **🔍 Architecture & State Debugger**: Live LangGraph state inspection, active thread metadata, and `BaseStore` namespace explorer.

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- **Python 3.13+** installed on your system.
- Recommended package manager: [uv](https://github.com/astral-sh/uv) (ultra-fast Python package manager).

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/MustafaKocamann/Deep-Agents-with-Langchain.git
cd Deep-Agents-with-Langchain

# Install all dependencies with uv (or pip)
uv sync
```

### 3. Configure API Keys

Create a `.env` file in the root directory:

```env
# Model Providers (Choose your preferred provider)
OPENROUTER_API_KEY=your_openrouter_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Tavily Real-time Search Tool
TAVILY_API_KEY=your_tavily_api_key_here
```

> [!TIP]
> You can obtain free OpenRouter API keys to access models like `nvidia/nemotron-3-ultra-550b-a55b:free` or `meta-llama/llama-3.3-70b-instruct:free`.

### 4. Run the Streamlit Chatbot

```bash
uv run streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

### 5. Launch Jupyter Lab / Notebooks

```bash
uv run jupyter lab
```
Navigate to `deepagents_lab/` to step through the notebooks interactively.

---

## 🔄 Backend Storage Comparison Matrix

| Backend Mode | Storage Medium | Persistence | Cross-Thread Sharing | Best Used For |
|---|---|---|---|---|
| **`StateBackend`** | LangGraph State (`files`) | Ephemeral (Session) | ❌ Thread-Isolated | Lightweight tasks, transient scratchpads, sandboxed safe execution |
| **`FilesystemBackend`** | Local Host Disk (`workspace/`) | Permanent (Disk) | ✅ Shared via Disk | Project source code, actual filesystem scripts, persistent repo files |
| **`StoreBackend`** | LangGraph `BaseStore` | Configurable (Memory/DB) | ✅ Shared via Namespace | User preferences, long-term cross-session memory, team knowledge bases |

---

## 🧩 Code Example: Creating a Deep Agent

Here is how simple it is to initialize a full Deep Agent with all 4 pillars using the library:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from pydantic import BaseModel, Field

load_dotenv()

# 1. Model Configuration
model = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# 2. Structured Output Schema for Subagents
class ResearchFindings(BaseModel):
    summary: str = Field(description="Summary of research findings")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    sources: list[str] = Field(description="Citations and URLs")

# 3. Subagent Definition
research_subagent = {
    "name": "researcher",
    "description": "Performs in-depth research on technical topics",
    "system_prompt": "Research the topic thoroughly and provide structured output.",
    "response_format": ResearchFindings,
}

# 4. Deep Agent Compilation
agent = create_deep_agent(
    model=model,
    backend=StateBackend(),
    middleware=[TodoListMiddleware()],   # Pillar 1: Planning (write_todos)
    memory=["/projects/AGENTS.md"],      # Pillar 4: Project Memory
    skills=["/skills/"],                 # Pillar 4: Modular Skills
    subagents=[research_subagent],       # Pillar 3: Subagent Delegation
    checkpointer=MemorySaver(),
)

# 5. Execution
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Plan and research LLM Gateways."}]},
    config={"configurable": {"thread_id": "demo-thread-1"}}
)

print(result["messages"][-1].content)
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a pull request:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/amazing-skill`)
3. Commit your Changes (`git commit -m 'Add new skill: docker-architect'`)
4. Push to the Branch (`git push origin feature/amazing-skill`)
5. Open a Pull Request

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/MustafaKocamann">Mustafa Kocaman</a> as part of the Deep Agents Book & Laboratory.</sub>
</div>
