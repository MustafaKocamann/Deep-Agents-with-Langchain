"""
Deep Agents Core Library
Encapsulates all Deep Agents concepts from deepagents_lab:
- 4 Architectural Pillars: Planning (write_todos), Filesystem (Context Offloading), Subagents (task), Context Engineering.
- 3 Backends: StateBackend (in-state), FilesystemBackend (real disk), StoreBackend (cross-thread store).
- Memory / Project Context: AGENTS.md
- Skills System: aws, langgraph, python, report-writer
- Subagents with Pydantic Structured Output: ResearchFindings
- Tavily Internet Search Tool
- LangGraph Checkpointer & InMemoryStore
"""

import os
import json
from pathlib import Path
from typing import Literal, Any, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Deep Agents
from deepagents import create_deep_agent
from deepagents.backends import StateBackend, FilesystemBackend, StoreBackend
from deepagents.backends.utils import create_file_data

# Tavily
from tavily import TavilyClient

# Base directories
BASE_DIR = Path(__file__).parent.resolve()
LAB_DIR = BASE_DIR / "deepagents_lab"
SKILLS_DIR = LAB_DIR / "skills"
PROJECTS_DIR = LAB_DIR / "projects"
WORKSPACE_DIR = LAB_DIR / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv()


# =====================================================================
# 1. Structured Output Schema for Subagents (from subagents.ipynb)
# =====================================================================
class ResearchFindings(BaseModel):
    """Structured findings returned by specialized research subagents."""
    summary: str = Field(description="Comprehensive summary and analysis of the research topic")
    confidence: float = Field(description="Confidence score from 0.0 (low) to 1.0 (high) in findings")
    sources: list[str] = Field(description="List of cited source URLs or primary references found")


# =====================================================================
# 2. Model Factory (OpenRouter, Groq, OpenAI)
# =====================================================================
def create_model(
    provider: str = "OpenRouter",
    model_name: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2
):
    """Builds a BaseChatModel instance based on provider and credentials."""
    if provider == "OpenRouter":
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        url = base_url or "https://openrouter.ai/api/v1"
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required to initialize OpenRouter model.")
        return ChatOpenAI(
            model=model_name,
            api_key=key,
            base_url=url,
            temperature=temperature
        )
    elif provider == "Groq":
        from langchain_groq import ChatGroq
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is required to initialize Groq model.")
        return ChatGroq(
            model=model_name,
            api_key=key,
            temperature=temperature
        )
    elif provider == "OpenAI":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required to initialize OpenAI model.")
        kwargs = {"model": model_name, "api_key": key, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =====================================================================
# 3. Custom Search Tool (from intro.ipynb & subagents.ipynb)
# =====================================================================
def create_internet_search_tool(
    api_key: Optional[str] = None,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False
):
    """Creates a Tavily web search tool with fallback handling."""
    tavily_key = api_key or os.getenv("TAVILY_API_KEY")

    def internet_search(query: str) -> str:
        """Run a web search using Tavily to retrieve current, real-time facts and citations."""
        if not tavily_key:
            return f"[Web Search Warning]: TAVILY_API_KEY not configured. Query '{query}' could not be executed."
        try:
            client = TavilyClient(api_key=tavily_key)
            results = client.search(
                query=query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic
            )
            # Format results cleanly for agent
            formatted = []
            for r in results.get("results", []):
                title = r.get("title", "No Title")
                url = r.get("url", "")
                content = r.get("content", "")
                formatted.append(f"Title: {title}\nURL: {url}\nSnippet: {content}")
            return "\n\n---\n\n".join(formatted) if formatted else "No results found for query."
        except Exception as e:
            return f"[Web Search Error]: Failed searching for '{query}': {str(e)}"

    return internet_search


# =====================================================================
# 4. Context Engineering: Memory & Skills Loaders
# =====================================================================
def get_agents_md_content() -> str:
    """Reads projects/AGENTS.md describing the deep agent architecture."""
    agents_path = PROJECTS_DIR / "AGENTS.md"
    if agents_path.exists():
        return agents_path.read_text(encoding="utf-8")
    return ""


def load_skills_files(selected_skills: Optional[list[str]] = None) -> dict[str, dict[str, Any]]:
    """
    Reads skills/<skill_name> files (SKILL.md, instructions.md, examples.md)
    and formats them as virtual file data dicts: { '/skills/...': create_file_data(...) }
    """
    files: dict[str, dict[str, Any]] = {}
    available_skills = ["langgraph", "python", "aws", "report-writer"]
    skills_to_load = selected_skills if selected_skills is not None else available_skills

    for skill_name in skills_to_load:
        skill_path = SKILLS_DIR / skill_name
        if not skill_path.exists():
            continue
        for file_path in skill_path.glob("*.md"):
            try:
                content = file_path.read_text(encoding="utf-8")
                virtual_path = f"/skills/{skill_name}/{file_path.name}"
                files[virtual_path] = create_file_data(content)
            except Exception:
                pass
    return files


def load_project_memory_files() -> dict[str, dict[str, Any]]:
    """Loads AGENTS.md into virtual file system at /projects/AGENTS.md."""
    content = get_agents_md_content()
    if content:
        return {"/projects/AGENTS.md": create_file_data(content)}
    return {}


# =====================================================================
# 5. Subagents Builder (from subagents.ipynb)
# =====================================================================
def build_subagents(
    model: Any,
    search_tool: Any,
    enable_structured: bool = True,
    selected_subagents: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """
    Builds specialized subagents:
    1. research-agent (general deep search)
    2. structured-researcher (with ResearchFindings Pydantic response format)
    3. code-expert (Python / LangGraph specialist)
    4. report-generator (Report-writer specialist)
    """
    all_subagents: dict[str, dict[str, Any]] = {}

    # 1. General Research Subagent
    all_subagents["research-agent"] = {
        "name": "research-agent",
        "description": "Delegated subagent used for deep, comprehensive research and gathering factual data.",
        "system_prompt": (
            "You are a specialized researcher. Use the internet_search tool to find accurate and "
            "up-to-date facts. Synthesize your findings clearly with citations and source URLs."
        ),
        "tools": [search_tool] if search_tool else [],
        "model": model,
    }

    # 2. Structured Researcher Subagent (Pydantic Output)
    structured_subagent: dict[str, Any] = {
        "name": "structured-researcher",
        "description": "Specialized researcher that returns structured output with a summary, confidence score, and sources.",
        "system_prompt": (
            "You are a meticulous researcher. Research the topic thoroughly using internet_search. "
            "Return your findings strictly following the required structured format with summary, confidence, and sources."
        ),
        "tools": [search_tool] if search_tool else [],
        "model": model,
    }
    if enable_structured:
        structured_subagent["response_format"] = ResearchFindings
    all_subagents["structured-researcher"] = structured_subagent

    # 3. Code Expert Subagent
    all_subagents["code-expert"] = {
        "name": "code-expert",
        "description": "Specialist in Python, LangChain, and LangGraph architectures, design patterns, and debugging.",
        "system_prompt": (
            "You are an expert Python and LangGraph architect. Provide concise, production-ready, "
            "idiomatic code with type annotations and clean structure."
        ),
        "tools": [],
        "model": model,
    }

    # 4. Report Generator Subagent
    all_subagents["report-generator"] = {
        "name": "report-generator",
        "description": "Technical writer that synthesizes complex findings into structured executive Markdown reports.",
        "system_prompt": (
            "You are a technical report synthesizer. Structure answers with Title, Executive Summary, "
            "Key Insights, Architecture / Methods, and Next Steps. Save drafts to files if appropriate."
        ),
        "tools": [],
        "model": model,
    }

    if selected_subagents is None:
        return list(all_subagents.values())
    return [all_subagents[k] for k in selected_subagents if k in all_subagents]


# =====================================================================
# 6. Deep Agent Orchestrator & Factory
# =====================================================================
class DeepAgentSession:
    """
    Manages an active Deep Agent instance, its backend, checkpointer,
    shared memory store, and conversation threads.
    """

    def __init__(
        self,
        model: Any,
        backend_type: Literal["StateBackend", "FilesystemBackend", "StoreBackend"] = "StateBackend",
        enable_planning: bool = True,
        enable_memory: bool = True,
        enable_skills: bool = True,
        enable_subagents: bool = True,
        enable_structured_subagent: bool = True,
        selected_skills: Optional[list[str]] = None,
        search_tool: Optional[Any] = None,
        system_prompt: Optional[str] = None
    ):
        self.model = model
        self.backend_type = backend_type
        self.enable_planning = enable_planning
        self.enable_memory = enable_memory
        self.enable_skills = enable_skills
        self.enable_subagents = enable_subagents
        self.enable_structured_subagent = enable_structured_subagent
        self.selected_skills = selected_skills or ["langgraph", "python", "aws", "report-writer"]
        self.search_tool = search_tool

        # Shared checkpointer for threads
        self.checkpointer = MemorySaver()

        # Shared InMemoryStore for cross-thread persistence (StoreBackend)
        self.store = InMemoryStore()

        # Virtual files cache for StateBackend
        self.virtual_files: dict[str, dict[str, Any]] = {}

        # Preload initial files into virtual_files and store
        self._initialize_context_files()

        # Initialize Backend
        self.backend = self._create_backend()

        # Custom system prompt
        base_prompt = (
            "You are an advanced Deep Agent with planning (write_todos), virtual filesystem "
            "context offloading (read_file, write_file, edit_file), and subagent delegation (task). "
            "Always follow the guidelines in AGENTS.md: plan first for complex requests, offload bulky "
            "findings to files, and use available skills when applicable."
        )
        if system_prompt:
            base_prompt = f"{base_prompt}\n\nUser Instructions:\n{system_prompt}"
        self.system_prompt = base_prompt

        # Compile agent
        self.agent = self._compile_agent()

    def _initialize_context_files(self):
        """Pre-seeds memory (AGENTS.md) and skills into virtual files and Store."""
        # 1. Project Memory
        if self.enable_memory:
            memory_files = load_project_memory_files()
            self.virtual_files.update(memory_files)
            # Also seed into store
            agents_text = get_agents_md_content()
            if agents_text:
                self.store.put(("memories",), "AGENTS.md", create_file_data(agents_text))
                self.store.put(("shared-memory",), "AGENTS.md", create_file_data(agents_text))

        # 2. Skills
        if self.enable_skills:
            skills_files = load_skills_files(self.selected_skills)
            self.virtual_files.update(skills_files)
            for path, f_data in skills_files.items():
                key = path.lstrip("/")
                self.store.put(("skills",), key, f_data)
                self.store.put(("shared-memory",), key, f_data)

        # 3. If FilesystemBackend, mirror memory & skills to real workspace disk
        if self.backend_type == "FilesystemBackend":
            if self.enable_memory:
                agents_text = get_agents_md_content()
                if agents_text:
                    dest = WORKSPACE_DIR / "projects" / "AGENTS.md"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(agents_text, encoding="utf-8")
            if self.enable_skills:
                skills_files = load_skills_files(self.selected_skills)
                for vpath, f_data in skills_files.items():
                    dest = WORKSPACE_DIR / vpath.lstrip("/")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(f_data.get("content", ""), encoding="utf-8")

    def _create_backend(self):
        """Instantiates the selected BackendProtocol implementation."""
        if self.backend_type == "FilesystemBackend":
            return FilesystemBackend(
                root_dir=str(WORKSPACE_DIR),
                virtual_mode=True
            )
        elif self.backend_type == "StoreBackend":
            return StoreBackend(
                store=self.store,
                namespace=lambda rt: ("shared-memory",)
            )
        else:  # StateBackend
            return StateBackend()

    def _compile_agent(self):
        """Assembles all components into create_deep_agent."""
        # Custom tools
        tools = [self.search_tool] if self.search_tool else []

        # Middlewares (Planning tool write_todos)
        middleware = []
        if self.enable_planning:
            middleware.append(TodoListMiddleware())

        # Subagents
        subagents = None
        if self.enable_subagents:
            subagents = build_subagents(
                model=self.model,
                search_tool=self.search_tool,
                enable_structured=self.enable_structured_subagent
            )

        # Memory & Skills configurations
        memory = ["/projects/AGENTS.md"] if self.enable_memory else None
        skills = ["/skills/"] if self.enable_skills else None

        agent = create_deep_agent(
            model=self.model,
            tools=tools,
            system_prompt=self.system_prompt,
            middleware=middleware,
            subagents=subagents,
            skills=skills,
            memory=memory,
            backend=self.backend,
            checkpointer=self.checkpointer,
            store=self.store,
        )
        return agent

    def get_current_files(self) -> dict[str, str]:
        """
        Returns a dictionary of {path: content} representing all files currently
        accessible in the active backend.
        """
        result_files: dict[str, str] = {}

        if self.backend_type == "FilesystemBackend":
            # Read real files from WORKSPACE_DIR
            if WORKSPACE_DIR.exists():
                for p in WORKSPACE_DIR.rglob("*"):
                    if p.is_file():
                        rel = "/" + p.relative_to(WORKSPACE_DIR).as_posix()
                        try:
                            result_files[rel] = p.read_text(encoding="utf-8", errors="replace")
                        except Exception as e:
                            result_files[rel] = f"[Error reading file: {e}]"
            return result_files

        elif self.backend_type == "StoreBackend":
            # Search store namespace
            try:
                items = self.store.search(("shared-memory",), limit=100)
                for it in items:
                    val = it.value
                    content = val.get("content", "") if isinstance(val, dict) else str(val)
                    key = it.key if it.key.startswith("/") else f"/{it.key}"
                    result_files[key] = content
            except Exception:
                pass
            return result_files

        else:  # StateBackend
            for path, f_data in self.virtual_files.items():
                if isinstance(f_data, dict):
                    result_files[path] = f_data.get("content", "")
                else:
                    result_files[path] = str(f_data)
            return result_files

    def write_virtual_file(self, path: str, content: str):
        """Directly writes or updates a file in the active backend."""
        clean_path = "/" + path.lstrip("/")
        file_data = create_file_data(content)

        if self.backend_type == "FilesystemBackend":
            local_path = WORKSPACE_DIR / clean_path.lstrip("/")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(content, encoding="utf-8")
        elif self.backend_type == "StoreBackend":
            key = clean_path.lstrip("/")
            self.store.put(("shared-memory",), key, file_data)
        else:
            self.virtual_files[clean_path] = file_data

    def run_message(self, user_text: str, thread_id: str) -> dict[str, Any]:
        """
        Executes a user turn in the agent.
        Carries forward virtual files in state if StateBackend is active.
        Extracts tool calls, subagent actions, planning steps, and final response.
        """
        input_payload: dict[str, Any] = {
            "messages": [HumanMessage(content=user_text)]
        }

        # If using StateBackend, pass along the virtual filesystem state
        if self.backend_type == "StateBackend":
            input_payload["files"] = self.virtual_files

        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the compiled deep agent
        result = self.agent.invoke(input_payload, config=config)

        # Update virtual files if StateBackend
        if self.backend_type == "StateBackend":
            updated_files = result.get("files", {})
            if updated_files:
                self.virtual_files.update(updated_files)

        # Parse messages
        all_messages = result.get("messages", [])
        final_answer = ""
        tool_steps: list[dict[str, Any]] = []
        latest_todos: list[dict[str, str]] = []

        # Check state todos if TodoListMiddleware stored them
        state_todos = result.get("todos", [])
        if state_todos:
            for td in state_todos:
                if isinstance(td, dict):
                    latest_todos.append(td)
                elif hasattr(td, "content") and hasattr(td, "status"):
                    latest_todos.append({"content": td.content, "status": td.status})

        for msg in all_messages:
            if isinstance(msg, AIMessage):
                if msg.content:
                    final_answer = msg.content
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        t_name = tc.get("name", "unknown")
                        t_args = tc.get("args", {})
                        tool_steps.append({
                            "type": "tool_call",
                            "name": t_name,
                            "args": t_args,
                            "id": tc.get("id", "")
                        })
                        # Extract write_todos tool calls
                        if t_name == "write_todos" and "todos" in t_args:
                            latest_todos = t_args["todos"]

            elif isinstance(msg, ToolMessage):
                tool_steps.append({
                    "type": "tool_result",
                    "name": msg.name,
                    "content": msg.content,
                    "id": msg.tool_call_id
                })

        return {
            "final_answer": final_answer,
            "tool_steps": tool_steps,
            "latest_todos": latest_todos,
            "files": self.get_current_files(),
            "raw_result": result
        }
