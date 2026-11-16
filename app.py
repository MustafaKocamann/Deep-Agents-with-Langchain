import os
import uuid
import json
from pathlib import Path
import streamlit as st

# Import core deep agents orchestrator
import deep_agents_core as core

# -----------------------------------------------------------------------------
# 1. Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deep Agents Lab — Interactive Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .pillar-chip {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .pillar-plan { background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .pillar-fs { background-color: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .pillar-sub { background-color: #FAF5FF; color: #7E22CE; border: 1px solid #E9D5FF; }
    .pillar-ctx { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    
    .todo-card {
        padding: 0.6rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .todo-completed { background-color: #F0FDF4; border-left: 4px solid #22C55E; color: #166534; }
    .todo-in_progress { background-color: #EFF6FF; border-left: 4px solid #3B82F6; color: #1E40AF; }
    .todo-pending { background-color: #F8FAFC; border-left: 4px solid #94A3B8; color: #475569; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session-{str(uuid.uuid4())[:8]}"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}  # {thread_id: list[dict]}

if st.session_state.thread_id not in st.session_state.chat_history:
    st.session_state.chat_history[st.session_state.thread_id] = []

if "session_todos" not in st.session_state:
    st.session_state.session_todos = {}  # {thread_id: list[dict]}

if "active_agent_session" not in st.session_state:
    st.session_state.active_agent_session = None

if "last_config_hash" not in st.session_state:
    st.session_state.last_config_hash = None


# -----------------------------------------------------------------------------
# 3. Sidebar: Configuration & Deep Agent Parameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 Model & Provider")
    provider = st.selectbox(
        "LLM Provider",
        options=["OpenRouter", "Groq", "OpenAI"],
        index=0
    )
    
    preset_models = {
        "OpenRouter": [
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o-mini"
        ],
        "Groq": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        "OpenAI": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo"
        ]
    }
    
    selected_preset = st.selectbox("Select Model", options=preset_models[provider] + ["Custom..."])
    if selected_preset == "Custom...":
        model_name = st.text_input("Custom Model Identifier", value="nvidia/nemotron-3-ultra-550b-a55b:free")
    else:
        model_name = selected_preset

    # API Keys with fallback to .env
    default_key = ""
    if provider == "OpenRouter":
        default_key = os.getenv("OPENROUTER_API_KEY", "")
    elif provider == "Groq":
        default_key = os.getenv("GROQ_API_KEY", "")
    elif provider == "OpenAI":
        default_key = os.getenv("OPENAI_API_KEY", "")

    api_key = st.text_input(
        f"{provider} API Key",
        value=default_key,
        type="password",
        help="Reads from your .env file automatically or you can enter a key here."
    )
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.markdown("---")
    st.markdown("### 💾 Storage Backend (backends.ipynb)")
    backend_choice = st.radio(
        "Choose Filesystem Backend",
        options=["StateBackend", "FilesystemBackend", "StoreBackend"],
        index=0,
        help=(
            "StateBackend: Virtual in-memory filesystem stored in conversation state.\n"
            "FilesystemBackend: Real disk files inside deepagents_lab/workspace.\n"
            "StoreBackend: Persistent LangGraph store shared across all threads!"
        )
    )

    if backend_choice == "StateBackend":
        st.caption("ℹ️ **StateBackend**: Ephemeral in-memory files. Pass `files` across conversation turns.")
    elif backend_choice == "FilesystemBackend":
        st.caption("ℹ️ **FilesystemBackend**: Direct read/write to real disk under `deepagents_lab/workspace/`.")
    else:
        st.caption("ℹ️ **StoreBackend**: Shared memory store. Files written in Thread A can be read in Thread B!")

    st.markdown("---")
    st.markdown("### 🧠 Context Engineering (context_engineering.ipynb)")
    enable_memory = st.checkbox("Inject Project Memory (`AGENTS.md`)", value=True, help="Injects AGENTS.md so the agent knows its own architecture.")
    enable_skills = st.checkbox("Enable Skills System (`/skills/`)", value=True, help="Provides skill documentation in virtual filesystem.")
    
    selected_skills = []
    if enable_skills:
        selected_skills = st.multiselect(
            "Active Skills",
            options=["langgraph", "python", "aws", "report-writer"],
            default=["langgraph", "python", "aws", "report-writer"]
        )

    st.markdown("---")
    st.markdown("### 👥 Subagents (subagents.ipynb)")
    enable_subagents = st.checkbox("Enable Subagents (`task` tool)", value=True, help="Allows main agent to spawn child agents with isolated contexts.")
    enable_structured_subagent = st.checkbox(
        "Enable Structured Findings (`ResearchFindings`)",
        value=True,
        help="Child subagent returns Pydantic structured output (summary, confidence, sources)."
    )

    st.markdown("---")
    st.markdown("### 📋 Planning & Tools")
    enable_planning = st.checkbox("Enable Planning Tool (`write_todos`)", value=True, help="Uses TodoListMiddleware to break tasks into steps.")
    enable_search = st.checkbox("Enable Web Search (`internet_search`)", value=True, help="Tavily client for real-time web search.")
    tavily_key = st.text_input("Tavily API Key", value=os.getenv("TAVILY_API_KEY", ""), type="password")

    st.markdown("---")
    st.markdown("### 🧵 Threads & Sessions")
    st.text(f"Current Thread ID:\n{st.session_state.thread_id}")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("➕ New Thread", use_container_width=True):
            new_id = f"session-{str(uuid.uuid4())[:8]}"
            st.session_state.thread_id = new_id
            st.session_state.chat_history[new_id] = []
            st.rerun()
    with col_t2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history[st.session_state.thread_id] = []
            st.session_state.session_todos[st.session_state.thread_id] = []
            st.rerun()


# -----------------------------------------------------------------------------
# 4. Agent Session Builder & Cache Management
# -----------------------------------------------------------------------------
config_hash = (
    provider, model_name, api_key, temperature, backend_choice,
    enable_memory, enable_skills, tuple(selected_skills),
    enable_subagents, enable_structured_subagent, enable_planning,
    enable_search, tavily_key
)

if st.session_state.active_agent_session is None or st.session_state.last_config_hash != config_hash:
    try:
        chat_model = core.create_model(
            provider=provider,
            model_name=model_name,
            api_key=api_key if api_key else None,
            temperature=temperature
        )
        search_tool = core.create_internet_search_tool(api_key=tavily_key) if enable_search else None
        
        st.session_state.active_agent_session = core.DeepAgentSession(
            model=chat_model,
            backend_type=backend_choice,
            enable_planning=enable_planning,
            enable_memory=enable_memory,
            enable_skills=enable_skills,
            enable_subagents=enable_subagents,
            enable_structured_subagent=enable_structured_subagent,
            selected_skills=selected_skills,
            search_tool=search_tool
        )
        st.session_state.last_config_hash = config_hash
    except Exception as e:
        st.sidebar.error(f"Error configuring agent: {e}")


agent_session: core.DeepAgentSession = st.session_state.active_agent_session


# -----------------------------------------------------------------------------
# 5. Top Banner & Four Architectural Pillars
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>🤖 Deep Agents Lab Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>A complete interactive chatbot implementing the full Deep Agents architecture: "
    "Planning, File Context Offloading, Subagents, and Context Engineering.</div>",
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<span class='pillar-chip pillar-plan'>1. Planning (write_todos)</span>", unsafe_allow_html=True)
    st.caption("Breaks complex tasks into observable todo steps.")
with c2:
    st.markdown("<span class='pillar-chip pillar-fs'>2. Filesystem (Context Offloading)</span>", unsafe_allow_html=True)
    st.caption(f"Virtual files via <b>{backend_choice}</b>.")
with c3:
    st.markdown("<span class='pillar-chip pillar-sub'>3. Subagents (task tool)</span>", unsafe_allow_html=True)
    st.caption("Delegates subtasks to child agents with isolated contexts.")
with c4:
    st.markdown("<span class='pillar-chip pillar-ctx'>4. Context Engineering</span>", unsafe_allow_html=True)
    st.caption("AGENTS.md project memory and modular skill instructions.")

st.markdown("---")


# -----------------------------------------------------------------------------
# 6. Main App Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_plan, tab_files, tab_skills, tab_arch = st.tabs([
    "💬 Chat & Execution",
    "📋 Live Plan & Todos",
    "🗂️ Virtual Filesystem",
    "📚 Context & Skills Library",
    "🔍 Architecture & State Debugger"
])


# =============================================================================
# TAB 1: Chat & Execution
# =============================================================================
with tab_chat:
    st.markdown("#### ⚡ Quick Prompt Suggestions")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    prompt_to_send = None

    with p_col1:
        if st.button("📝 File Offloading & Verification", use_container_width=True):
            prompt_to_send = (
                "Create a file at /notes/todo.txt with exactly this content:\n"
                "1. Record video\n2. Edit video\n3. Upload video\n"
                "Then read it back to me verbatim."
            )
    with p_col2:
        if st.button("🔬 Structured Research Subagent", use_container_width=True):
            prompt_to_send = (
                "Delegate a research task to your structured-researcher subagent to investigate "
                "'LLM Gateways and routing architectures'. Return the findings with summary, confidence score, and sources."
            )
    with p_col3:
        if st.button("📊 Report Writer Skill", use_container_width=True):
            prompt_to_send = (
                "Using your report-writer skill, write a comprehensive markdown report about "
                "'Deep Agents vs Standard LangChain Agents' and save it to /reports/deep-agents-comparison.md."
            )
    with p_col4:
        if st.button("🧠 Context Memory Check", use_container_width=True):
            prompt_to_send = (
                "What is in your memory? Who are you and what architectural guidelines from AGENTS.md do you follow?"
            )

    # Render Chat History
    current_messages = st.session_state.chat_history.get(st.session_state.thread_id, [])

    for msg in current_messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)
            
            # If assistant message has execution trace, render it in an expander
            if role == "assistant" and "tool_steps" in msg and msg["tool_steps"]:
                with st.expander("🛠️ View Agent Execution Trace & Tool Calls", expanded=False):
                    for step in msg["tool_steps"]:
                        st_type = step.get("type")
                        if st_type == "tool_call":
                            t_name = step.get("name", "tool")
                            t_args = step.get("args", {})
                            st.markdown(f"**Tool Call:** `{t_name}`")
                            st.json(t_args)
                        elif st_type == "tool_result":
                            res_content = step.get("content", "")
                            st.markdown(f"**Tool Output (`{step.get('name')}`):**")
                            # Try parsing json if structured
                            try:
                                parsed = json.loads(res_content)
                                st.json(parsed)
                            except Exception:
                                st.code(str(res_content)[:1000], language="markdown")

    # Chat Input
    user_input = st.chat_input("Ask Deep Agent anything (e.g. research a topic, write code, plan a project, manage files)...")
    if prompt_to_send:
        user_input = prompt_to_send

    if user_input:
        if not api_key:
            st.error(f"Please provide an API Key for {provider} in the sidebar.")
        elif agent_session is None:
            st.error("Agent session is not initialized. Please check configuration.")
        else:
            # 1. Append user message to UI
            st.session_state.chat_history[st.session_state.thread_id].append({
                "role": "user",
                "content": user_input
            })
            with st.chat_message("user"):
                st.markdown(user_input)

            # 2. Invoke Deep Agent
            with st.chat_message("assistant"):
                with st.spinner("🤖 Deep Agent reasoning, planning, and executing tools..."):
                    try:
                        res = agent_session.run_message(
                            user_text=user_input,
                            thread_id=st.session_state.thread_id
                        )
                        
                        final_answer = res["final_answer"]
                        tool_steps = res["tool_steps"]
                        latest_todos = res["latest_todos"]

                        # Update session todos if any
                        if latest_todos:
                            st.session_state.session_todos[st.session_state.thread_id] = latest_todos

                        # Render answer
                        st.markdown(final_answer if final_answer else "*(Task completed without additional text)*")

                        # Render execution trace
                        if tool_steps:
                            with st.expander("🛠️ View Agent Execution Trace & Tool Calls", expanded=True):
                                for step in tool_steps:
                                    if step.get("type") == "tool_call":
                                        st.markdown(f"🔧 **Invoking Tool:** `{step.get('name')}`")
                                        st.json(step.get("args", {}))
                                    elif step.get("type") == "tool_result":
                                        st.markdown(f"📥 **Result from** `{step.get('name')}`:")
                                        st.code(str(step.get("content", ""))[:1200], language="markdown")

                        # Append to chat history
                        st.session_state.chat_history[st.session_state.thread_id].append({
                            "role": "assistant",
                            "content": final_answer,
                            "tool_steps": tool_steps
                        })
                        st.rerun()

                    except Exception as err:
                        st.error(f"Execution Error: {str(err)}")


# =============================================================================
# TAB 2: Live Plan & Todos Tracker
# =============================================================================
with tab_plan:
    st.markdown("### 📋 Live Plan & Todo List (Pillar 1: `write_todos`)")
    st.caption("Deep agents maintain explicit planning state to coordinate multi-step workflows.")

    current_todos = st.session_state.session_todos.get(st.session_state.thread_id, [])

    if not current_todos:
        st.info("No active plan generated yet for this thread. Ask the agent to perform a multi-step task (e.g. 'Plan and build a research report on AWS').")
    else:
        completed_count = sum(1 for t in current_todos if t.get("status") == "completed")
        total_count = len(current_todos)
        st.progress(completed_count / max(1, total_count), text=f"Progress: {completed_count}/{total_count} steps completed")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown("##### ⏳ Pending")
            for t in current_todos:
                if t.get("status") == "pending":
                    st.markdown(f"<div class='todo-card todo-pending'><span>{t.get('content')}</span><b>Pending</b></div>", unsafe_allow_html=True)
        with col_p2:
            st.markdown("##### 🔄 In Progress")
            for t in current_todos:
                if t.get("status") == "in_progress":
                    st.markdown(f"<div class='todo-card todo-in_progress'><span>{t.get('content')}</span><b>In Progress</b></div>", unsafe_allow_html=True)
        with col_p3:
            st.markdown("##### ✅ Completed")
            for t in current_todos:
                if t.get("status") == "completed":
                    st.markdown(f"<div class='todo-card todo-completed'><span>{t.get('content')}</span><b>Done</b></div>", unsafe_allow_html=True)


# =============================================================================
# TAB 3: Virtual Filesystem Explorer & Editor
# =============================================================================
with tab_files:
    st.markdown(f"### 🗂️ Virtual Filesystem Inspector ({backend_choice})")
    st.caption("Pillar 2: Context Offloading. Files created, read, or edited by the agent live here.")

    if agent_session:
        current_files = agent_session.get_current_files()
    else:
        current_files = {}

    f_col1, f_col2 = st.columns([1, 2])

    with f_col1:
        st.markdown("#### 📁 File Tree")
        if not current_files:
            st.warning("No files found in the current backend.")
            selected_file = None
        else:
            file_paths = sorted(list(current_files.keys()))
            selected_file = st.selectbox("Select a file to inspect or edit:", options=file_paths)

        st.markdown("---")
        st.markdown("#### ➕ Create / Inject File")
        new_file_path = st.text_input("New File Path (e.g. /notes/test.txt)", value="/notes/new_note.md")
        new_file_content = st.text_area("File Content", value="# New Document\nWritten from UI.")
        if st.button("Save File to Backend"):
            if agent_session and new_file_path:
                agent_session.write_virtual_file(new_file_path, new_file_content)
                st.success(f"File '{new_file_path}' written to {backend_choice}!")
                st.rerun()

    with f_col2:
        st.markdown("#### 📄 File Content")
        if selected_file and selected_file in current_files:
            content = current_files[selected_file]
            st.text(f"Path: {selected_file}  |  Size: {len(content)} characters")
            
            # Editing capability
            edit_mode = st.toggle("✏️ Edit File", value=False)
            if edit_mode:
                edited_content = st.text_area("Edit text:", value=content, height=450)
                if st.button("💾 Commit Edit"):
                    agent_session.write_virtual_file(selected_file, edited_content)
                    st.success("File updated successfully!")
                    st.rerun()
            else:
                if selected_file.endswith(".md") or selected_file.endswith(".txt"):
                    st.markdown(content)
                else:
                    st.code(content, language="python" if selected_file.endswith(".py") else "text")

            st.download_button(
                label=f"⬇️ Download {Path(selected_file).name}",
                data=content,
                file_name=Path(selected_file).name,
                mime="text/plain"
            )
        else:
            st.info("Select a file from the list to view its contents.")


# =============================================================================
# TAB 4: Context & Skills Library
# =============================================================================
with tab_skills:
    st.markdown("### 📚 Context & Skills Knowledge Base (context_engineering.ipynb)")
    
    subtab_agents, subtab_skills = st.tabs(["📄 AGENTS.md (Project Context)", "⚡ Available Skills (/skills/)"])

    with subtab_agents:
        st.markdown("#### `projects/AGENTS.md`")
        st.caption("Loaded automatically into agent memory when `memory=['/projects/AGENTS.md']` is enabled.")
        agents_text = core.get_agents_md_content()
        if agents_text:
            st.markdown(agents_text)
        else:
            st.warning("AGENTS.md not found in deepagents_lab/projects.")

    with subtab_skills:
        st.markdown("#### Skills Explorer")
        skill_choice = st.selectbox("Select a Skill to Inspect:", options=["report-writer", "langgraph", "python", "aws"])
        
        skill_folder = core.SKILLS_DIR / skill_choice
        if skill_folder.exists():
            s_c1, s_c2, s_c3 = st.tabs(["SKILL.md", "instructions.md", "examples.md"])
            with s_c1:
                p = skill_folder / "SKILL.md"
                if p.exists():
                    st.markdown(p.read_text(encoding="utf-8"))
            with s_c2:
                p = skill_folder / "instructions.md"
                if p.exists():
                    st.markdown(p.read_text(encoding="utf-8"))
            with s_c3:
                p = skill_folder / "examples.md"
                if p.exists():
                    st.markdown(p.read_text(encoding="utf-8"))


# =============================================================================
# TAB 5: Architecture & State Debugger
# =============================================================================
with tab_arch:
    st.markdown("### 🔍 Deep Agents Architecture & State Debugger")
    
    st.markdown("""
    ```
                        ┌───────────────────────────────────────────┐
                        │                Deep Agent                 │
                        │       (create_deep_agent on LangGraph)    │
                        └─────────────────────┬─────────────────────┘
                                              │
            ┌──────────────────┬──────────────┴───────────────┬──────────────────┐
            ▼                  ▼                              ▼                  ▼
      ┌───────────┐      ┌───────────┐                  ┌─────────────┐    ┌─────────────┐
      │ Planning  │      │   Files   │                  │  Subagents  │    │   Custom    │
      │ write_    │      │ State /   │                  │ task tool:  │    │   Tools     │
      │ todos     │      │ Store / FS│                  │ isolated    │    │ Tavily web  │
      │           │      │ backend   │                  │ context     │    │ search      │
      └───────────┘      └───────────┘                  └─────────────┘    └─────────────┘
    ```
    """)

    st.markdown("#### ⚙️ Active Session State")
    st.write({
        "thread_id": st.session_state.thread_id,
        "backend_type": backend_choice,
        "provider": provider,
        "model": model_name,
        "planning_enabled": enable_planning,
        "memory_enabled": enable_memory,
        "skills_enabled": enable_skills,
        "subagents_enabled": enable_subagents,
        "total_messages_in_thread": len(st.session_state.chat_history.get(st.session_state.thread_id, [])),
        "total_files_in_backend": len(agent_session.get_current_files()) if agent_session else 0
    })

    if backend_choice == "StoreBackend" and agent_session:
        st.markdown("#### 📦 Store Items (`shared-memory` namespace)")
        try:
            raw_items = agent_session.store.search(("shared-memory",), limit=20)
            st.json([{"key": it.key, "namespace": it.namespace} for it in raw_items])
        except Exception as e:
            st.text(f"Could not read store: {e}")
