import os
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import uuid
from huggingface_hub import InferenceClient

# Helper to safely extract assistant text from HF text generation response
def _extract_assistant_text(text_resp) -> str:
    if isinstance(text_resp, str) and text_resp.strip():
        return text_resp.strip()
    return ""

# --- Streamlit Setup ---
load_dotenv()
st.set_page_config(page_title="GPT-OSS-20B Chat", page_icon="🤖", layout="wide")

# --- Persistence helpers (JSON on local disk) ---
PERSIST_DIR = Path.home() / ".gpt_oss_chat"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_FILE = PERSIST_DIR / "chats.json"

def load_persisted_state() -> dict:
    """Load chats from disk."""
    try:
        if PERSIST_FILE.exists():
            data = json.loads(PERSIST_FILE.read_text())
            if isinstance(data, dict) and data.get("version") == 1:
                return data
    except Exception as e:
        st.warning(f"Failed to load persisted state: {str(e)}")
    return {"version": 1, "chats": {}, "active_chat_id": None}

def save_persisted_state(chats: dict, active_chat_id: str | None) -> bool:
    """Persist chats in a JSON-safe structure."""
    def _to_safe_messages(msgs):
        safe = []
        for m in msgs or []:
            if isinstance(m, dict):
                q = m.get("q") or m.get("question") or (m.get("role") == "user" and m.get("content"))
                a = m.get("a") or m.get("answer") or (m.get("role") == "assistant" and m.get("content"))
                safe.append({"q": q, "a": a})
            elif isinstance(m, (list, tuple)) and len(m) >= 2:
                q, a = m[0], m[1]
                safe.append({"q": q, "a": a})
            else:
                safe.append({"q": None, "a": None})
        return safe

    safe_chats = {}
    for cid, chat in chats.items():
        title = chat.get("title", "New Chat") if isinstance(chat, dict) else "New Chat"
        msgs = chat.get("messages", []) if isinstance(chat, dict) else []
        safe_chats[cid] = {"title": title, "messages": _to_safe_messages(msgs)}

    for attempt in range(3):
        try:
            payload = {"version": 1, "chats": safe_chats, "active_chat_id": active_chat_id}
            PERSIST_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            return True
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed to save chat state: {str(e)}")
            import time
            time.sleep(0.5 * (attempt + 1))
    return False

# Consolidated CSS
st.markdown(
    """
    <style>
    :root {
        --brand-blue: #6366f1;
        --brand-purple: #a855f7;
        --sidebar-bg: #f0f4ff;
        --main-bg: #ffffff;
        --button-gradient: linear-gradient(to right, var(--brand-blue), var(--brand-purple));
        --title-gradient: linear-gradient(to right, #6366f1, #a855f7);
        --radius: 8px;
    }
    .block-container { max-width: 860px; padding-top: 4rem; }
    html, body, [data-testid="stAppViewContainer"] { font-size: 15px; }
    hr { display: none !important; }
    section[data-testid="stSidebar"] { background: var(--sidebar-bg) !important; }
    section[data-testid="stSidebar"] .block-container { background: transparent; padding: 14px; margin: 0; }
    section[data-testid="stSidebar"] h2 { font-weight: 700; margin-bottom: 0.25rem; }
    section[data-testid="stSidebar"] .stMarkdown { opacity: 0.95; }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        border-radius: var(--radius); border: 1px solid #e0e7ff; background: white;
    }
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: var(--radius); font-weight: 600; transition: all 0.2s ease;
        background: var(--button-gradient); color: white !important; border: none;
    }
    section[data-testid="stSidebar"] .stButton > button:hover { filter: brightness(1.1); }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: var(--button-gradient); opacity: 0.8;
    }
    div[data-testid="stAppViewContainer"] > .main { background: var(--main-bg); }
    h1 {
        background: var(--title-gradient); color: white; padding: 12px 20px;
        border-radius: var(--radius); text-align: center; font-size: 1.8rem !important;
    }
    h1 + div[data-testid="stMarkdownContainer"] p {
        text-align: center; color: #6b7280; margin-top: -10px; font-size: 0.95rem;
    }
    .stChatMessage div[data-testid="stMarkdownContainer"] { font-size: 0.98rem; }
    .stChatMessage[aria-label="assistant"] div[data-testid="stMarkdownContainer"] {
        background: white; border: 1px solid #e0e7ff; border-left: 4px solid var(--brand-blue);
        border-radius: var(--radius); padding: 12px;
    }
    .stChatMessage[aria-label="user"] div[data-testid="stMarkdownContainer"] {
        background: #f9fafb; border: 1px solid #e0e7ff; border-right: 4px solid var(--brand-purple);
        border-radius: var(--radius); padding: 12px;
    }
    div[data-testid="stChatInput"] textarea {
        min-height: 42px !important; font-size: 0.98rem; border-radius: var(--radius);
        background: #f0f4ff; border: 1px solid #c7d2fe;
    }
    div[data-baseweb="notification"] { border-radius: var(--radius); }
    div.stExpander { border-radius: var(--radius); border: 1px solid #e0e7ff; }
    section[data-testid="stSidebar"] .stButton { margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Multi-chat state ---
if "hydrated" not in st.session_state:
    persisted = load_persisted_state()
    st.session_state.chats = persisted.get("chats", {}) or {}
    st.session_state.active_chat_id = persisted.get("active_chat_id")
    st.session_state.hydrated = True
    if not st.session_state.chats:
        _id = str(uuid.uuid4())
        st.session_state.chats[_id] = {"title": "New Chat", "messages": []}
        st.session_state.active_chat_id = _id
        save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)

def _create_new_chat(title: str | None = None):
    chat_id = str(uuid.uuid4())
    if not title:
        title = f"New Chat {len(st.session_state.chats)+1}"
    st.session_state.chats[chat_id] = {"title": title, "messages": []}
    st.session_state.active_chat_id = chat_id
    if not save_persisted_state(st.session_state.chats, st.session_state.active_chat_id):
        st.error("Failed to create new chat due to persistence error.")
    else:
        st.success("New chat created!")

def _get_active_chat():
    cid = st.session_state.active_chat_id
    if cid and cid in st.session_state.chats:
        return st.session_state.chats[cid]
    if not st.session_state.chats:
        _create_new_chat("New Chat")
    else:
        st.session_state.active_chat_id = next(iter(st.session_state.chats))
    return st.session_state.chats[st.session_state.active_chat_id]

# Sidebar: chat manager
with st.sidebar:
    st.header("🤖 GPT-OSS-20B Chat")
    if "reasoning_level" not in st.session_state:
        st.session_state.reasoning_level = "Medium"
    st.caption("Reasoning Level")
    st.session_state.reasoning_level = st.selectbox(
        "Reasoning Level",
        options=["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(st.session_state.reasoning_level),
        label_visibility="collapsed",
        key="reasoning_level_select"
    )
    if st.button("New Chat", key="new_chat", use_container_width=True):
        _create_new_chat(f"Chat {len(st.session_state.chats)+1}")
        st.rerun()

    st.subheader("Conversations")
    if st.button("Clear Current Chat", key="clear_chat", use_container_width=True):
        active_chat = _get_active_chat()
        active_chat["messages"] = []
        save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)
        st.rerun()

    # Inline rename state
    if "rename_id" not in st.session_state:
        st.session_state.rename_id = None
        st.session_state.rename_value = ""

    # List chats with robust rename/delete behavior
    for cid, data in list(st.session_state.chats.items()):
        title = data.get("title") or f"Chat {cid[:8]}"
        btn_type = "primary" if cid == st.session_state.active_chat_id else "secondary"
        if st.button(title, key=f"chat_btn_{cid}", use_container_width=True, type=btn_type):
            st.session_state.active_chat_id = cid
            save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Rename", key=f"ren_{cid}", use_container_width=True, type="secondary"):
                st.session_state.rename_id = cid
                st.session_state.rename_value = title
                st.rerun()
        with c2:
            if st.button("Delete", key=f"del_{cid}", use_container_width=True, type="secondary"):
                if cid in st.session_state.chats:
                    was_active = (st.session_state.active_chat_id == cid)
                    st.session_state.chats.pop(cid, None)
                    if was_active:
                        st.session_state.active_chat_id = next(iter(st.session_state.chats), None)
                    save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)
                    st.rerun()

        # Inline rename UI
        if st.session_state.rename_id == cid:
            new_title = st.text_input(
                "Rename chat",
                value=st.session_state.rename_value,
                key=f"ren_input_{cid}",
                label_visibility="collapsed",
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("Save", key=f"ren_save_{cid}", use_container_width=True, type="secondary"):
                    if new_title.strip():
                        st.session_state.chats[cid]["title"] = new_title.strip()
                        st.session_state.rename_id = None
                        st.session_state.rename_value = ""
                        save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)
                        st.rerun()
                    else:
                        st.warning("Title cannot be empty")
            with rc2:
                if st.button("Cancel", key=f"ren_cancel_{cid}", use_container_width=True, type="secondary"):
                    st.session_state.rename_id = None
                    st.session_state.rename_value = ""
                    st.rerun()

# Main chat interface
st.markdown(
    """
    <h1>🤖 GPT-OSS-20B Chat</h1>
    <p>Conversational AI powered by open-source 20B model</p>
    """,
    unsafe_allow_html=True,
)

if not os.getenv("HF_TOKEN"):
    st.error("Missing HF_TOKEN environment variable. Create a token at https://huggingface.co/settings/tokens and set HF_TOKEN.")
    st.stop()

try:
    client = InferenceClient("openai/gpt-oss-20b", token=os.getenv("HF_TOKEN"))
except Exception as e:
    st.warning(f"Primary model unavailable ({str(e)}). Falling back to microsoft/DialoGPT-medium")
    try:
        client = InferenceClient("microsoft/DialoGPT-medium", token=os.getenv("HF_TOKEN"))
    except Exception as fallback_e:
        st.error(f"Fallback failed: {str(fallback_e)}")
        client = None

active_chat = _get_active_chat()
for entry in active_chat["messages"]:
    q = entry.get("q") or (entry.get("role") == "user" and entry.get("content"))
    a = entry.get("a") or (entry.get("role") == "assistant" and entry.get("content"))
    if q and a:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

query = st.chat_input("Type your message here...", key=f"query_{st.session_state.active_chat_id}")

if query:
    active_chat = _get_active_chat()
    insertion_idx = len(active_chat["messages"])
    active_chat["messages"].append({"q": query, "a": ""})
    save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("🤖 Generating answer..."):
        if client is None:
            st.error("No valid client available. Check token and model status.")
        else:
            system = {
                "Low": "Reasoning: low",
                "Medium": "Reasoning: medium",
                "High": "Reasoning: high"
            }[st.session_state.reasoning_level]
            # Build prompt with conversation history
            prompt_text = f"System: You are a helpful assistant. {system}\n"
            for entry in active_chat["messages"]:
                q = entry.get("q", "")
                a = entry.get("a", "")
                if q:
                    prompt_text += f"User: {q}\n"
                if a:
                    prompt_text += f"Assistant: {a}\n"
            prompt_text += "Assistant:"

            try:
                resp = client.text_generation(
                    prompt_text,
                    max_new_tokens=2048,
                    temperature=0.2,
                    stream=True
                )
                out, box = "", st.empty()
                for token in resp:
                    out += token
                    box.markdown(out + "▌")
                box.markdown(out)
                answer = out
            except Exception as e:
                st.error(f"Generation failed: {str(e)}. Check model endpoint status or try again later.")
                answer = "Failed to generate response."

            with st.chat_message("assistant"):
                st.markdown(answer)
            active_chat["messages"][insertion_idx] = {"q": query, "a": answer}
            save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)