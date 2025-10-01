import os
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import uuid
from huggingface_hub import InferenceClient

# Helper to safely extract assistant text from HF chat response
def _extract_assistant_text(chat_resp) -> str:
    try:
        choice = chat_resp.choices[0]
        msg = choice.message
        if isinstance(msg, dict):
            content = msg.get("content") or msg.get("reasoning_content")
        else:
            content = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None)
        if content and isinstance(content, str):
            return content.strip()
    except Exception:
        pass
    try:
        return str(chat_resp).strip()
    except Exception:
        return ""

# --- Streamlit Setup ---
load_dotenv()
st.set_page_config(page_title="GPT-OSS-20B Chat", page_icon="🤖", layout="wide")

# Modern UI CSS with light/dark mode support
CSS = """
<style>
    /* CSS Variables for theme support */
    :root {
        --background-color: #ffffff;
        --text-color: #262730;
        --card-background: #f8f9fa;
        --border-color: #e9ecef;
        --info-box-bg: #e3f2fd;
        --info-box-border: #2196f3;
    }
    
    .stApp {
        --background-color: #ffffff;
        --text-color: #262730;
        --card-background: #f8f9fa;
        --border-color: #e9ecef;
        --info-box-bg: #e3f2fd;
        --info-box-border: #2196f3;
    }
    
    .stApp[data-theme="dark"] {
        --background-color: #0e1117;
        --text-color: #fafafa;
        --card-background: #262730;
        --border-color: #464646;
        --info-box-bg: #1e3a5f;
        --info-box-border: #4fc3f7;
    }
    
    .stApp[data-theme="dark"] [data-testid="stAppViewContainer"] {
        background-color: var(--background-color) !important;
    }
    
    .stApp[data-theme="dark"] .main .block-container {
        background-color: var(--background-color) !important;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        padding: 1.5rem 2rem !important;
        border-radius: 10px !important;
        margin-bottom: 1.5rem !important;
        color: white !important;
        text-align: center !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    .main-header h1 {
        margin: 0 !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
    }
    
    .main-header p {
        margin: 0.3rem 0 0 0 !important;
        font-size: 1rem !important;
        opacity: 0.9 !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    [data-testid="stSidebar"] .stButton > button {
        padding: 6px 10px !important;
        border-radius: 10px !important;
        font-size: 13px !important;
    }
    
    #convo-list .element-container { margin: 0 !important; padding: 0 !important; }
    #convo-list .stButton { margin: 0 !important; }
    #convo-list .conv-title .stButton > button { box-shadow: none !important; }
    #convo-list .conv-row { margin-top: -10px !important; }
    #convo-list .conv-row [data-testid="column"] { padding: 0 !important; }
    
    [data-testid="chatAvatarIcon-user"],[data-testid="chatAvatarIcon-assistant"]{display:none!important}
    .stChatMessage[data-testid="user-message"]{display:flex!important;flex-direction:row-reverse!important;justify-content:flex-end!important;margin:8px 0!important}
    .stChatMessage[data-testid="assistant-message"]{display:flex!important;flex-direction:row!important;justify-content:flex-start!important;margin:8px 0!important}
    .stChatMessage[data-testid="user-message"] .stMarkdown{background:#667eea!important;color:#fff!important;padding:10px 14px!important;border-radius:16px 16px 4px 16px!important;max-width:62%!important;margin-left:auto!important}
    .stChatMessage[data-testid="assistant-message"] .stMarkdown{background:var(--card-background)!important;color:var(--text-color)!important;padding:10px 14px!important;border-radius:16px 16px 16px 4px!important;max-width:62%!important;margin-right:auto!important;border:1px solid var(--border-color)!important}

    .stChatInput{background:var(--background-color)!important}
    .stChatInput>div{background:var(--background-color)!important;border:1px solid var(--border-color)!important;border-radius:12px;padding:6px}
    .stChatInput textarea,.stChatInput input{font-size:14px;color:var(--text-color)!important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Persistence helpers
PERSIST_FILE = Path("conversations.json")

def _load():
    try:
        return json.load(open(PERSIST_FILE,"r",encoding="utf-8")) if PERSIST_FILE.exists() else {}
    except:
        return {}

def _save(d):
    try:
        json.dump(d, open(PERSIST_FILE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    except:
        pass

S = st.session_state
if "conversations" not in S: S.conversations = _load()
if "cur" not in S: S.cur = next(iter(S.conversations), None)
if "hf" not in S: S.hf = os.getenv("HF_TOKEN", "")
if "rename_id" not in S: S.rename_id = None
if "rename_value" not in S: S.rename_value = ""

with st.sidebar:
    st.title("🤖 GPT-OSS-20B Chat")
    level = st.selectbox("Reasoning Level", ["Low","Medium","High"], index=1)
    
    if st.button("➕ New Chat", use_container_width=True):
        i = str(uuid.uuid4())
        S.conversations[i] = {"title":"New Chat","messages":[]}
        S.cur = i
        _save(S.conversations)
        st.rerun()
    
    st.markdown("### Conversations")
    st.markdown('<div id="convo-list">', unsafe_allow_html=True)
    if not S.conversations:
        st.info("No conversations yet. Start a new chat!")
    
    for i, c in list(S.conversations.items()):
        st.markdown('<div class="conv-group">', unsafe_allow_html=True)
        st.markdown('<div class="conv-title">', unsafe_allow_html=True)
        if st.button(c.get("title","New Chat"), key=f"sel_{i}", use_container_width=True):
            S.cur = i
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="conv-row">', unsafe_allow_html=True)
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("Rename", key=f"ren_{i}", use_container_width=True):
                S.rename_id = i
                S.rename_value = c.get("title","New Chat")
        with col_right:
            if st.button("Delete", key=f"del_{i}", use_container_width=True):
                S.conversations.pop(i, None)
                S.cur = next(iter(S.conversations), None)
                _save(S.conversations)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        if S.rename_id == i:
            new_title = st.text_input("Rename conversation", value=S.rename_value, key=f"ren_input_{i}")
            rcol1, rcol2 = st.columns(2)
            if rcol1.button("Save", key=f"ren_save_{i}", use_container_width=True):
                title = (new_title or "").strip() or c.get("title","New Chat")
                S.conversations[i]["title"] = title
                _save(S.conversations)
                S.rename_id = None
                S.rename_value = ""
                st.rerun()
            if rcol2.button("Cancel", key=f"ren_cancel_{i}", use_container_width=True):
                S.rename_id = None
                S.rename_value = ""
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🤖 GPT-OSS-20B Chat</h1>
    <p>Open-source 20B reasoning chat assistant</p>
</div>
""", unsafe_allow_html=True)

if not S.hf:
    st.warning("⚠️ Set HF_TOKEN in environment.")
    st.stop()

try:
    client = InferenceClient("openai/gpt-oss-20b", token=S.hf)
except Exception as e:
    st.error(str(e))
    st.stop()

msgs = S.conversations.get(S.cur, {}).get("messages", []) if S.cur else []
for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Type your message here..."):
    if not S.cur:
        S.cur = str(uuid.uuid4())
        S.conversations[S.cur] = {"title":"New Chat","messages":[]}
    
    msgs.append({"role":"user","content":prompt})
    S.conversations[S.cur]["messages"] = msgs
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            sys = {"Low":"Reasoning: low","Medium":"Reasoning: medium","High":"Reasoning: high"}[level]
            messages = [{"role":"system","content":f"You are a helpful assistant. {sys}"}] + msgs
            
            # Try chat_completion first, fallback to text_generation
            answer = ""
            try:
                chat_resp = client.chat_completion(
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7,
                    stream=False
                )
                answer = _extract_assistant_text(chat_resp)
            except AttributeError:
                # Fallback: use text_generation for older API
                prompt_text = f"System: You are a helpful assistant. {sys}\n\n"
                for msg in msgs:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    prompt_text += f"{role}: {msg['content']}\n"
                prompt_text += "Assistant:"
                answer = client.text_generation(
                    prompt_text,
                    temperature=0.7,
                    max_new_tokens=1000,
                    stream=False
                )
            
            st.markdown(answer)
            
            msgs.append({"role":"assistant","content":answer})
            if len(msgs)==2:
                S.conversations[S.cur]["title"] = msgs[0]["content"][:30]+("..." if len(msgs[0]["content"])>30 else "")
            S.conversations[S.cur]["messages"] = msgs
            _save(S.conversations)
        except Exception as e:
            st.error(str(e))

with st.container():
    st.markdown('<div id="clear-chat">', unsafe_allow_html=True)
    if S.cur and st.button("🗑️ Clear Current Chat", use_container_width=True):
        S.conversations[S.cur]["messages"] = []
        _save(S.conversations)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
