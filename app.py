import streamlit as st
import os, json, uuid
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub import HfFolder
load_dotenv()
st.set_page_config(page_title="GPT-2 Chat", page_icon="🤖", layout="wide")

# Fallback: try to read HF token from a local api.txt if present (never committed)
def _fallback_read_hf_token():
    try:
        if os.path.exists("api.txt"):
            txt = open("api.txt","r",encoding="utf-8").read()
            for part in txt.replace("\n"," ").split():
                if part.startswith("hf_") and len(part) > 10:
                    return part.strip()
            for ln in txt.splitlines():
                if "HF_TOKEN" in ln and "=" in ln:
                    return ln.split("=",1)[1].strip()
    except Exception:
        pass
    return ""

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
        --primary-gradient: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Streamlit light mode (default) */
    .stApp {
        --background-color: #ffffff;
        --text-color: #262730;
        --card-background: #f8f9fa;
        --border-color: #e9ecef;
        --info-box-bg: #e3f2fd;
        --info-box-border: #2196f3;
    }
    
    /* Streamlit dark mode */
    .stApp[data-theme="dark"] {
        --background-color: #0e1117;
        --text-color: #fafafa;
        --card-background: #262730;
        --border-color: #464646;
        --info-box-bg: #1e3a5f;
        --info-box-border: #4fc3f7;
    }
    
    /* App container background */
    .stApp[data-theme="dark"] [data-testid="stAppViewContainer"] {
        background-color: var(--background-color) !important;
    }
    
    /* Main content area */
    .stApp[data-theme="dark"] .main .block-container {
        background-color: var(--background-color) !important;
        padding: 0 !important;
    }
    
    /* Main header */
    .main-header {
        background: var(--primary-gradient) !important;
        padding: 1rem 1.2rem !important;
        border-radius: 6px !important;
        margin-bottom: 0.5rem !important;
        color: white !important;
        text-align: center !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    .main-header h1 {
        margin: 0 !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
    }
    
    .main-header p {
        margin: 0.1rem 0 0 0 !important;
        font-size: 0.8rem !important;
        opacity: 0.9 !important;
        line-height: 1.1 !important;
    }
    
    /* Session card */
    .session-card {
        background: var(--card-background);
        padding: 0.8rem !important;
        border-radius: 4px !important;
        border-left: 2px solid #667eea;
        margin: 0.3rem 0 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        color: var(--text-color);
        max-width: 800px !important;
        font-size: 0.8rem !important;
        line-height: 1.2 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 18px;
        padding: 0.3rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        padding: 0.2rem 0.6rem !important;
        border-radius: 6px !important;
        font-size: 0.75rem !important;
        line-height: 1.1 !important;
    }
    
    /* Conversation list */
    #convo-list .element-container { margin: 0 !important; padding: 0 !important; }
    #convo-list .stButton { margin: 0 !important; }
    #convo-list .stButton > button { margin: 0 !important; }
    #convo-list .conv-title { margin: 0 !important; }
    #convo-list .conv-group { display: flex; flex-direction: column; gap: 0 !important; margin: 0 !important; padding: 0 !important; }
    #convo-list .conv-row { margin: 0 !important; padding: 0 !important; }
    #convo-list .conv-row [data-testid="column"] { padding: 0 !important; margin: 0 !important; }
    #convo-list .conv-title .stButton > button { box-shadow: none !important; background: var(--card-background) !important; color: var(--text-color) !important; border-radius: 6px !important; padding: 0.3rem 0.6rem !important; }
    #convo-list .conv-actions .stButton > button { padding: 0.15rem 0.4rem !important; font-size: 0.7rem !important; border-radius: 4px !important; }
    #convo-list .conv-actions { margin: 0 !important; margin-top: 0 !important; }
    
    /* Info box */
    .info-box {
        background: var(--info-box-bg);
        border-left: 2px solid var(--info-box-border);
        padding: 0.4rem 0.6rem !important;
        border-radius: 3px;
        margin: 0.5rem 0 !important;
        color: var(--text-color);
        font-size: 0.8rem !important;
        text-align: center !important;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 4px;
    }
    
    .status-active { background: #00b894; animation: pulse 2s infinite; }
    .status-inactive { background: #ddd; }
    
    @keyframes pulse { 0%{opacity:1} 50%{opacity:0.5} 100%{opacity:1} }
    
    /* Dark mode overrides */
    .stApp[data-theme="dark"] .stAlert {
        background-color: var(--card-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    .stApp[data-theme="dark"] .stMarkdown { color: var(--text-color) !important; }
    .stApp[data-theme="dark"] .stText { color: var(--text-color) !important; }
    .stApp[data-theme="dark"] .stAlert[data-baseweb="notification"][data-severity="success"] { background-color: #22543d !important; color: #9ae6b4 !important; border: 1px solid #38a169 !important; }
    .stApp[data-theme="dark"] .stAlert[data-baseweb="notification"][data-severity="error"] { background-color: #742a2a !important; color: #feb2b2 !important; border: 1px solid #e53e3e !important; }
    
    /* Chat bubbles */
    [data-testid="chatAvatarIcon-user"],[data-testid="chatAvatarIcon-assistant"]{display:none!important}
    .stChatMessage[data-testid="user-message"]{display:flex!important;flex-direction:row-reverse!important;justify-content:flex-end!important;margin:4px 0!important}
    .stChatMessage[data-testid="assistant-message"]{display:flex!important;flex-direction:row!important;justify-content:flex-start!important;margin:4px 0!important}
    .stChatMessage[data-testid="user-message"] .stMarkdown{background:#667eea!important;color:#fff!important;padding:6px 10px!important;border-radius:10px 10px 3px 10px!important;max-width:65%!important;margin-left:auto!important}
    .stChatMessage[data-testid="assistant-message"] .stMarkdown{background:var(--card-background)!important;color:var(--text-color)!important;padding:6px 10px!important;border-radius:10px 10px 10px 3px!important;max-width:65%!important;margin-right:auto!important;border:1px solid var(--border-color)!important}
    
    /* Chat input */
    .stChatInput{background:var(--background-color)!important}
    .stChatInput>div{background:var(--background-color)!important;border:1px solid var(--border-color)!important;border-radius:8px;padding:3px}
    .stChatInput textarea,.stChatInput input{font-size:0.85rem;color:var(--text-color)!important}
    
    /* Sidebar adjustments */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { margin: 0 !important; padding: 0 !important; }
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 { margin: 0 !important; padding: 0 !important; font-size: 1.1rem !important; }
    [data-testid="stSidebar"] .stSelectbox { margin: 0 !important; padding: 0 !important; }
    [data-testid="stSidebar"] .stTextInput { margin: 0 !important; padding: 0 !important; }
    [data-testid="stSidebar"] .stCaption { margin: 0.1rem 0 !important; padding: 0 !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Helpers
def _load():
    try: return json.load(open("conversations.json","r",encoding="utf-8")) if os.path.exists("conversations.json") else {}
    except: return {}

def _save(d):
    try: json.dump(d, open("conversations.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    except: pass

S = st.session_state
if "conversations" not in S: S.conversations = _load()
if "cur" not in S: S.cur = next(iter(S.conversations), None)


if "hf" not in S:
    S.hf = (
        os.getenv("HF_TOKEN", "")
        or _fallback_read_hf_token()
        or HfFolder.get_token()
        or ""
    )
if S.hf:
    os.environ["HF_TOKEN"] = S.hf
if "rename_id" not in S: S.rename_id = None
if "rename_value" not in S: S.rename_value = ""
if "confirm_delete_id" not in S: S.confirm_delete_id = None
VERSION = "ui-rename-delete+token-ctrl v4"

with st.sidebar:
    st.markdown('<div><h3>🤖 GPT-2 Chat</h3></div>', unsafe_allow_html=True)
    level = st.selectbox("Reasoning Level", ["Low","Medium","High"], index=1, help="Select the reasoning complexity for responses.")
    # Always allow overriding token from the UI (deployment-friendly)
    token_input = st.text_input("HF Token", value=S.hf, type="password", help="Paste your Hugging Face or provider API token. This overrides .env/api.txt.")
    if token_input != S.hf:
        S.hf = token_input.strip()
        if S.hf:
            os.environ["HF_TOKEN"] = S.hf
    st.caption(f"Token: {'Set' if S.hf else 'Not set'}")
    save_env = st.checkbox("Save token to .env (local only)")
    if save_env and S.hf and st.button("Save HF_TOKEN", use_container_width=True):
        try:
            env_path = ".env"
            lines = []
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("HF_TOKEN=")]
            lines.append(f"HF_TOKEN={S.hf}")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            st.success("Saved HF_TOKEN to .env")
        except Exception as e:
            st.error(f"Failed to save .env: {e}")
    if st.button("Reload .env", use_container_width=True):
        load_dotenv(override=True)
        S.hf = os.getenv("HF_TOKEN", "") or S.hf
        if S.hf:
            os.environ["HF_TOKEN"] = S.hf
        st.rerun()
    if st.button("➕ New Chat", use_container_width=True):
        i = str(uuid.uuid4()); S.conversations[i] = {"title":"New Chat","messages":[]}; S.cur = i; _save(S.conversations); st.rerun()
    st.markdown('<div><h4>Conversations</h4></div>', unsafe_allow_html=True)
    st.markdown('<div id="convo-list">', unsafe_allow_html=True)
    if not S.conversations: st.markdown('<div class="info-box">No conversations yet. Start a new chat!</div>', unsafe_allow_html=True)
    for i, c in list(S.conversations.items()):
        st.markdown('<div class="conv-group">', unsafe_allow_html=True)
        st.markdown('<div class="conv-title">', unsafe_allow_html=True)
        if st.button(c.get("title","New Chat"), key=f"sel_{i}", use_container_width=True): S.cur = i; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="conv-row conv-actions">', unsafe_allow_html=True)
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("Rename", key=f"ren_{i}", use_container_width=True):
                S.rename_id = i; S.rename_value = c.get("title","New Chat")
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
                S.rename_id = None; S.rename_value = ""
                st.rerun()
            if rcol2.button("Cancel", key=f"ren_cancel_{i}", use_container_width=True):
                S.rename_id = None; S.rename_value = ""
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🤖 GPT-2 Chat</h1>
    <p>Conversational AI powered by GPT-2 model</p>
</div>
""", unsafe_allow_html=True)

if not S.hf:
    st.markdown('<div class="info-box">⚠️ Set HF_TOKEN in the sidebar to enable chatting.</div>', unsafe_allow_html=True)
    client = None
else:
    try:
        try:
            client = InferenceClient(provider="auto", token=S.hf)
        except TypeError:
            client = InferenceClient(token=S.hf)
    except Exception as e:
        client = None
        st.error(str(e))

msgs = S.conversations.get(S.cur, {}).get("messages", []) if S.cur else []
for m in msgs:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Type your message here..."):
    if not S.cur:
        S.cur = str(uuid.uuid4()); S.conversations[S.cur] = {"title":"New Chat","messages":[]}
    msgs.append({"role":"user","content":prompt}); S.conversations[S.cur]["messages"] = msgs
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            if client is None:
                st.error("No valid client available. Check token and model status.")
            else:
                sys = {"Low":"Reasoning: low","Medium":"Reasoning: medium","High":"Reasoning: high"}[level]
                # Build the initial prompt with conversation history
                prompt_text = f"System: You are a helpful assistant. {sys}\n\n"
                for msg in msgs:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    prompt_text += f"{role}: {msg['content']}\n"
                prompt_text += "Assistant:"
                
                # Use text_generation directly
                # Prefer conversational task; on 404 or unsupported, fall back to HF text_generation
                target_model = "deepseek-ai/DeepSeek-V3-0324"
                fallback_model = "HuggingFaceH4/zephyr-7b-beta"
                out = ""
                used_fallback = False
                try:
                    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                        resp = client.chat.completions.create(
                            model=target_model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt_text},
                            ],
                            stream=True,
                            temperature=0.7,
                            max_tokens=1000,
                        )
                        box = st.empty()
                        for chunk in resp:
                            delta = (
                                getattr(chunk, "choices", [None])[0]
                                and getattr(chunk.choices[0], "delta", None)
                            )
                            token_text = (getattr(delta, "content", None) or "") if delta else ""
                            out += token_text
                            box.markdown(out + "▌")
                        box.markdown(out)
                    else:
                        raise AttributeError("chat API not available")
                except Exception as e_conv:
                    used_fallback = True
                    out = ""
                    resp = client.text_generation(
                        prompt=prompt_text,
                        model=fallback_model,
                        temperature=0.7,
                        max_new_tokens=1000,
                        stream=True
                    )
                    box = st.empty()
                    for token in resp:
                        out += token
                        box.markdown(out + "▌")
                    box.markdown(out)
                
                msgs.append({"role":"assistant","content":out})
                if len(msgs) == 2: 
                    S.conversations[S.cur]["title"] = msgs[0]["content"][:30] + ("..." if len(msgs[0]["content"]) > 30 else "")
                S.conversations[S.cur]["messages"] = msgs
                _save(S.conversations)
        except Exception as e:
            st.error(f"Generation failed: {str(e)}. Check model endpoint status or try again later. Request ID: {getattr(e, 'request_id', 'N/A')}")

with st.container():
    st.markdown('<div id="clear-chat">', unsafe_allow_html=True)
    if S.cur and st.button("🗑️ Clear Current Chat", use_container_width=True):
        S.conversations[S.cur]["messages"] = []; _save(S.conversations); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
