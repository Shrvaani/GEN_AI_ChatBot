import streamlit as st
import os, json, uuid
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
st.set_page_config(page_title="GPT-OSS-20B Chat", page_icon="🤖", layout="wide")

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
    
    /* Streamlit light mode (default) */
    .stApp {
        --background-color: #ffffff;
        --text-color: #262730;
        --card-background: #f8f9fa;
        --border-color: #e9ecef;
        --info-box-bg: #e3f2fd;
        --info-box-border: #2196f3;
    }
    
    /* Streamlit dark mode detection */
    .stApp[data-theme="dark"] {
        --background-color: #0e1117;
        --text-color: #fafafa;
        --card-background: #262730;
        --border-color: #464646;
        --info-box-bg: #1e3a5f;
        --info-box-border: #4fc3f7;
    }
    
    /* App container background - only for dark mode */
    .stApp[data-theme="dark"] [data-testid="stAppViewContainer"] {
        background-color: var(--background-color) !important;
    }
    
    /* Main content area - only for dark mode */
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
        max-height: 120px !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    .main-header h1 {
        margin: 0 !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
        text-align: center !important;
        width: 100% !important;
    }
    
    .main-header p {
        margin: 0.3rem 0 0 0 !important;
        font-size: 1rem !important;
        opacity: 0.9 !important;
        line-height: 1.3 !important;
        text-align: center !important;
        width: 100% !important;
    }
    
    .session-card {
        background: var(--card-background);
        padding: 1.2rem !important;
        border-radius: 8px !important;
        border-left: 4px solid #667eea;
        margin: 0.6rem 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: var(--text-color);
        max-width: 800px !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
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
    
    .info-box {
        background: var(--info-box-bg);
        border-left: 4px solid var(--info-box-border);
        padding: 0.75rem 1rem !important;
        border-radius: 5px;
        margin: 1rem 0 !important;
        color: var(--text-color);
        max-width: none !important;
        width: 100% !important;
        text-align: center !important;
        font-size: 0.9rem !important;
    }
    
    .progress-container {
        background: var(--card-background);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--text-color);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active { background: #00b894; animation: pulse 2s infinite; }
    .status-inactive { background: #ddd; }
    
    @keyframes pulse { 0%{opacity:1} 50%{opacity:.5} 100%{opacity:1} }
    
    /* Dark mode specific overrides for Streamlit elements */
    .stApp[data-theme="dark"] .stAlert {
        background-color: var(--card-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    .stApp[data-theme="dark"] .stMarkdown { color: var(--text-color) !important; }
    .stApp[data-theme="dark"] .stText { color: var(--text-color) !important; }
    .stApp[data-theme="dark"] .element-container .stAlert { background-color: #2d3748 !important; color: #e2e8f0 !important; border: 1px solid #4a5568 !important; }
    .stApp[data-theme="dark"] .stAlert[data-baseweb="notification"][data-severity="success"] { background-color: #22543d !important; color: #9ae6b4 !important; border: 1px solid #38a169 !important; }
    .stApp[data-theme="dark"] .stAlert[data-baseweb="notification"][data-severity="error"] { background-color: #742a2a !important; color: #feb2b2 !important; border: 1px solid #e53e3e !important; }
    .stApp[data-theme="dark"] .stAlert[data-baseweb="notification"][data-severity="warning"] { background-color: #744210 !important; color: #faf089 !important; border: 1px solid #d69e2e !important; }
    .stApp[data-theme="dark"] .stAlert[data-baseweb="notification"][data-severity="info"] { background-color: #2a4365 !important; color: #90cdf4 !important; border: 1px solid #3182ce !important; }

    /* Chat bubbles aligned with theme */
    [data-testid="chatAvatarIcon-user"],[data-testid="chatAvatarIcon-assistant"]{display:none!important}
    .stChatMessage[data-testid="user-message"]{display:flex!important;flex-direction:row-reverse!important;justify-content:flex-end!important;margin:8px 0!important}
    .stChatMessage[data-testid="assistant-message"]{display:flex!important;flex-direction:row!important;justify-content:flex-start!important;margin:8px 0!important}
    .stChatMessage[data-testid="user-message"] .stMarkdown{background:#667eea!important;color:#fff!important;padding:10px 14px!important;border-radius:16px 16px 4px 16px!important;max-width:62%!important;margin-left:auto!important}
    .stChatMessage[data-testid="assistant-message"] .stMarkdown{background:var(--card-background)!important;color:var(--text-color)!important;padding:10px 14px!important;border-radius:16px 16px 16px 4px!important;max-width:62%!important;margin-right:auto!important;border:1px solid var(--border-color)!important}

    /* Chat input */
    [data-testid="stChatInputContainer"], .stChatInput{background:var(--background-color)!important}
    .stChatInput>div{background:var(--background-color)!important;border:1px solid var(--border-color)!important;border-radius:12px;padding:6px}
    .stChatInput textarea,.stChatInput input{font-size:14px;color:var(--text-color)!important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# tiny helpers
def _load():
    try: return json.load(open("conversations.json","r",encoding="utf-8")) if os.path.exists("conversations.json") else {}
    except: return {}

def _save(d):
    try: json.dump(d, open("conversations.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    except: pass

S = st.session_state
if "conversations" not in S: S.conversations = _load()
if "cur" not in S: S.cur = next(iter(S.conversations), None)
if "hf" not in S: S.hf = os.getenv("HF_TOKEN", "")

with st.sidebar:
    st.title("🤖 GPT-OSS-20B Chat")
    level = st.selectbox("Reasoning Level", ["Low","Medium","High"], index=1)
    if st.button("➕ New Chat", use_container_width=True):
        i = str(uuid.uuid4()); S.conversations[i] = {"title":"New Chat","messages":[]}; S.cur = i; _save(S.conversations); st.rerun()
    st.markdown("### Conversations")
    st.markdown('<div id="convo-list">', unsafe_allow_html=True)
    if not S.conversations: st.info("No conversations yet. Start a new chat!")
    for i, c in list(S.conversations.items()):
        col1, col2 = st.columns([4,1])
        if col1.button(c.get("title","New Chat"), key=f"sel_{i}", use_container_width=True): S.cur = i; st.rerun()
        if col2.button("🗑️", key=f"del_{i}"):
            S.conversations.pop(i, None); S.cur = next(iter(S.conversations), None); _save(S.conversations); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Main Header replacing plain title
st.markdown("""
<div class="main-header">
    <h1>🤖 GPT-OSS-20B Chat</h1>
    <p>Open-source 20B reasoning chat assistant</p>
</div>
""", unsafe_allow_html=True)

if not S.hf: st.warning("⚠️ Set HF_TOKEN in environment."); st.stop()
try: client = InferenceClient("openai/gpt-oss-20b", token=S.hf)
except Exception as e: st.error(str(e)); st.stop()

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
            sys = {"Low":"Reasoning: low","Medium":"Reasoning: medium","High":"Reasoning: high"}[level]
            resp = client.chat_completion(messages=[{"role":"system","content":f"You are a helpful assistant. {sys}"}]+msgs, temperature=0.7, max_tokens=1000, stream=True)
            out, box = "", st.empty()
            for ch in resp:
                t = getattr(getattr(ch.choices[0],"delta",object()),"content",None)
                if t is None and hasattr(ch,"generated_text"): out = ch.generated_text
                elif t: out += t
                box.markdown(out+"▌")
            box.markdown(out); msgs.append({"role":"assistant","content":out})
            if len(msgs)==2: S.conversations[S.cur]["title"] = msgs[0]["content"][:30]+("..." if len(msgs[0]["content"])>30 else "")
            S.conversations[S.cur]["messages"] = msgs; _save(S.conversations)
        except Exception as e: st.error(str(e))

with st.container():
    st.markdown('<div id="clear-chat">', unsafe_allow_html=True)
    if S.cur and st.button("🗑️ Clear Current Chat"): S.conversations[S.cur]["messages"] = []; _save(S.conversations); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


    