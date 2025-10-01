import os
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import uuid
from huggingface_hub import InferenceClient
from requests.exceptions import HTTPError

# -----------------------------
# Helpers
# -----------------------------
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

def pick_working_model(token: str, candidates):
    """
    Return the first model that is deployed on HF Serverless Inference API.
    If none work, return gpt2.
    """
    probe = InferenceClient(token=token)
    for mid in candidates:
        try:
            status = probe.get_model_status(mid)  # checks deployment
            return mid
        except Exception:
            continue
    return "openai-community/gpt2"  # fallback always works

# -----------------------------
# Streamlit setup
# -----------------------------
load_dotenv()
st.set_page_config(page_title="HF Chat (Serverless)", page_icon="🤖", layout="wide")

CSS = """
<style>
:root { --background-color:#ffffff; --text-color:#262730; --card-background:#f8f9fa; --border-color:#e9ecef; }
.stApp[data-theme="dark"] { --background-color:#0e1117; --text-color:#fafafa; --card-background:#262730; --border-color:#464646; }
.main-header { background: linear-gradient(90deg, #667eea, #764ba2); padding: 1.5rem 2rem; border-radius: 10px; margin-bottom: 1.5rem; color: white; text-align: center; }
.main-header h1 { margin: 0; font-size: 2rem; font-weight: 600; }
.stButton > button { background: linear-gradient(90deg, #667eea, #764ba2); color: white; border: none; border-radius: 25px; padding: .5rem 1.5rem; font-weight: bold; transition: .3s; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,.2); }
[data-testid="chatAvatarIcon-user"],[data-testid="chatAvatarIcon-assistant"]{display:none!important}
.stChatMessage[data-testid="user-message"]{display:flex!important;flex-direction:row-reverse!important;justify-content:flex-end!important;margin:8px 0!important}
.stChatMessage[data-testid="assistant-message"]{display:flex!important;flex-direction:row!important;justify-content:flex-start!important;margin:8px 0!important}
.stChatMessage[data-testid="user-message"] .stMarkdown{background:#667eea!important;color:#fff!important;padding:10px 14px!important;border-radius:16px 16px 4px 16px!important;max-width:62%!important;margin-left:auto!important}
.stChatMessage[data-testid="assistant-message"] .stMarkdown{background:var(--card-background)!important;color:var(--text-color)!important;padding:10px 14px!important;border-radius:16px 16px 16px 4px!important;max-width:62%!important;margin-right:auto!important;border:1px solid var(--border-color)!important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    st.warning("⚠️ Set HF_TOKEN in your environment (Hugging Face access token).")
    st.stop()

# Candidate models (try in order)
MODEL_CANDIDATES = [
    "HuggingFaceTB/SmolLM3-3B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]

SELECTED_MODEL = pick_working_model(HF_TOKEN, MODEL_CANDIDATES)

# UI header
st.markdown(f"""
<div class="main-header">
  <h1>🤖 HF Chat (Serverless)</h1>
  <p>Model: <code>{SELECTED_MODEL}</code></p>
</div>
""", unsafe_allow_html=True)

# Persistence
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

with st.sidebar:
    st.title("📂 Chats")
    level = st.selectbox("Reasoning Level", ["Low","Medium","High"], index=1)

    if st.button("➕ New Chat", use_container_width=True):
        i = str(uuid.uuid4())
        S.conversations[i] = {"title":"New Chat","messages":[]}
        S.cur = i
        _save(S.conversations)
        st.rerun()

    st.markdown("### Conversations")
    if not S.conversations:
        st.info("No conversations yet. Start a new chat!")

    for i, c in list(S.conversations.items()):
        if st.button(c.get("title","New Chat"), key=f"sel_{i}", use_container_width=True):
            S.cur = i
            st.rerun()
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

# Create client
try:
    client = InferenceClient(SELECTED_MODEL, token=HF_TOKEN)
except Exception as e:
    st.error(f"Failed to init InferenceClient: {e}")
    st.stop()

# Render history
msgs = S.conversations.get(S.cur, {}).get("messages", []) if S.cur else []
for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
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

            answer = ""
            try:
                # Try chat
                chat_resp = client.chat_completion(
                    messages=messages,
                    max_tokens=800,
                    temperature=0.7,
                    stream=False
                )
                answer = _extract_assistant_text(chat_resp)
            except Exception:
                # Fallback to text generation
                prompt_text = f"System: You are a helpful assistant. {sys}\n\n"
                for msg in msgs:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    prompt_text += f"{role}: {msg['content']}\n"
                prompt_text += "Assistant:"

                answer = client.text_generation(
                    prompt_text,
                    temperature=0.7,
                    max_new_tokens=800,
                    stream=False
                )

            st.markdown(answer)
            msgs.append({"role":"assistant","content":answer})
            if len(msgs)==2:
                S.conversations[S.cur]["title"] = msgs[0]["content"][:30]+("..." if len(msgs[0]["content"])>30 else "")
            S.conversations[S.cur]["messages"] = msgs
            _save(S.conversations)

        except HTTPError as e:
            st.error(f"Inference API error: {e}")
        except Exception as e:
            st.error(str(e))

# Clear chat
if S.cur and st.button("🗑️ Clear Current Chat", use_container_width=True):
    S.conversations[S.cur]["messages"] = []
    _save(S.conversations)
    st.rerun()
