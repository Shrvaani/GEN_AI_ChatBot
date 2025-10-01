import os
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import uuid
from huggingface_hub import InferenceClient
from requests.exceptions import HTTPError

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
st.set_page_config(page_title="T5-Small Chat", page_icon="🤖", layout="wide")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = "google-t5/t5-small"   # ✅ works with hf-inference

if not HF_TOKEN:
    st.warning("⚠️ Please set HF_TOKEN in Streamlit secrets or .env file.")
    st.stop()

# Init Hugging Face client
try:
    client = InferenceClient(MODEL_ID, token=HF_TOKEN)
except Exception as e:
    st.error(f"Could not init InferenceClient: {e}")
    st.stop()

# -----------------------------
# Persistence
# -----------------------------
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

# -----------------------------
# Sidebar: Chat list
# -----------------------------
with st.sidebar:
    st.title("📂 Conversations")

    if st.button("➕ New Chat", use_container_width=True):
        i = str(uuid.uuid4())
        S.conversations[i] = {"title":"New Chat","messages":[]}
        S.cur = i
        _save(S.conversations)
        st.rerun()

    st.markdown("### History")
    if not S.conversations:
        st.info("No chats yet. Start a new one!")

    for i, c in list(S.conversations.items()):
        if st.button(c.get("title","New Chat"), key=f"sel_{i}", use_container_width=True):
            S.cur = i
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rename", key=f"ren_{i}", use_container_width=True):
                S.rename_id = i
                S.rename_value = c.get("title","New Chat")
        with col2:
            if st.button("Delete", key=f"del_{i}", use_container_width=True):
                S.conversations.pop(i, None)
                S.cur = next(iter(S.conversations), None)
                _save(S.conversations)
                st.rerun()

        if getattr(S,"rename_id",None) == i:
            new_title = st.text_input("Rename", value=S.rename_value, key=f"ren_input_{i}")
            if st.button("Save", key=f"save_{i}"):
                S.conversations[i]["title"] = new_title or "New Chat"
                _save(S.conversations)
                S.rename_id = None
                st.rerun()

# -----------------------------
# Header
# -----------------------------
st.markdown(f"""
<div style="background: linear-gradient(90deg, #667eea, #764ba2); padding: 1rem; border-radius: 8px; margin-bottom: 1.2rem; color: white; text-align:center;">
  <h1>🤖 T5-Small Chat</h1>
  <p>Powered by {MODEL_ID} on HF Inference API</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Main Chat UI
# -----------------------------
msgs = S.conversations.get(S.cur, {}).get("messages", []) if S.cur else []
for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Type your message..."):
    if not S.cur:
        S.cur = str(uuid.uuid4())
        S.conversations[S.cur] = {"title":"New Chat","messages":[]}

    msgs.append({"role":"user","content":prompt})
    S.conversations[S.cur]["messages"] = msgs

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Call T5 via text-to-text
            result = client.text_to_text(prompt, max_new_tokens=200)
            answer = result.strip() if isinstance(result, str) else str(result)

            st.markdown(answer)
            msgs.append({"role":"assistant","content":answer})

            if len(msgs)==2:  # auto-title chat
                S.conversations[S.cur]["title"] = msgs[0]["content"][:30]+("..." if len(msgs[0]["content"])>30 else "")

            S.conversations[S.cur]["messages"] = msgs
            _save(S.conversations)

        except HTTPError as e:
            st.error(f"Inference API error: {e}")
        except Exception as e:
            st.error(str(e))

# -----------------------------
# Clear chat
# -----------------------------
if S.cur and st.button("🗑️ Clear Current Chat", use_container_width=True):
    S.conversations[S.cur]["messages"] = []
    _save(S.conversations)
    st.rerun()
