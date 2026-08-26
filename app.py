import os
import time

import streamlit as st

from data_loader import get_catalog, validate_columns
import rule_engine
import llm_engine

st.set_page_config(
    page_title="Odia Bibhaba Library - Interactive Catalog",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",  # user can reopen with the ">>" control
)

DATA_PATH = os.environ.get("CATALOG_PATH", os.path.join("data", "catalog.ods"))

# ----------------------------------------------------------------------------
# Sidebar: config + data status
# ----------------------------------------------------------------------------
api_key = os.environ.get("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")

# Hardcode your preferred fallback model 
model = "gpt-4o-mini" 

with st.sidebar:
    st.divider()
    uploaded = st.file_uploader("Replace catalog file (.xlsx / .ods / .csv)", type=["xlsx", "ods", "csv"])
    if uploaded is not None:
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state["data_path"] = save_path
        st.success(f"Loaded {uploaded.name}")

    data_path = st.session_state.get("data_path", DATA_PATH)

    st.divider()
    if "fast_path_hits" not in st.session_state:
        st.session_state.fast_path_hits = 0
        st.session_state.llm_hits = 0
    st.metric("Answered via pandas (free)", st.session_state.fast_path_hits)
    st.metric("Answered via LLM", st.session_state.llm_hits)

# ----------------------------------------------------------------------------
# Load & cache the catalog exactly once
# ----------------------------------------------------------------------------
try:
    df = get_catalog(data_path)
except Exception as e:
    st.error(f"Couldn't load catalog file at `{data_path}`: {e}")
    st.stop()

missing = validate_columns(df)
if missing:
    st.warning(
        f"The loaded file is missing expected column(s): {', '.join(missing)}. "
        "Some questions may not work correctly until this is fixed."
    )

st.title("📚 Odia Bibhaba Library - Interactive Catalog")
st.caption(
    f"{len(df)} books loaded..." #from `{os.path.basename(data_path)}` — "
    "Ask questions in plain English."
)

with st.expander("Preview catalog data"):
    preview_cols = [c for c in df.columns if not c.startswith("_")]
    
    preview_df = df[preview_cols]
    st.dataframe(preview_df, use_container_width=True)

# ----------------------------------------------------------------------------
# Chat state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": (
            "Hi! Ask me things like *\"How many books are in the Poetry category?\"*, "
            "*\"List books by Gobinda Rath\"*, *\"Which books need review?\"*, or "
            "*\"What's the oldest book in the catalog?\"*"
        )}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------------------------------------------------
# Handle new question
# ----------------------------------------------------------------------------
question = st.chat_input("Ask a question about the catalog...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            reply = rule_engine.answer(question, df)
            source = "pandas"
            if reply is None:
                reply = llm_engine.answer(question, df, api_key=api_key, model=model)
                source = "llm"
            elapsed = time.time() - start

        st.markdown(reply)
        badge = "🟢 Internal (instant, free)" if source == "pandas" else "🔵 with LLM "
        st.caption(f"{badge} · {elapsed:.2f}s")

        if source == "pandas":
            st.session_state.fast_path_hits += 1
        else:
            st.session_state.llm_hits += 1

    st.session_state.messages.append({"role": "assistant", "content": reply})
