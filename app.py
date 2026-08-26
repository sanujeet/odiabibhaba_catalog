import os
import time

import streamlit as st

from data_loader import get_catalog, validate_columns
import rule_engine
import llm_engine

st.set_page_config(
    page_title="Odia Bibhaba - Interactive Catalog",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",  # user can reopen with the ">>" control
)

DATA_PATH = os.environ.get("CATALOG_PATH", os.path.join("data", "catalog.ods"))

# ----------------------------------------------------------------------------
# Background Config (Sidebar removed)
# ----------------------------------------------------------------------------
api_key = os.environ.get("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
model = "gpt-4o-mini"
data_path = DATA_PATH  # Locks the app to always use the default catalog.ods

if "fast_path_hits" not in st.session_state:
    st.session_state.fast_path_hits = 0
    st.session_state.llm_hits = 0

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
            "Ask things like *\"How many books are in the Poetry category?\"*, "
            "*\"List books by Gobinda Rath\"*, *\"Which books need review?\"*, or "
            "*\"What's the oldest book in the catalog?\"*"
        )}
    ]

# Create a fixed-height scrollable box for the chat (adjust height as needed)
#chat_container = st.container(height=500)

#with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ----------------------------------------------------------------------------
# Handle new question
# ----------------------------------------------------------------------------
question = st.chat_input("Ask a question about the catalog...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Make sure new messages are ALSO drawn inside the fixed container
    with chat_container:
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
