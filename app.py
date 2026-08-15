"""Streamlit chat UI: ask about the fusion episodes, get cited answers,
leave thumbs feedback. Every conversation is logged to Postgres.

    uv run --env-file .env streamlit run app.py
"""

import streamlit as st

from fusionrag import db
from fusionrag.rag import RAG

st.set_page_config(page_title="fusion-rag", page_icon="⚛️")


@st.cache_resource
def get_rag():
    return RAG()


@st.cache_resource
def get_db():
    return db.connect()


def send_feedback(conversation_id, key):
    value = st.session_state[key]
    if value is not None:
        db.log_feedback(get_db(), conversation_id, thumbs_up=value == 1)


st.title("⚛️ fusion-rag")
st.caption(
    "Ask three fusion experts anything — answers cite the exact moment in "
    "the episode. Corpus: Lex Fridman #112 (Hutchinson), #353 (Whyte), "
    "#485 (Kirtley)."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.feedback(
                "thumbs",
                key=f"fb-{msg['conversation_id']}",
                on_change=send_feedback,
                args=(msg["conversation_id"], f"fb-{msg['conversation_id']}"),
            )

if question := st.chat_input("e.g. Why is helium-3 a good fusion fuel?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("searching the episodes…"):
            result = get_rag().ask(question)
        conversation_id = db.log_conversation(get_db(), result)
        st.markdown(result["answer"])
        st.caption(
            f"{result['latency_s']:.1f}s · "
            f"{result['prompt_tokens'] + result['completion_tokens']} tokens · "
            f"${result['cost_usd']:.4f}"
        )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "conversation_id": conversation_id,
        }
    )
    st.rerun()
