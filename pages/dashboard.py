"""Monitoring dashboard: reads Postgres, 6 charts + headline stats.

    uv run --env-file .env streamlit run app.py   (then pick "dashboard")
"""

import pandas as pd
import streamlit as st

from fusionrag import db

st.set_page_config(page_title="fusion-rag — monitoring", page_icon="📊", layout="wide")


@st.cache_resource
def get_db():
    return db.connect()


def query(sql):
    with get_db().cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


st.title("📊 Monitoring")

conv = query("SELECT * FROM conversations ORDER BY ts")
fb = query("SELECT * FROM feedback")

if conv.empty:
    st.info("No conversations logged yet — ask something in the chat first.")
    st.stop()

conv["ts"] = pd.to_datetime(conv["ts"])
conv["day"] = conv["ts"].dt.date
conv["tokens"] = conv["prompt_tokens"] + conv["completion_tokens"]

ups = int(fb["thumbs_up"].sum()) if not fb.empty else 0
downs = len(fb) - ups

m1, m2, m3, m4 = st.columns(4)
m1.metric("Questions", len(conv))
m2.metric("Total cost", f"${conv['cost_usd'].sum():.2f}")
m3.metric("Median latency", f"{conv['latency_s'].median():.1f}s")
m4.metric("Feedback 👍 / 👎", f"{ups} / {downs}")

left, right = st.columns(2)

with left:
    st.subheader("Questions over time")
    per_day = conv.groupby("day").size().rename("questions").reset_index()
    st.bar_chart(per_day, x="day", y="questions")

    st.subheader("Answer latency")
    bins = pd.cut(conv["latency_s"], bins=[0, 2, 4, 6, 8, 10, 15, 30, 120])
    latency = bins.value_counts().sort_index()
    latency.index = [f"{int(i.left)}–{int(i.right)}s" for i in latency.index]
    st.bar_chart(latency.rename("answers"))

    st.subheader("Episodes cited (retrieved chunks)")
    cited = (
        conv["retrieved_chunk_ids"]
        .explode()
        .dropna()
        .str.split("-")
        .str[0]
        .value_counts()
        .rename("chunks retrieved")
    )
    cited.index = "#" + cited.index
    st.bar_chart(cited)

with right:
    st.subheader("Feedback")
    feedback = pd.Series({"👍 up": ups, "👎 down": downs}, name="votes")
    st.bar_chart(feedback)

    st.subheader("Cost per day (USD)")
    cost = conv.groupby("day")["cost_usd"].sum().rename("cost_usd").reset_index()
    st.bar_chart(cost, x="day", y="cost_usd")

    st.subheader("Retrieval score distribution")
    score_bins = pd.cut(conv["avg_retrieval_score"].dropna(), bins=10)
    scores = score_bins.value_counts().sort_index()
    scores.index = [f"{i.left:.2f}" for i in scores.index]
    st.bar_chart(scores.rename("questions"))

st.subheader("Recent conversations")
st.dataframe(
    conv[["ts", "question", "latency_s", "tokens", "cost_usd"]]
    .sort_values("ts", ascending=False)
    .head(20),
    hide_index=True,
)
