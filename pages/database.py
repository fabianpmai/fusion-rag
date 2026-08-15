"""Raw view of the Postgres tables (conversations + feedback)."""

import pandas as pd
import streamlit as st

from fusionrag import db

st.set_page_config(page_title="fusion-rag — database", page_icon="🗄️", layout="wide")


@st.cache_resource
def get_db():
    return db.connect()


def query(sql):
    with get_db().cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


st.title("🗄️ Database")
if st.button("Refresh"):
    st.rerun()

conv = query("SELECT * FROM conversations ORDER BY id DESC")
fb = query("SELECT * FROM feedback ORDER BY id DESC")

st.subheader(f"conversations ({len(conv)} rows)")
st.dataframe(conv, hide_index=True)

st.subheader(f"feedback ({len(fb)} rows)")
st.dataframe(fb, hide_index=True)
