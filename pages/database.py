"""Raw view of the Postgres tables (conversations + feedback)."""

import streamlit as st

from fusionrag import db

st.set_page_config(page_title="fusion-rag — database", page_icon="🗄️", layout="wide")

st.title("🗄️ Database")
if st.button("Refresh"):
    st.rerun()

conv = db.query("SELECT * FROM conversations ORDER BY id DESC")
fb = db.query("SELECT * FROM feedback ORDER BY id DESC")

st.subheader(f"conversations ({len(conv)} rows)")
st.dataframe(conv, hide_index=True)

st.subheader(f"feedback ({len(fb)} rows)")
st.dataframe(fb, hide_index=True)
