import streamlit as st

from rag.pipeline import answer

st.set_page_config(page_title="CellFix Assistant", page_icon="📱")
st.title("📱 CellFix Assistant")
st.caption("A RAG system built from scratch")

question = st.text_input("Ask a question about CellFix")
if question:
    with st.spinner("Searching documents..."):
        reply, hits = answer(question)
    st.markdown(reply)
    with st.expander("Retrieved chunks"):
        for h in hits:
            st.markdown(f"**{h['source']}** (distance {h['distance']:.3f})")
            st.text(h["text"])
