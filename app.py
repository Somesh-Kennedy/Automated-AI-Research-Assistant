"""Streamlit front end:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from research_assistant.arxiv_client import fetch_papers
from research_assistant.clustering import cluster_papers

st.set_page_config(page_title="Automated AI Research Assistant", layout="wide")
st.title("Automated AI Research Assistant")
st.caption("Retrieve many arXiv papers for a topic, group them into themes, then optionally have an agent write the summary.")

with st.sidebar:
    topic = st.text_input("Research topic", "retrieval augmented generation")
    n = st.slider("Papers to retrieve", 20, 200, 100, step=10)
    summarize = st.checkbox("Also write an agent summary (needs an LLM key)")
    go = st.button("Run")

if go and topic.strip():
    with st.spinner("Retrieving papers from arXiv..."):
        papers = fetch_papers(topic, n)
    if len(papers) < 4:
        st.warning(f"Only {len(papers)} papers found. Try broader keywords.")
        st.stop()
    result = cluster_papers(papers)
    st.subheader(f"{len(papers)} papers, {result.k} themes (silhouette {result.silhouette:.2f})")

    rank = {c.id: f"Theme {i}" for i, c in enumerate(result.clusters, 1)}      # themes numbered biggest first
    df = pd.DataFrame({"x": result.coords[:, 0], "y": result.coords[:, 1],
                       "theme": [rank[label] for label in result.labels]})
    st.scatter_chart(df, x="x", y="y", color="theme")
    for n_, c in enumerate(result.clusters, 1):
        with st.expander(f"Theme {n_}: {', '.join(c.top_terms[:4])}  ({c.size} papers)"):
            for p in c.papers:
                st.markdown(f"- [{p.title}]({p.link}) ({p.published[:4]})")
    if summarize:
        from research_assistant.crew import run
        with st.spinner("Agents are writing the summary..."):
            st.markdown(run(topic, n))
