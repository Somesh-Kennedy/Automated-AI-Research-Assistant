import streamlit as st
from crew import crew
from agents import llm
import os

st.set_page_config(
    page_title="Automated AI Research Assistant",
    layout="wide",
    page_icon="🤖"
)

# CSS styling
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.markdown('<div class="title">🤖 Automated AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Research with arXiv, get structured summaries, and ask follow-up questions</div>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Controls")
    topic = st.text_input("🔍 Enter a research topic")
    if st.button("Run Research"):
        if topic.strip():
            with st.spinner("Researching and writing summary... Please wait ⏳"):
                result = crew.kickoff(inputs={"topic": topic})

            if os.path.exists("research_summary1.md"):
                with open("research_summary1.md", "r", encoding="utf-8") as f:
                    summary_content = f.read()
                st.session_state["summary_content"] = summary_content
            else:
                st.error("❌ Summary file not found.")
        else:
            st.warning("Please enter a topic before running research.")

    if st.button("Clear Summary"):
        st.session_state.pop("summary_content", None)
        st.success("Summary cleared. Enter a new topic.")

if "summary_content" in st.session_state:
    col1, col2 = st.columns([2, 1])

    # summary column
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"📄 {topic} Research Summary")
        st.markdown(st.session_state["summary_content"], unsafe_allow_html=True)

        # Download button
        st.download_button(
            "📥 Download Summary",
            st.session_state["summary_content"],
            file_name="research_summary.md",
            mime="text/markdown"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Q/A column
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💬 Ask Questions")
        query = st.text_area("Enter your question:", height=100)

        if st.button("Get Answer"):
            if query.strip():
                with st.spinner("Thinking..."):
                    full_prompt = f"""
                    You are answering a question based on the following summary:

                    Summary:
                    {st.session_state["summary_content"]}

                    Question:
                    {query}

                    Please provide a clear, concise, and helpful answer.
                    """
                    response = llm.call(full_prompt)

                    if isinstance(response, dict):
                        output_text = response.get("output") or response.get("content") or str(response)
                    else:
                        output_text = str(response)

                st.markdown("### 📝 Answer")
                st.markdown(f'<div class="answer-box">{output_text}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a question first.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
        <div class="info-box">
            👈 Enter a topic in the sidebar and click <b>Run Research</b> to begin.
        </div>
        """, unsafe_allow_html=True)

