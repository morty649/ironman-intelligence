"""Streamlit UI for Iron Man RAG System"""

import streamlit as st
import time
from rag import retrieve, generate_answer

st.set_page_config(
    page_title="🦾 Iron Man Intelligence",
    page_icon="🤖",
    layout="centered"
)

# Simple styling
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #b30000;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []


@st.cache_resource
def initialize_system():
    """Warm up embedding + vector connection"""
    return True


def main():
    init_session_state()
    initialize_system()

    st.title(" Iron Man (2008) Intelligence Engine")
    st.markdown("Ask questions about Iron Man (2008)")

    st.markdown("---")

    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="How does Tony escape the cave?"
        )
        submit = st.form_submit_button("🔍 Search")

    if submit and question:
        with st.spinner("Analyzing cinematic data..."):
            start_time = time.time()

            docs = retrieve(question)
            answer = generate_answer(question, docs)

            elapsed_time = time.time() - start_time

            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "time": elapsed_time
            })

            st.markdown("### 💡 Answer")
            st.success(answer)

            with st.expander("📄 Source Scenes"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(
                        f"**Scene {doc.metadata['scene_number']} "
                        f"| {doc.metadata['location']}**"
                    )
                    st.text_area(
                        f"Scene {i}",
                        doc.page_content[:800] + "...",
                        height=150,
                        disabled=True
                    )

            st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    # History
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕘 Recent Queries")

        for item in reversed(st.session_state.history[-3:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer'][:200]}...")
            st.caption(f"{item['time']:.2f}s")
            st.markdown("")


if __name__ == "__main__":
    main()
