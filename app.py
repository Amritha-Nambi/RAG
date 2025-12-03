# app.py
import streamlit as st
from src.chatbot import HarryPotterChatbot

# Page Configuration
st.set_page_config(page_title="Hogwarts AI Librarian", page_icon="")

# Custom CSS for chat
st.markdown("""
<style>
    .stChatMessage { font-family: 'Georgia', sans-serif; }
    .stMarkdown h1 { color: #740001; } /* Gryffindor Red */
</style>
""", unsafe_allow_html=True)

st.title("Hogwarts AI Librarian")
st.markdown("Ask any question about the Wizarding World!")

# Initialize Chatbot (Cached so it doesn't reload on every click)
@st.cache_resource
def load_bot():
    return HarryPotterChatbot()

if "bot" not in st.session_state:
    st.session_state.bot = load_bot()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What is a Horcrux?"):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Get Answer
    with st.chat_message("assistant"):
        with st.spinner("Consulting the library..."):
            result = st.session_state.bot.query(prompt, top_k=10, verbose=False)
            response = result['answer']
            
            st.markdown(response)
            
            # Optional: Show Sources in an Expander
            with st.expander("View Retrieved Context (Source of Truth)"):
                for rank, (idx, score) in enumerate(result['retrieved_chunks'], 1):
                    chunk = st.session_state.bot.chunks[idx]
                    source = chunk.metadata.get('source', 'Unknown')
                    st.markdown(f"**{rank}. {source}** (Score: {score:.3f})")
                    st.caption(chunk.page_content[:200] + "...")

    # 3. Save Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": response})