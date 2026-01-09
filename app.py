import streamlit as st
from src.rag import load_models, build_rag_chain

# 1. Setup Page
st.set_page_config(page_title="IITB RAG")
st.title("🎓 IIT Bombay Chatbot")

# 2. Load Models 
# this tells Streamlit: "Run this function once to load the AI, and then keep it in memory. Don't load it again unless the app restarts."
@st.cache_resource 
def get_chain():
    embd_model, llm = load_models()
    return build_rag_chain(llm, embd_model)

try:
    chain = get_chain()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. Handle Input
if user_input := st.chat_input("Ask a question..."):
    # Display User Message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(user_input)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

