import streamlit as st
from src.rag import load_models, build_rag_chain

# page configuration
st.set_page_config(
    page_title="IITB RAG Chatbot",
    page_icon="🎓",
    layout="wide"
)

# sidebar
with st.sidebar:
    st.header("ℹ️ About This Chatbot")
    
    st.markdown("""
    ### 🎓 IIT Bombay RAG Chatbot
    
    This is a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions based on IIT Bombay documentation.
    """)
    
    st.divider()
    
    st.markdown("""
    ### ⚙️ How It Works
    - Searches through IIT Bombay documents
    - Retrieves relevant information
    - Generates accurate responses
    """)
    
    st.divider()
    
    st.markdown("""
    ### ⚠️ Important Notes
    - **No Memory**: This chatbot does NOT remember previous conversations
    - Each question is processed independently
    - Context is not carried between sessions
    """)
    
    st.divider()
    
    st.markdown("""
    ### 📚 Data Source
    IIT Bombay Official Documentation
    """)

# title
st.title("🎓 IIT Bombay Chatbot")
st.markdown("Ask questions about IIT Bombay policies, programs, and procedures.")

# load models (cached)
@st.cache_resource 
def get_chain():
    embd_model, llm = load_models()
    return build_rag_chain(llm, embd_model)

try:
    chain = get_chain()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# handles user's input
if user_input := st.chat_input("Ask a question about IIT Bombay..."):
    # display user's message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            response = chain.invoke(user_input)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

