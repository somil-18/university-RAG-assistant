import streamlit as st
from src.rag import generate_response


# page setup
st.set_page_config(page_title="IITB RAG")
st.title("🎓 IIT Bombay Chatbot")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# handle input
if user_input := st.chat_input("Ask a question..."):
    # display user message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # generate and stream response
    with st.chat_message("assistant"):
        try:
            response = st.write_stream(generate_response(user_input))
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error generating response: {e}")

