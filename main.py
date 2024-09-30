# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from openai import OpenAI

from langchain_utils import invoke_chain

st.title("DataManagement Chatbot")



# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key="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")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

# Initialize chat history
if "messages" not in st.session_state:
    print("Creating session state")
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            print("Session state:",st.session_state.messages)
            response = invoke_chain(prompt,st.session_state.messages)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})