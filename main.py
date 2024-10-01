import streamlit as st
from openai import OpenAI

from langchain_utils import invoke_chain,generate_download_link


st.title("DataManagement AI")


# Add a title and subtitle in the sidebar
st.sidebar.title("Chatbot Settings")
st.sidebar.subheader("Customize your interaction")

# Model selection dropdown
model = st.sidebar.selectbox(
    "Select Model:",
    ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"],
    index=0
)


# Temperature control (for creativity)
temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Save chat history as a text file
if st.sidebar.button("Download Chat History"):
    chat_history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages])
    st.download_button("Download", chat_history, "chat_history.txt")


# Reset Button
if st.sidebar.button("Reset Conversation"):
    st.session_state["messages"] = []

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
    with st.chat_message("user", avatar='img/user-icon.png'):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Generating response..."):
        with st.chat_message("assistant",avatar='img/bkofkgl.png'):
            print("Session state:",st.session_state.messages)
            response = invoke_chain(prompt,st.session_state.messages)
            st.markdown(response)
                # If the query result DataFrame is valid, provide a download link
            href = f'<a href="#" >Download Data as CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
   
    st.session_state.messages.append({"role": "assistant", "content": response})