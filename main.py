import streamlit as st
import pandas as pd
from io import StringIO
import ast




st.set_page_config(layout="wide",
                    page_title = "DataManagement AI",
                    page_icon = "img/bkofkgl.png")
from openai import OpenAI

from langchain_utils import invoke_chain,create_chart



st.title("DataManagement AI")

st.logo(
    image = 'img/bklogo.png',
    link="https://bk.rw/personal",
    icon_image=None,
    
)


with st.sidebar:
    # Add a title and subtitle in the sidebar
    st.header("Chatbot Settings")
    # Model selection dropdown
    model = st.selectbox(
        "Model:",
        ["Model 1", "Model 2", "Model 3"],
        index=0
    )
    # Temperature control (for creativity)
    temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.5, step=0.1,)
    # Save chat history as a text file
    if st.button("Download Chat History",use_container_width=True):
        chat_history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages])
        st.download_button("Download", chat_history, "chat_history.txt")
    # Reset Button
    if st.button("Reset Conversation",use_container_width=True):
        st.session_state["messages"] = []

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key="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")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

# Set a default link for dowloads
if "Link" not in st.session_state:
    st.session_state['Link'] = ''

# Initialize chat history
if "messages" not in st.session_state:
    print("Creating session state\n")
    st.session_state.messages = []

# Display chat messages from history on app rerun
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar='img/user-icon.png' if message["role"] == "user" else 'img/bkofkgl.png'):
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
            print(f"Session state: {st.session_state.messages}\n")
            response = invoke_chain(prompt,st.session_state.messages)

            if isinstance(response,str):
                st.markdown(response)

            else:
                st.markdown(response[0])
                # Dynamically extract the download link from the response
                if response[1]!='' or response[2] !='':
                    # Create the href dynamically with the extracted link
                    href = response[1]
                    # Display the download link in the markdown
                    st.markdown(href, unsafe_allow_html=True)
  
                    # Split the string into a list of values
                    results_list = ast.literal_eval(response[2])
                    column_names = ast.literal_eval(response[4])
                    data_columns = ast.literal_eval(response[5])

                    # Convert to DataFrame
                    df = pd.DataFrame(results_list)
                    if response[3] == "none":
                        print("No chart needed, End of Chain\n")
                    else:
                        create_chart(response[3],results_list,column_names,data_columns)
   
    st.session_state.messages.append({"role": "assistant", "content": response if isinstance(response, str) else response[0]})


