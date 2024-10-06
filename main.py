import streamlit as st
import pandas as pd
from io import StringIO
import ast



st.set_page_config(layout="wide",
                    page_title = "DataManagement AI",
                    page_icon = "img/bkofkgl.png")
from openai import OpenAI

from langchain_utils import invoke_chain



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
        ["Churn Prediction", "Loan Performance", "Bank overview"],
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
    print("Creating session state")
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
            print("Session state:",st.session_state.messages)
            response = invoke_chain(prompt,st.session_state.messages)

            if isinstance(response,str):
                st.markdown(response)

            else:
                st.markdown(response[0])
                # Dynamically extract the download link from the response
                if response[1]!='':
                    # Create the href dynamically with the extracted link
                    href = response[1]
                    # Display the download link in the markdown
                    st.markdown(href, unsafe_allow_html=True)
  
                    # Split the string into a list of values
                    print(response[2])
                    results_list = ast.literal_eval(response[2])

                    # Convert to DataFrame
                    df = pd.DataFrame(results_list, columns=['Month', 'Customers'])
        
                    # Convert data types (optional, if you want numeric columns)
                    df = df.astype({'Month': int, 'Customers': int})
                    print(df)

                    st.line_chart(df.set_index('Month')['Customers'])
   
    st.session_state.messages.append({"role": "assistant", "content": response if isinstance(response, str) else response[0]})
