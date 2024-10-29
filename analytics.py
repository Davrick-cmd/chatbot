import streamlit as st
import pandas as pd
from io import StringIO
import ast
import base64
from openai import OpenAI
from langchain_utils import invoke_chain, create_chart, create_interactive_visuals

def show_analytics():
    # Page configuration
    # st.set_page_config(layout="wide", page_title="DataManagement AI", page_icon="img/bkofkgl.png",
    #                    menu_items={'Get Help': 'mailto:john@example.com',
    #                                'About': "#### This is DataManagement cool app!"})

    # Title and Logo
    st.title("DataManagement AI")
    st.logo(image='img/bklogo.png', link="https://bk.rw/personal")

    # Sidebar for Navigation and Chatbot Settings
    with st.sidebar:
        st.header("Chatbot Settings")
        temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.0, step=0.1, disabled=True)
        
        if st.button("Download Chat History", use_container_width=True):
            chat_history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages])
            st.download_button("Download", chat_history, "chat_history.txt")
            
        if st.button("Reset Conversation", use_container_width=True):
            st.session_state["messages"] = []

    # API key for OpenAI and Session State Initialization
    client = OpenAI(api_key="your_openai_api_key_here")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar='img/user-icon.png' if message["role"] == "user" else 'img/bkofkgl.png'):
            st.markdown(message["content"])

    # Accept user input for Analytics Chatbot
    if prompt := st.chat_input(f"Hi {st.session_state.firstname}, what analytics insights can I help you explore today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar='img/user-icon.png'):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            with st.chat_message("assistant", avatar='img/bkofkgl.png'):
                response = invoke_chain(prompt, st.session_state.messages)
                
                if isinstance(response, str):
                    st.markdown(response)
                else:
                    st.markdown(response[0])
                    
                    # Process CSV data if href is available
                    if response[1]:
                        href = response[1]
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        href = None

                    if href:
                        try:
                            base64_data = href.split("base64,")[1].split('"')[0]
                            csv_data = base64.b64decode(base64_data).decode("utf-8")
                            df = pd.read_csv(StringIO(csv_data))
                            
                            if not df.empty:
                                create_interactive_visuals(df)
                        except (IndexError, ValueError):
                            st.error("Failed to process CSV data. Please check the href content.")
                    
                    # Chart generation logic
                    column_names = ast.literal_eval(response[4]) if response[4] else []
                    data_columns = ast.literal_eval(response[5])
                    
                    if response[3] != "none" and 3 <= len(df) <= 24:
                        create_chart(response[3], df, column_names, data_columns)
                        
        st.session_state.messages.append({"role": "assistant", "content": response if isinstance(response, str) else response[0]})

    # Informational message when there are no messages in session state
    if len(st.session_state.messages) == 0:
        st.markdown("""
            ### How can I help you?
            <ul>
                <li>🗃️ <strong>Data Queries</strong>: Ask about customer, account, or transaction data.</li>
                <li>📥 <strong>Download Results</strong>: Get your query results in CSV format.</li>
                <li>📊 <strong>Data Visualizations</strong>: Get charts summarizing your data.</li>
                <li>📈 <strong>Data Insights</strong>: Gain insights on churn, channel usage, performance indicators, and more.</li>
                <li>🔮 <strong>Forecasting & Predictions</strong>: Get forecasts and predictions based on historical data trends.</li>
                <li>📝 <strong>Chat Assistance</strong>: Get answers about the bank's business processes, news, and more.</li>
            </ul>
        """, unsafe_allow_html=True)
