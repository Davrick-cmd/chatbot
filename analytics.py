import streamlit as st
import pandas as pd
from io import StringIO
import ast
import base64
from openai import OpenAI
from langchain_utils import invoke_chain, create_chart, create_interactive_visuals
from typing import Optional, Dict, List, Union
from config import OPENAI_API_KEY  # Move API key to separate config file
from pathlib import Path

# Constants
ASSETS_DIR = Path("img")
USER_AVATAR = ASSETS_DIR / "user-icon.png"
BOT_AVATAR = ASSETS_DIR / "bkofkgl.png"
LOGO = ASSETS_DIR / "bklogo.png"

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def process_csv_data(href: str) -> Optional[pd.DataFrame]:
    """Process CSV data from base64 encoded href string."""
    try:
        base64_data = href.split("base64,")[1].split('"')[0]
        csv_data = base64.b64decode(base64_data).decode("utf-8")
        return pd.read_csv(StringIO(csv_data))
    except (IndexError, ValueError) as e:
        st.error(f"Failed to process CSV data: {str(e)}")
        return None

def handle_response(response: Union[str, List]):
    """Handle the chatbot response and generate visualizations."""
    if isinstance(response, str):
        st.markdown(response)
        return response

    message, href, *rest = response
    st.markdown(message)
    
    if href:
        st.markdown(href, unsafe_allow_html=True)
        df = process_csv_data(href)
        
        if df is not None and not df.empty:
            # Create a container for visualizations
            viz_container = st.container()
            
            # Create columns for buttons
            col1, col2 = st.columns(2)
            
            # Interactive visualization button in first column
            with col1:
                if st.button("Generate Interactive Visualizations 📊"):
                    with viz_container:
                        with st.spinner("Generating interactive visualizations..."):
                            # Wait for the visualization to complete
                            visuals = create_interactive_visuals(df)
                            if visuals:
                                st.success("Visualizations generated successfully!")
            
            # Chart generation logic in second column
            with col2:
                column_names = ast.literal_eval(rest[2]) if rest[2] else []
                data_columns = ast.literal_eval(rest[3])
                
                if rest[1] != "none" and 3 <= len(df) <= 24:
                    create_chart(rest[1], df, column_names, data_columns)
    
    return message

def render_welcome_message():
    """Render the welcome message with capabilities."""
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

def render_sidebar():
    """Render sidebar with settings and controls."""
    with st.sidebar:
        
        st.header("Chatbot Settings")
        temperature = st.slider(
            "Creativity (Temperature)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.1, 
            disabled=True
        )
        
        st.divider()
        
        # Download chat history
        if st.button("📥 Download Chat History", use_container_width=True):
            chat_history = "\n".join([
                f'{msg["role"]}: {msg["content"]}' 
                for msg in st.session_state.messages
            ])
            st.download_button(
                "Download",
                chat_history,
                "chat_history.txt",
                mime="text/plain"
            )
        
        # Reset conversation
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def show_analytics():
    """Main function to display the analytics dashboard."""
    initialize_session_state()
    
    # Title and Logo
    # col1, col2, col3 = st.columns([1,2,1])
    # with col2:
    #     st.image(str(LOGO), width=400, use_column_width=True)
    st.title("DataManagement AI")

    # Render sidebar
    render_sidebar()

    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Display chat messages from history
    for message in st.session_state.messages:
        avatar = str(USER_AVATAR if message["role"] == "user" else BOT_AVATAR)
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input(
        f"Hi {st.session_state.get('firstname', 'there')}, "
        "what analytics insights can I help you explore today?"
    )

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=str(USER_AVATAR)):
            st.markdown(prompt)

        # Generate and display response
        with st.spinner("Generating response..."):
            with st.chat_message("assistant", avatar=str(BOT_AVATAR)):
                response = invoke_chain(prompt, st.session_state.messages)
                final_response = handle_response(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response
                })

    # Show welcome message if no messages
    if not st.session_state.messages:
        render_welcome_message()

if __name__ == "__main__":
    show_analytics()
