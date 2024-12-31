import streamlit as st
from streamlit_feedback import streamlit_feedback
import pandas as pd
from io import StringIO
import base64
from openai import OpenAI
from langchain_utils import invoke_chain, create_chart, create_interactive_visuals, log_conversation_details
from typing import Optional, Dict, List, Union
from config import OPENAI_API_KEY
from pathlib import Path
import random

# Constants
ASSETS_DIR = Path("img")
USER_AVATAR = ASSETS_DIR / "user-icon.png"
BOT_AVATAR = ASSETS_DIR / "bkofkgl.png"
LOGO = ASSETS_DIR / "bklogo.png"

def _submit_feedback(user_response, emoji=None):
    st.toast(f"Feedback submitted: {user_response}", icon=emoji)
    print(f"Feedback submitted: {user_response}", emoji)
    
    # Mapping of emojis to words
    emoji_to_word = {
        "😀": "Very Happy",
        "🙂": "Happy",
        "😐": "Neutral",
        "🙁": "Unhappy",
        "😞": "Very Unhappy",
        # Add more emojis and their corresponding words as needed
    }

    # Convert emoji score to words
    feedback_score_word = emoji_to_word.get(user_response['score'], "Unknown")

    # Log the feedback with the word representation
    log_conversation_details(
        user_id=st.session_state.get('username', 'anonymous'),
        question=st.session_state.current_message['prompt'],
        sql_query=st.session_state.current_message['query'],
        answer=st.session_state.current_message['message'],
        feedback=feedback_score_word,  # Use the word representation
        feedback_comment=user_response['text']
    )
    
    return user_response.update({"some metadata": 123})

# Session State Management
def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Data Processing
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
        return response, None

    message, href, data_str, chart_type, column_names, data_column, query = response
    st.markdown(message)
    
    if href:
        st.markdown(href, unsafe_allow_html=True)
        df = process_csv_data(href)
        st.session_state['Link'] = href
        
        if df is not None and not df.empty:
            create_interactive_visuals(df)
            if chart_type != "none" and 3 <= len(df) <= 24:
                create_chart(chart_type, df)
    
    return message, query

# UI Components
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
        st.divider()
        
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
        
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown("<h5 style='color: gray;'>For support, contact: <a href='mailto:datamanagementai.bk.rw'>datamanagementai.bk.rw</a></h5>", unsafe_allow_html=True)



# Main Application
def show_analytics():
    """Main function to display the analytics dashboard."""
    initialize_session_state()
    
    st.title("DataManagement AI")
    render_sidebar()

    client = OpenAI(api_key=OPENAI_API_KEY)

    messages = st.session_state.messages

    # Display chat messages
    for n,message in enumerate(messages):
        avatar = str(USER_AVATAR if message["role"] == "user" else BOT_AVATAR)
        with st.chat_message(message["role"], avatar=avatar):
            # Display message content
            st.markdown(message["content"])            
            # Display feedback only if it exists for assistant messages
            if message["role"] == "assistant" and n >= 1:
                feedback_key = f"feedback_{int(n/2)}"

                if feedback_key not in st.session_state:
                    st.session_state[feedback_key] = None
                streamlit_feedback(
                    feedback_type="faces",
                    optional_text_label="[Optional] provide extra information",
                    on_submit=_submit_feedback,
                    key=feedback_key,
                )

    # Chat input handling
    prompt = st.chat_input(
        f"Hi {st.session_state.get('firstname', 'there')}, "
        "what analytics insights can I help you explore today?"
    )

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=str(USER_AVATAR)):
            st.markdown(prompt)

        # Generate response
        with st.spinner("Generating response..."):
            with st.chat_message("assistant", avatar=str(BOT_AVATAR)):
                response = invoke_chain(prompt, st.session_state.messages, st.session_state.get('username', 'anonymous'))
                message, query = handle_response(response)

                
                # Store current message
                st.session_state.current_message = {
                    'message': message,
                    'query': query,
                    'prompt': prompt
                }
                
                # Add message to session state immediately
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message
                })
                
                # Log conversation without feedback
                log_conversation_details(
                    user_id=st.session_state.get('username', 'anonymous'),
                    question=prompt,
                    sql_query=query,
                    answer=message,
                    feedback=None,
                    feedback_comment=None
                )

                st.rerun()


    # Show welcome message for new sessions
    if not st.session_state.messages:
        render_welcome_message()

if __name__ == "__main__":
    show_analytics()