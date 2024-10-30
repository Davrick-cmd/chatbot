# app.py
import streamlit as st
from utils.config import AppConfig

# Page configuration
AppConfig.setup_page()

from supabase import create_client, Client
import base64
from predictions import show_predictions
from analytics import show_analytics
from blog_home import blog_home
from utils.auth import AuthManager




# Initialize Supabase Connection
@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

# Initialize session state
def init_session_state():
    defaults = {
        # Auth states
        "authenticated": False,
        "username": "",
        "page": "Login",
        "firstname": None,
        
        # Chatbot states
        "messages": [],
        "openai_model": "gpt-4-mini",
        "Link": '',
        
        # Prediction states
        "forecast_results": None,
        "model_option": "Prophet",
        "periods": 30,
        "uploaded_file_path": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

class Navigation:
    @staticmethod
    def render_sidebar():
        with st.sidebar:
            st.title(f"Welcome, {st.session_state.get('firstname', 'User')}!")
            
            # Navigation menu with icons
            st.session_state["page"] = st.radio(
                "Navigation",
                options=[
                    "🏠 Home",
                    "📊 Analytics",
                    "🔮 Predictions"
                ],
                index=1,
                format_func=lambda x: x.split()[1]  # Remove emoji from label
            )
            
            # Profile section
            with st.expander("👤 Profile"):
                st.write(f"Email: {st.session_state['username']}")
                # Add more profile info here
            
            st.divider()
            
            # Settings and Help
            with st.expander("⚙️ Settings"):
                st.toggle("Dark Mode")
                st.selectbox("Language", ["English", "Spanish", "French"])
            
            # Logout button
            if st.button("🚪 Logout", type="primary"):
                AuthManager.logout(supabase)

def main_page():
    if st.session_state.get("authenticated", False):
        Navigation.render_sidebar()
        
        # Route to appropriate page
        pages = {
            "🏠 Home": blog_home,
            "📊 Analytics": show_analytics,
            "🔮 Predictions": show_predictions
        }
        
        current_page = st.session_state["page"]
        if current_page in pages:
            pages[current_page]()
            
    else:
        AuthManager.render_login_page(supabase)

if __name__ == "__main__":
    init_session_state()
    main_page()
