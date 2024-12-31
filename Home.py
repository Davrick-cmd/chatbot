# app.py
import streamlit as st
from utils.config import AppConfig

# Page configuration
AppConfig.setup_page()

import base64
from predictions import show_predictions
from analytics import show_analytics
from blog_home import blog_home
from utils.auth import AuthManager
from pathlib import Path
from utils.db import DatabaseManager
from admin import admin_dashboard
from streamlit.runtime.scriptrunner import get_script_run_ctx
import streamlit.components.v1 as components


# Constants
ASSETS_DIR = Path("img")
LOGO = ASSETS_DIR / "bklogo1.png"

# Initialize session state
def init_session_state():
    # Get current session ID
    ctx = get_script_run_ctx()
    
    # If this is just a browser refresh and we already have authentication
    if (is_browser_refresh() and 
        '_session_id' in st.session_state and 
        'authenticated' in st.session_state and 
        st.session_state.authenticated):
        # Update session ID but keep other state
        st.session_state['_session_id'] = ctx.session_id
        return

    # If it's a new session, initialize everything
    defaults = {
        # Session tracking
        '_session_id': ctx.session_id,
        
        # Auth states
        "authenticated": False,
        "username": "",
        "page": "Login",
        "firstname": None,
        "lastname": None,
        "role": None,
        "status": None,
        
        # Chatbot states
        "messages": [],
        "openai_model": "gpt-4-mini",
        "Link": '',
        "generate_visuals": False,
        "feedback": None,
        "feedback_comment": None,
        
        # Prediction states
        "forecast_results": None,
        "model_option": "Prophet",
        "periods": 30,
        "uploaded_file_path": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Add this JavaScript to prevent complete page reloads
def inject_refresh_handler():
    components.html(
        """
        <script>
            // Intercept browser refresh
            window.addEventListener('beforeunload', function(e) {
                // Cancel the event
                e.preventDefault();
                // Chrome requires returnValue to be set
                e.returnValue = '';
            });
        </script>
        """,
        height=0
    )

class Navigation:
    @staticmethod
    def render_sidebar():
        with st.sidebar:
            st.image(str(LOGO), use_column_width=True)

            # Profile section
            with st.expander(f"👤 {st.session_state.get('firstname', 'User')} {st.session_state.get('lastname', '')}!"):
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📧 Email**")
                    st.markdown("**🏢 Department**")
                    st.markdown("**👥 Role**")
                with col2:
                    st.markdown(f"`{st.session_state['username']}`")
                    st.markdown(f"`{st.session_state.get('department', 'N/A')}`")
                    st.markdown(f"`{st.session_state.get('role', 'N/A')}`")
                st.markdown("---")

                # Add more profile info here
            
            # Check for admin access and add admin button
            db = DatabaseManager()
            
            # Navigation menu with icons
            options = [
                "🏠 Home",
                "📊 Analytics",
                "🔮 Predictions"
            ]
            
            # Only add Admin option if user is admin
            if db.is_admin(st.session_state.get("username")):
                options.append("👑 Admin")
                
            st.session_state["page"] = st.radio(
                "Navigation",
                options=options,
                index=1,
                label_visibility="collapsed"
            )
            
            # st.divider()
            
            # # Settings and Help
            # with st.expander("⚙️ Settings"):
            #     st.toggle("Dark Mode")
            #     st.selectbox("Language", ["English", "Spanish", "French"])
            
            # Logout button
            # st.markdown("<h5 style='color: gray;'>For support, contact: <a href='mailto:datamanagementai.bk.rw'>datamanagementai.bk.rw</a></h5>", unsafe_allow_html=True)
            if st.button("🚪 Logout", type="primary"):
                AuthManager.logout()

# Add this function to detect if it's a browser refresh
def is_browser_refresh():
    ctx = get_script_run_ctx()
    return ctx.session_id != st.session_state.get('_session_id', None)

def main_page():
    # Add the refresh handler
    inject_refresh_handler()
    
    if st.session_state.get("authenticated", False):
        Navigation.render_sidebar()
        
        # Route to appropriate page
        pages = {
            "🏠 Home": blog_home,
            "📊 Analytics": show_analytics,
            "🔮 Predictions": show_predictions,
            "👑 Admin": admin_dashboard
        }
        
        current_page = st.session_state["page"]
        if current_page in pages:
            try:
                pages[current_page]()
            except Exception as e:
                st.error(f"Error loading page: {str(e)}")
                # Log the error here if needed
    else:
        auth_manager = AuthManager()
        auth_manager.render_login_page()

if __name__ == "__main__":
    init_session_state()
    main_page()
