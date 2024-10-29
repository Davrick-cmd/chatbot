# app.py
import streamlit as st

st.set_page_config(layout="wide", page_title="DataManagement AI", page_icon="img/bkofkgl.png",
                   menu_items={'Get Help': 'mailto:john@example.com',
                               'About': "#### This is DataManagement cool app!"})

from supabase import create_client, Client
from predictions import show_predictions  # import the show_predictions function
from analytics import show_analytics      # import the show_analytics function
import base64

# Page configuration


# Initialize Supabase Connection
@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

# # Initialize session state variables if they don't exist

# login session variables
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False  # Default to not authenticated
if "username" not in st.session_state:
    st.session_state["username"] = ""  # Default to an empty username
if "page" not in st.session_state:
    st.session_state["page"] = "Login"  # Default to Login page
if "firstname" not in st.session_state:
    st.session_state['firstname']=None

# Chatbot Session variables

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize messages list
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"  # Default model
if "Link" not in st.session_state:
    st.session_state["Link"] = ''  # Default Link

# Predicition sessions variables

if "forecast_results" not in st.session_state:
    st.session_state['forecast_results'] = None
if "model_option" not in st.session_state:
    st.session_state['model_option'] = "Prophet"
if "periods" not in st.session_state:
    st.session_state['periods'] = 30
# Initialize session state to track the file path
if "uploaded_file_path" not in st.session_state:
    st.session_state['uploaded_file_path'] = None

# Helper to encode background images
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Logout function
def logout():
    supabase.auth.sign_out()
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.session_state["page"] = "Login"
    st.session_state["firstname"] = None
    st.success("Successfully logged out!")
    st.rerun()

# Authentication Page with custom login and signup forms
def login_page():
    st.markdown("<h1 class='title'>Welcome to DataManagement AI</h1>", unsafe_allow_html=True)
    
    tab_login, tab_signup = st.tabs(["Login", "Signup"])
    with tab_login:
        st.write("Please log in with your credentials:")
        login_username = st.text_input("Email", key="login_username", placeholder="Enter your email")
        login_password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
        
        if st.button("Login"):
            if login_username and login_password:
                try:
                    response = supabase.auth.sign_in_with_password({"email": login_username, "password": login_password})
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = response.user.email
                    user_id = response.user.id
                    profile_response = supabase.from_("profiles").select("first_name, last_name").eq("id", user_id).execute()
                    st.session_state["firstname"] = profile_response.data[0]["first_name"]
                    st.session_state["page"] = "Analytics"
                    st.success(f"Welcome, {st.session_state['username']}! Redirecting to Analytics...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
            else:
                st.warning("Please enter both email and password.")
    
    with tab_signup:
        st.write("Create a new account:")
        signup_username = st.text_input("Email", key="signup_username", placeholder="Enter your email")
        signup_password = st.text_input("Password", type="password", key="signup_password", placeholder="Create a password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password", placeholder="Confirm your password")
        
        if st.button("Sign up"):
            if signup_username and signup_password and confirm_password:
                if signup_password == confirm_password:
                    try:
                        supabase.auth.sign_up({"email": signup_username, "password": signup_password})
                        st.success("Signup successful! Please log in.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Signup failed: {str(e)}")
                else:
                    st.warning("Passwords do not match.")
            else:
                st.warning("Please fill in all fields.")

# Main App Page with Sidebar Navigation
def main_page():
    if st.session_state["authenticated"]:
        # Sidebar with custom-styled page selectors

        st.sidebar.title(f"Welcome, {st.session_state['firstname']}!")
        
        # Sidebar radio buttons styled as page selectors
        st.session_state["page"] = st.sidebar.radio(
            "Navigation",
            ["Home", "Analytics", "Predictions"],index=1
        )

        # Logout button at the bottom of the sidebar
        st.sidebar.write("---")
        if st.sidebar.button("Logout"):
            logout()
        
        # Main content based on selected page
        if st.session_state["page"] == "Home":
            st.title("DataManagement AI Home")
            st.write("Explore analytics and data-driven insights with ease.")
        
        elif st.session_state["page"] == "Analytics":
            show_analytics()  # Calls the function from analytics.py
        
        elif st.session_state["page"] == "Predictions":
            show_predictions()  # Calls the function from predictions.py

    else:
        login_page()  # Redirects to login page if not authenticated

# Main app execution
if __name__ == "__main__":
    main_page()
