# app.py
import streamlit as st
from supabase import create_client, Client


# Page configuration
st.set_page_config(layout="wide", page_title="DataManagement AI", page_icon="img/bkofkgl.png",
                   menu_items={'Get Help': 'mailto:john@example.com',
                               'About': "#### This is DataManagement cool app!"})


# Initialize Supabase Connection
@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()



# Initialize session state variables if they don't exist
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False  # Default to not authenticated
if "username" not in st.session_state:
    st.session_state["username"] = ""  # Default to an empty username
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize messages list
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"  # Default model
if "Link" not in st.session_state:
    st.session_state["Link"] = ''  # Default Link
if "page" not in st.session_state:
    st.session_state["page"] = "Login"  # Default to Login page

if "firstname" not in st.session_state:
    st.session_state['firstname']=None

# Logout function
def logout():
    supabase.auth.sign_out()  # Sign out from Supabase
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["page"] = "Login"
    st.session_state['firstname']=None
    st.success("Successfully logged out!")
    st.rerun()  # Rerun to refresh the app state

# Authentication Page with custom login and signup forms
# @st.cache_resource
def login_page():
    st.title("Welcome to DataManagement AI 📊")
    st.subheader("Your AI-driven data management partner")

    # Tabs for Login and Signup
    tab_login, tab_signup = st.tabs(["Login", "Signup"])

    with tab_login:
        st.write("Please log in with your credentials:")
        login_username = st.text_input("email", key="login_username", placeholder="Enter your email")
        login_password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
        
        if st.button("Login"):
            if login_username and login_password:
                try:
                    # Attempt login with Supabase
                    response = supabase.auth.sign_in_with_password({"email": login_username, "password": login_password})
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = response.user.email
                    user_id = response.user.id
                    profile_response = supabase.from_("profiles").select("first_name, last_name").eq("id", user_id).execute()
                    st.session_state['firstname']  = profile_response.data[0]["first_name"]
                    st.session_state["page"] = "Analytics"  # Set page to Analytics after login
                    st.success(f"Welcome, {st.session_state['username']}! Redirecting to Analytics...")
                    st.switch_page("pages/analytics.py")
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
                        # Register new user in Supabase
                        user = supabase.auth.sign_up({"email": signup_username, "password": signup_password})
                        st.success("Signup successful! Please log in.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Signup failed: {str(e)}")
                else:
                    st.warning("Passwords do not match.")
            else:
                st.warning("Please fill in all fields.")

# Dashboard/Homepage for Authenticated Users
# @st.cache_resource
def dashboard():
    st.title("DataManagement AI")
    st.write("Explore analytics and data-driven insights with ease.")

    # Logout button in the main content area
    if st.button("Logout",use_container_width=True):
        logout()  # Call the logout function


# Main Logic
if st.session_state["authenticated"]:
    dashboard()
else:
    login_page()
