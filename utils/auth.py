import streamlit as st
import time
from .db import DatabaseManager

class AuthManager:
    def __init__(self):
        self.db = DatabaseManager()

    def render_login_page(self):
        st.markdown("""
            <style>
                .title {
                    text-align: center;
                    padding: 10px;
                }
                .stTabs {
                    background-color: rgba(255, 255, 255, 0.1);
                    padding: 20px;
                    border-radius: 10px;
                }


            </style>
        """, unsafe_allow_html=True)
        

        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            # Add width parameter to automatically scale the image
            st.image("img/bklogo3.png")  # Adjust the width value as needed

        with col2:
            st.markdown("<h1 class='title'>DataManagement AI</h1>", unsafe_allow_html=True)
        
        tab_login, tab_signup = st.tabs(["Login", "Signup"])
        
        with tab_login:
            self._render_login_form()
            
        with tab_signup:
            self._render_signup_form()
    
    def _render_login_form(self):
        with st.form("login_form"):
            st.write("Please log in with your credentials:")
            login_username = st.text_input("Email", placeholder="Enter your email")
            login_password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("Login")
            if submitted:
                self._handle_login(login_username, login_password)
    
    def _render_signup_form(self):
        with st.form("signup_form"):
            st.write("Create a new account:")
            signup_username = st.text_input("Email", placeholder="Enter your BK email (example@bk.rw)")
            signup_password = st.text_input("Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            first_name = st.text_input("First Name", placeholder="Enter your first name")
            last_name = st.text_input("Last Name", placeholder="Enter your last name")
            
            submitted = st.form_submit_button("Sign up")
            if submitted:
                if not signup_username.endswith("@bk.rw"):
                    st.warning("Please use your BK email address (@bk.rw)")
                else:
                    self._handle_signup(signup_username, signup_password, confirm_password, first_name, last_name)
    
    def _handle_login(self, username: str, password: str):
        if username and password:
            user_data = self.db.verify_user(username, password)
            if user_data:
                first_name, last_name, department,role, status,*_ = user_data
                if status == 'pending':
                    st.error("Your account is pending approval. Please wait for admin confirmation.")
                    return
                
                # Update last signin time
                self.db.update_last_signin(username)
                
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["firstname"] = first_name
                st.session_state["lastname"] = last_name
                st.session_state["role"] = role
                st.session_state["department"] = department
                st.session_state["page"] = "📊 Analytics"
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid email or password")
        else:
            st.warning("Please enter both email and password.")
    
    def _handle_signup(self, username: str, password: str, confirm_password: str, 
                      first_name: str, last_name: str):
        if all([username, password, confirm_password, first_name, last_name]):
            if password == confirm_password:
                if self.db.create_user(username, password, first_name, last_name):
                    st.success("Account created successfully! Please wait for admin approval before logging in.")
                else:
                    st.error("Email already exists")
            else:
                st.warning("Passwords do not match.")
        else:
            st.warning("Please fill in all fields.")
    
    @staticmethod
    def logout():
        for key in ["authenticated", "username", "page", "firstname", "lastname"]:
            st.session_state[key] = None
        st.success("Successfully logged out!")
        st.rerun() 