import streamlit as st

class AuthManager:
    @staticmethod
    def render_login_page(supabase):
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
            AuthManager._render_login_form(supabase)
            
        with tab_signup:
            AuthManager._render_signup_form(supabase)
    
    @staticmethod
    def _render_login_form(supabase):
        with st.form("login_form"):
            st.write("Please log in with your credentials:")
            login_username = st.text_input("Email", placeholder="Enter your email")
            login_password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("Login")
            if submitted:
                AuthManager._handle_login(supabase, login_username, login_password)
    
    @staticmethod
    def _render_signup_form(supabase):
        with st.form("signup_form"):
            st.write("Create a new account:")
            signup_username = st.text_input("Email", placeholder="Enter your email")
            signup_password = st.text_input("Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            first_name = st.text_input("First Name", placeholder="Enter your first name")
            last_name = st.text_input("Last Name", placeholder="Enter your last name")
            
            submitted = st.form_submit_button("Sign up")
            if submitted:
                AuthManager._handle_signup(supabase, signup_username, signup_password, confirm_password, first_name, last_name)
    
    @staticmethod
    def _handle_login(supabase, username, password):
        if username and password:
            try:
                response = supabase.auth.sign_in_with_password({
                    "email": username,
                    "password": password
                })
                AuthManager._update_session_after_login(response, supabase)
                st.success("Login successful! Redirecting...")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {str(e)}")
        else:
            st.warning("Please enter both email and password.")
    
    @staticmethod
    def _handle_signup(supabase, username, password, confirm_password, first_name, last_name):
        if all([username, password, confirm_password, first_name, last_name]):
            if password == confirm_password:
                try:
                    response = supabase.auth.sign_up({
                        "email": username,
                        "password": password,
                        "options": {
                            "data": {
                                "first_name": first_name,
                                "last_name": last_name
                            }
                        }
                    })
                    st.success("Signup successful! Please check your email to verify your account.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Signup failed: {str(e)}")
            else:
                st.warning("Passwords do not match.")
        else:
            st.warning("Please fill in all fields.")
    
    @staticmethod
    def _update_session_after_login(response, supabase):
        st.session_state["authenticated"] = True
        st.session_state["username"] = response.user.email
        user_id = response.user.id
        profile_response = supabase.from_("profiles").select("first_name, last_name").eq("id", user_id).execute()
        st.session_state["firstname"] = profile_response.data[0]["first_name"]
        st.session_state["lastname"] = profile_response.data[0]["last_name"]
        st.session_state["page"] = "📊 Analytics"
    
    @staticmethod
    def logout(supabase):
        supabase.auth.sign_out()
        for key in ["authenticated", "username", "page", "firstname"]:
            st.session_state[key] = None
        st.success("Successfully logged out!")
        st.rerun() 