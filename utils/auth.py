import streamlit as st
import time
from .db import DatabaseManager
from ldap3 import Server, Connection, ALL, NTLM, SUBTREE, SIMPLE
import logging
from .config import Config

class AuthManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.logger = logging.getLogger(__name__)

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
        
        # tab_login, tab_signup = st.tabs(["Login", "Signup"])
                # Only show login tab
        tab_login = st.tabs(["Login"])[0]
        
        with tab_login:
            self._render_login_form()
            
        # with tab_signup:
        #     self._render_signup_form()
    
    def _render_login_form(self):
        with st.form("login_form"):
            st.write("Please log in with your credentials:")
            login_username = st.text_input("Email", placeholder="Enter your email")
            login_password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("Login")
            if submitted:
                self._handle_login(login_username, login_password)
    
    # def _render_signup_form(self):
    #     with st.form("signup_form"):
    #         st.write("Create a new account:")
    #         signup_username = st.text_input("Email", placeholder="Enter your BK email (example@bk.rw)")
    #         signup_password = st.text_input("Password", type="password", placeholder="Create a password")
    #         confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
    #         first_name = st.text_input("First Name", placeholder="Enter your first name")
    #         last_name = st.text_input("Last Name", placeholder="Enter your last name")
            
    #         submitted = st.form_submit_button("Sign up")
    #         if submitted:
    #             if not signup_username.endswith("@bk.rw"):
    #                 st.warning("Please use your BK email address (@bk.rw)")
    #             else:
    #                 self._handle_signup(signup_username, signup_password, confirm_password, first_name, last_name)
    
    # def _handle_login(self, username: str, password: str):
    #     if username and password:
    #         # Remove @domain.com if present
    #         username = username.split('@')[0] if '@' in username else username
    #         # First try LDAP authentication
    #         ldap_success, ldap_user_data = self._authenticate_ldap(username, password)
    #         print(ldap_success, ldap_user_data)
            
    #         if ldap_success:
    #             # Check if user exists in local DB
    #             user_data = self.db.verify_user(username, password)
                
    #             if not user_data:
    #                 # Create user in local DB if they don't exist
    #                 self.db.create_user(
    #                     email=username,
    #                     password=password,
    #                     first_name=ldap_user_data.get('first_name', ''),
    #                     last_name=ldap_user_data.get('last_name', ''),
    #                     department=ldap_user_data.get('department', ''),
    #                     # status='active'  # Auto-approve LDAP users
    #                 )
    #                 user_data = self.db.verify_user(username, password)
                
    #             if user_data:
    #                 first_name, last_name, department, role, status, *_ = user_data
    #                 first_name, last_name, department, role, status, consent_signed, *_ = user_data
    #                 if status == 'pending':
    #                     st.success("Your account is pending approval. Please wait for admin confirmation.")
    #                     return
                    
    #                 # Update last signin time
    #                 self.db.update_last_signin(username)
                    
    #                 st.session_state["authenticated"] = True
    #                 st.session_state["username"] = username
    #                 st.session_state["firstname"] = first_name
    #                 st.session_state["lastname"] = last_name
    #                 st.session_state["role"] = role
    #                 st.session_state["department"] = department
    #                 st.session_state["page"] = "📊 Analytics"
    #                 st.success("Login successful! Redirecting...")
    #                 st.rerun()
    #         else:
    #             st.error("Invalid email or password")
    #     else:
    #         st.warning("Please enter both email and password.")

    def _handle_login(self, username: str, password: str):
        if username and password:
            # Remove @domain.com if present
            username = username.split('@')[0] if '@' in username else username
            # First try LDAP authentication
            ldap_success, ldap_user_data = self._authenticate_ldap(username, password)
            print(ldap_success, ldap_user_data)
            
            if ldap_success:  # to be reversed
                # Check if user exists in local DB
                user_data = self.db.verify_user(username, password)
                
                if not user_data:
                    # Create user in local DB if they don't exist
                    self.db.create_user(
                        email=username,
                        password=password,
                        first_name=ldap_user_data.get('first_name', ''),
                        last_name=ldap_user_data.get('last_name', ''),
                        department=ldap_user_data.get('department', ''),
                    )
                    user_data = self.db.verify_user(username, password)
                
                if user_data:
                    first_name, last_name, department, role, status, consent_signed, *_ = user_data
                    
                    if status == 'pending':
                        st.success("Your account is pending approval. Please wait for admin confirmation.")
                        return
                    
                    if not consent_signed:
                        self._render_consent_form(username)
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

    def _render_consent_form(self, username: str):
        """
        Display a consent form for the user and update the database when signed.
        """
        st.write("Consent Form")
        
        # Detailed Consent Content
        st.markdown("""
        ### **Consent for AI Chatbot Usage**
        
        By engaging with this chatbot, you acknowledge and agree to the following terms:
        
        1. **Purpose of the Chatbot**  
        This chatbot is designed to assist with your queries, provide recommendations, and offer support. Its responses are generated based on the information you provide and pre-trained models.  
        
        2. **Data Collection**  
        - Information you provide during your interaction with the chatbot may be collected and used to enhance the quality of service and for troubleshooting purposes.  
        - Any personal or sensitive information you share will be handled with strict confidentiality and in compliance with applicable data protection regulations.  
        
        3. **Use of Data**  
        - Your data may be used to improve the chatbot’s functionality, personalize your experience, and refine its performance.  
        - Data shared will not be sold or shared with third parties without your explicit consent, except as required by law.  
        
        4. **Limitations of AI Responses**  
        - The chatbot’s responses are generated by artificial intelligence and should not be considered as professional advice.  
        - For critical decisions or detailed guidance, consult a qualified expert or professional.  
        
        5. **User Responsibility**  
        - Avoid sharing confidential or sensitive personal information during your interaction.  
        - Use the chatbot responsibly and refrain from any misuse or abusive behavior.  
        
        6. **Consent Confirmation**  
        By continuing to use this chatbot, you confirm that:  
        - You have read and understood the terms and conditions outlined above.  
        - You agree to the collection and use of your data as described.  
        
        If you do not agree to these terms, you are advised to discontinue using this chatbot.
        """)

        # Checkbox for Consent
        consent_given = st.checkbox("I agree to the terms and conditions")

        # Process Consent
        if consent_given:
            self.db.update_consent(username)
            st.success("Thank you for signing the consent form! Redirecting...")
            time.sleep(1)
            st.rerun()  # You can adjust or remove this as per your app flow
        else:
            st.warning("You must agree to the terms and conditions to proceed.")

        



    # Add this method to `DatabaseManager` to update the consent_signed field
    def update_consent(self, username: str):
        """
        Update the consent_signed field for the user.
        """
        query = "UPDATE [users__] SET consent_signed = 1 WHERE email = ?"
        self.execute_query(query, [username])

    
    # def _handle_signup(self, username: str, password: str, confirm_password: str, 
    #                   first_name: str, last_name: str):
    #     if all([username, password, confirm_password, first_name, last_name]):
    #         if password == confirm_password:
    #             if self.db.create_user(username, password, first_name, last_name):
    #                 st.success("Account created successfully! Please wait for admin approval before logging in.")
    #             else:
    #                 st.error("Email already exists")
    #         else:
    #             st.warning("Passwords do not match.")
    #     else:
    #         st.warning("Please fill in all fields.")
    
    @staticmethod
    def logout():
        for key in ["authenticated", "username", "page", "firstname", "lastname"]:
            st.session_state[key] = None
        st.success("Successfully logged out!")
        st.rerun() 

    def _authenticate_ldap(self, username: str, password: str) -> tuple[bool, dict]:
        """
        Authenticate user against AD/LDAP server
        Returns (success_bool, user_data_dict)
        """
        try:
            self.logger.debug(f"Starting LDAP authentication for {username}")
            server = Server(Config.AD_SERVER, get_info=ALL)
            
            # Extract the username part before @bk.rw
            ldap_username = username.split('@')[0] if '@' in username else username
            
            # Construct the userPrincipalName
            user_principal = f"{ldap_username}@{Config.AD_DOMAIN}"
            
            self.logger.debug(f"Attempting connection with user: {user_principal}")
            
            conn = Connection(
                server,
                user=user_principal,
                password=password,
                authentication=SIMPLE,
                auto_bind=True,
                read_only=True,
                receive_timeout=Config.AD_TIMEOUT
            )
            
            self.logger.debug("Connection established successfully")
            
            # Search for user details
            search_filter = f"(&(objectClass=user)(sAMAccountName={username}))"
            conn.search(
                Config.AD_SEARCH_BASE,
                search_filter,
                attributes=['givenName', 'sn', 'department', 'mail']
            )

            if len(conn.entries) > 0:
                user_data = {
                    'first_name': conn.entries[0].givenName.value,
                    'last_name': conn.entries[0].sn.value,
                    'department': conn.entries[0].department.value,
                    'email': conn.entries[0].mail.value
                }
                return True, user_data
            
            return True, {}
        except Exception as e:
            self.logger.error(f"LDAP authentication failed for user {username}: {e}")
            return False, {}