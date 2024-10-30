import streamlit as st
from utils.db import DatabaseManager
from datetime import datetime

def admin_dashboard():
    if not st.session_state.get("authenticated"):
        st.warning("Please login first")
        st.stop()
    
    # Check if user is admin
    db = DatabaseManager()
    if not db.is_admin(st.session_state.get("username")):
        st.error("Unauthorized access")
        st.stop()
    
    st.title("Admin Dashboard")
    
    # Create tabs for different admin functions
    tab1, tab2 = st.tabs(["Pending Approvals", "All Users"])
    
    with tab1:
        st.header("Pending User Approvals")
        pending_users = db.get_pending_users()
        
        # Define available roles and departments
        roles = ["User", "Manager", "Analyst", "Developer"]  # Customize these
        departments = ["IT", "Finance", "HR", "Operations", "Marketing"]  # Customize these
        
        if not pending_users:
            st.info("No pending approvals")
        else:
            for user in pending_users:
                with st.container():
                    col1, col2, col3 = st.columns([2,2,1])
                    with col1:
                        st.write(f"**Name:** {user.first_name} {user.last_name}")
                        st.write(f"**Email:** {user.email}")
                    with col2:
                        # Add role and department selection
                        selected_role = st.selectbox(
                            "Role", 
                            options=roles,
                            key=f"role_{user.id}"
                        )
                        selected_dept = st.selectbox(
                            "Department", 
                            options=departments,
                            key=f"dept_{user.id}"
                        )
                    with col3:
                        if st.button("Approve", key=f"approve_{user.id}"):
                            if db.approve_user(user.id, selected_role, selected_dept):
                                st.success("User approved!")
                                st.rerun()
                    st.divider()
    
    with tab2:
        st.header("All Users")
        users = db.get_all_users()
        
        # Create a DataFrame for better display
        user_data = [{
            'Name': f"{user.first_name} {user.last_name}",
            'Email': user.email,
            'Department': user.department or 'N/A',
            'Role': user.role or 'N/A',
            'Status': user.status,
            'Created': user.created_at.strftime("%Y-%m-%d"),
            'Last Sign-in': user.last_signin.strftime("%Y-%m-%d %H:%M") if user.last_signin else 'Never'
        } for user in users]
        
        st.dataframe(user_data, use_container_width=True)

if __name__ == "__main__":
    admin_dashboard() 