import streamlit as st
from utils.db import DatabaseManager
from datetime import datetime
import time
import pandas as pd

def log_admin_action(admin_user, action, affected_user, details):
    db = DatabaseManager()
    db.log_admin_action({
        'admin_user': admin_user,
        'action': action,
        'affected_user': affected_user,
        'details': details,
        'timestamp': datetime.now()
    })

def convert_to_csv(users):
    df = pd.DataFrame([{
        'First Name': user.first_name,
        'Last Name': user.last_name,
        'Email': user.email,
        'Department': user.department,
        'Role': user.role,
        'Status': user.status,
        'Created': user.created_at,
        'Last Sign-in': user.last_signin
    } for user in users])
    return df.to_csv(index=False)

@st.cache_data(ttl=1800)  # Cache data for 5 minutes
def get_users_data():
    db = DatabaseManager()
    return db.get_all_users()

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
    tab1, tab2, tab3 = st.tabs(["Pending Approvals", "All Users", "Analytics"])
    
    with tab1:
        st.header("Pending User Approvals")
        pending_users = db.get_pending_users()
        
        # Define available roles and departments
        roles = ["Admin","User", "Manager", "Analyst", "Developer"]
        departments = ["IT", "Finance", "HR", "Operations", "Marketing","DataManagement","Retention"]
        
        if not pending_users:
            st.info("No pending approvals")
        else:
            for user in pending_users:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2,2,0.5,0.5])
                    with col1:
                        st.write(f"**Name:** {user.first_name} {user.last_name}")
                        st.write(f"**Email:** {user.email}")
                    with col2:
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
                        if st.button("✅ Approve", key=f"approve_{user.id}"):
                            if db.approve_user(user.id, selected_role, selected_dept):
                                st.success("User approved!")
                                time.sleep(3)
                                st.rerun()
                    with col4:
                        if st.button("❌ Deny", key=f"deny_{user.id}", type="secondary"):
                            if db.delete_user(user.id):
                                st.error("User denied and removed")
                                time.sleep(3)
                                st.rerun()
                    st.divider()
    
    with tab2:
        st.header("All Users")
        
        # Add search and filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("🔍 Search users", placeholder="Name or email...")
        with col2:
            filter_dept = st.multiselect("Filter by Department", departments)
        with col3:
            filter_role = st.multiselect("Filter by Role", roles)
            
        # Filter users based on search and filters
        filtered_users = [
            user for user in get_users_data()
            if (not search_term or 
                search_term.lower() in user.email.lower() or 
                search_term.lower() in f"{user.first_name} {user.last_name}".lower())
            and (not filter_dept or user.department in filter_dept)
            and (not filter_role or user.role in filter_role)
        ]
        
        # Create a DataFrame for better display
        user_data = [{
            'Select': False if user.email == st.session_state.get('username') else False,
            'Name': f"{user.first_name} {user.last_name}",
            'Email': user.email,
            'Department': user.department or 'N/A',
            'Role': user.role or 'N/A',
            'Status': user.status,
            'Created': user.created_at.strftime("%Y-%m-%d"),
            'Last Sign-in': user.last_signin.strftime("%Y-%m-%d %H:%M") if user.last_signin else 'Never',
        } for user in filtered_users]
        
        # Display interactive dataframe
        edited_df = st.data_editor(
            user_data,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select to edit user",
                    width="small",
                    default=False
                ),
                "Department": st.column_config.SelectboxColumn(
                    "Department",
                    options=departments,
                    width="medium"
                ),
                "Role": st.column_config.SelectboxColumn(
                    "Role",
                    options=roles,
                    width="small"
                ),
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["pending", "approved"],
                    width="small"
                )
            },
            disabled=["Name", "Email", "Created", "Last Sign-in"] if not any(row["Select"] for row in user_data) else [],
            hide_index=True,
            use_container_width=True,
            key="user_editor",
            on_change=None
        )
        
        # Handle update and delete actions
        for user in filtered_users:
            user_entry = next((item for item in edited_df if item['Email'] == user.email), None)
            if user_entry and user_entry['Select']:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Update User", key=f"update_{user.email}"):
                        old_data = {
                            'department': user.department,
                            'role': user.role,
                            'status': user.status
                        }
                        if db.update_user(
                            user.id, 
                            department=user_entry['Department'],
                            role=user_entry['Role'],
                            status=user_entry['Status']
                        ):
                            log_admin_action(
                                st.session_state.get('username'),
                                'update',
                                user.email,
                                f"Changed from {old_data} to {user_entry}"
                            )
                            st.success(f"User {user.first_name} {user.last_name} updated")
                            # Clear the cache to refresh the data
                            get_users_data.clear()
                with col2:
                    if st.button("🗑️ Delete User", key=f"delete_{user.email}", type="secondary"):
                        if db.delete_user(user.id):
                            st.error(f"User {user.first_name} {user.last_name} deleted")
                            # Clear the cache to refresh the data
                            get_users_data.clear()
        
        # Add export buttons
        col1, col2 = st.columns([4,1])
        with col2:
            if st.button("📥 Export to CSV"):
                csv = convert_to_csv(filtered_users)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="users_export.csv",
                    mime="text/csv"
                )
        
        with st.expander("📤 Bulk Import Users"):
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                if st.button("Import Users"):
                    success_count = 0
                    for _, row in df.iterrows():
                        if db.create_user(row):
                            success_count += 1
                    st.success(f"Successfully imported {success_count} users!")
    
    with tab3:
        st.header("User Analytics")
        
        # Get all users first
        users = db.get_all_users()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", len(users))
        with col2:
            st.metric("Pending Approvals", len(pending_users))
        with col3:
            active_users = sum(1 for user in users if user.last_signin)
            st.metric("Active Users", active_users)
        with col4:
            departments_count = {}
            for user in users:
                departments_count[user.department] = departments_count.get(user.department, 0) + 1
            
            # Create a pie chart of department distribution
            st.write("Department Distribution")
            st.bar_chart(departments_count)

if __name__ == "__main__":
    admin_dashboard()  