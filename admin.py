import streamlit as st
from utils.db import DatabaseManager
from datetime import datetime
import time
import pandas as pd
import json
from notifications import send_notification

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

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_users_data():
    db = DatabaseManager()
    return db.get_all_users()

@st.cache_data(ttl=300)
def get_pending_users():
    db = DatabaseManager()
    return db.get_pending_users()

@st.cache_data(ttl=300)
def get_user_analytics():
    users = get_users_data()
    pending = get_pending_users()
    active_users = sum(1 for user in users if user.last_signin)
    departments_count = {}
    for user in users:
        departments_count[user.department] = departments_count.get(user.department, 0) + 1
    
    return {
        "total_users": len(users),
        "pending_users": len(pending),
        "active_users": active_users,
        "departments_count": departments_count
    }

def clear_user_caches():
    get_users_data.clear()
    get_pending_users.clear()
    get_user_analytics.clear()

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
        roles = ["User", "Manager", "Analyst", "Developer","Admin"]
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
                        # selected_dept = st.selectbox(
                        #     "Department", 
                        #     options=departments,
                        #     key=f"dept_{user.id}"
                        # )
                    with col3:
                        if st.button("✅ Approve", key=f"approve_{user.id}"):
                            if db.approve_user(user.id, selected_role):
                                send_notification(user.first_name, user.email, "approved.")
                                st.success("User approved!")
                                clear_user_caches()
                                time.sleep(2)
                                st.rerun()
                    with col4:
                        if st.button("❌ Deny", key=f"deny_{user.id}", type="secondary"):
                            if db.delete_user(user.id):
                                # Send notification email
                                send_notification(user.first_name, user.email, "declined.")
                                st.error("User denied and removed")
                                clear_user_caches()
                                time.sleep(2)
                                st.rerun()
                    st.divider()
    
    with tab2:
        st.header("All Users")
        
        # Initialize session state variables if they don't exist
        if "search_term" not in st.session_state:
            st.session_state.search_term = ""
        if "filter_dept" not in st.session_state:
            st.session_state.filter_dept = []
        if "filter_role" not in st.session_state:
            st.session_state.filter_role = []
        
        # Use session state in filters
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("🔍 Search users", 
                                       value=st.session_state.search_term,
                                       key="search_input")
            st.session_state.search_term = search_term
        with col2:
            filter_dept = st.multiselect("Filter by Department", 
                                        departments,
                                        default=st.session_state.filter_dept,
                                        key="dept_filter")
            st.session_state.filter_dept = filter_dept
        with col3:
            filter_role = st.multiselect("Filter by Role", roles, 
                                        default=st.session_state.filter_role,
                                        key="role_filter")
            st.session_state.filter_role = filter_role
            
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
        
        # Add key to data editor and use on_change callback
        if "edited_data" not in st.session_state:
            st.session_state.edited_data = user_data

        edited_df = st.data_editor(
            st.session_state.edited_data,
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
            disabled=["Name", "Email", "Created", "Last Sign-in"] if not any(row["Select"] for row in st.session_state.edited_data) else [],
            hide_index=True,
            use_container_width=True,
            key="user_editor",
            on_change=None
        )

        # Only update session state if data actually changed
        if edited_df != st.session_state.edited_data:
            st.session_state.edited_data = edited_df
        
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
                            clear_user_caches()
                            log_admin_action(
                                st.session_state.get('username'),
                                'update',
                                user.email,
                                f"Changed from {old_data} to {user_entry}"
                            )
                            st.success(f"User {user.first_name} {user.last_name} updated")
                            time.sleep(1)
                            st.rerun()
                with col2:
                    if st.button("🗑️ Delete User", key=f"delete_{user.email}", type="secondary"):
                        if db.delete_user(user.id):
                            clear_user_caches()
                            st.success(f"User {user.first_name} {user.last_name} deleted")
                            time.sleep(1)
                            st.rerun()
        
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
        
        # Create subtabs for different types of analytics
        analytics_tab1, analytics_tab2 = st.tabs(["User Statistics", "Conversation Analytics"])
        
        with analytics_tab1:
            # Your existing analytics code
            analytics = get_user_analytics()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Users", analytics["total_users"])
            with col2:
                st.metric("Pending Approvals", analytics["pending_users"])
            with col3:
                st.metric("Active Users", analytics["active_users"])
            with col4:
                st.write("Department Distribution")
                st.bar_chart(analytics["departments_count"])
        
        with analytics_tab2:
            st.subheader("Conversation Analytics")
            
            # Add conversation data loading function
            @st.cache_data(ttl=300)
            def get_conversation_data():
                conversations = []
                try:
                    with open('logs/conversation.log', 'r') as file:
                        for line in file:
                            try:
                                json_str = line.split(' - ', 1)[1]
                                conversation = json.loads(json_str)
                                conversations.append(conversation)
                            except:
                                continue
                except FileNotFoundError:
                    st.error("Conversation log file not found")
                    return []
                
                return conversations
            
            conversations = get_conversation_data()
            
            if conversations:
                # Key metrics
                col1, col2, col3 = st.columns(3)
                
                total_conversations = len(conversations)
                unique_users = len(set(conv['user_id'] for conv in conversations))
                feedback_stats = {
                    'Very Happy': sum(1 for conv in conversations if conv.get('feedback') == 'Very Happy'),
                    'Happy': sum(1 for conv in conversations if conv.get('feedback') == 'Happy'),
                    'no_feedback': sum(1 for conv in conversations if conv.get('feedback') is None),
                    'Neutral': sum(1 for conv in conversations if conv.get('feedback') == 'Neutral'),
                    'Unhappy': sum(1 for conv in conversations if conv.get('feedback') == 'Unhappy'),
                    'Very Unhappy': sum(1 for conv in conversations if conv.get('feedback') == 'Very Unhappy')

                }
                
                with col1:
                    st.metric("Total Conversations", total_conversations)
                with col2:
                    st.metric("Unique Users", unique_users)
                with col3:
                    feedback_rate = ((feedback_stats['Very Happy'] + feedback_stats['Happy'] + feedback_stats['Neutral'] + feedback_stats['Unhappy'] + feedback_stats['Very Unhappy']) / total_conversations * 100)
                    st.metric("Feedback Rate", f"{feedback_rate:.1f}%")
                
                # Feedback Distribution
                st.subheader("Feedback Distribution")
                feedback_data = pd.DataFrame({
                    'Feedback': ['Very Happy', 'Happy', 'Neutral','Unhappy', 'Very Unhappy','No Feedback'],
                    'Count': [
                        feedback_stats['Very Happy'],
                        feedback_stats['Happy'],
                        feedback_stats['Neutral'],
                        feedback_stats['Unhappy'],
                        feedback_stats['Very Unhappy'],
                        feedback_stats['no_feedback']
                    ]
                })
                st.bar_chart(feedback_data.set_index('Feedback'))
                
                # User Activity Analysis
                st.subheader("User Activity")
                
                # Group conversations by user
                user_activity = {}
                for conv in conversations:
                    user_activity[conv['user_id']] = user_activity.get(conv['user_id'], 0) + 1
                
                # Create DataFrame for top users
                top_users_df = pd.DataFrame({
                    'User': list(user_activity.keys()),
                    'Conversations': list(user_activity.values())
                }).sort_values('Conversations', ascending=False).head(10)
                
                st.bar_chart(top_users_df.set_index('User'))
                
                # Daily Activity Chart
                st.subheader("Daily Activity")
                
                # Convert timestamps to datetime
                for conv in conversations:
                    conv['datetime'] = datetime.fromisoformat(conv['timestamp'].replace('Z', '+00:00'))
                
                # Group by date
                daily_activity = {}
                for conv in conversations:
                    date = conv['datetime'].date()
                    daily_activity[date] = daily_activity.get(date, 0) + 1
                
                # Create DataFrame for daily activity
                daily_df = pd.DataFrame({
                    'Date': list(daily_activity.keys()),
                    'Conversations': list(daily_activity.values())
                }).sort_values('Date')
                
                st.line_chart(daily_df.set_index('Date'))
                
                # Recent Conversations Table
                with st.expander("Recent Conversations"):
                    st.dataframe(
                        pd.DataFrame([{
                            'Time': conv['datetime'].strftime('%Y-%m-%d %H:%M'),
                            'User': conv['user_id'],
                            'Question': conv['question'],
                            'Feedback': conv.get('feedback', 'None')
                        } for conv in sorted(conversations, 
                                          key=lambda x: x['datetime'], 
                                          reverse=True)[:20]]),
                        use_container_width=True
                    )
            else:
                st.info("No conversation data available")

if __name__ == "__main__":
    admin_dashboard()
