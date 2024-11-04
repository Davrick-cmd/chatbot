import streamlit as st
from utils.db import DatabaseManager
import re
from datetime import datetime

def display_article(article, db):
    post, author_first_name, author_last_name, author_email = article
    
    st.subheader(post.title)
    st.write(post.summary)
    st.write(f"By: {author_first_name} {author_last_name}")
    st.write(f"Published: {post.created_at.strftime('%Y-%m-%d %H:%M')}")
    
    # Only show link button if link exists
    if post.link:
        st.link_button("Read Article 📄", post.link, type="primary")
    
    # Create a container for likes and comments
    col1, col2, col3 = st.columns([1, 2, 1], gap="small")
    
    with col1:
        # Existing likes functionality
        user_liked = False
        if st.session_state.get("authenticated"):
            user_liked = db.has_user_liked_post(post.id, st.session_state["username"])
        
        like_emoji = "❤️" if user_liked else "🤍"
        if st.button(f"{like_emoji} {post.likes_count} likes", key=f"like_{post.id}"):
            if not st.session_state.get("authenticated"):
                st.warning("Please log in to like posts.")
            else:
                success, new_count = db.toggle_blog_like(post.id, st.session_state["username"])
                if success:
                    st.rerun()
    
    with col2:
        comments = db.get_blog_comments(post.id)
        with st.expander(f"💬 {len(comments)} comments"):
            st.write("#### Comments")
            
            # Display existing comments
            for comment, commenter_first, commenter_last in comments:
                st.write(f"**{commenter_first} {commenter_last}:** {comment.content}")
                st.write(f"*{comment.created_at.strftime('%Y-%m-%d %H:%M')}*")
                st.write("---")
            
            st.write("Join the discussion!")
            comment_key = f"comment_{post.id}"
            new_comment = st.text_area("Write your comment:", height=100, key=comment_key)
            if st.button("Post Comment", key=f"button_{post.id}"):
                if not st.session_state.get("authenticated"):
                    st.warning("Please log in to post a comment.")
                elif not new_comment.strip():
                    st.warning("Please write a comment before posting.")
                else:
                    if db.add_blog_comment(post.id, st.session_state["username"], new_comment):
                        st.success("Comment posted successfully!")
                        # Clear the input by updating session state
                        st.session_state[comment_key] = ""
                        st.rerun()
                    else:
                        st.error("Failed to post comment. Please try again.")
    
    # Add delete button for article owner
    with col3:
        if st.session_state.get("authenticated") and st.session_state["username"] == author_email:
            if st.button("🗑️ Delete", key=f"delete_{post.id}"):
                if db.delete_blog_post(post.id):
                    st.success("Article deleted successfully!")
                    st.rerun()
                else:
                    st.error("Failed to delete article. Please try again.")
    
    st.write("---")

def blog_home():
    db = DatabaseManager()
    
    st.title("Welcome to the DataManagement AI Blog!")
    
    # Articles section
    with st.container():
        st.write("### Latest Articles")
        articles = db.get_blog_posts()
        for article in articles:
            display_article(article, db)
    
    # Content submission section
    with st.container():
        st.write("### Share an Article")
        if not st.session_state.get("authenticated"):
            st.warning("Please log in to share articles.")
        else:
            title = st.text_input("Article Title:")
            summary = st.text_area("Article Summary (brief overview):", height=100)
            link = st.text_input("Article Link (URL) - Optional:")
            
            if st.button("Share"):
                if not all([title.strip(), summary.strip()]):
                    st.warning("Please fill in the required fields (title and summary).")
                elif link and not is_valid_url(link):
                    st.warning("Please enter a valid URL or leave it empty.")
                else:
                    post_id = db.create_blog_post(
                        title=title,
                        summary=summary,
                        link=link if link.strip() else None,
                        author_email=st.session_state["username"]
                    )
                    if post_id:
                        st.success("Article shared successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to share article. Please try again.")

# Add this helper function to validate URLs
def is_valid_url(url):
    try:
        # Basic URL validation using regex
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None
    except:
        return False

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def handle_subscription(email):
    if not email:
        st.warning("Please enter an email address.")
        return
    if not validate_email(email):
        st.warning("Please enter a valid email address.")
        return
    # Add subscription logic here
    st.success(f"Thank you for subscribing, {email}!")

def initialize_session_state():
    if 'published_articles' not in st.session_state:
        st.session_state.published_articles = []
    if 'subscribed_emails' not in st.session_state:
        st.session_state.subscribed_emails = set()

def handle_content_submission(content):
    if not content.strip():
        st.warning("Please write something before publishing.")
        return
    
    # Add content validation
    if len(content) < 50:
        st.warning("Content must be at least 50 characters long.")
        return
        
    # Here you would typically save to a database
    st.success("Your content has been published successfully!")
    with st.expander("View Your Published Article"):
        st.write(content)




