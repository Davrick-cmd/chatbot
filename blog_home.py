import streamlit as st
import re
# Temporary list of articles until database integration
ARTICLES = [
    {
        "title": "Getting Started with DataManagement AI",
        "summary": "Learn how to leverage our AI tools for better data management and analytics.",
        "link": "#",
        "likes": 42,
        "comments": [
            "Great article! Very informative.",
            "This helped me understand the basics.",
            "Looking forward to more content like this!"
        ]
    },
    {
        "title": "Best Practices for Data Analytics",
        "summary": "Discover proven strategies and techniques for effective data analysis.",
        "link": "#",
        "likes": 38,
        "comments": [
            "These practices have improved my workflow.",
            "Would love to see more advanced topics."
        ]
    },
    {
        "title": "Understanding Predictive Analytics",
        "summary": "A comprehensive guide to predictive modeling and forecasting.",
        "link": "#",
        "likes": 35,
        "comments": [
            "Very comprehensive overview of the subject."
        ]
    }
]


def display_article(article):
    st.subheader(article["title"])
    st.write(article["summary"])
    st.markdown(f"[Read more...]({article['link']})")
    
    # Display likes and comments count in the same line with less spacing
    col1, col2 = st.columns([1, 2], gap="small")
    with col1:
        likes_count = article["likes"]
        like_button = st.button(f"👍 {likes_count} likes", key=f"like_{article['title']}")
        if like_button:
            article["likes"] += 1
            st.rerun()
    
    with col2:
        comments = article.get("comments", [])
        comments_count = len(comments)
        with st.expander(f"💬 {comments_count} comments"):
            st.write("#### Comments")
            
            # Display existing comments
            for comment in comments:
                st.write(f"• {comment}")
            
            st.write("---")
            st.write("Join the discussion!")
            new_comment = st.text_area("Write your comment:", height=100, key=f"comment_{article['title']}")
            if st.button("Post Comment", key=f"button_{article['title']}"):
                if not st.session_state.get("authenticated", False):
                    st.warning("Please log in to post a comment.")
                elif not new_comment.strip():
                    st.warning("Please write a comment before posting.")
                else:
                    article["comments"].append(new_comment)
                    st.success("Comment posted successfully!")
                    st.rerun()
    
    st.write("---")  # Divider between articles

def blog_home():
    """Render the blog-like home page with content submission."""
    # Page header
    st.title("Welcome to the DataManagement AI Blog!")
    
    # Articles section
    with st.container():
        st.write("### Latest Articles")
        for article in ARTICLES:
            display_article(article)
    
    # Content submission section
    with st.container():
        st.write("### Publish Your Article")
        st.write("Share your thoughts and articles with the community!")
        content = st.text_area("Write your article here:", height=300)
        if st.button("Publish"):
            handle_content_submission(content)
    
    # Newsletter subscription section
    with st.container():
        st.write("### Subscribe to Our Newsletter")
        st.write("Stay updated with the latest articles and insights!")
        email = st.text_input("Enter your email to subscribe:")
        if st.button("Subscribe"):
            handle_subscription(email)

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




