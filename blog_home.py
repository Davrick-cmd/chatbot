import streamlit as st

def blog_home():
    """Render the blog-like home page with content submission."""
    st.title("Welcome to the DataManagement AI Blog!")
    st.write("### Latest Articles")
    
    # Example articles with real links
    articles = [
        {
            "title": "Understanding Data Management: A Beginner's Guide",
            "summary": "This article covers the basics of data management, why it's important, and how to get started.",
            "link": "https://tdwi.org/events/onsite-education/2024/02/understanding-data-management-trends-in-2024.aspx"
        },
        {
            "title": "How AI is Transforming Data Analytics",
            "summary": "Explore how artificial intelligence is revolutionizing the way we analyze and interpret data.",
            "link": "https://www.datanami.com/2024/01/15/how-ai-is-transforming-data-analytics/"
        },
        {
            "title": "Best Practices for Data Security",
            "summary": "Learn about the essential practices for ensuring your data remains secure and protected.",
            "link": "https://www.sans.org/white-papers/40399/"
        },
        {
            "title": "Data Visualization Techniques for Better Insights",
            "summary": "A guide to effective data visualization techniques to help you derive better insights from your data.",
            "link": "https://www.datapine.com/blog/data-visualization-techniques/"
        }
    ]
    
    for article in articles:
        st.subheader(article["title"])
        st.write(article["summary"])
        st.markdown(f"[Read more...]({article['link']})")
        st.write("---")  # Divider between articles

    st.write("### Publish Your Article")
    st.write("Share your thoughts and articles with the community!")
    
    # Text area for content submission
    content = st.text_area("Write your article here:", height=300)
    
    if st.button("Publish"):
        if content:
            # Here you would typically send the content to a database or perform an action
            st.success("Your content has been published successfully!")
            st.text_area("Your Published Article:", content, height=300, disabled=True)  # Display the published content
        else:
            st.warning("Please write something before publishing.")

    st.write("### Subscribe to Our Newsletter")
    st.write("Stay updated with the latest articles and insights!")
    email = st.text_input("Enter your email to subscribe:")
    if st.button("Subscribe"):
        if email:
            st.success(f"Thank you for subscribing, {email}!")
        else:
            st.warning("Please enter a valid email address.")
