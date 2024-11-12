import streamlit as st

class AppConfig:
    @staticmethod
    def setup_page():
        st.set_page_config(
            layout="wide",
            page_title="DataManagement AI",
            page_icon="./img/bkofkgl.png",
            menu_items={
                'Get Help': 'mailto:john@example.com',
                'About': "#### This is DataManagement cool app!"
            }
        )
        
        # Add custom CSS
        st.markdown("""
            <style>
                .stRadio [role=radiogroup] {
                    gap: 1rem;
                }
                .stButton button {
                    width: 100%;
                }
                .sidebar .sidebar-content {
                    background-color: #f0f2f6;
                }
                /* Add more custom CSS here */
            </style>
        """, unsafe_allow_html=True) 

class Config:
    AD_TIMEOUT = 10  # seconds
    AD_SERVER = "bk.local"
    AD_DOMAIN = "bk.local"
    AD_USER = 'htwahirwa'
    AD_USER_PASSWORD = "Twahih199400#"
    AD_TIMEOUT=10
    AD_SEARCH_BASE='DC=bk,DC=local'