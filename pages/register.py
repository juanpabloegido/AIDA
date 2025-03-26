import streamlit as st
from auth import register

st.set_page_config(
    page_title="Register - AIDA",
    page_icon="💊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTitle {
        color: #2e4d7b;
    }
    #MainMenu {visibility: hidden}
    #header {visibility: hidden}
    #footer {visibility: hidden}
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Logo
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("https://www.atida.com/static/version1741757720/frontend/Interactiv4/mifarmaHyva/es_ES/images/logo.svg",
             width=200)

# Registration form
register() 