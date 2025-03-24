import streamlit as st
from database import add_user,update_otp
import re
def navigate_to_page(page_name):
    st.session_state["current_page"] = page_name
    st.experimental_rerun()
def validate_mail(mail):
    valid = re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', mail)
    return True
def signup_page():
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url("https://img.freepik.com/free-vector/bokeh-bubbles-sun-flash-blue-background_1182-1745.jpg?w=360");  
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    with st.form(key="signup_form"):
        col1,col2 = st.columns([10,1])
        col1.title("Sign Up Here!!")
        if col2.form_submit_button("🏠"):
            navigate_to_page("home")
        name=st.text_input("Name")
        email = st.text_input("Email", key="signup_email")
        otp=None
        col1,col2=st.columns([1,1])
        password = col1.text_input("Create a Password", type="password", key="signup_password")
        retyped_password = col2.text_input("Retype Password", type="password", key="signup_retyped_password")
        col1,col2,col3 = st.columns([1,1,1])
        with col1:
            if st.form_submit_button("Sign Up") and validate_mail(email)!=None and len(password)>=6 and password==retyped_password and name:
                try:
                    add_user(name,email,otp,password)
                    update_otp(email,otp)
                    st.success("Account created successfully!!")
                    navigate_to_page("login")
                except Exception as e:
                    st.error('Already have an account with this email address. Please login.')
            elif validate_mail(email)==None:
                st.error("Invalid email address. Please enter a valid email address.")
            elif password!=retyped_password:
                st.error("Passwords do not match.")
            elif len(password)<6 and len(password)!=0:
                st.error("Password must be at least 6 characters long.")
        with col3:
            if st.form_submit_button("Already have an account🤔"):
                navigate_to_page("login")