import streamlit as st
from database import fetch_otp, fetch_user
import random
def navigate_to_page(page_name):
    st.session_state["current_page"] = page_name
    st.experimental_rerun()
def otp_page():
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url("https://www.mintzglobalscreening.com/wp-content/uploads/2019/06/Comment-preparer-verification-antecedents.jpeg");  
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 255, 255, 0.6);
            background-blend-mode: overlay;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    user=st.session_state["current_user"]
    col1,col2,col3=st.columns([1,10,1])
    with col2.form(key="otp_form"):
        st.title("Verify Credentials")
        otp = st.text_input("Enter OTP",placeholder="Please enter the OTP sent to your registered email address")
        k=fetch_otp(user)
        col1,col2,col3=st.columns([1,2,1])
        if col1.form_submit_button("Submit",type='primary'):
            email = st.session_state["current_user"]
            if int(otp) == int(fetch_otp(user)[0]):
                user = fetch_user(email)
                st.success(f"Welcome back, {user[1]}!")
                navigate_to_page("user_home")
            else:
                st.error("Invalid OTP. Please try again.")
        if col3.form_submit_button("Resend OTP",type='primary'):
            navigate_to_page("login")