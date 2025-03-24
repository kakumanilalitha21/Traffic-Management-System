import streamlit as st
from database import authenticate_user,update_otp
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import random
def send_alert_email(to_email, subject, message, from_email, from_password):
    # Set up the SMTP server
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    
    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))
    
    try:
        # Connect to the server and send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Alert email sent successfully!")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")

def navigate_to_page(page_name):
    st.session_state["current_page"] = page_name
    st.experimental_rerun()

def login_page():
    # Center the login form using Streamlit form layout
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url("https://img.freepik.com/premium-photo/abstract-futuristic-technology-background-with-fractal-horizon_476363-2135.jpg");  
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    with st.form(key="login_form"):
        # Title
        col1,col2=st.columns([10,1])
        col1.title("Login Here!!")
        if col2.form_submit_button("🏠"):
            navigate_to_page("home")

        # Email and Password inputs
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        # Submit button inside the form
        col1,col2,col3=st.columns([1,4,1])
        with col1:
            if st.form_submit_button("Login"):
                if authenticate_user(email, password):
                    otp = random.randint(100000, 999999)
                    update_otp(email, otp)
                    to_email=email
                    subject = "OTP for Traffic Management System"
                    message = f"Hello,\n\nYour OTP for Traffic Management System is {otp}.\n\nThank you."
                    from_email = 'lalithakakumani21@gmail.com'
                    from_password = 'ifzwfytjzznilzpx'  
                    # Send the alert email
                    send_alert_email(to_email, subject, message, from_email, from_password)
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = email
                    navigate_to_page("otp")
                else:
                    st.error("Invalid email or password.")
        with col3:
            if st.form_submit_button("Sign Up🤔"):
                navigate_to_page("signup")