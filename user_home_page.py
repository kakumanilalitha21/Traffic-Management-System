import streamlit as st
#import option_menu from streamlit
from streamlit_option_menu import option_menu
from database import fetch_user
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import ensemble
data=pd.read_csv('cleaned_traffic_volume.csv')
y= data['traffic_volume']
x= data.drop(columns=['traffic_volume'],axis=1)
names=x.columns
x=scale(x)
x=pd.DataFrame( x,columns=names)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
Rand=ensemble.RandomForestRegressor()
Rand.fit(x_train, y_train)
p3=Rand.predict(x_train)

def navigate_to_page(page_name):
    st.session_state["current_page"] = page_name
    st.experimental_rerun()

def user_home_page():
    st.sidebar.image("https://static.vecteezy.com/system/resources/thumbnails/043/988/973/small_2x/traffic-light-displaying-all-signals-on-transparent-background-png.png",width=300)
    with st.sidebar:
        select = option_menu(
            "Dashboard",
            ['Home',"Traffic Volume","Logout"],
            icons=['house','car-front-fill','unlock-fill'],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
        )
    if select == 'Home':
        st.markdown(
                    f"""
                    <div style="text-align: center; padding: 10px; background-color: #beeb0e ; border-radius: 15px;">
                        <p style="color: black; font-size: 38px;"><b>📢 AI Based Traffic Management System</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown(
            """
            <style>
            /* Apply background image to the main content area */
            .main {
                background-image: url("https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjU0NmJhdGNoMy1teW50LTM0LWJhZGdld2F0ZXJjb2xvcl8xLmpwZw.jpg");  
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            </style>
            """,
            unsafe_allow_html=True
            )
        st.image('https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/dd1da458-be98-4295-a51e-26955afa21df/d1z7j2f-6b514eb3-a683-4dce-ad95-8098b2e66960.gif?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2RkMWRhNDU4LWJlOTgtNDI5NS1hNTFlLTI2OTU1YWZhMjFkZlwvZDF6N2oyZi02YjUxNGViMy1hNjgzLTRkY2UtYWQ5NS04MDk4YjJlNjY5NjAuZ2lmIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.wUKvo7XyHdNq7sfldphuYiqFHonrbKvdN1WsBdFdTTY',use_column_width=True)
    elif select == 'Traffic Volume':
        
        st.markdown(
            """
            <style>
            /* Apply background image to the main content area */
            .main {
                background-image: url("https://img.freepik.com/free-photo/minimalist-blue-white-wave-background_1017-46756.jpg");  
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            </style>
            """,
            unsafe_allow_html=True
            )
        text=""
        k=0
        with st.form(key="traffic-volume-form"):
            st.title("Traffic Volume Prediction")
            col1, col2 = st.columns(2)
            holiday = col1.selectbox("Is it a holiday?", ['Yes', 'No'])
            if holiday == 'Yes':
                holiday = 1
            else:
                holiday = 0
            temp = col2.number_input("Temperature (in Celsius)")
            rain = col1.number_input("Rain (in mm)")
            snow = col2.number_input("Snow (in mm)")
            weather = col1.selectbox("Weather", ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain','Smoke','Snow','Squad11', 'Thunderstorm'])
            values=['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain','Smoke','Snow','Squad11', 'Thunderstorm']
            weather = values.index(weather)
            #fetching the current date
            now = datetime.datetime.now()
            today = datetime.date.today()
            date=col2.date_input("Select a Date", min_value=today)
            if date:
                day=date.day
                month=date.month
                year=date.year
            else:
                day=now.day
                year=now.year
                month=now.month
            hours=now.hour
            minutes=now.minute
            seconds=now.second
            input_features = [holiday, temp, rain, snow, weather, year, month, day, hours, minutes, seconds]
            if st.form_submit_button(label="Predict Traffic Volume"):
                features_values = np.array(input_features).reshape(1, -1)
                prediction = Rand.predict(features_values)
                text = f"Estimated Traffic Volume is: {prediction[0]}"
                if prediction[0] < 500:
                    k=0
                    text += " (Low Traffic)"
                elif prediction[0] < 2000:
                    k=1
                    text += " (Medium Traffic)"
                else:
                    k=2
                    text += " (High Traffic)"
        if text:
            if k==0:
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 5px; background-color: rgba(51, 51, 51, 0.5); border-radius: 10px;">
                        <p style="color: white; font-size: 25px;">{text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif k==1:
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 5px; background-color: rgba(0, 255, 0, 0.7); border-radius: 10px;">
                        <p style="color: black; font-size: 25px;">{text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif k==2:
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 5px; background-color: rgba(255, 0, 0, 0.7); border-radius: 10px;">
                        <p style="color: black; font-size: 25px;">{text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 5px; background-color: rgba(51, 51, 51, 0.7); border-radius: 10px;">
                        <p style="color: white; font-size: 30px;">No Traffic 🚫</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
  
    elif select == 'Logout':
        st.session_state["logged_in"] = False
        st.session_state["current_user"] = None
        navigate_to_page("home")
