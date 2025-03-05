import streamlit as st
#import option_menu from streamlit
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import datetime
import cv2
import streamlit as st
import tempfile
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import gdown
import pickle

# Google Drive file ID extracted from the link
file_id = "11bsWFQYp33d38U-MqzgHKjIOU2DimobF"
output = "model.pkl"  # Save as model.pkl

# Construct the download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file
gdown.download(url, output, quiet=False)

# Load the model.pkl file
with open(output, "rb") as file:
    model = pickle.load(file)
encoder = joblib.load('encoder.pkl')

def navigate_to_page(page_name):
    st.session_state["current_page"] = page_name
    st.experimental_rerun()

def user_home_page():
    st.sidebar.image("https://static.vecteezy.com/system/resources/thumbnails/043/988/973/small_2x/traffic-light-displaying-all-signals-on-transparent-background-png.png",width=300)
    with st.sidebar:
        select = option_menu(
            "Dashboard",
            ['Home',"Traffic Volume","Real Time Traffic","Logout"],
            icons=['house','car-front-fill','stoplights-fill','unlock-fill'],
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
                prediction = model.predict(features_values)
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
    elif select == 'Real Time Traffic':
        st.markdown(
            """
            <style>
            /* Apply background image to the main content area */
            .main {
                background-image: url("https://img.freepik.com/free-vector/abstract-watercolor-background_23-2149056656.jpg");  
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            </style>
            """,
            unsafe_allow_html=True
            )
        # Load YOLOv8 model
        model1 = YOLO("yolov8n.pt")
        # Define object categories
        grouped_categories = {
            "Vehicles": ["car", "truck", "bus", "train", "motorcycle", "bicycle", "boat", "airplane"],
            "Electronics": ["laptop", "mouse", "keyboard", "cell phone", "tv", "remote", "microwave", "oven"],
            "Animals": ["dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "Furniture": ["chair", "couch", "bed", "dining table", "potted plant", "toilet"],
            "Food Items": ["bottle", "cup", "wine glass", "banana", "apple", "sandwich", "pizza", "donut", "cake"],
            "Sports & Outdoor": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "tennis racket"],
            "Accessories": ["backpack", "umbrella", "handbag", "tie", "suitcase", "watch", "glasses"],
        }

        def classify_objects(detected_objects):
            category_counts = {category: 0 for category in grouped_categories.keys()}
            
            for obj, count in detected_objects.items():
                for category, items in grouped_categories.items():
                    if obj in items:
                        category_counts[category] += count
                        break
            
            return category_counts

        def detect_objects(frame):
            results = model1(frame)
            detected_objects = {}
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, cls in zip(boxes, classes):
                    label = model1.names[int(cls)]
                    detected_objects[label] = detected_objects.get(label, 0) + 1
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return frame, detected_objects

        #title
        st.markdown(f"<h1 style='text-align: center; color: red;'>Traffic Monitoring</h1>", unsafe_allow_html=True)
        option = st.selectbox("Choose Input Source", ("Webcam", "Upload Video"))

        video_file = None
        if option == "Upload Video":
            video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
        col1,col2,col3=st.columns([2,1,2])
        if col2.button("Start Detection",type='primary'):
            if option == "Webcam":
                cap = cv2.VideoCapture(0)
            elif video_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())
                cap = cv2.VideoCapture(tfile.name)
            else:
                st.error("Please upload a video file.")
                exit()

            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame, detected_objects = detect_objects(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create a blank canvas for object count display
                overlay = np.ones((frame.shape[0], 300, 3), dtype=np.uint8) * 255
                y_offset = 50
                cv2.putText(overlay, "Objects Count", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                for idx, (obj, count) in enumerate(detected_objects.items()):
                    cv2.putText(overlay, f"{obj}: {count}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 40
                
                # Classify objects into categories
                category_counts = classify_objects(detected_objects)
                
                # Create another blank canvas for categorized counts
                category_overlay = np.ones((frame.shape[0], 300, 3), dtype=np.uint8) * 255
                y_offset = 50
                cv2.putText(category_overlay, "Category Report", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                for idx, (category, count) in enumerate(category_counts.items()):
                    cv2.putText(category_overlay, f"{category}: {count}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    y_offset += 40
                
                # Concatenate frames for display
                combined_frame = np.hstack((frame, overlay, category_overlay))
                
                stframe.image(combined_frame, channels="RGB")
            
            cap.release()
            cv2.destroyAllWindows()


    elif select == 'Logout':
        st.session_state["logged_in"] = False
        st.session_state["current_user"] = None
        navigate_to_page("home")
