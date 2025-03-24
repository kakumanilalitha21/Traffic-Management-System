import streamlit as st
#import option_menu from streamlit
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import datetime
import cv2
import streamlit as st
import tempfile
from ultralytics import YOLO
import numpy as np
import pickle
import gdown
import os
from pathlib import Path
from typing import NamedTuple
from io import BytesIO
import streamlit as st
from sample_utils.download import download_file
HERE = Path(__file__).parent
ROOT = HERE.parent

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)
file_id = "11bsWFQYp33d38U-MqzgHKjIOU2DimobF"
output = "model.pkl"

url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, output, quiet=False)
# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Create temporary folder if doesn't exists
if not os.path.exists('./temp'):
   os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button == True:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

def processVideo(video_file, score_threshold):
    
    # Write the file into disk
    write_bytesio_to_file(temp_file_input, video_file)
    
    videoCapture = cv2.VideoCapture(temp_file_input)

    # Check the video
    if (videoCapture.isOpened() == False):
        st.error('Error opening the video file')
    else:
        _width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        _height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = videoCapture.get(cv2.CAP_PROP_FPS)
        _frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        _duration = _frame_count/_fps
        _duration_minutes = int(_duration/60)
        _duration_seconds = int(_duration%60)
        _duration_strings = str(_duration_minutes) + ":" + str(_duration_seconds)
        inferenceBarText = "Performing inference on video, please wait."
        inferenceBar = st.progress(0, text=inferenceBarText)

        imageLocation = st.empty()

        # Issue with opencv-python with pip doesn't support h264 codec due to license, so we cant show the mp4 video on the streamlit in the cloud
        # If you can install the opencv through conda using this command, maybe you can render the video for the streamlit
        # $ conda install -c conda-forge opencv
        # fourcc_mp4 = cv2.VideoWriter_fourcc(*'h264')
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

        # Read until video is completed
        _frame_counter = 0
        while(videoCapture.isOpened()):
            ret, frame = videoCapture.read()
            if ret == True:
                
                # Convert color-chanel
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform inference
                _image = np.array(frame)

                image_resized = cv2.resize(_image, (640, 640), interpolation = cv2.INTER_AREA)
                results = net.predict(image_resized, conf=score_threshold)
                
                # Save the results
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    detections = [
                    Detection(
                        class_id=int(_box.cls),
                        label=CLASSES[int(_box.cls)],
                        score=float(_box.conf),
                        box=_box.xyxy[0].astype(int),
                        )
                        for _box in boxes
                    ]

                annotated_frame = results[0].plot()
                _image_pred = cv2.resize(annotated_frame, (_width, _height), interpolation = cv2.INTER_AREA)

                print(_image_pred.shape)
                
                # Write the image to file
                _out_frame = cv2.cvtColor(_image_pred, cv2.COLOR_RGB2BGR)
                cv2writer.write(_out_frame)
                
                # Display the image
                imageLocation.image(_image_pred)

                _frame_counter = _frame_counter + 1
                inferenceBar.progress(_frame_counter/_frame_count, text=inferenceBarText)
            
            # Break the loop
            else:
                inferenceBar.empty()
                break

        # When everything done, release the video capture object
        videoCapture.release()
        cv2writer.release()

    # Download button for the video
    st.success("Video Processed!")

    col1, col2 = st.columns(2)
    with col1:
        # Also rerun the appplication after download
        with open(temp_file_infer, "rb") as f:
            st.download_button(
                label="Download Prediction Video",
                data=f,
                file_name="RDD_Prediction.mp4",
                mime="video/mp4",
                use_container_width=True
            )
            
    with col2:
        if st.button('Restart Apps', use_container_width=True, type="primary"):
            # Rerun the application
            st.rerun()

# Google Drive file ID extracted from the link
output = "model.pkl"  # Save as model.pkl


# Download the file

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
            ['Home',"Traffic Volume","Real Time Traffic","Signal Control","Incident Prevntion","Logout"],
            icons=['house','car-front-fill','stoplights-fill','bell-fill','car-front-fill','unlock-fill'],
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

        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
        col1,col2,col3=st.columns([2,1,2])
        if col2.button("Start Detection",type='primary'):
            if video_file is not None:
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
    elif select=='Signal Control':
        st.markdown(
            """
            <style>
            /* Apply background image to the main content area */
            .main {
                background-image: url("");  
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            </style>
            """,
            unsafe_allow_html=True
            )
        model1 = YOLO("yolov8n.pt")

        # Define object categories
        vehicle_classes = ["car", "truck", "bus", "train", "motorcycle", "bicycle"]

        def detect_vehicles(frame):
            results = model1(frame)
            vehicle_count = 0
            
            for result in results:
                classes = result.boxes.cls.cpu().numpy()
                for cls in classes:
                    label = model1.names[int(cls)]
                    if label in vehicle_classes:
                        vehicle_count += 1
            
            return vehicle_count
        st.markdown("<h1 style='text-align: center; color: red;'>Traffic Sign Control Monitoring</h1>", unsafe_allow_html=True)

        video_files = st.file_uploader("Upload up to 4 Videos", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True)
        col1,col2,col3=st.columns([2,2,1])
        button=col2.button('Detect',type='primary')
        if button:
            if len(video_files) > 4:
                st.error("You can upload a maximum of 4 videos.")
            else:
                vehicle_counts = []
                
                for video_file in video_files:
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(video_file.read())
                    cap = cv2.VideoCapture(tfile.name)
                    
                    ret, frame = cap.read()  # Read only the first frame
                    cap.release()
                    
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        vehicle_count = detect_vehicles(frame)
                        vehicle_counts.append(vehicle_count)
                    else:
                        st.error(f"Could not read the first frame of {video_file.name}.")

                # Display results
                if vehicle_counts:
                    total_vehicles = sum(vehicle_counts)
                    time_allocations = [int((count / total_vehicles) * 100) if total_vehicles > 0 else 0 for count in vehicle_counts]
                    st.markdown(
                        """
                        <style>
                        /* Background Image */
                        .main {
                            background-image: url('background.jpg');  
                            background-size: cover;
                            background-position: center;
                            background-repeat: no-repeat;
                        }
                        /* Styling for columns */
                        .col-container {
                            background-color:red;
                            display: flex;
                            justify-content: space-around;
                            text-align: center;
                            margin-top: 20px;
                        }
                        .col-box {
                            width: 200px;
                            padding: 15px;
                            border-radius: 15px;
                            background-color: #f8f9fa;
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                            text-align: center;
                        }
                        .icon {
                            font-size: 40px;
                            color: #ff5733;
                        }
                        .count-text {
                            font-size: 20px;
                            font-weight: bold;
                            color: #2e86c1;
                        }
                        .time-text {
                            font-size: 16px;
                            font-weight: bold;
                            color: #28a745;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    try:
                    # Display Results in Columns if button is pressed
                        if button and vehicle_counts and len(vehicle_counts) ==4:
                            directions = ['North', 'South', 'East', 'West']
                            icons = ["\u2B06", "\u2B07", "\u27A1", "\u2B05"]  # Unicode arrows
                            
                            st.markdown("<div class='col-container'>", unsafe_allow_html=True)
                            cols = st.columns(4) 
                            for i, col in enumerate(cols):
                                with col:
                                    st.markdown(f"<h2 style='text-align: center;'>{icons[i]}</h2>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; font-size: 20px; color: #2e86c1;'><b>🚗 {vehicle_counts[i]} Vehicles</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; font-size: 16px; color: #28a745;'><b>⏳ {time_allocations[i]} Sec</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; font-size: 18px;'><b>{directions[i]}</b></p>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.error('You can upload a maximum of 4 videos')
                    except:
                        st.error('You can upload a maximum of 4 videos')
    elif select=='Incident Prevntion':
        st.markdown(f"<h1 style='text-align: center; color: red;'>Pothole Detection</h1>", unsafe_allow_html=True)
        st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url('https://static.vecteezy.com/system/resources/thumbnails/038/971/230/small/ai-generated-damaged-american-road-surface-with-deep-pothole-ruined-street-in-urgent-need-of-repair-photo.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 255, 255, 0.6);
            background-blend-mode: overlay;
        }
        </style>
        """,
        unsafe_allow_html=True,
        )
        #give the option to upload the image or video
        video_file = st.file_uploader("Upload Pothole Video", type=".mp4")
        score_threshold = 0.10
        col1,col2,col3=st.columns([2,2,1])
        button=col2.button('Start Detection',key='processing_button',type='primary')
        if video_file is not None:
                processVideo(video_file, score_threshold)
        
        


        
