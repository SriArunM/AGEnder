import sqlite3
import os
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.metrics import MeanAbsoluteError
from datetime import datetime
from fpdf import FPDF
import time
from PIL import Image


IMAGE_SIZE = 128
GENDER_MAPPING = ["Male", "Female"]

st.markdown(
    """
    <style>
        .stButton > button:hover {
            background-color: #4CAF50;
            color: white;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 120px;
            background-color: black;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position above the button */
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
""",
    unsafe_allow_html=True,
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def frame_crop_head(frame, expansion=0.05):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        x1 = max(0, x - int(w * expansion))
        y1 = max(0, y - int(h * expansion))
        x2 = min(frame.shape[1], x + w + int(w * expansion))
        y2 = min(frame.shape[0], y + h + int(h * expansion))
        head = frame[y1:y2, x1:x2]
        return head
    return None


def crop_head(image_path, expansion=0.05):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        x1 = max(0, x - int(w * expansion))
        y1 = max(0, y - int(h * expansion))
        x2 = min(image.shape[1], x + w + int(w * expansion))
        y2 = min(image.shape[0], y + h + int(h * expansion))
        head = image[y1:y2, x1:x2]
        return head
    return None


@st.cache_resource
def load_model_once():
    model = load_model(
        "AgeGenderModel.h5",
        custom_objects={"mae": MeanAbsoluteError()},
    )
    return model


def preprocess_image(cropped_image):
    img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_age_and_gender(img_array, model):
    predictions = model.predict(img_array)
    gender_prob = predictions[0][0]
    predicted_gender = GENDER_MAPPING[int(round(gender_prob[0]))]
    predicted_age = predictions[1][0]
    return predicted_gender, predicted_age


def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB,
            predicted_age REAL,
            predicted_gender TEXT,
            feedback TEXT,
            feedback_rating INTEGER,
            timestamp DATETIME
        )
    """
    )
    conn.commit()
    conn.close()


def save_to_db(image_data, predicted_age, predicted_gender, feedback, feedback_rating):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (image, predicted_age, predicted_gender, feedback, feedback_rating, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (
            image_data,
            predicted_age,
            predicted_gender,
            feedback,
            feedback_rating,
            datetime.now(),
        ),
    )
    conn.commit()
    conn.close()


def generate_pdf_report(image_data, predicted_age, predicted_gender):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Age and Gender Prediction Report", ln=True, align="C")
    pdf.ln(10)

    with open("temp_uploaded_image.jpg", "wb") as f:
        f.write(image_data)
    pdf.image("temp_uploaded_image.jpg", x=10, y=30, w=100)

    pdf.set_font("Arial", size=12)
    pdf.ln(110)
    pdf.cell(200, 10, txt=f"Predicted Gender: {predicted_gender}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Age: {predicted_age:.2f} years", ln=True)

    pdf_path = "prediction_report.pdf"
    pdf.output(pdf_path)
    return pdf_path


def main():
    st.title("Age and Gender Prediction with Report Generation")
    st.image("Logo.png", use_container_width=True)
    st.write(
        "Upload an image or capture an image using your webcam to predict age and gender."
    )

    init_db()

    if "upload_cropped_head" not in st.session_state:
        st.session_state.upload_cropped_head = None
    if "upload_predicted_age" not in st.session_state:
        st.session_state.upload_predicted_age = None
    if "upload_predicted_gender" not in st.session_state:
        st.session_state.upload_predicted_gender = None
    if "upload_show_cropped" not in st.session_state:
        st.session_state.upload_show_cropped = False
    if "upload_generate_report" not in st.session_state:
        st.session_state.upload_generate_report = False

    if "capture_cropped_head" not in st.session_state:
        st.session_state.capture_cropped_head = None
    if "capture_predicted_age" not in st.session_state:
        st.session_state.capture_predicted_age = None
    if "capture_predicted_gender" not in st.session_state:
        st.session_state.capture_predicted_gender = None
    if "capture_show_cropped" not in st.session_state:
        st.session_state.capture_show_cropped = False
    if "capture_generate_report" not in st.session_state:
        st.session_state.capture_generate_report = False

    model = load_model_once()

    option = st.sidebar.selectbox(
        "Choose Image Source", ["Upload Image", "Capture Image"]
    )

    if option == "Upload Image":
        uploaded_image = st.file_uploader(
            "Upload an Image", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image:

            img = Image.open(uploaded_image)
            img = img.convert("RGB")

            temp_file_path = "temp_uploaded_image.jpg"
            img.save(temp_file_path, "JPEG")

            st.image(temp_file_path, caption="Uploaded Image", use_container_width=True)

            model = load_model_once()

            show_cropped_button = st.checkbox("Show Cropped Image Button")

            if st.button("Make Prediction"):

                with st.spinner("Making prediction..."):
                    time.sleep(1)
                    st.session_state.upload_cropped_head = crop_head(temp_file_path)
                    if st.session_state.upload_cropped_head is not None:
                        img_array = preprocess_image(
                            st.session_state.upload_cropped_head
                        )
                        (
                            st.session_state.upload_predicted_gender,
                            st.session_state.upload_predicted_age,
                        ) = predict_age_and_gender(img_array, model)

                        with open(temp_file_path, "rb") as f:
                            image_data = f.read()
                        save_to_db(
                            image_data,
                            st.session_state.upload_predicted_age.item(),
                            st.session_state.upload_predicted_gender,
                            feedback="",
                            feedback_rating=0,
                        )

                        st.write(
                            f"**Predicted Gender:** {st.session_state.upload_predicted_gender}"
                        )
                        st.write(
                            f"**Predicted Age:** {st.session_state.upload_predicted_age.item():.2f} years"
                        )

            if show_cropped_button and st.session_state.upload_cropped_head is not None:
                if st.button("Show Cropped Image"):
                    st.session_state.upload_show_cropped = True

            if st.session_state.upload_show_cropped:
                st.image(
                    st.session_state.upload_cropped_head,
                    caption="Cropped Face Image",
                    use_container_width=True,
                )

            if (
                st.session_state.upload_predicted_age is not None
                and st.session_state.upload_predicted_gender is not None
            ):
                feedback = st.text_area(
                    "Provide feedback about the prediction (optional)"
                )
                feedback_rating = st.radio(
                    "Rate the accuracy of the prediction", [1, 2, 3, 4, 5]
                )

                if st.button("Submit Feedback"):
                    with open(temp_file_path, "rb") as f:
                        image_data = f.read()

                    save_to_db(
                        image_data,
                        st.session_state.upload_predicted_age.item(),
                        st.session_state.upload_predicted_gender,
                        feedback,
                        feedback_rating,
                    )

                    st.success("Thank you for your feedback!")

            if (
                st.session_state.upload_predicted_age is not None
                and st.session_state.upload_predicted_gender is not None
            ):
                if st.button("Download Report"):
                    st.session_state.upload_generate_report = True

            if st.session_state.upload_generate_report:
                with open(temp_file_path, "rb") as f:
                    image_data = f.read()

                pdf_path = generate_pdf_report(
                    image_data,
                    st.session_state.upload_predicted_age.item(),
                    st.session_state.upload_predicted_gender,
                )

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download Report",
                        f,
                        file_name="prediction_report.pdf",
                        mime="application/pdf",
                    )

    if option == "Capture Image":
        st.write("Capture an image using your webcam.")

        if st.button("Capture Image"):

            video_capture = cv2.VideoCapture(0)

            if not video_capture.isOpened():
                st.error("Could not open webcam.")
            else:
                ret, frame = video_capture.read()

                video_capture.release()

                if ret:
                    st.session_state.captured_image = frame
                    st.image(
                        frame,
                        channels="BGR",
                        caption="Captured Image",
                        use_container_width=True,
                    )
                    st.success(
                        "Image captured successfully! Click 'Make Prediction' to proceed."
                    )

        if st.session_state.captured_image is not None and st.button("Make Prediction"):

            with st.spinner("Making prediction..."):
                time.sleep(1)
                st.session_state.capture_cropped_head = frame_crop_head(
                    st.session_state.captured_image
                )
                if st.session_state.capture_cropped_head is not None:
                    img_array = preprocess_image(st.session_state.capture_cropped_head)
                    (
                        st.session_state.capture_predicted_gender,
                        st.session_state.capture_predicted_age,
                    ) = predict_age_and_gender(img_array, model)

                    st.write(
                        f"**Predicted Gender:** {st.session_state.capture_predicted_gender}"
                    )
                    st.write(
                        f"**Predicted Age:** {st.session_state.capture_predicted_age.item():.2f} years"
                    )

                    image_data = cv2.imencode(".jpg", st.session_state.captured_image)[
                        1
                    ].tobytes()
                    save_to_db(
                        image_data,
                        st.session_state.capture_predicted_age.item(),
                        st.session_state.capture_predicted_gender,
                        feedback="",
                        feedback_rating=0,
                    )

                    # # Provide options for feedback and report
                    # feedback = st.text_area(
                    #     "Provide feedback about the prediction (optional)"
                    # )
                    # feedback_rating = st.radio(
                    #     "Rate the accuracy of the prediction", [1, 2, 3, 4, 5]
                    # )

                    # if st.button("Submit Feedback"):
                    #     save_to_db(
                    #         image_data,
                    #         st.session_state.capture_predicted_age.item(),
                    #         st.session_state.capture_predicted_gender,
                    #         feedback,
                    #         feedback_rating,
                    #     )
                    #     st.success("Thank you for your feedback!")

                    # # Show the cropped image if the button is pressed
                    # if st.button("Show Cropped Image"):
                    #     st.session_state.capture_show_cropped = True

                    # if st.session_state.capture_show_cropped:
                    #     st.image(
                    #         st.session_state.capture_cropped_head,
                    #         caption="Cropped Face Image",
                    #         use_container_width =True,
                    #     )

                    # # Generate and download the report
                    # if st.button("Download Report"):
                    #     st.session_state.capture_generate_report = True

                    # if st.session_state.capture_generate_report:
                    #     pdf_path = generate_pdf_report(
                    #         image_data,
                    #         st.session_state.capture_predicted_age.item(),
                    #         st.session_state.capture_predicted_gender,
                    #     )

                    #     # Allow download of the PDF report
                    #     with open(pdf_path, "rb") as f:
                    #         st.download_button(
                    #             "Download Report",
                    #             f,
                    #             file_name="prediction_report.pdf",
                    #             mime="application/pdf",
                    #         )

                else:
                    st.error("No face detected. Please try again.")


if __name__ == "__main__":
    main()
