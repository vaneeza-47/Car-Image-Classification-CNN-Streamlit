# import required libararies for the streamlit web application
import streamlit as st # Main Streamlit libraray for creating web apps
import tensorflow as tf #Deep learning framework
from tensorflow.keras.models import load_model # Function to load saved keras model
from PIL import Image # python image library for image processing
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2 # OPencv for computer vision tasks
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

CLASS_NAMES= ['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova'] 

@st.cache_resource
def load_trained_model():
    model = load_model("model.h5")
    return model

def preprocess_image(image):
    # preprocess the uploaded image to match the model's expected format
    # Resize the image to 128 * 128 pixels (the size of the model was trained on)
    image = image.resize((128,128))
    # Conver the PIL image to a numpy array
    img_array = img_to_array(image)
    # normalize the values from 0-255 to 0-1 (divided by 225)
    img_array = img_array/255.0
    # add a batch dimension to make it comptible with the model (shape: [1,128,128,3])
    img_array = np.expand_dims(img_array,axis=0)
    return img_array



def predict_car_class(model,image):
    #preprocess the uploaded image to match model input requirements
    proccessed_image = preprocess_image(image)
    # use the model to make predictions on the processed image
    predictions = model.predict(proccessed_image)

    # Find the index of the class with the highest probablity
    predicted_class_idx = np.argmax(predictions[0])
    #get the car class name corrsponding to the predicted index
    predicted_class = CLASS_NAMES[predicted_class_idx]

    #Calculate Confidence percentage 
    confidence = float(predictions[0][predicted_class_idx]) * 100
    
    return predicted_class, confidence, predictions[0], predicted_class_idx





#Configure the streamlit page settings
st.set_page_config (
    page_title="Car Image Classifier",
    page_icon="🚗",
    layout="wide"
)
# creaye the main tittle and header section
st.title("🚗 Car Image Classifier") # Display the main title with car emoji
st.markdown("---")
st.markdown("Upload a car image and the AI will classify into one of 7 categories!")

# Load the trained model with a loading spinner
with st.spinner("Loading the trained model"):
    model = load_trained_model()

#check if model loading was successful 
if model is None:
    st.error("Failed to load the model. Please make sure 'CarModel.h5' is in current directory")
    st.stop()


st.success("Model Loaded Successfully")

# Create the main content section for image upload
st.header(" Upload Image")

# create a file uploader widget for users to upload car images
uploaded_file = st.file_uploader(
    "Choose a car image... ", # label for the upload button
    type=['jpg', 'jpeg', 'png'], # allowed file types
    help="Upload an image of car to classify" 
)

# Check if a file has been uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with col2:
        if st.button("🚀 Classify Image", type="primary"):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, all_predictions, predicted_class_idx = predict_car_class(model, image)

            st.success(f"### ✅ Predicted Car: {predicted_class}")
            st.metric("Confidence", f"{confidence:.2f}%")

            # Progress bar
            st.progress(confidence / 100)

            # Top 3 predictions as metrics
            st.subheader("🏆 Top 3 Predictions")
            sorted_indices = np.argsort(all_predictions)[::-1]
            for i in range(3):
                idx = sorted_indices[i]
                prob = all_predictions[idx] * 100
                st.metric(f"{i+1}. {CLASS_NAMES[idx]}", f"{prob:.2f}%")

            # Interactive Probability Chart
            st.subheader("📊 All Class Probabilities")
            prob_dict = {CLASS_NAMES[i]: all_predictions[i] * 100 for i in range(len(CLASS_NAMES))}

            # Prepare data for Altair chart
            df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability": [p * 100 for p in all_predictions]
            })

            # Highlight the predicted class
            df["Color"] = ["#FF4B4B" if i == predicted_class_idx else "#4DA6FF" for i in range(len(df))]

            # Create an Altair bar chart (visually close to st.bar_chart)
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Class", sort=None, title=None),
                    y=alt.Y("Probability", title="Probability (%)"),
                    color=alt.Color("Color", scale=None, legend=None),
                    tooltip=["Class", alt.Tooltip("Probability", format=".2f")]
                )
                .properties(width="container", height=350)
            )

            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)



st.markdown("---", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; padding: 15px; color: #666; font-size: 16px; font-family: 'Arial';">
        <p style="margin: 5px; font-weight: 500;">
             Crafted with ❤️ by <b style="color:#4CAF50;">Vaneeza Aftab</b> using 
            <span style="color:#FF4B4B;">Streamlit</span> & 
            <span style="color:#007ACC;">TensorFlow</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

