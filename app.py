import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import cv2

# Load the model
model = load_model("emotion_recognition_model.keras")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit UI
st.title("Emotion Recognition App")
st.write("Upload an image, and the model will predict the emotion.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Convert to an OpenCV image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((48, 48))                  # Resize to model input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)           # Add batch dimension
    image /= 255                                    # Normalize

    # Make prediction
    prediction = model.predict(image)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]

    st.write(f"Predicted Emotion: **{emotion}**")
