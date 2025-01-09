import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
from PIL import Image

# Load the trained model
model = load_model('models/model.keras')

# Class labels for FER-2013 dataset
class_labels = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Happiness', 'Surprise', 'Neutral']

# Function to preprocess the image
def preprocess_image(img):
    # Convert image to grayscale
    img = img.convert('L')
    img = img.resize((48, 48))
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array.astype('float32')  # Normalize the image
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit user interface
st.title("Emotion Detection from Facial Expression")
st.write("Upload an image to detect emotion.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the image
    st.image(uploaded_image, caption='Uploaded Image', width=250)    
    # Process the image
    img = Image.open(uploaded_image)
    processed_img = preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)
    emotion = class_labels[predicted_class[0]]
    
    # Display the result
    st.write(f"Predicted Emotion: {emotion}")
