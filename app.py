import streamlit as st

st.set_page_config(page_title="Plant Disease Class Prediction", page_icon="🌿", layout="centered")

import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_model.keras')
    return model

model = load_model()

# Class names corresponding to model output indices
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

st.title("🌿 Plant Disease Class Prediction")
st.markdown(
    """
    Upload an image of a plant leaf, and the app will predict the disease class.
    """
)

with st.sidebar:
    st.header("Instructions")
    st.write(
        """
        1. Click the 'Browse files' button to upload an image (jpg, jpeg, png).
        2. Wait for the prediction to appear.
        3. View the predicted disease class below the image.
        """
    )
    st.write("---")
    st.write("Developed with Streamlit and TensorFlow")

uploaded_file = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1, 2])

    with col1:
        # Ensure image is in RGB mode before displaying
        image = image.convert("RGB")
        # Display the image directly without converting to bytes to avoid encoding issues
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.subheader("Prediction Result")
        with st.spinner('Predicting...'):
            # Preprocess the image to match model input
            image_resized = image.resize((128, 128))
            input_arr = np.array(image_resized)
            input_arr = np.expand_dims(input_arr, axis=0)  # Create batch dimension

            # Predict the class
            prediction = model.predict(input_arr)
            result_index = np.argmax(prediction)
            predicted_class = class_names[result_index]

            st.success(f"Predicted Class: **{predicted_class}**")