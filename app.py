import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/teeth_classifier_mobilenetv2.h5")



model = load_model()
class_labels = ['MC', 'OLP', 'CoS', 'OC', 'OT', 'CaS', 'Gum']  # update based on your dataset

# Title and instructions
st.title("Dental Condition Classifier")
st.markdown("Upload a dental image, and the model will classify its condition.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # Show result
    st.success(f"Prediction: **{pred_class}** with {confidence:.2f}% confidence")
