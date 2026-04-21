# -------------------------------
# Streamlit App for CIFAR-10 Classifier
# -------------------------------

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_93.h5")

model = load_model()

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="AI Image Classifier", layout="centered")
st.title("🖼️ Image Classifier (CIFAR-10)")
st.write("Upload an image and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# -------------------------------
# Prediction Function
# -------------------------------
def predict(img):
    img = img.resize((128, 128))  # same as training
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[0][predicted_class])

    return class_names[predicted_class], confidence, predictions[0]

# -------------------------------
# Run Prediction
# -------------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    label, confidence, probs = predict(img)

    st.success(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")

    st.subheader("Class Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {probs[i]*100:.2f}%")
