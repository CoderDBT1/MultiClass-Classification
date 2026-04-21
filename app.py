import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #ffffff;
}
.subtitle {
    text-align: center;
    color: #aaaaaa;
    margin-bottom: 30px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #161b22;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_93.h5")

model = load_model()

# -------------------------------
# Class Names + Emojis
# -------------------------------
class_names = [
    'butterfly', 'cat', 'chicken', 'cow', 'dog',
    'elephant', 'horse', 'ragno', 'sheep', 'squirrel'
]

emoji_map = {
    'butterfly': '🦋',
    'cat': '🐱',
    'chicken': '🐔',
    'cow': '🐄',
    'dog': '🐶',
    'elephant': '🐘',
    'horse': '🐴',
    'ragno': '🕷️',   # spider
    'sheep': '🐑',
    'squirrel': '🐿️'
}

# -------------------------------
# Header
# -------------------------------
st.markdown('<div class="title">🖼️ CNN Image Classifier by Debarshi</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and let my AI predict the animal</div>', unsafe_allow_html=True)

# -------------------------------
# Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# Prediction Function
# -------------------------------
def predict(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[0][predicted_class])

    return class_names[predicted_class], confidence, predictions[0]

# -------------------------------
# Output Section
# -------------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.spinner("Analyzing image..."):
            label, confidence, probs = predict(img)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        emoji = emoji_map.get(label, "")
        st.success(f"Prediction: {emoji} {label.capitalize()}")
        st.metric("Confidence", f"{confidence*100:.2f}%")

        st.subheader("📊 Probability Distribution")

        # Show all classes
        for i, class_name in enumerate(class_names):
            emoji = emoji_map.get(class_name, "")
            st.progress(float(probs[i]))
            st.caption(f"{emoji} {class_name.capitalize()} — {probs[i]*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)
