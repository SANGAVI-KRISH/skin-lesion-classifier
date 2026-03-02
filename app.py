import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ✅ IMPORTANT: use standalone keras for Keras 3 compatibility
import keras

# ---------------- Page Settings ----------------
st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="centered")

# ---------------- Class Full Names ----------------
CLASS_NAMES = {
    "akiec": "Actinic Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevus (Mole)",
    "vasc":  "Vascular Lesion"
}

# ---------------- Clinical Category ----------------
CATEGORY = {
    "mel":   "Malignant",
    "bcc":   "Malignant",
    "akiec": "Pre-Malignant",
    "bkl":   "Benign",
    "df":    "Benign",
    "nv":    "Benign",
    "vasc":  "Benign"
}

CLASSES = list(CLASS_NAMES.keys())

BASE_DIR = os.path.dirname(__file__)
MODEL_KERAS = os.path.join(BASE_DIR, "skin_lesion_model.keras")
MODEL_H5    = os.path.join(BASE_DIR, "skin_lesion_model.h5")

# ---------------- Load Model ----------------
@st.cache_resource
def get_model():
    # Try .keras first (best for Keras 3), then fallback to .h5
    if os.path.exists(MODEL_KERAS):
        return keras.saving.load_model(MODEL_KERAS, compile=False)
    if os.path.exists(MODEL_H5):
        return keras.saving.load_model(MODEL_H5, compile=False)
    raise FileNotFoundError("No model file found. Put skin_lesion_model.keras or skin_lesion_model.h5 next to app.py")

def get_input_size(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    h = int(shape[1])
    w = int(shape[2])
    return h, w

def preprocess_pil(pil_img: Image.Image, size_hw):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (size_hw[1], size_hw[0]))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- UI ----------------
st.title("🩺 Skin Lesion Classification System")
st.write("Upload a dermoscopic image to predict the skin lesion type.")
st.caption("⚠ This system is for educational purposes only and not a medical diagnosis.")

# Load model
try:
    model = get_model()
except Exception as e:
    st.error("Model could not be loaded.")
    st.write("✅ Fix checklist:")
    st.write("- Ensure the model file is in the same folder as app.py")
    st.write("- Ensure requirements.txt matches the deployment stack (TF 2.16.1 + Keras 3.3.3)")
    st.code(f"Looking for:\n{MODEL_KERAS}\n{MODEL_H5}")
    st.exception(e)
    st.stop()

# Input size
h, w = get_input_size(model)

uploaded = st.file_uploader("Upload Image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        pil_img = Image.open(uploaded)
    except Exception as e:
        st.error("Could not read the uploaded image. Please upload a valid JPG/PNG.")
        st.exception(e)
        st.stop()

    st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        x = preprocess_pil(pil_img, (h, w))
        probs = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(probs))
    predicted_code = CLASSES[idx]
    predicted_name = CLASS_NAMES[predicted_code]
    confidence = float(np.max(probs))
    clinical_type = CATEGORY[predicted_code]

    st.subheader("Prediction Result")

    if clinical_type == "Malignant":
        st.error(f"Clinical Assessment: **{clinical_type} Lesion**")
    elif clinical_type == "Pre-Malignant":
        st.warning(f"Clinical Assessment: **{clinical_type} Lesion**")
    else:
        st.success(f"Clinical Assessment: **{clinical_type} Lesion**")

    st.success(f"Detected Lesion Type: **{predicted_name}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

    st.subheader("Detailed Probabilities")
    for code, p in sorted(zip(CLASSES, probs), key=lambda t: t[1], reverse=True):
        st.write(f"{CLASS_NAMES[code]} ({CATEGORY[code]}) : {float(p) * 100:.2f}%")