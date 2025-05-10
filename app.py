import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model
# Load the CNN model
cnn_model = load_model('cnn_model.h5')
# Load the individual models (KNN, SVM, Random Forest)
knn_model = joblib.load('knn_model.pkl')
svm_model = joblib.load('svm_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# Function to extract features using CNN model and predict tumor
def predict_tumor(img_array):
    # Extract features using CNN model
    cnn_features = cnn_model.predict(img_array)

    # Flatten the CNN features for traditional ML models (KNN, SVM, etc.)
    flat_img_array = cnn_features.flatten().reshape(1, -1)

    # Predict using each model (KNN, SVM, Random Forest)
    knn_pred = knn_model.predict(flat_img_array)
    svm_pred = svm_model.predict(flat_img_array)
    rf_pred = rf_model.predict(flat_img_array)

    # Voting logic: return the majority class
    prediction = np.round((knn_pred + svm_pred + rf_pred) / 3)
    return prediction

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")
st.title("🧠 Brain Tumor Detector")
st.write("Upload an MRI scan, and the model will predict if a tumor is present.")

# File uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")  # Ensure image is in RGB mode
    st.image(img, caption='🖼 Uploaded Image', use_container_width=True)

    # Preprocess the image
    img = img.resize((150, 150))  # Resize to the input size expected by the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    # Call the predict_tumor function to get the prediction
    prediction = predict_tumor(img_array)
    result = "✅ Tumor Detected!" if prediction[0] == 1 else "❌ No Tumor Detected."
    
    # Show result
    st.subheader("Prediction Result:")
    st.success(result)  