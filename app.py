import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # For loading Keras model
from PIL import Image
import numpy as np
import json
import pandas as pd
import os

# --- Call st.set_page_config() as the VERY FIRST Streamlit command ---
st.set_page_config(layout="wide", page_title="Pneumonia Detection (Custom CNN)")

# --- Configuration: Paths to your model and class names file ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "custom_cnn_pneumonia_model.keras" # Your custom CNN model file
CLASS_NAMES_FILENAME = "class_names_pneumonia.json"

MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
CLASS_NAMES_PATH = os.path.join(BASE_DIR, CLASS_NAMES_FILENAME)

# --- Parameters (MUST match your custom CNN training) ---
IMG_WIDTH = 150 # As used in your custom CNN training
IMG_HEIGHT = 150
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# --- Load Model and Class Names ---
@st.cache_resource
def load_custom_model_and_classes():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            return None, ["Error"]
        if not os.path.exists(CLASS_NAMES_PATH):
            st.error(f"Class names file not found at: {CLASS_NAMES_PATH}")
            return None, ["Error"]

        model = load_model(MODEL_PATH) # Standard Keras model loading
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        print("Custom CNN Model and class names loaded successfully.")
        return model, class_names
    except Exception as e:
        st.error(f"An error occurred while loading the model or class names: {e}")
        return None, [f"Error loading files: {e}"]

model, class_names = load_custom_model_and_classes()

# --- Preprocessing Function (MUST match your custom CNN training) ---
def preprocess_image_for_custom_cnn(image_pil):
    """
    Preprocesses a PIL Image for the custom CNN prediction.
    1. Resize to IMAGE_SIZE.
    2. Convert to RGB (if not already).
    3. Convert to NumPy array.
    4. Rescale pixel values to [0,1] (as done in training).
    5. Expand dimensions for batch.
    """
    img_resized = image_pil.resize(IMAGE_SIZE, Image.LANCZOS)

    if img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')

    img_array = np.array(img_resized)
    
    # Rescale pixel values (0-255 to 0-1)
    # This matches layers.Rescaling(1./255) used in training
    img_array_rescaled = img_array / 255.0
    
    img_batch = np.expand_dims(img_array_rescaled, axis=0)
    return img_batch


def predict_custom_cnn(image_pil):
    """
    Takes a PIL image, preprocesses it for the custom CNN, and returns
    the predicted class name and probabilities.
    """
    if model is None or (isinstance(class_names, list) and class_names and "Error" in class_names[0]):
        return "Error: Setup incomplete", {"Error": 1.0}

    try:
        preprocessed_image = preprocess_image_for_custom_cnn(image_pil)
        predictions_array = model.predict(preprocessed_image)[0] # Output is (1,1) for binary sigmoid

        # For binary classification with sigmoid, predictions_array[0] is the probability of the positive class
        # class_names[0] is 'NORMAL', class_names[1] is 'PNEUMONIA'
        # If prediction > 0.5, it's PNEUMONIA (class 1)
        
        prob_pneumonia = float(predictions_array[0])
        prob_normal = 1.0 - prob_pneumonia

        if prob_pneumonia > 0.5:
            predicted_class_name = class_names[1] # PNEUMONIA
        else:
            predicted_class_name = class_names[0] # NORMAL
            
        probabilities = {
            class_names[0]: prob_normal, # NORMAL
            class_names[1]: prob_pneumonia # PNEUMONIA
        }
        
        return predicted_class_name, probabilities
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "Error during prediction", {"Error": 1.0}

# --- Streamlit UI Design ---
# (Using the same nice CSS and layout from before, adapted slightly)

st.markdown("""
    <style>
    .main-header { font-size: 2.5em !important; font-weight: bold; color: #0072B2; text-align: center; padding: 20px 0 10px 0;}
    .sub-header { font-size: 1.2em; color: #56B4E9; text-align: center; margin-bottom: 30px;}
    .stFileUploader > label { font-size: 1.1em !important; }
    .stFileUploader > div > button { background-color: #0072B2; color: white; border-radius: 5px; padding: 8px 15px; }
    .stFileUploader > div > button:hover { background-color: #005C8A; }
    .prediction-text { font-size: 1.5em; font-weight: bold; margin-top: 20px; text-align: center; }
    .normal-prediction { color: #009E73; } /* Greenish Teal */
    .pneumonia-prediction { color: #D55E00; } /* Vermillion */
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='main-header'>🩺 Pneumonia Detection from Chest X-Rays (Custom CNN)</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>Upload a chest X-ray image to classify it as Normal or Pneumonia using a custom-trained CNN.</p>",
    unsafe_allow_html=True
)

st.sidebar.title("About This App")
st.sidebar.info(
    "This application uses a **Custom Convolutional Neural Network (CNN)** "
    "trained from scratch on the Chest X-Ray (Pneumonia) dataset from Kaggle. "
    "The model classifies images into 'NORMAL' or 'PNEUMONIA'."
)
st.sidebar.markdown("---")
st.sidebar.subheader("How to Use:")
st.sidebar.markdown("1. Click **'Browse files'** or drag & drop an X-ray image.")
st.sidebar.markdown("2. The model analyzes the image.")
st.sidebar.markdown("3. View the prediction and confidence scores.")
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ **Disclaimer:** For educational/demonstration purposes only. "
                   "**Not for medical diagnosis.** Always consult a qualified medical professional.")

# Main Content Area
if model is None or (isinstance(class_names, list) and class_names and "Error" in class_names[0]):
    st.error("Application setup failed: Model or class names could not be loaded. Please check the file paths and ensure the model was trained correctly.")
else:
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a chest X-ray."
    )

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)

        col1, col2 = st.columns([2, 3]) # Image column, Results column

        with col1:
            st.image(image_pil, caption="Uploaded X-Ray", use_column_width='always')

        with col2:
            with st.spinner("🩻 Analyzing X-Ray... Please wait."):
                predicted_class_name, probabilities = predict_custom_cnn(image_pil)

            if "Error" not in predicted_class_name:
                prediction_color_class = "normal-prediction" if predicted_class_name == "NORMAL" else "pneumonia-prediction"
                
                st.markdown(
                    f"<p class='prediction-text {prediction_color_class}'>Prediction: {predicted_class_name}</p>",
                    unsafe_allow_html=True
                )
                
                st.subheader("Confidence Scores:")
                probs_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
                probs_df = probs_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)
                
                st.table(probs_df.style.format({"Probability": "{:.2%}"})
                                     .highlight_max(subset=['Probability'], color='#ADEBAD' if predicted_class_name == "NORMAL" else '#FFB6C1', axis=0) # Light green for normal, light red for pneumonia
                                     .set_properties(**{'width': '150px'}))

                st.subheader("Probabilities Chart:")
                chart_data = pd.DataFrame.from_dict(probabilities, orient='index', columns=['Probability'])
                st.bar_chart(chart_data, height=250)

                if predicted_class_name == "NORMAL":
                    st.success("The model suggests the X-ray appears NORMAL.")
                else:
                    st.error(f"The model suggests signs of PNEUMONIA. " # Using st.error for more visual alarm
                               "This is not a diagnosis. Please consult a medical expert.")
            else:
                # Error message already displayed by predict_custom_cnn
                pass 
    else:
        st.info("👈 Upload an image to begin classification.")

st.markdown("---")
st.markdown("<p style='text-align:center; color:grey; font-size:0.9em;'>"
            "Custom CNN Chest X-Ray Classifier | For Educational Purposes"
            "</p>", unsafe_allow_html=True)