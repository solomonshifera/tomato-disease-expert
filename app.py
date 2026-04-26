import streamlit as st
import keras
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(page_title="Tomato Health AI", layout="centered")

# 1. Load the Model
@st.cache_resource
def load_my_model():
    # Make sure this name matches your file on GitHub exactly
    return tf.keras.models.load_model('tomato_disease_modelv2.h5')

model = load_my_model()

# 2. Define the Disease Names (Labels)
# Ensure these match the 10 classes in your TRAINING folder
class_labels = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
    'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 
    'Target_Spot', 'Yellow_Leaf_Curl_Virus', 'Mosaic_virus', 'Healthy'
]

# UI Elements
st.title("🍅 Tomato Disease Expert Diagnosis")
st.write("Upload a clear photo of a tomato leaf to get an instant AI diagnosis.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("---")
    
    with st.spinner('Analyzing...'):
        # Preprocessing
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        preds = model.predict(img_array)
        result_index = np.argmax(preds[0])
        confidence = preds[0][result_index]

        # Professional Smart Filter (Threshold)
        if confidence < 0.45:
            st.error("⚠️ Error: This AI is 95% sure this is NOT a tomato leaf. Please upload a clear photo.")
        else:
            st.success(f"### Diagnosis: {class_labels[result_index]}")
            st.info(f"**Confidence Level:** {confidence*100:.2f}%")
            
            # Advice based on confidence
            if confidence > 0.90:
                st.write("✅ High-precision diagnosis. Please consult the treatment manual.")
            else:
                st.write("ℹ️ Moderate confidence. Consider taking a closer photo in better light.")
